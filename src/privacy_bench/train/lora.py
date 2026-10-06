import argparse
import json
import math
import os
import platform
import random
import time
from pathlib import Path

import torch
import transformers
import yaml
from packaging import version
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from .citations import aggregate, score_citation
from .data import (
    LORA_TARGETS,
    OVERRIDES,
    apply_overrides,
    dev_rows,
    flat_defaults,
    join_example,
    pad_batch,
    read_jsonl,
    sft_rows,
    text_hash,
)


def find_bundle(explicit: str | None) -> Path:
    candidates = [explicit, os.environ.get("PALR_BUNDLE")]
    for c in candidates:
        if c and (Path(c) / "bundle_manifest.json").exists():
            return Path(c)
    root = Path("/kaggle/input")
    if root.exists():
        for manifest in sorted(root.rglob("bundle_manifest.json")):
            return manifest.parent
    raise FileNotFoundError("bundle_manifest.json not found; pass --bundle or set PALR_BUNDLE")


def parse_sets(pairs: list[str]) -> dict:
    out = {}
    for pair in pairs:
        key, _, raw = pair.partition("=")
        if key not in OVERRIDES:
            raise KeyError(f"unknown override key: {key}")
        out[key] = yaml.safe_load(raw)
    return out


def environment() -> dict:
    import peft

    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "gpu_count": torch.cuda.device_count(),
    }
    return info


def load_model(cfg: dict, model_override: str | None, device: str):
    bm = cfg["base_model"]
    name = model_override or bm["name"]
    revision = None if model_override else bm["revision"]
    dtype = getattr(torch, bm["dtype"]) if device == "cuda" else torch.float32
    dtype_key = "dtype" if version.parse(transformers.__version__) >= version.parse("4.56") else "torch_dtype"
    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        name, revision=revision, attn_implementation=bm["attn_implementation"], **{dtype_key: dtype}
    )
    model.to(device)
    resolved = getattr(model.config, "_commit_hash", None)
    return tokenizer, model, dtype, resolved


def add_lora(model, cfg: dict, dtype):
    from peft import LoraConfig, get_peft_model

    lc = cfg["lora"]
    if cfg["train"]["gradient_checkpointing"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model.config.use_cache = False
    config = LoraConfig(
        r=lc["r"],
        lora_alpha=lc["r"] * lc["alpha_multiplier"],
        lora_dropout=lc["dropout"],
        target_modules=LORA_TARGETS[lc["target"]],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    if dtype == torch.float16:
        for p in model.parameters():
            if p.requires_grad:
                p.data = p.data.float()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return model, trainable, total


def prompt_text(tokenizer, msgs: list[dict]) -> str:
    return tokenizer.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)


def encode(tokenizer, rows: list[dict], max_length: int) -> tuple[list[dict], int]:
    examples, dropped = [], 0
    for row in rows:
        prompt = prompt_text(tokenizer, row["messages"])
        full = tokenizer.apply_chat_template(row["messages"], tokenize=False)
        if not full.startswith(prompt):
            raise ValueError("chat template prompt is not a prefix of the full conversation")
        p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        c_ids = tokenizer(full[len(prompt):], add_special_tokens=False)["input_ids"]
        ex = join_example(p_ids, c_ids, max_length)
        if ex is None:
            dropped += 1
        else:
            examples.append(ex)
    return examples, dropped


def to_tensors(batch: dict, device: str) -> dict:
    return {k: torch.tensor(v, dtype=torch.long, device=device) for k, v in batch.items()}


def autocast(device: str, enabled: bool):
    return torch.autocast(device_type=device, dtype=torch.float16, enabled=enabled)


@torch.no_grad()
def dev_loss(model, examples, pad_id, batch_size, device, fp16) -> float:
    model.eval()
    total, count = 0.0, 0
    for i in range(0, len(examples), batch_size):
        batch = to_tensors(pad_batch(examples[i:i + batch_size], pad_id), device)
        with autocast(device, fp16):
            out = model(**batch)
        n = int((batch["labels"][:, 1:] != -100).sum())
        total += float(out.loss) * n
        count += n
    model.train()
    return total / max(count, 1)


@torch.no_grad()
def generate(model, tokenizer, rows, cfg, device, fp16) -> tuple[list[dict], dict]:
    ec = cfg["eval"]
    model.eval()
    use_cache = model.config.use_cache
    model.config.use_cache = True
    tokenizer.padding_side = "left"
    eos = [tokenizer.eos_token_id]
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end, int) and im_end != tokenizer.unk_token_id:
        eos.append(im_end)
    stops = set(eos) | {tokenizer.pad_token_id}
    outputs, new_tokens = [], 0
    start = time.perf_counter()
    for i in range(0, len(rows), ec["gen_batch_size"]):
        chunk = rows[i:i + ec["gen_batch_size"]]
        texts = [prompt_text(tokenizer, r["messages"]) for r in chunk]
        enc = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        with autocast(device, fp16):
            out = model.generate(
                **enc,
                max_new_tokens=ec["max_new_tokens"],
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=sorted(set(eos)),
                use_cache=True,
            )
        gen = out[:, enc["input_ids"].shape[1]:].tolist()
        for r, ids in zip(chunk, gen):
            n = next((k + 1 for k, t in enumerate(ids) if t in stops), len(ids))
            new_tokens += n
            ids = ids[:n]
            outputs.append({"row": r, "prediction": tokenizer.decode(ids, skip_special_tokens=True).strip(), "new_tokens": n})
    seconds = time.perf_counter() - start
    tokenizer.padding_side = "right"
    model.config.use_cache = use_cache
    model.train()
    return outputs, {"gen_seconds": seconds, "mean_new_tokens": new_tokens / max(len(outputs), 1)}


def score_predictions(outputs: list[dict], valid: set[str]) -> tuple[list[dict], dict]:
    records = []
    for o in outputs:
        r = o["row"]
        s = score_citation(o["prediction"], r["uid"], valid)
        records.append({
            "eval_id": r["eval_id"],
            "fact_id": r["fact_id"],
            "uid": r["uid"],
            "question": r["question"],
            "reference_answer": r["reference_answer"],
            "prediction": o["prediction"],
            "new_tokens": o["new_tokens"],
            **s,
        })
    return records, aggregate([{k: rec[k] for k in ("hit", "tag_present", "has_nonexistent", "wrong_existing")} for rec in records])


def evaluate(model, tokenizer, dev, dev_examples, cfg, valid, device, fp16, run, step, prefix="dev") -> tuple[dict, list[dict]]:
    loss = dev_loss(model, dev_examples, tokenizer.pad_token_id, cfg["eval"]["loss_batch_size"], device, fp16)
    outputs, timing = generate(model, tokenizer, dev, cfg, device, fp16)
    records, metrics = score_predictions(outputs, valid)
    metrics = {f"{prefix}/{k}": v for k, v in {**metrics, **timing, "loss": loss}.items()}
    run.log(metrics, step=step)
    return metrics, records


def train(model, tokenizer, examples, dev, dev_examples, cfg, valid, device, fp16, run) -> int:
    tc = cfg["train"]
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=tc["learning_rate"], weight_decay=tc["weight_decay"])
    micro_per_epoch = math.ceil(len(examples) / tc["batch_size"])
    steps_per_epoch = math.ceil(micro_per_epoch / tc["grad_accum"])
    total_steps = steps_per_epoch * tc["epochs"]
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(round(tc["warmup_ratio"] * total_steps)), total_steps)
    scaler = torch.amp.GradScaler(device, enabled=fp16)
    run.config.update({"steps_per_epoch": steps_per_epoch, "total_steps": total_steps}, allow_val_change=True)
    model.train()
    step, window, nonfinite, skipped = 0, [], 0, 0
    start = time.perf_counter()
    for epoch in range(tc["epochs"]):
        order = list(range(len(examples)))
        random.Random(tc["seed"] + epoch).shuffle(order)
        batches = [order[i:i + tc["batch_size"]] for i in range(0, len(order), tc["batch_size"])]
        for i, idx in enumerate(batches):
            batch = to_tensors(pad_batch([examples[j] for j in idx], tokenizer.pad_token_id), device)
            with autocast(device, fp16):
                loss = model(**batch).loss
            if not torch.isfinite(loss):
                nonfinite += 1
            window.append(loss.detach().item())
            scaler.scale(loss / tc["grad_accum"]).backward()
            if (i + 1) % tc["grad_accum"] == 0 or i + 1 == len(batches):
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(params, tc["max_grad_norm"])
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if fp16 and scaler.get_scale() < scale_before:
                    skipped += 1
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                step += 1
                if step % tc["log_every"] == 0 or step == total_steps:
                    run.log({
                        "train/loss": sum(window) / len(window),
                        "train/lr": scheduler.get_last_lr()[0],
                        "train/grad_norm": float(grad_norm),
                        "train/loss_scale": scaler.get_scale() if fp16 else 1.0,
                        "train/epoch": epoch + (i + 1) / len(batches),
                        "train/nonfinite_losses": nonfinite,
                        "train/skipped_steps": skipped,
                    }, step=step)
                    window = []
        epoch_metrics = {"dev/loss": dev_loss(model, dev_examples, tokenizer.pad_token_id, cfg["eval"]["loss_batch_size"], device, fp16),
                         "train/epoch": epoch + 1}
        run.log(epoch_metrics, step=step)
        if cfg["eval"]["generate_every_epoch"] and epoch + 1 < tc["epochs"]:
            evaluate(model, tokenizer, dev, dev_examples, cfg, valid, device, fp16, run, step, prefix=f"dev_epoch{epoch + 1}")
    run.summary.update({"train/seconds": time.perf_counter() - start, "train/nonfinite_losses": nonfinite, "train/skipped_steps": skipped})
    return step


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA fine-tuning of the base model with dev-para evaluation")
    parser.add_argument("--bundle")
    parser.add_argument("--model", help="local model path that replaces the configured base model, for smoke tests")
    parser.add_argument("--output-dir")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    parser.add_argument("--base-only", action="store_true", help="evaluate the base model on dev-para without training")
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-dev", type=int)
    parser.add_argument("--name")
    parser.add_argument("--tags", nargs="*", default=[])
    args = parser.parse_args()

    import wandb

    bundle = find_bundle(args.bundle)
    cfg = yaml.safe_load((bundle / "train.yaml").read_text(encoding="utf-8"))
    manifest = json.loads((bundle / "bundle_manifest.json").read_text(encoding="utf-8"))
    flat = {**flat_defaults(cfg), **parse_sets(args.set)}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fp16 = device == "cuda"

    run = wandb.init(
        project=cfg["wandb"]["project"],
        job_type="eval-base" if args.base_only else cfg["wandb"]["job_type"],
        name=args.name,
        tags=args.tags + (["base"] if args.base_only else ["lora"]),
        config={
            **flat,
            "base_only": args.base_only,
            "base_model": args.model or cfg["base_model"]["name"],
            "run_version": cfg["run_version"],
            "system_prompt_hash": text_hash(cfg["system_prompt"]),
            "bundle_git_commit": manifest["git"]["commit"],
            "bundle_git_dirty": manifest["git"]["dirty"],
            "limit_train": args.limit_train,
            "limit_dev": args.limit_dev,
            "env": environment(),
        },
    )
    cfg = apply_overrides(cfg, {k: run.config[k] for k in OVERRIDES})
    transformers.set_seed(cfg["train"]["seed"])

    tokenizer, model, dtype, resolved = load_model(cfg, args.model, device)
    tokenizer.padding_side = "right"
    run.config.update({"base_model_commit": resolved}, allow_val_change=True)
    valid = {p["uid"] for p in json.loads((bundle / "data" / "provisions.json").read_text(encoding="utf-8"))}

    facts = read_jsonl(bundle / "data" / "train.jsonl")[: args.limit_train]
    dev = dev_rows(read_jsonl(bundle / "data" / "dev_para.jsonl"), cfg["system_prompt"])[: args.limit_dev]
    dev_examples, _ = encode(tokenizer, dev, cfg["data"]["max_length"])

    step = 0
    if not args.base_only:
        rows = sft_rows(facts, cfg["system_prompt"], cfg["data"]["phrasings"])
        examples, dropped = encode(tokenizer, rows, cfg["data"]["max_length"])
        model, trainable, total = add_lora(model, cfg, dtype)
        run.config.update({
            "train_examples": len(examples),
            "train_dropped_over_length": dropped,
            "trainable_params": trainable,
            "total_params": total,
        }, allow_val_change=True)
        step = train(model, tokenizer, examples, dev, dev_examples, cfg, valid, device, fp16, run)

    metrics, records = evaluate(model, tokenizer, dev, dev_examples, cfg, valid, device, fp16, run, step)
    run.summary.update(metrics)

    out_dir = Path(args.output_dir or cfg["output"]["dir"]) / (run.name or run.id)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "dev_predictions.jsonl", "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    (out_dir / "resolved_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    columns = ["eval_id", "uid", "question", "reference_answer", "prediction", "cited", "hit", "has_nonexistent"]
    table = wandb.Table(columns=columns)
    for rec in records:
        table.add_data(*[json.dumps(rec[c]) if isinstance(rec[c], list) else rec[c] for c in columns])
    run.log({"dev/predictions": table}, step=step)

    if not args.base_only and cfg["output"]["save_adapter"]:
        model.save_pretrained(out_dir / "adapter")
        if cfg["output"]["log_adapter_artifact"]:
            artifact = wandb.Artifact(f"adapter-{run.id}", type="model", metadata={**metrics, "config": cfg})
            artifact.add_dir(str(out_dir / "adapter"))
            run.log_artifact(artifact)
    print(json.dumps(metrics, indent=2))
    run.finish()


if __name__ == "__main__":
    main()
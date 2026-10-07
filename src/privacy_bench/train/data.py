import copy
import hashlib
import json
import random
from pathlib import Path

LORA_TARGETS = {
    "attention": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "all_linear": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
}

OVERRIDES = {
    "learning_rate": ("train", "learning_rate"),
    "epochs": ("train", "epochs"),
    "batch_size": ("train", "batch_size"),
    "grad_accum": ("train", "grad_accum"),
    "warmup_ratio": ("train", "warmup_ratio"),
    "weight_decay": ("train", "weight_decay"),
    "seed": ("train", "seed"),
    "lora_r": ("lora", "r"),
    "lora_alpha_multiplier": ("lora", "alpha_multiplier"),
    "lora_dropout": ("lora", "dropout"),
    "lora_target": ("lora", "target"),
    "phrasings": ("data", "phrasings"),
    "max_length": ("data", "max_length"),
    "base_dtype": ("base_model", "dtype"),
}


def read_jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def flat_defaults(cfg: dict) -> dict:
    return {key: cfg[section][field] for key, (section, field) in OVERRIDES.items()}

def coerce(value, default):
    if isinstance(default, bool) or default is None:
        return value
    if isinstance(default, float):
        return float(value)
    if isinstance(default, int):
        return int(value)
    return value

def apply_overrides(cfg: dict, flat: dict) -> dict:
    out = copy.deepcopy(cfg)
    unknown = sorted(set(flat) - set(OVERRIDES))
    if unknown:
        raise KeyError(f"unknown override keys: {unknown}")
    for key, value in flat.items():
        section, field = OVERRIDES[key]
        out[section][field] = coerce(value, cfg[section][field])
    if out["lora"]["target"] not in LORA_TARGETS:
        raise ValueError(f"unknown lora target: {out['lora']['target']}")
    if out["base_model"]["dtype"] not in ("float16", "float32"):
        raise ValueError("base_dtype must be float16 or float32")
    if out["data"]["phrasings"] not in (1, 2):
        raise ValueError("phrasings must be 1 or 2")
    return out


def messages(system_prompt: str, question: str, answer: str | None = None) -> list[dict]:
    msgs = [{"role": "system", "content": system_prompt}, {"role": "user", "content": question}]
    if answer is not None:
        msgs.append({"role": "assistant", "content": answer})
    return msgs


def sft_rows(facts: list[dict], system_prompt: str, phrasings: int) -> list[dict]:
    rows = []
    for fact in facts:
        for idx, question in enumerate(fact["questions"][:phrasings]):
            rows.append({
                "fact_id": fact["fact_id"],
                "uid": fact["uid"],
                "phrasing": idx,
                "messages": messages(system_prompt, question, fact["answer"]),
            })
    return rows


def dev_rows(items: list[dict], system_prompt: str) -> list[dict]:
    return [
        {
            "eval_id": item["eval_id"],
            "fact_id": item["fact_id"],
            "uid": item["uid"],
            "display": item["display"],
            "question": item["question"],
            "reference_answer": item["reference_answer"],
            "messages": messages(system_prompt, item["question"], item["reference_answer"]),
        }
        for item in items
    ]


def probe_items(facts: list[dict], exclude_fact_ids: set[str], size: int, seed: int) -> list[dict]:
    pool = sorted((f for f in facts if f["fact_id"] not in exclude_fact_ids), key=lambda f: f["fact_id"])
    chosen = sorted(random.Random(seed).sample(pool, min(size, len(pool))), key=lambda f: f["fact_id"])
    return [
        {
            "eval_id": f"probe-{i:04d}",
            "fact_id": f["fact_id"],
            "uid": f["uid"],
            "display": f["display"],
            "question": f["questions"][0],
            "reference_answer": f["answer"],
        }
        for i, f in enumerate(chosen, 1)
    ]


def join_example(prompt_ids: list[int], completion_ids: list[int], max_length: int) -> dict | None:
    if len(prompt_ids) + len(completion_ids) > max_length:
        return None
    return {
        "input_ids": prompt_ids + completion_ids,
        "labels": [-100] * len(prompt_ids) + completion_ids,
    }


def pad_batch(examples: list[dict], pad_id: int) -> dict:
    width = max(len(e["input_ids"]) for e in examples)
    batch = {"input_ids": [], "attention_mask": [], "labels": []}
    for e in examples:
        gap = width - len(e["input_ids"])
        batch["input_ids"].append(e["input_ids"] + [pad_id] * gap)
        batch["attention_mask"].append([1] * len(e["input_ids"]) + [0] * gap)
        batch["labels"].append(e["labels"] + [-100] * gap)
    return batch


def provision_index(sections: list[dict]) -> list[dict]:
    return [{"uid": s["uid"], "display": s["display"]} for s in sections]
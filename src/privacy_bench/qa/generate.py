import argparse
import json
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import yaml

from ..data.build_split import allocate, compile_leak_rules, read_jsonl, write_jsonl
from ..data.text_norm import match_key
from .prompts import SYSTEM_PROMPT, build_output_schema, build_user_prompt, prompt_hash
from .validate import check_item

PLACEHOLDER = "REPLACE_WITH"
OUTPUT_FILES = {"train": "train_facts.jsonl", "holdout_h1": "h1_questions.jsonl", "holdout_h2": "h2_questions.jsonl"}


@dataclass
class Job:
    key: str
    row: dict
    chunk: int
    n_chunks: int
    text: str
    quota: int
    n_questions: int
    citation: str


def chunk_text(text: str, max_chars: int) -> list[str]:
    chunks, current, size = [], [], 0
    for para in text.split("\n"):
        if current and size + len(para) + 1 > max_chars:
            chunks.append("\n".join(current))
            current, size = [], 0
        current.append(para)
        size += len(para) + 1
    if current:
        chunks.append("\n".join(current))
    return chunks


def build_jobs(rows: list[dict], manifest: dict, cfg: dict) -> list[Job]:
    qa = manifest["qa"]
    jobs = []
    for row in rows:
        chunks = chunk_text(row["text"], cfg["run"]["chunk_max_chars"])
        minimum = 1 if row["quota"] >= len(chunks) else 0
        quotas = allocate([float(len(c)) for c in chunks], row["quota"], minimum)
        for i, (chunk, quota) in enumerate(zip(chunks, quotas)):
            if quota == 0:
                continue
            jobs.append(
                Job(
                    key=f"{row['uid'].replace(':', '_')}__{i:02d}",
                    row=row,
                    chunk=i,
                    n_chunks=len(chunks),
                    text=chunk,
                    quota=quota,
                    n_questions=qa["train_phrasings_per_fact"] if row["split"] == "train" else 1,
                    citation=qa["citation_format"].format(display=row["display"]),
                )
            )
    return jobs


def call_model(client, cfg: dict, schema: dict, job: Job, n: int, covered: list[str]) -> tuple[list[dict], dict]:
    gen = cfg["generator"]
    output_config = {"format": {"type": "json_schema", "schema": schema}}
    if gen.get("effort"):
        output_config["effort"] = gen["effort"]
    resp = client.messages.create(
        model=gen["model"],
        max_tokens=gen["max_tokens"],
        system=SYSTEM_PROMPT,
        output_config=output_config,
        messages=[{"role": "user", "content": build_user_prompt(job, n, covered)}],
    )
    text = "".join(b.text for b in resp.content if b.type == "text")
    facts, parse_error = [], None
    if resp.stop_reason == "max_tokens":
        parse_error = "truncated"
    else:
        try:
            facts = json.loads(text).get("facts", [])
        except (json.JSONDecodeError, AttributeError) as exc:
            parse_error = type(exc).__name__
    usage = {
        "model": resp.model,
        "input_tokens": resp.usage.input_tokens,
        "output_tokens": resp.usage.output_tokens,
        "stop_reason": resp.stop_reason,
        "parse_error": parse_error,
    }
    return facts, usage


def run_job(job: Job, client, cfg: dict, manifest: dict, leak_rules, cache_dir: Path) -> dict:
    path = cache_dir / f"{job.key}.json"
    current = prompt_hash()
    state = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    if state is None or state.get("prompt_hash") != current:
        state = {"key": job.key, "prompt_hash": current, "accepted": [], "rejected": [], "calls": []}

    schema = build_output_schema(manifest["qa"]["question_types"], job.n_questions)
    for _ in range(cfg["run"]["max_rounds"]):
        deficit = job.quota - len(state["accepted"])
        if deficit <= 0:
            break
        n = min(deficit, cfg["run"]["max_items_per_call"])
        covered = [f["statement"] for f in state["accepted"]]
        facts, usage = call_model(client, cfg, schema, job, n, covered)
        state["calls"].append(
            {**usage, "requested": n, "returned": len(facts), "time": time.strftime("%Y-%m-%dT%H:%M:%S")})
        seen = {match_key(s) for s in covered}
        for item in facts:
            errors = check_item(
                item,
                source_text=job.text,
                citation=job.citation,
                n_questions=job.n_questions,
                question_types=manifest["qa"]["question_types"],
                leak_rules=leak_rules,
            )
            if not errors and match_key(item["statement"]) in seen:
                errors = ["duplicate_statement"]
            if errors:
                state["rejected"].append({**item, "errors": errors})
            elif len(state["accepted"]) < job.quota:
                state["accepted"].append(item)
                seen.add(match_key(item["statement"]))
        path.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")
    return state


def assemble(jobs: list[Job], states: list[dict], meta: dict) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {split: [] for split in OUTPUT_FILES}
    for job, state in zip(jobs, states):
        for n, item in enumerate(state["accepted"], 1):
            out[job.row["split"]].append(
                {
                    "fact_id": f"{job.row['uid']}#{job.chunk:02d}-{n:03d}",
                    "uid": job.row["uid"],
                    "display": job.row["display"],
                    "unit": job.row["unit"],
                    "split": job.row["split"],
                    "question_type": item["question_type"],
                    "statement": item["statement"],
                    "questions": item["questions"],
                    "answer": item["answer"],
                    "evidence": item["evidence"],
                    **meta,
                }
            )
    return out


def summarise(jobs: list[Job], states: list[dict], meta: dict) -> dict:
    requested, accepted = Counter(), Counter()
    reasons, tokens = Counter(), Counter()
    shortfall = []
    for job, state in zip(jobs, states):
        split = job.row["split"]
        requested[split] += job.quota
        accepted[split] += len(state["accepted"])
        if len(state["accepted"]) < job.quota:
            shortfall.append({"key": job.key, "quota": job.quota, "accepted": len(state["accepted"])})
        for item in state["rejected"]:
            reasons.update(item["errors"])
        for call in state["calls"]:
            tokens["input"] += call["input_tokens"]
            tokens["output"] += call["output_tokens"]
            tokens["calls"] += 1
    return {
        **meta,
        "jobs": len(jobs),
        "requested": dict(requested),
        "accepted": dict(accepted),
        "rejected_by_reason": dict(reasons),
        "shortfall": shortfall,
        "usage": dict(tokens),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Generate QA items from masked Act provisions with an LLM.")
    ap.add_argument("--config", type=Path, default=Path("configs/qa_generation.yaml"))
    ap.add_argument("--split", choices=["all", *OUTPUT_FILES], default="all")
    ap.add_argument("--uids", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    paths = {k: Path(v) for k, v in cfg["paths"].items()}
    manifest = yaml.safe_load(paths["manifest"].read_text(encoding="utf-8"))
    rows = read_jsonl(paths["inputs"])
    if args.split != "all":
        rows = [r for r in rows if r["split"] == args.split]
    if args.uids:
        wanted = set(args.uids.split(","))
        rows = [r for r in rows if r["uid"] in wanted]
    jobs = build_jobs(rows, manifest, cfg)
    if args.limit:
        jobs = jobs[: args.limit]

    if args.dry_run:
        quota = Counter()
        for job in jobs:
            quota[job.row["split"]] += job.quota
        print(f"jobs={len(jobs)} quota={dict(quota)} max_job_quota={max(j.quota for j in jobs)} "
              f"max_chunk_chars={max(len(j.text) for j in jobs)} prompt_hash={prompt_hash()}")
        return 0

    if str(cfg["generator"]["model"]).startswith(PLACEHOLDER):
        raise SystemExit("Set generator.model in the config to a current Anthropic model ID before running.")

    import anthropic
    from dotenv import load_dotenv

    load_dotenv()
    client = anthropic.Anthropic(max_retries=cfg["generator"]["max_retries"])
    leak_rules = compile_leak_rules(manifest)
    false_positive = {u for u, e in manifest["leakage"].get("source_review", {}).items() if
                      e["action"] == "false_positive"}
    paths["cache_dir"].mkdir(parents=True, exist_ok=True)

    states: list[dict | None] = [None] * len(jobs)
    with ThreadPoolExecutor(max_workers=cfg["run"]["max_workers"]) as pool:
        futures = {}
        for i, job in enumerate(jobs):
            rules = leak_rules if job.row["split"] == "train" and job.row["uid"] not in false_positive else None
            futures[pool.submit(run_job, job, client, cfg, manifest, rules, paths["cache_dir"])] = i
        for done, fut in enumerate(as_completed(futures), 1):
            i = futures[fut]
            states[i] = fut.result()
            print(f"[{done}/{len(jobs)}] {jobs[i].key}: {len(states[i]['accepted'])}/{jobs[i].quota}", flush=True)

    meta = {
        "generator_model": cfg["generator"]["model"],
        "generator_effort": cfg["generator"].get("effort"),
        "prompt_version": cfg["prompt_version"],
        "prompt_hash": prompt_hash(),
    }
    for split, items in assemble(jobs, states, meta).items():
        if items:
            write_jsonl(paths["out_dir"] / OUTPUT_FILES[split], items)
    rejects = [{"key": j.key, **r} for j, s in zip(jobs, states) for r in s["rejected"]]
    write_jsonl(paths["rejects"], rejects)
    report = summarise(jobs, states, meta)
    paths["report"].parent.mkdir(parents=True, exist_ok=True)
    paths["report"].write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"accepted={report['accepted']} requested={report['requested']} shortfall_jobs={len(report['shortfall'])} usage={report['usage']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

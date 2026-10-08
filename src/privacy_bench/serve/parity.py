import argparse
import json
import statistics
import time
import urllib.request
from pathlib import Path

import yaml

from ..train.citations import aggregate, score_citation
from ..train.data import read_jsonl, sha256_file, write_jsonl


def chat(server: str, system_prompt: str, question: str, max_tokens: int, timeout: float) -> dict:
    body = {
        "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": question}],
        "temperature": 0,
        "top_k": 1,
        "max_tokens": max_tokens,
        "seed": 0,
    }
    req = urllib.request.Request(
        f"{server.rstrip('/')}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    elapsed = time.perf_counter() - start
    timings = payload.get("timings", {})
    return {
        "prediction": payload["choices"][0]["message"]["content"].strip(),
        "completion_tokens": payload.get("usage", {}).get("completion_tokens"),
        "total_seconds": elapsed,
        "prompt_ms": timings.get("prompt_ms"),
        "predicted_ms": timings.get("predicted_ms"),
    }


def compare(reference: list[dict], candidate: list[dict]) -> dict:
    ref = {r["eval_id"]: r for r in reference}
    cand = {c["eval_id"]: c for c in candidate}
    if set(ref) != set(cand):
        raise ValueError("reference and candidate cover different items")
    ids = sorted(ref)
    ref_hit = sum(ref[i]["hit"] for i in ids) / len(ids)
    cand_hit = sum(cand[i]["hit"] for i in ids) / len(ids)
    return {
        "n": len(ids),
        "reference_citation_hit": ref_hit,
        "candidate_citation_hit": cand_hit,
        "hit_diff": cand_hit - ref_hit,
        "hit_agreement": sum(ref[i]["hit"] == cand[i]["hit"] for i in ids) / len(ids),
        "exact_text_match": sum(ref[i]["prediction"].strip() == cand[i]["prediction"].strip() for i in ids) / len(ids),
        "flipped_to_miss": sorted(i for i in ids if ref[i]["hit"] and not cand[i]["hit"]),
        "flipped_to_hit": sorted(i for i in ids if cand[i]["hit"] and not ref[i]["hit"]),
    }


def gate(metrics: dict, max_hit_diff: float, min_agreement: float) -> dict:
    checks = {
        "hit_diff_within_limit": abs(metrics["hit_diff"]) <= max_hit_diff,
        "agreement_above_limit": metrics["hit_agreement"] >= min_agreement,
    }
    return {"passed": all(checks.values()), "checks": checks, "max_hit_diff": max_hit_diff, "min_agreement": min_agreement}


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, round(q * (len(ordered) - 1))))
    return ordered[k]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare GGUF outputs served by llama.cpp with the PEFT model's dev-para outputs")
    parser.add_argument("--model", default="a1", choices=["a1"])
    parser.add_argument("--config", default="configs/serve.yaml")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    pc = cfg["parity"]
    mc = cfg["models"][args.model]
    train_cfg = yaml.safe_load(Path(pc["train_config"]).read_text(encoding="utf-8"))
    system_prompt = train_cfg["system_prompt"]
    max_tokens = train_cfg["eval"]["max_new_tokens"]
    valid = {s["uid"] for s in read_jsonl(pc["sections"])}

    dev = read_jsonl(pc["dev"])[: args.limit]
    reference = [r for r in read_jsonl(mc["reference_predictions"]) if r["eval_id"] in {d["eval_id"] for d in dev}]

    records = []
    for item in dev:
        out = chat(pc["server"], system_prompt, item["question"], max_tokens, pc["timeout_seconds"])
        records.append({"eval_id": item["eval_id"], "uid": item["uid"], **out, **score_citation(out["prediction"], item["uid"], valid)})

    metrics = compare(reference, records)
    totals = [r["total_seconds"] for r in records]
    report = {
        "model": args.model,
        "gguf": mc["gguf"],
        "gguf_sha256": sha256_file(mc["gguf"]) if Path(mc["gguf"]).exists() else None,
        "outtype": cfg["gguf"]["outtype"],
        "server": pc["server"],
        "parity": metrics,
        "gate": gate(metrics, pc["max_hit_diff"], pc["min_agreement"]),
        "candidate_metrics": aggregate(records),
        "latency": {
            "total_seconds_median": statistics.median(totals),
            "total_seconds_p95": percentile(totals, 0.95),
            "mean_completion_tokens": statistics.mean(r["completion_tokens"] or 0 for r in records),
        },
    }
    out_dir = Path(mc["gguf"]).parent
    write_jsonl(out_dir / "parity_predictions.jsonl", records)
    (out_dir / "parity.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("parity", "gate", "latency")}, indent=2))


if __name__ == "__main__":
    main()
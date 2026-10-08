import argparse
import json
import statistics
from pathlib import Path

import yaml

from ..train.data import read_jsonl
from .retrieve import Retriever


def section_ranks(chunks: list[dict]) -> list[str]:
    seen = []
    for c in chunks:
        if c["uid"] not in seen:
            seen.append(c["uid"])
    return seen


def metrics_for(gold: list[str], retrieved: list[list[dict]], ks: list[int]) -> dict:
    out = {f"recall@{k}": sum(g in {c["uid"] for c in r[:k]} for g, r in zip(gold, retrieved)) / len(gold) for k in ks}
    rr = []
    for g, r in zip(gold, retrieved):
        ranks = [c["uid"] for c in r]
        rr.append(1.0 / (ranks.index(g) + 1) if g in ranks else 0.0)
    out["mrr"] = sum(rr) / len(rr)
    out["unique_sections@max_k"] = statistics.mean(len(section_ranks(r)) for r in retrieved)
    return out


def select(results: dict, primary_k: int, ks: list[int], tolerance: float) -> dict:
    method = max(results, key=lambda m: (results[m][f"recall@{primary_k}"], results[m]["mrr"]))
    best = max(results[method][f"recall@{k}"] for k in ks)
    k = min(k for k in ks if results[method][f"recall@{k}"] >= best - tolerance)
    return {"method": method, "k": k, "recall_at_k": results[method][f"recall@{k}"]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Score retrieval methods on dev-para and select method and k")
    parser.add_argument("--config", default="configs/rag.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    sc = cfg["selection"]

    retriever = Retriever.load(cfg)
    dev = read_jsonl(sc["dev"])
    gold = [d["uid"] for d in dev]
    max_k = max(sc["ks"])
    results, latency = {}, {}
    for method in sc["methods"]:
        runs = [retriever.search(d["question"], method, max_k) for d in dev]
        results[method] = metrics_for(gold, [r["chunks"] for r in runs], sc["ks"])
        ms = sorted(r["retrieval_ms"] for r in runs)
        latency[method] = {"median_ms": statistics.median(ms), "p95_ms": ms[int(0.95 * (len(ms) - 1))]}

    report = {
        "n": len(dev),
        "index": json.loads((Path(cfg["index_dir"]) / "manifest.json").read_text(encoding="utf-8")),
        "results": results,
        "latency": latency,
        "selection_rule": f"highest recall@{sc['primary_k']} (MRR breaks ties); smallest k within {sc['tolerance']} of that method's best recall",
        "selected": select(results, sc["primary_k"], sc["ks"], sc["tolerance"]),
    }
    Path(sc["out"]).parent.mkdir(parents=True, exist_ok=True)
    Path(sc["out"]).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("results", "latency", "selected")}, indent=2))


if __name__ == "__main__":
    main()

import json
import urllib.error
import urllib.request

from ..serve.parity import percentile
from ..train.citations import aggregate, score_citation


def ask(api: str, question: str, arm: str, timeout: float) -> dict:
    req = urllib.request.Request(
        f"{api.rstrip('/')}/answer",
        data=json.dumps({"question": question, "arm": arm}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def measure(arm: str, items: list[dict], valid: set[str], ask_fn, warmup: int) -> dict:
    for item in items[:warmup]:
        try:
            ask_fn(item["question"], arm)
        except (urllib.error.URLError, OSError):
            pass
    records, failures = [], 0
    for item in items:
        try:
            out = ask_fn(item["question"], arm)
        except (urllib.error.URLError, OSError, KeyError, ValueError):
            failures += 1
            continue
        records.append({
            "eval_id": item["eval_id"],
            "answer": out["answer"],
            "total_ms": out["latency"]["total_ms"],
            **score_citation(out["answer"], item["uid"], valid),
        })
    totals = [r["total_ms"] for r in records]
    metrics = {
        "arm": arm,
        "requested": len(items),
        "failures": failures,
        **aggregate(records),
        "p50_ms": percentile(totals, 0.5) if totals else None,
        "p95_ms": percentile(totals, 0.95) if totals else None,
    }
    return {"metrics": metrics, "records": records}


def evaluate_gate(candidate: dict, production: dict, parity: dict | None, cfg: dict) -> dict:
    parity_ok = (
        parity is not None
        and abs(parity["hit_diff"]) <= cfg["max_hit_diff"]
        and parity["hit_agreement"] >= cfg["min_agreement"]
    )
    latency_ok = (
        candidate["failures"] == 0
        and candidate["p95_ms"] is not None
        and production["p95_ms"] is not None
        and candidate["p95_ms"] <= cfg["max_latency_ratio"] * production["p95_ms"]
    )
    checks = {
        "conversion_parity": parity_ok,
        "quality_floor": candidate.get("citation_hit", 0.0) > production.get("citation_hit", 0.0),
        "format_compliance": candidate.get("tag_present", 0.0) >= cfg["min_tag_present"],
        "operational": latency_ok,
    }
    return {"passed": all(checks.values()), "checks": checks}

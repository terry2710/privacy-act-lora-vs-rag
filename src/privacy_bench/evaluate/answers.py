import argparse
import json
import time
from pathlib import Path

import yaml

from ..registry.gate import ask
from ..train.citations import score_citation
from ..train.data import read_jsonl


def answer_record(item: dict, set_name: str, arm: str, out: dict, valid: set[str]) -> dict:
    record = {
        "eval_id": item["eval_id"],
        "set": set_name,
        "arm": arm,
        "question": item["question"],
        "answer": out["answer"],
        "retrieved": [r["uid"] for r in out.get("retrieved", [])],
        "latency": out["latency"],
        "tokens": out.get("tokens", {}),
    }
    gold = item.get("uid")
    if gold:
        record["gold_uid"] = gold
        record["citation"] = score_citation(out["answer"], gold, valid)
        record["retrieval_hit"] = gold in record["retrieved"] if record["retrieved"] else None
    else:
        record["citation"] = score_citation(out["answer"], "", valid)
    return record


def collect(items: list[dict], set_name: str, arm: str, ask_fn, out_path: Path, valid: set[str], warmup: int) -> dict:
    done = set()
    if out_path.exists():
        done = {r["eval_id"] for r in read_jsonl(out_path)}
    todo = [i for i in items if i["eval_id"] not in done]
    for item in todo[:warmup]:
        try:
            ask_fn(item["question"], arm)
        except OSError:
            pass
    failures = []
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a", encoding="utf-8") as f:
        for item in todo:
            try:
                out = ask_fn(item["question"], arm)
            except (OSError, ValueError, KeyError) as exc:
                failures.append({"eval_id": item["eval_id"], "error": str(exc)[:200]})
                continue
            f.write(json.dumps(answer_record(item, set_name, arm, out, valid), ensure_ascii=False) + "\n")
            f.flush()
    return {"set": set_name, "arm": arm, "items": len(items), "already_done": len(done), "new": len(todo) - len(failures), "failures": failures}


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect answers from the answer service for evaluation sets and arms, resuming where a run stopped")
    parser.add_argument("--config", default="configs/evaluation.yaml")
    parser.add_argument("--sets", nargs="*")
    parser.add_argument("--arms", nargs="*")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    valid = {s["uid"] for s in read_jsonl(cfg["sections"])}

    def ask_fn(question: str, arm: str) -> dict:
        return ask(cfg["api"], question, arm, cfg["timeout_seconds"])

    for set_name in args.sets or cfg["sets"]:
        items = read_jsonl(Path(cfg["eval_dir"]) / f"{set_name}.jsonl")
        for arm in args.arms or cfg["arms"]:
            start = time.perf_counter()
            out_path = Path(cfg["answers_dir"]) / set_name / f"{arm}.jsonl"
            summary = collect(items, set_name, arm, ask_fn, out_path, valid, cfg["warmup"])
            summary["seconds"] = round(time.perf_counter() - start, 1)
            print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()

import argparse
import csv
import json
import random
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

from ..train.data import read_jsonl, write_jsonl
from .calibrate import ADJ_FIELDS, adjudication, final_labels, report, sample
from .llm import judge_one, make_client
from .rubric import build_task, prompt_hash


def load_evidence(paths: list[str]) -> dict[str, str]:
    evidence = {}
    for path in paths:
        for row in read_jsonl(path):
            evidence[row["fact_id"]] = row["evidence"]
    return evidence


def load_answers(answers_dir: Path, sets: list[str], arms: list[str]) -> dict[tuple[str, str], list[dict]]:
    out = {}
    for set_name in sets:
        for arm in arms:
            path = answers_dir / set_name / f"{arm}.jsonl"
            if path.exists():
                out[(set_name, arm)] = read_jsonl(path)
    return out


def load_items(eval_dir: Path, sets: list[str]) -> dict[str, dict]:
    return {item["eval_id"]: item for s in sets for item in read_jsonl(eval_dir / f"{s}.jsonl")}


def to_tasks(answers: list[dict], items: dict[str, dict], evidence: dict[str, str]) -> list[dict]:
    return [build_task(items[a["eval_id"]], a, evidence) for a in answers]


def run_judges(jobs: list[tuple[str, dict]], cfg: dict, clients: dict, hashes: dict, cache_dir: Path) -> list[dict]:
    results = []
    with ThreadPoolExecutor(max_workers=cfg["run"]["max_workers"]) as pool:
        futures = [
            pool.submit(judge_one, name, cfg["judges"][name], clients[name], task, cache_dir, hashes[name])
            for name, task in jobs
        ]
        for done, fut in enumerate(as_completed(futures), 1):
            results.append(fut.result())
            if done % 25 == 0 or done == len(futures):
                print(f"[{done}/{len(futures)}] judgements done", flush=True)
    return sorted(results, key=lambda r: (r["key"], r["judge"]))


def usage(results: list[dict]) -> dict:
    out = {}
    for r in results:
        u = out.setdefault(r["judge"], Counter())
        u.update({"items": 1, "calls": r["calls"], "input_tokens": r["input_tokens"], "output_tokens": r["output_tokens"]})
    return {
        j: {**u, "input_per_item": round(u["input_tokens"] / u["items"], 1), "output_per_item": round(u["output_tokens"] / u["items"], 1)}
        for j, u in out.items()
    }


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def summarise(results: list[dict]) -> dict:
    out = {}
    for r in results:
        slot = out.setdefault(r["set"], {}).setdefault(r["arm"], Counter())
        slot["n"] += 1
        if r["label"] is None:
            slot["no_answer"] += 1
        elif r["kind"] == "qa":
            slot[f"accuracy_{r['label']}"] += 1
            slot["contradicts_reference"] += r["verdict"]["contradicts_reference"]
        else:
            slot[r["label"]] += 1
            slot["premise_corrected"] += r["verdict"]["premise_corrected"]
            slot["invents_content"] += r["verdict"]["invents_content"]
    final = {}
    for set_name, arms in out.items():
        for arm, c in arms.items():
            scored = c["n"] - c["no_answer"]
            row = dict(c)
            if set_name == "s_neg":
                row["pass_rate"] = round(c["pass"] / scored, 4) if scored else None
            else:
                row["mean_accuracy"] = round((c["accuracy_1"] + 2 * c["accuracy_2"]) / scored, 4) if scored else None
                row["fully_correct_rate"] = round(c["accuracy_2"] / scored, 4) if scored else None
            final.setdefault(set_name, {})[arm] = row
    return final


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="LLM-as-judge scoring with a two-judge calibration and human adjudication of disagreements.")
    ap.add_argument("command", choices=["pilot", "calibrate", "calibrate-score", "run"])
    ap.add_argument("--config", type=Path, default=Path("configs/judge.yaml"))
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--sets", nargs="*")
    ap.add_argument("--arms", nargs="*")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    paths = {k: v if isinstance(v, list) else Path(v) for k, v in cfg["paths"].items()}
    cal = cfg["calibration"]
    cal_dir = paths["calibration_dir"]
    judges = [cfg["roles"]["primary"], cfg["roles"]["secondary"]]
    hashes = {name: prompt_hash(jcfg, cfg["prompt_version"]) for name, jcfg in cfg["judges"].items()}
    sets = args.sets or cfg["sets"]
    arms = args.arms or cfg["arms"]
    evidence = load_evidence(paths["evidence"])
    items = load_items(paths["eval_dir"], cfg["sets"])

    def calibration_tasks() -> list[dict]:
        answers = load_answers(paths["answers_dir"], list(cal["per_set"]), cal["arms"])
        return to_tasks(sample(answers, cal["arms"], cal["per_set"], cfg["run"]["seed"]), items, evidence)

    def clients_for(names: list[str]) -> dict:
        from dotenv import load_dotenv

        load_dotenv()
        return {n: make_client(cfg["judges"][n], cfg["run"]["max_retries"]) for n in names}

    if args.command == "pilot":
        tasks = calibration_tasks()
        random.Random(cfg["run"]["seed"]).shuffle(tasks)
        tasks = tasks[: args.limit]
        results = run_judges([(j, t) for j in judges for t in tasks], cfg, clients_for(judges), hashes, paths["cache_dir"])
        full_items = sum(len(read_jsonl(paths["eval_dir"] / f"{s}.jsonl")) for s in cfg["sets"]) * len(cfg["arms"])
        u = usage(results)
        print(json.dumps({
            "pilot_items": len(tasks),
            "labels": {j: dict(Counter(str(r["label"]) for r in results if r["judge"] == j)) for j in judges},
            "usage": u,
            "projected_tokens": {
                "calibration": {j: {"input": round(u[j]["input_per_item"] * sum(cal["per_set"].values()) * len(cal["arms"])),
                                    "output": round(u[j]["output_per_item"] * sum(cal["per_set"].values()) * len(cal["arms"]))} for j in judges},
                "full_run_primary": {"items": full_items, "input": round(u[judges[0]]["input_per_item"] * full_items),
                                     "output": round(u[judges[0]]["output_per_item"] * full_items)},
            },
        }, indent=2))
        return 0

    if args.command == "calibrate":
        if (cal_dir / "adjudication.csv").exists():
            raise SystemExit(f"{cal_dir / 'adjudication.csv'} already exists; refusing to overwrite possible labels.")
        tasks = calibration_tasks()
        results = run_judges([(j, t) for j in judges for t in tasks], cfg, clients_for(judges), hashes, paths["cache_dir"])
        cal_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(cal_dir / "judgements.jsonl", results)
        rows, key = adjudication(tasks, results, judges, random.Random(cfg["run"]["seed"]))
        with (cal_dir / "adjudication.csv").open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=ADJ_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        (cal_dir / "adjudication_key.json").write_text(json.dumps(key, indent=1), encoding="utf-8")
        prelim = report(tasks, results, judges, None, cal)
        print(json.dumps({"judge_agreement": prelim["judge_agreement"], "usage": usage(results)}, indent=2))
        print(f"{len(rows)} disagreements written to {cal_dir / 'adjudication.csv'}")
        return 0

    if args.command == "calibrate-score":
        tasks = calibration_tasks()
        results = read_jsonl(cal_dir / "judgements.jsonl")
        key = json.loads((cal_dir / "adjudication_key.json").read_text(encoding="utf-8"))
        final = final_labels(tasks, results, judges, key, read_csv(cal_dir / "adjudication.csv"))
        out = report(tasks, results, judges, final, cal)
        out["prompt_version"] = cfg["prompt_version"]
        out["prompt_hashes"] = {j: hashes[j] for j in judges}
        out["models"] = {j: cfg["judges"][j]["model"] for j in judges}
        out["usage"] = usage(results)
        write_jsonl(cal_dir / "final_labels.jsonl", [{"key": k, "label": v} for k, v in sorted(final.items())])
        (cal_dir / "report.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(json.dumps({k: out[k] for k in ("judge_agreement", "vs_final", "by_arm", "gate")}, indent=2))
        return 0

    report_path = cal_dir / "report.json"
    if not report_path.exists():
        raise SystemExit("run calibrate and calibrate-score first")
    cal_report = json.loads(report_path.read_text(encoding="utf-8"))
    primary = judges[0]
    if not cal_report["gate"]["passed"] or cal_report["prompt_hashes"][primary] != hashes[primary]:
        raise SystemExit("calibration gate did not pass for the current primary judge prompt and config")
    answers = load_answers(paths["answers_dir"], sets, arms)
    missing = [f"{s}/{a}" for s in sets for a in arms if (s, a) not in answers]
    if missing:
        raise SystemExit(f"answers missing for: {missing}")
    tasks = [t for group in answers.values() for t in to_tasks(group, items, evidence)]
    results = run_judges([(primary, t) for t in tasks], cfg, clients_for([primary]), hashes, paths["cache_dir"])
    for (set_name, arm) in answers:
        write_jsonl(paths["judgements_dir"] / set_name / f"{arm}.jsonl", [r for r in results if r["set"] == set_name and r["arm"] == arm])
    summary_path = paths["judgements_dir"] / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    for set_name, by_arm in summarise(results).items():
        summary.setdefault(set_name, {}).update(by_arm)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"usage": usage(results), "summary": summarise(results)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

from ..data.build_split import read_jsonl
from .generate import OUTPUT_FILES

STRATA = {
    "verified_pass_train": 40,
    "verified_pass_holdout": 15,
    "verified_fail": 25,
    "paraphrase_scenario": 20,
    "neg_unverifiable": 20,
}
CRITERIA = {
    "qa": "ok if every claim in the answer is supported by the source text and the answer fully answers every question",
    "paraphrase": "ok if the new question asks for exactly the same fact as the training question and the reference answer fully answers it",
    "neg": "ok if the premise is correct, i.e. the Privacy Act 1988 really contains no such provision, body or mechanism",
}
SHEET_FIELDS = ["check_id", "kind", "criterion", "provision", "question", "answer", "source_text", "label", "note"]
LABELS = {"ok", "problem"}
WINDOW = 4000


def source_window(text: str, evidence: str, width: int = WINDOW) -> str:
    if len(text) <= width:
        return text
    pos = text.lower().find(evidence[:60].lower())
    start = max(0, min(pos - width // 2, len(text) - width)) if pos >= 0 else 0
    return ("..." if start else "") + text[start : start + width] + ("..." if start + width < len(text) else "")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return round(centre - half, 4), round(centre + half, 4)


def cohen_kappa(pairs: list[tuple[bool, bool]]) -> float | None:
    n = len(pairs)
    if n == 0:
        return None
    po = sum(a == b for a, b in pairs) / n
    pa, pb = sum(a for a, _ in pairs) / n, sum(b for _, b in pairs) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return round((po - pe) / (1 - pe), 4) if pe < 1 else None


def build_rows(generated: list[dict], verdicts: dict[str, str], sources: dict[str, str], s_para: list[dict],
               train: dict[str, dict], s_neg: list[dict], rng: random.Random) -> tuple[list[dict], dict]:
    pools = {
        "verified_pass_train": [g for g in generated if g["split"] == "train" and verdicts[g["fact_id"]] == "pass"],
        "verified_pass_holdout": [g for g in generated if g["split"] != "train" and verdicts[g["fact_id"]] == "pass"],
        "verified_fail": [g for g in generated if verdicts[g["fact_id"]] != "pass"],
        "paraphrase_scenario": [p for p in s_para if p["paraphrase_type"] == "scenario"],
        "neg_unverifiable": [n for n in s_neg if n["category"] != "nonexistent_provision"],
    }
    entries = []
    for stratum, size in STRATA.items():
        for item in rng.sample(pools[stratum], min(size, len(pools[stratum]))):
            entries.append((stratum, item))
    rng.shuffle(entries)
    rows, key = [], {}
    for n, (stratum, item) in enumerate(entries, 1):
        check_id = f"chk-{n:03d}"
        if stratum.startswith("verified"):
            row = {
                "kind": "qa",
                "provision": item["display"],
                "question": " || ".join(item["questions"]),
                "answer": item["answer"],
                "source_text": source_window(sources[item["uid"]], item["evidence"]),
            }
            key[check_id] = {"stratum": stratum, "id": item["fact_id"], "verifier_verdict": verdicts[item["fact_id"]]}
        elif stratum == "paraphrase_scenario":
            fact = train[item["fact_id"]]
            row = {
                "kind": "paraphrase",
                "provision": item["display"],
                "question": item["question"],
                "answer": item["reference_answer"],
                "source_text": "Training questions: " + " || ".join(fact["questions"]),
            }
            key[check_id] = {"stratum": stratum, "id": item["eval_id"]}
        else:
            row = {"kind": "neg", "provision": "", "question": item["question"], "answer": "", "source_text": "Premise: " + item["premise"]}
            key[check_id] = {"stratum": stratum, "id": item["eval_id"]}
        rows.append({"check_id": check_id, "criterion": CRITERIA[row["kind"]], "label": "", "note": "", **row})
    return rows, key


def score(rows: list[dict], key: dict) -> dict:
    bad = [r["check_id"] for r in rows if r["label"].strip().lower() not in LABELS]
    if bad:
        raise ValueError(f"label must be 'ok' or 'problem' for: {bad}")
    human_ok = {r["check_id"]: r["label"].strip().lower() == "ok" for r in rows}
    by_stratum: dict[str, Counter] = {}
    for cid, meta in key.items():
        by_stratum.setdefault(meta["stratum"], Counter())["ok" if human_ok[cid] else "problem"] += 1
    strata = {}
    for name, c in by_stratum.items():
        n = c["ok"] + c["problem"]
        strata[name] = {"n": n, "ok": c["ok"], "problem": c["problem"], "ok_rate": round(c["ok"] / n, 4), "ok_rate_ci95": wilson(c["ok"], n)}
    pairs = [(meta["verifier_verdict"] == "pass", human_ok[cid]) for cid, meta in key.items() if "verifier_verdict" in meta]
    confusion = Counter((("verifier_pass" if v else "verifier_fail"), ("human_ok" if h else "human_problem")) for v, h in pairs)
    return {
        "strata": strata,
        "verifier_vs_human": {
            "n": len(pairs),
            "confusion": {f"{a}/{b}": n for (a, b), n in sorted(confusion.items())},
            "raw_agreement": round(sum(v == h for v, h in pairs) / len(pairs), 4) if pairs else None,
            "cohen_kappa_on_sample": cohen_kappa(pairs),
            "note": "Verifier failures are oversampled; read per-stratum rates for population estimates.",
        },
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Export a blind human spot-check sheet and score it once labelled.")
    ap.add_argument("command", choices=["export", "score"])
    ap.add_argument("--dir", type=Path, default=Path("data/audit/spot_check"))
    ap.add_argument("--seed", type=int, default=20261005)
    args = ap.parse_args(argv)
    sheet, key_path, results = args.dir / "sheet.csv", args.dir / "key.json", args.dir / "results.json"

    if args.command == "export":
        if sheet.exists():
            raise SystemExit(f"{sheet} already exists; refusing to overwrite a sheet that may contain labels.")
        generated = [it for name in OUTPUT_FILES.values() for it in read_jsonl(Path("data/qa/generated") / name)]
        verdicts = {r["fact_id"]: r["verdict"] for r in read_jsonl(Path("data/audit/verification_results.jsonl"))}
        sources = {r["uid"]: r["text"] for r in read_jsonl(Path("data/processed/generation_inputs.jsonl"))}
        train = {it["fact_id"]: it for it in read_jsonl(Path("data/qa/final/train.jsonl"))}
        rows, key = build_rows(generated, verdicts, sources, read_jsonl(Path("data/qa/eval/s_para.jsonl")), train,
                               read_jsonl(Path("data/qa/eval/s_neg.jsonl")), random.Random(args.seed))
        args.dir.mkdir(parents=True, exist_ok=True)
        with sheet.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=SHEET_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        key_path.write_text(json.dumps(key, indent=1), encoding="utf-8")
        print(f"exported {len(rows)} rows to {sheet}: {dict(Counter(m['stratum'] for m in key.values()))}")
        return 0

    with sheet.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    report = score(rows, json.loads(key_path.read_text(encoding="utf-8")))
    results.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for name, s in report["strata"].items():
        print(f"{name}: ok {s['ok']}/{s['n']} ({s['ok_rate']:.1%}, 95% CI {s['ok_rate_ci95']})")
    print(f"verifier_vs_human: {report['verifier_vs_human']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
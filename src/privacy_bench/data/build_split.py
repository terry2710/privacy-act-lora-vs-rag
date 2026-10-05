import argparse
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import yaml

QUOTA_KEY = {"train": "facts", "holdout_h1": "questions", "holdout_h2": "questions"}
HOLDOUT_SPLITS = {"holdout_h1", "holdout_h2"}
PLAN_FIELDS = (
    "uid", "display", "unit", "part", "division", "subdivision", "title",
    "split", "split_reason", "n_chars", "weight", "quota",
    "leak_hits", "source_review", "masked_lines",
)
GEN_FIELDS = (
    "uid", "display", "unit", "split",
    "schedule_title", "part_title", "division_title", "subdivision_title", "title", "quota",
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def verify_source(manifest: dict) -> str:
    src = manifest["source"]
    digest = hashlib.sha256(Path(src["docx"]).read_bytes()).hexdigest()
    if digest != src["docx_sha256"]:
        raise ValueError(f"DOCX checksum {digest} does not match manifest value {src['docx_sha256']}")
    return digest


def assign_splits(sections: list[dict], manifest: dict) -> list[dict]:
    units = manifest["units"]
    overrides = {o["uid"]: o for o in manifest.get("section_overrides", [])}
    present = {s["unit"] for s in sections}
    unconfigured = sorted(present - units.keys())
    unknown = sorted(units.keys() - present)
    stray = sorted(overrides.keys() - {s["uid"] for s in sections})
    if unconfigured or unknown or stray:
        raise ValueError(
            f"manifest mismatch: unconfigured units {unconfigured}, unknown units {unknown}, unknown overrides {stray}"
        )
    for s in sections:
        rule = overrides.get(s["uid"], units[s["unit"]])
        s["split"] = rule["split"]
        s["split_reason"] = rule.get("reason")
    return sections


def check_holdout_patterns(sections: list[dict], manifest: dict) -> None:
    patterns = [re.compile(p) for p in manifest["leakage"]["holdout_uid_patterns"]]
    errors = []
    for s in sections:
        matched = any(p.search(s["uid"]) for p in patterns)
        if matched != (s["split"] in HOLDOUT_SPLITS):
            errors.append(f"{s['uid']} (split={s['split']}, pattern_match={matched})")
    if errors:
        raise ValueError("holdout_uid_patterns inconsistent with split assignment: " + "; ".join(errors))


def allocate(weights: list[float], total: int, minimum: int = 1) -> list[int]:
    n = len(weights)
    if n == 0:
        return []
    if total < minimum * n:
        raise ValueError(f"quota {total} is below minimum {minimum} x {n} sections")
    remaining = total - minimum * n
    wsum = sum(weights)
    raw = [remaining * w / wsum if wsum else 0.0 for w in weights]
    counts = [math.floor(r) for r in raw]
    order = sorted(range(n), key=lambda i: raw[i] - counts[i], reverse=True)
    for i in order[: remaining - sum(counts)]:
        counts[i] += 1
    return [minimum + c for c in counts]


def allocate_quotas(sections: list[dict], manifest: dict) -> list[dict]:
    cfg = manifest["allocation"]
    guide = re.compile(cfg["guide_title_pattern"], re.IGNORECASE)
    groups: dict[str, list[dict]] = defaultdict(list)
    for s in sections:
        s["weight"], s["quota"] = 0.0, 0
        unit_split = manifest["units"][s["unit"]]["split"]
        if s["split"] in QUOTA_KEY and s["split"] == unit_split:
            factor = cfg["guide_weight"] if guide.search(s["title"]) else 1.0
            s["weight"] = round(s["n_chars"] * factor, 1)
            groups[s["unit"]].append(s)
    for unit, items in groups.items():
        rule = manifest["units"][unit]
        total = rule[QUOTA_KEY[rule["split"]]]
        quotas = allocate([s["weight"] for s in items], total, cfg["min_per_section"])
        for s, q in zip(items, quotas):
            s["quota"] = q
    return sections


def compile_leak_rules(manifest: dict) -> list[re.Pattern]:
    lk = manifest["leakage"]
    rules = [re.compile(p) for p in lk["holdout_citation_patterns"]]
    rules += [re.compile(rf"\b{re.escape(t)}(?:e?s)?\b", re.IGNORECASE) for t in lk["holdout_terms"]]
    return rules


def find_hits(line: str, rules: list[re.Pattern]) -> set[str]:
    return {m.group(0) for rx in rules for m in rx.finditer(line)}


def block_lines(lines: list[str], block: dict, uid: str) -> list[int]:
    start = next((i for i, line in enumerate(lines) if line.startswith(block["from"])), None)
    if start is None:
        raise ValueError(f"{uid}: mask block start {block['from']!r} not found")
    end = next((i for i in range(start, len(lines)) if lines[i].startswith(block["to"])), None)
    if end is None:
        raise ValueError(f"{uid}: mask block end {block['to']!r} not found after start")
    return list(range(start, end + 1))


def scan_leakage(sections: list[dict], manifest: dict) -> list[dict]:
    rules = compile_leak_rules(manifest)
    review = manifest["leakage"].get("source_review", {})
    stray = sorted(set(review) - {s["uid"] for s in sections})
    if stray:
        raise ValueError(f"source_review references unknown uids {stray}")
    for s in sections:
        s["leak_hits"], s["masked_lines"], s["source_review"] = [], [], None
        if s["split"] != "train":
            continue
        lines = s["text"].split("\n")
        flagged = {i: h for i, h in ((i, find_hits(line, rules)) for i, line in enumerate(lines)) if h}
        entry = review.get(s["uid"])
        if not flagged and entry is None:
            continue
        s["leak_hits"] = sorted(set().union(*flagged.values()))
        action = entry["action"] if entry else "mask"
        if action == "false_positive":
            s["source_review"] = action
            continue
        if action != "mask":
            raise ValueError(f"{s['uid']}: unknown source_review action {action!r}")
        masked = set(flagged)
        for block in (entry or {}).get("blocks", []):
            masked.update(block_lines(lines, block, s["uid"]))
        s["source_review"] = action
        s["masked_lines"] = sorted(masked)
    return sections


def build_generation_inputs(sections: list[dict]) -> list[dict]:
    rows = []
    for s in sections:
        if not s["quota"]:
            continue
        drop = set(s["masked_lines"])
        kept = [line for i, line in enumerate(s["text"].split("\n")) if i not in drop]
        row = {k: s.get(k) for k in GEN_FIELDS}
        row["text"] = "\n".join(kept)
        rows.append(row)
    return rows


def summarise(sections: list[dict], manifest: dict, digest: str) -> dict:
    by_split: dict[str, dict[str, int]] = defaultdict(lambda: {"sections": 0, "chars": 0, "quota": 0})
    by_unit: dict[str, dict] = {}
    for s in sections:
        agg = by_split[s["split"]]
        agg["sections"] += 1
        agg["chars"] += s["n_chars"]
        agg["quota"] += s["quota"]
        u = by_unit.setdefault(s["unit"], {"sections": 0, "chars": 0, "quota": 0})
        if s["quota"]:
            u["sections"] += 1
            u["chars"] += s["n_chars"]
            u["quota"] += s["quota"]
    for u in by_unit.values():
        u["chars_per_item"] = round(u["chars"] / u["quota"]) if u["quota"] else None
    reviewed = [s for s in sections if s["source_review"]]
    review = manifest["leakage"].get("source_review", {})
    by_uid = {s["uid"]: s for s in sections}
    return {
        "docx_sha256": digest,
        "by_split": dict(by_split),
        "by_unit": by_unit,
        "masked_line_total": sum(len(s["masked_lines"]) for s in sections),
        "source_review": [
            {
                "uid": s["uid"],
                "title": s["title"],
                "action": s["source_review"],
                "hits": s["leak_hits"],
                "masked": [s["text"].split("\n")[i] for i in s["masked_lines"]],
            }
            for s in reviewed
        ],
        "unused_source_review": sorted(uid for uid in review if not by_uid[uid]["leak_hits"]),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Assign Act sections to splits, allocate QA quotas and mask held-out content.")
    ap.add_argument("--manifest", type=Path, default=Path("configs/split_manifest.yaml"))
    ap.add_argument("--sections", type=Path, default=Path("data/processed/sections.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("data/processed/split_plan.jsonl"))
    ap.add_argument("--gen-out", type=Path, default=Path("data/processed/generation_inputs.jsonl"))
    ap.add_argument("--report", type=Path, default=Path("data/audit/split_report.json"))
    args = ap.parse_args(argv)

    manifest = yaml.safe_load(args.manifest.read_text(encoding="utf-8"))
    digest = verify_source(manifest)
    sections = assign_splits(read_jsonl(args.sections), manifest)
    check_holdout_patterns(sections, manifest)
    allocate_quotas(sections, manifest)
    scan_leakage(sections, manifest)
    report = summarise(sections, manifest, digest)

    write_jsonl(args.out, [{k: s.get(k) for k in PLAN_FIELDS} for s in sections])
    write_jsonl(args.gen_out, build_generation_inputs(sections))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    q = {k: v["quota"] for k, v in report["by_split"].items()}
    print(
        f"train facts={q.get('train', 0)} h1={q.get('holdout_h1', 0)} h2={q.get('holdout_h2', 0)}; "
        f"reviewed train sections={len(report['source_review'])}, masked lines={report['masked_line_total']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
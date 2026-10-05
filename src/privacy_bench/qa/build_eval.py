import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

from ..data.build_split import allocate, read_jsonl, write_jsonl
from .embed import EmbeddingCache, embed_texts
from .generate import OUTPUT_FILES

APP_REF = re.compile(r"\b(?:Australian Privacy Principle|APP)\s+(\d+)\b")
SECTION_REF = re.compile(r"\bsection\s+(\d+[A-Z]*)\b")
DIVISION_REF = re.compile(r"\bDivision\s+(\d+[A-Z]*)\s+of\s+Part\s+([IVX]+[A-Z]*)\b")
PART_REF = re.compile(r"\bPart\s+([IVX]+[A-Z]*)\b")
SCHEDULE_REF = re.compile(r"\bSchedule\s+(\d+)\b")
NEG_CATEGORIES = {"nonexistent_provision", "foreign_law", "fabricated_mechanism"}
NEG_EXPECTED = "State that the Privacy Act 1988 contains no such provision or mechanism, and do not invent one."


def provision_refs(question: str, sections: dict[str, dict]) -> list[tuple[str, bool]]:
    units = {s["unit"] for s in sections.values()}
    divisions = {(s["part"], s["division"]) for s in sections.values() if s["schedule"] is None and s["division"]}
    refs = [(f"APP {m[1]}", f"sch1:app{m[1]}" in sections) for m in APP_REF.finditer(question)]
    refs += [(f"s {m[1]}", f"s:{m[1]}" in sections) for m in SECTION_REF.finditer(question)]
    refs += [(f"Division {m[1]} of Part {m[2]}", (m[2], m[1]) in divisions) for m in DIVISION_REF.finditer(question)]
    refs += [(f"Part {m[1]}", f"Part {m[1]}" in units) for m in PART_REF.finditer(DIVISION_REF.sub(" ", question))]
    refs += [(f"Schedule {m[1]}", f"Sch {m[1]}" in units) for m in SCHEDULE_REF.finditer(question)]
    return refs


def load_neg(path: Path, sections: dict[str, dict]) -> list[dict]:
    rows = yaml.safe_load(path.read_text(encoding="utf-8"))["items"]
    out, errors = [], []
    for i, row in enumerate(rows, 1):
        refs = provision_refs(row["question"], sections)
        existing = [r for r, exists in refs if exists]
        if row["category"] not in NEG_CATEGORIES:
            errors.append(f"item {i}: unknown category {row['category']!r}")
        elif row["category"] == "nonexistent_provision" and (not refs or existing):
            errors.append(f"item {i}: needs only non-existent provision references, found {refs}")
        elif existing:
            errors.append(f"item {i}: cites existing provisions {existing}")
        out.append({
            "eval_id": f"s_neg-{i:04d}",
            "set": "s_neg",
            "category": row["category"],
            "question": row["question"],
            "premise": row["premise"],
            "expected_behaviour": NEG_EXPECTED,
            "provision_refs": [r for r, _ in refs],
        })
    if errors:
        raise ValueError("invalid S-neg items:\n" + "\n".join(errors))
    return out


def group_starts(counts: list[int]) -> np.ndarray:
    return np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(int)


def item_similarity(matrix: np.ndarray, counts: list[int]) -> np.ndarray:
    starts = group_starts(counts)
    sim = matrix @ matrix.T
    sim = np.maximum.reduceat(sim, starts, axis=1)
    return np.maximum.reduceat(sim, starts, axis=0)


def cross_similarity(a: np.ndarray, a_counts: list[int], b: np.ndarray, b_counts: list[int]) -> np.ndarray:
    sim = b @ a.T
    sim = np.maximum.reduceat(sim, group_starts(a_counts), axis=1)
    return np.maximum.reduceat(sim, group_starts(b_counts), axis=0)


def greedy_dedup(sim: np.ndarray, threshold: float) -> tuple[list[int], list[tuple[int, int, float]]]:
    kept: list[int] = []
    dropped: list[tuple[int, int, float]] = []
    for i in range(sim.shape[0]):
        if kept:
            idx = np.array(kept)
            j = int(idx[np.argmax(sim[i, idx])])
            if sim[i, j] >= threshold:
                dropped.append((i, j, float(sim[i, j])))
                continue
        kept.append(i)
    return kept, dropped


def stratified_sample(items: list[dict], n: int, rng: random.Random, exclude: set[str]) -> list[dict]:
    pool: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        if item["fact_id"] not in exclude:
            pool[item["unit"]].append(item)
    units = sorted(pool)
    quotas = allocate([float(len(pool[u])) for u in units], n, minimum=0)
    chosen = [it for u, q in zip(units, quotas) for it in rng.sample(pool[u], q)]
    return sorted(chosen, key=lambda it: it["fact_id"])


def eval_record(set_name: str, idx: int, item: dict, question: str, **extra) -> dict:
    return {
        "eval_id": f"{set_name}-{idx:04d}",
        "set": set_name,
        "fact_id": item["fact_id"],
        "uid": item["uid"],
        "display": item["display"],
        "unit": item["unit"],
        "question_type": item["question_type"],
        "question": question,
        "reference_answer": item["answer"],
        **extra,
    }

def annotate_parallels(records: list[dict], parallels: dict[str, dict], train_ids: set[str]) -> list[dict]:
    known = {r["fact_id"] for r in records}
    bad = sorted(k for k, v in parallels.items() if k not in known or v["train_fact"] not in train_ids)
    if bad:
        raise ValueError(f"holdout_parallels references unknown holdout or training facts: {bad}")
    for r in records:
        r["parallel_to_train"] = parallels.get(r["fact_id"], {}).get("train_fact")
    return records

def question_rows(items: list[dict]) -> tuple[list[str], list[int]]:
    return [q for it in items for q in it["questions"]], [len(it["questions"]) for it in items]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Deduplicate verified QA, audit semantic leakage, and build evaluation sets.")
    ap.add_argument("--config", type=Path, default=Path("configs/eval_sets.yaml"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    paths = {k: Path(v) for k, v in cfg["paths"].items()}
    manifest = yaml.safe_load(paths["manifest"].read_text(encoding="utf-8"))
    dedup_cfg, eval_cfg = manifest["dedup"], manifest["eval"]
    sections = {s["uid"]: s for s in read_jsonl(paths["sections"])}
    verified = {split: read_jsonl(paths["verified_dir"] / name) for split, name in OUTPUT_FILES.items()}
    neg = load_neg(paths["neg_source"], sections)

    if args.dry_run:
        print(f"verified={ {k: len(v) for k, v in verified.items()} } s_neg={len(neg)} {dict(Counter(r['category'] for r in neg))}")
        return 0

    import openai
    from dotenv import load_dotenv

    load_dotenv()
    emb = cfg["embedding"]
    cache = EmbeddingCache(paths["embed_cache_dir"] / f"{emb['model']}.npz")
    all_texts = [q for items in verified.values() for q in question_rows(items)[0]]
    embed_texts(all_texts, cache, openai.OpenAI(), emb["model"], emb["batch_size"])

    kept_items: dict[str, list[dict]] = {}
    matrices: dict[str, tuple[np.ndarray, list[int]]] = {}
    dedup_log = []
    for split, items in verified.items():
        texts, counts = question_rows(items)
        matrix = cache.matrix(texts)
        kept, dropped = greedy_dedup(item_similarity(matrix, counts), dedup_cfg["within_split_min_cosine"])
        for i, j, cos in dropped:
            dedup_log.append({
                "split": split,
                "dropped": items[i]["fact_id"],
                "kept": items[j]["fact_id"],
                "cosine": round(cos, 4),
                "dropped_question": items[i]["questions"][0],
                "kept_question": items[j]["questions"][0],
            })
        kept_items[split] = [items[i] for i in kept]
        offsets = group_starts(counts)
        rows = [r for i in kept for r in range(offsets[i], offsets[i] + counts[i])]
        matrices[split] = (matrix[rows], [counts[i] for i in kept])

    leakage = []
    train_m, train_c = matrices["train"]
    for split in ("holdout_h1", "holdout_h2"):
        hold_m, hold_c = matrices[split]
        sim = cross_similarity(train_m, train_c, hold_m, hold_c)
        for i, j in zip(*np.nonzero(sim >= dedup_cfg["leakage_review_min_cosine"])):
            hold, train = kept_items[split][i], kept_items["train"][j]
            leakage.append({
                "holdout_split": split,
                "holdout_id": hold["fact_id"],
                "train_id": train["fact_id"],
                "cosine": round(float(sim[i, j]), 4),
                "holdout_question": hold["questions"][0],
                "train_question": train["questions"][0],
            })
    leakage.sort(key=lambda r: -r["cosine"])

    rng = random.Random(manifest["seed"])
    train = kept_items["train"]
    s_mem_facts = stratified_sample(train, eval_cfg["s_mem"]["facts"], rng, set())
    dev_facts = stratified_sample(train, eval_cfg["dev_para"]["facts"], rng, {it["fact_id"] for it in s_mem_facts})
    s_mem = []
    for n, item in enumerate(s_mem_facts, 1):
        k = rng.randrange(len(item["questions"]))
        s_mem.append(eval_record("s_mem", n, item, item["questions"][k], source_phrasing=k))
    tag_divs = set(eval_cfg["h1"].get("tag_divisions", []))
    h1 = [
        eval_record("h1", n, it, it["questions"][0], division=sections[it["uid"]]["division"],
                    tagged=sections[it["uid"]]["division"] in tag_divs)
        for n, it in enumerate(kept_items["holdout_h1"], 1)
    ]
    h2 = [eval_record("h2", n, it, it["questions"][0]) for n, it in enumerate(kept_items["holdout_h2"], 1)]
    annotate_parallels(h1 + h2, manifest["leakage"].get("holdout_parallels", {}), {it["fact_id"] for it in train})

    write_jsonl(paths["final_dir"] / "train.jsonl", train)
    for name, rows in (("s_mem", s_mem), ("h1", h1), ("h2", h2), ("s_neg", neg)):
        write_jsonl(paths["eval_dir"] / f"{name}.jsonl", rows)
    selection = {"seed": manifest["seed"], "s_mem": [it["fact_id"] for it in s_mem_facts], "dev_para": [it["fact_id"] for it in dev_facts]}
    (paths["eval_dir"] / "selection.json").write_text(json.dumps(selection, indent=1), encoding="utf-8")
    write_jsonl(paths["dedup_log"], dedup_log)
    write_jsonl(paths["leakage_review"], leakage)

    report = {
        "embedding_model": emb["model"],
        "thresholds": {k: dedup_cfg[k] for k in ("within_split_min_cosine", "leakage_review_min_cosine")},
        "verified": {k: len(v) for k, v in verified.items()},
        "after_dedup": {k: len(v) for k, v in kept_items.items()},
        "dedup_dropped": len(dedup_log),
        "leakage_pairs_for_review": len(leakage),
        "eval_sizes": {"s_mem": len(s_mem),
                       "dev_para_facts": len(dev_facts),
                       "h1": len(h1),
                       "h1_tagged": sum(r["tagged"] for r in h1),
                       "h2": len(h2),
                       "s_neg": len(neg),
                       "holdout_parallel_to_train": sum(bool(r["parallel_to_train"]) for r in h1 + h2)},
        "s_mem_by_unit": dict(Counter(it["unit"] for it in s_mem_facts)),
        "dev_para_by_unit": dict(Counter(it["unit"] for it in dev_facts)),
        "s_neg_by_category": dict(Counter(r["category"] for r in neg)),
    }
    paths["report"].parent.mkdir(parents=True, exist_ok=True)
    paths["report"].write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"after_dedup={report['after_dedup']} dropped={len(dedup_log)} leakage_pairs={len(leakage)} eval_sizes={report['eval_sizes']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
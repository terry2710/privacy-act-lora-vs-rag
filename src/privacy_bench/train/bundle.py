import argparse
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import yaml

from .citations import score_citation
from .data import provision_index, read_jsonl, sha256_file, text_hash


def git_state() -> dict:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "--no-optional-locks", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": dirty}


def check_inputs(facts: list[dict], dev: list[dict], provisions: list[dict], holdout_patterns: list[str]) -> dict:
    valid = {p["uid"] for p in provisions}
    holdout = [re.compile(p) for p in holdout_patterns]
    problems = []
    fact_ids = {f["fact_id"] for f in facts}
    for f in facts:
        if f["uid"] not in valid:
            problems.append(f"train uid not in provisions: {f['uid']}")
        if any(p.search(f["uid"]) for p in holdout):
            problems.append(f"held-out uid in train: {f['uid']}")
        if len(f["questions"]) < 2:
            problems.append(f"fewer than two phrasings: {f['fact_id']}")
        if not score_citation(f["answer"], f["uid"], valid)["hit"]:
            problems.append(f"train answer citation not parsed: {f['fact_id']}")
    for d in dev:
        if d["fact_id"] not in fact_ids:
            problems.append(f"dev fact not in train: {d['fact_id']}")
        if not score_citation(d["reference_answer"], d["uid"], valid)["hit"]:
            problems.append(f"dev reference citation not parsed: {d['eval_id']}")
    if problems:
        raise ValueError("bundle checks failed:\n" + "\n".join(problems[:20]))
    return {
        "train_facts": len(facts),
        "train_pairs_two_phrasings": sum(min(2, len(f["questions"])) for f in facts),
        "dev_items": len(dev),
        "provisions": len(provisions),
    }


def build(cfg: dict, config_path: str, owner: str | None) -> Path:
    b = cfg["bundle"]
    out = Path(b["out_dir"])
    if out.exists():
        shutil.rmtree(out)
    (out / "data").mkdir(parents=True)

    facts = read_jsonl(b["train"])
    dev = read_jsonl(b["dev"])
    provisions = provision_index(read_jsonl(b["sections"]))
    manifest = yaml.safe_load(Path(b["split_manifest"]).read_text(encoding="utf-8"))
    counts = check_inputs(facts, dev, provisions, manifest["leakage"]["holdout_uid_patterns"])

    shutil.copyfile(b["train"], out / "data" / "train.jsonl")
    shutil.copyfile(b["dev"], out / "data" / "dev_para.jsonl")
    (out / "data" / "provisions.json").write_text(json.dumps(provisions, indent=1), encoding="utf-8")
    shutil.copyfile(config_path, out / "train.yaml")
    for rel in b["package_files"]:
        dest = out / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(rel, dest)

    files = sorted(p for p in out.rglob("*") if p.is_file())
    record = {
        "bundle": b["name"],
        "run_version": cfg["run_version"],
        "base_model": cfg["base_model"]["name"],
        "system_prompt_hash": text_hash(cfg["system_prompt"]),
        "counts": counts,
        "git": git_state(),
        "files": {p.relative_to(out).as_posix(): sha256_file(p) for p in files},
    }
    (out / "bundle_manifest.json").write_text(json.dumps(record, indent=2), encoding="utf-8")

    if owner:
        meta = {"title": b["title"], "id": f"{owner}/{b['name']}", "licenses": [{"name": "CC-BY-4.0"}]}
        (out / "dataset-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    archive = shutil.make_archive(str(out), "zip", root_dir=out)
    print(json.dumps({"out_dir": str(out), "zip": archive, **counts, "git": record["git"]}, indent=2))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Kaggle dataset bundle for LoRA training")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--owner", default=os.environ.get("KAGGLE_USERNAME"))
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    build(cfg, args.config, args.owner)


if __name__ == "__main__":
    main()
import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

from ..data.build_split import read_jsonl, write_jsonl
from .generate import OUTPUT_FILES, Job, build_jobs

ISSUE_TYPES = ["none", "unsupported", "incorrect", "incomplete", "question_mismatch", "ambiguous_question", "trivial"]

SYSTEM_PROMPT = """You are auditing a question-answering dataset built from the Privacy Act 1988 (Cth). For each item you receive the provision text it was generated from, one or more phrasings of a question, and an answer. Judge each item strictly against the provision text, not against outside knowledge.

An item passes only if all of the following hold:
1. Every claim in the answer is stated in, or directly entailed by, the provision text.
2. The answer correctly and completely answers every phrasing of the question, and does not omit a condition or qualification in the provision that the question requires.
3. Each phrasing is self-contained and unambiguous for a reader who has not seen the provision text.
4. The item tests a substantive fact rather than something trivial, such as merely restating a heading.

If any condition fails, the verdict is "fail" and issue_type is the single most important problem:
- unsupported: the answer contains a claim not found in the provision text
- incorrect: the answer contradicts the provision text
- incomplete: the answer omits a condition, exception or element the question requires
- question_mismatch: at least one phrasing asks for something the answer does not address
- ambiguous_question: a phrasing cannot be understood or answered without the provision text
- trivial: the item tests nothing substantive
If the verdict is "pass", issue_type must be "none".

Keep each note under 25 words. Return exactly one result for every item id."""

USER_TEMPLATE = """Provision: {display} ({title})

<provision>
{text}
</provision>

Items to audit:
{items}"""


def output_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "verdict": {"type": "string", "enum": ["pass", "fail"]},
                        "issue_type": {"type": "string", "enum": ISSUE_TYPES},
                        "note": {"type": "string"},
                    },
                    "required": ["id", "verdict", "issue_type", "note"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["results"],
        "additionalProperties": False,
    }


def prompt_hash(model: str, effort: str | None) -> str:
    payload = SYSTEM_PROMPT + USER_TEMPLATE + json.dumps(output_schema(), sort_keys=True) + model + str(effort)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def strip_citation(answer: str) -> str:
    return answer.rsplit("[Source:", 1)[0].strip()


def chunk_key(fact_id: str) -> str:
    uid, rest = fact_id.split("#", 1)
    return f"{uid.replace(':', '_')}__{rest.split('-', 1)[0]}"


def normalise(result: dict) -> tuple[str, str]:
    verdict, issue = result.get("verdict"), result.get("issue_type")
    if (verdict == "pass") != (issue == "none"):
        return "fail", "inconsistent"
    return verdict, issue


def load_items(generated_dir: Path) -> list[dict]:
    items = []
    for name in OUTPUT_FILES.values():
        path = generated_dir / name
        if path.exists():
            items.extend(read_jsonl(path))
    return items


def group_items(items: list[dict], jobs: list[Job]) -> list[tuple[Job, list[dict]]]:
    by_key = {job.key: job for job in jobs}
    groups: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        groups[chunk_key(item["fact_id"])].append(item)
    missing = sorted(groups.keys() - by_key.keys())
    if missing:
        raise ValueError(f"no source chunk for keys {missing}")
    return [(by_key[k], groups[k]) for k in sorted(groups)]


def call_verifier(client, cfg: dict, job: Job, items: list[dict]) -> tuple[list[dict], dict]:
    ver = cfg["verifier"]
    payload = [{"id": it["fact_id"], "questions": it["questions"], "answer": strip_citation(it["answer"])} for it in items]
    extra = {"reasoning": {"effort": ver["reasoning_effort"]}} if ver.get("reasoning_effort") else {}
    resp = client.responses.create(
        model=ver["model"],
        instructions=SYSTEM_PROMPT,
        input=USER_TEMPLATE.format(
            display=job.row["display"],
            title=job.row["title"],
            text=job.text,
            items=json.dumps(payload, ensure_ascii=False, indent=1),
        ),
        text={"format": {"type": "json_schema", "name": "verification", "schema": output_schema(), "strict": True}},
        max_output_tokens=ver["max_output_tokens"],
        store=False,
        **extra,
    )
    results, parse_error = [], None
    if resp.status != "completed":
        parse_error = f"status_{resp.status}"
    else:
        try:
            results = json.loads(resp.output_text).get("results", [])
        except (json.JSONDecodeError, AttributeError) as exc:
            parse_error = type(exc).__name__
    usage = {
        "model": resp.model,
        "input_tokens": resp.usage.input_tokens,
        "output_tokens": resp.usage.output_tokens,
        "status": resp.status,
        "parse_error": parse_error,
    }
    return results, usage


def run_job(job: Job, items: list[dict], client, cfg: dict, cache_dir: Path, phash: str) -> dict:
    path = cache_dir / f"{job.key}.json"
    state = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    if state is None or state.get("prompt_hash") != phash:
        state = {"key": job.key, "prompt_hash": phash, "results": {}, "calls": []}
    known = {it["fact_id"] for it in items}
    for _ in range(cfg["run"]["max_rounds"]):
        pending = [it for it in items if it["fact_id"] not in state["results"]]
        if not pending:
            break
        results, usage = call_verifier(client, cfg, job, pending)
        state["calls"].append({**usage, "requested": len(pending), "returned": len(results)})
        for res in results:
            rid = res.get("id")
            if rid in known and rid not in state["results"]:
                verdict, issue = normalise(res)
                state["results"][rid] = {"verdict": verdict, "issue_type": issue, "note": res.get("note", "")}
        path.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")
    return state


def assemble(groups, states, meta: dict) -> tuple[dict[str, list[dict]], list[dict]]:
    verified: dict[str, list[dict]] = {split: [] for split in OUTPUT_FILES}
    results = []
    for (job, items), state in zip(groups, states):
        for item in items:
            res = state["results"].get(item["fact_id"], {"verdict": "unverified", "issue_type": "unverified", "note": ""})
            results.append({"fact_id": item["fact_id"], "uid": item["uid"], "unit": item["unit"], "split": item["split"], **res})
            if res["verdict"] == "pass":
                verified[item["split"]].append({**item, **meta})
    return verified, results


def pass_rate(counter: Counter) -> float:
    total = sum(counter.values())
    return round(counter["pass"] / total, 4) if total else 0.0


def summarise(results: list[dict], states: list[dict], meta: dict) -> dict:
    by_split: dict[str, Counter] = defaultdict(Counter)
    by_unit: dict[str, Counter] = defaultdict(Counter)
    issues: Counter = Counter()
    tokens: Counter = Counter()
    for res in results:
        by_split[res["split"]][res["verdict"]] += 1
        by_unit[res["unit"]][res["verdict"]] += 1
        if res["verdict"] != "pass":
            issues[res["issue_type"]] += 1
    for state in states:
        for call in state["calls"]:
            tokens["input"] += call["input_tokens"]
            tokens["output"] += call["output_tokens"]
            tokens["calls"] += 1
    return {
        **meta,
        "by_split": {k: {**dict(v), "pass_rate": pass_rate(v)} for k, v in by_split.items()},
        "by_unit": {k: {**dict(v), "pass_rate": pass_rate(v)} for k, v in sorted(by_unit.items())},
        "issues": dict(issues),
        "usage": dict(tokens),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Audit generated QA items against their source provisions with a second LLM.")
    ap.add_argument("--config", type=Path, default=Path("configs/qa_verification.yaml"))
    ap.add_argument("--uids", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    paths = {k: Path(v) for k, v in cfg["paths"].items()}
    gen_cfg = yaml.safe_load(paths["generation_config"].read_text(encoding="utf-8"))
    manifest = yaml.safe_load(Path(gen_cfg["paths"]["manifest"]).read_text(encoding="utf-8"))
    jobs = build_jobs(read_jsonl(Path(gen_cfg["paths"]["inputs"])), manifest, gen_cfg)
    items = load_items(paths["generated_dir"])
    if args.uids:
        wanted = set(args.uids.split(","))
        items = [it for it in items if it["uid"] in wanted]
    groups = group_items(items, jobs)
    if args.limit:
        groups = groups[: args.limit]

    phash = prompt_hash(cfg["verifier"]["model"], cfg["verifier"].get("reasoning_effort"))
    if args.dry_run:
        n_items = sum(len(its) for _, its in groups)
        print(f"groups={len(groups)} items={n_items} max_items_per_call={max(len(its) for _, its in groups)} prompt_hash={phash}")
        return 0

    import openai
    from dotenv import load_dotenv

    load_dotenv()
    client = openai.OpenAI(max_retries=cfg["verifier"]["max_retries"])
    paths["cache_dir"].mkdir(parents=True, exist_ok=True)

    states: list[dict | None] = [None] * len(groups)
    with ThreadPoolExecutor(max_workers=cfg["run"]["max_workers"]) as pool:
        futures = {
            pool.submit(run_job, job, its, client, cfg, paths["cache_dir"], phash): i
            for i, (job, its) in enumerate(groups)
        }
        for done, fut in enumerate(as_completed(futures), 1):
            i = futures[fut]
            states[i] = fut.result()
            n_pass = sum(r["verdict"] == "pass" for r in states[i]["results"].values())
            print(f"[{done}/{len(groups)}] {groups[i][0].key}: pass {n_pass}/{len(groups[i][1])}", flush=True)

    meta = {
        "verifier_model": cfg["verifier"]["model"],
        "verifier_reasoning_effort": cfg["verifier"].get("reasoning_effort"),
        "verifier_prompt_version": cfg["prompt_version"],
        "verifier_prompt_hash": phash,
    }
    verified, results = assemble(groups, states, meta)
    for split, rows in verified.items():
        if rows:
            write_jsonl(paths["out_dir"] / OUTPUT_FILES[split], rows)
    write_jsonl(paths["results"], results)
    report = summarise(results, states, meta)
    paths["report"].parent.mkdir(parents=True, exist_ok=True)
    paths["report"].write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"by_split={report['by_split']} issues={report['issues']} usage={report['usage']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
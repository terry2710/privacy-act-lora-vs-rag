import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

from ..data.build_split import read_jsonl, write_jsonl
from ..data.text_norm import match_key
from .build_eval import eval_record
from .validate import IDENTIFIER, POINTER, YES_NO_ANSWER, YES_NO_QUESTION, strip_external_refs
from .verify import strip_citation

TYPES = ["lexical", "syntactic", "scenario"]
WORD = re.compile(r"[a-z0-9]+")
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

SYSTEM_PROMPT = """You write held-out test questions for a question-answering benchmark on the Privacy Act 1988 (Cth). For each fact you receive its statement, the reference answer, the questions already used in training, and the types of new question required. Write one new question per required type.

Rules for every new question:
1. It must ask for exactly the same fact, so that the reference answer fully and correctly answers it without any additional information.
2. It must not reuse distinctive wording or sentence structure from the training questions. Change the vocabulary and the framing.
3. It must be self-contained, and must not contain any section, subsection, paragraph, clause, Part, Division, Schedule or Australian Privacy Principle number, nor phrases such as "this section" or "this Act". Refer to "the Privacy Act" where needed.
4. If the reference answer begins with "Yes" or "No", the question must be a yes/no question. Otherwise it must not be a yes/no question.

Question types:
- lexical: a direct question that expresses the same request with different key words and phrasing.
- syntactic: a direct question with a clearly different grammatical structure, for example starting from the condition, the actor or the consequence instead.
- scenario: a short concrete situation involving a generic actor (for example "a credit provider", "an agency", "an individual"), followed by a question whose answer is exactly the fact. Do not name real people or companies, and do not add details that change or extend the fact."""

USER_TEMPLATE = """Facts:
{facts}"""


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
                        "phrasings": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "type": {"type": "string", "enum": TYPES},
                                    "question": {"type": "string"},
                                },
                                "required": ["type", "question"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": ["id", "phrasings"],
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


def assign_types(fact_ids: list[str], per_fact: int) -> dict[str, list[str]]:
    return {fid: [TYPES[(i + k) % len(TYPES)] for k in range(per_fact)] for i, fid in enumerate(fact_ids)}


def tokens(text: str) -> list[str]:
    return WORD.findall(text.lower())


def ngram_jaccard(a: str, b: str, n: int = 4) -> float:
    ta, tb = tokens(a), tokens(b)
    ga = {tuple(ta[i : i + n]) for i in range(len(ta) - n + 1)}
    gb = {tuple(tb[i : i + n]) for i in range(len(tb) - n + 1)}
    return len(ga & gb) / len(ga | gb) if ga | gb else 0.0


def rouge_l(a: str, b: str) -> float:
    ta, tb = tokens(a), tokens(b)
    if not ta or not tb:
        return 0.0
    prev = [0] * (len(tb) + 1)
    for x in ta:
        cur = [0]
        for j, y in enumerate(tb, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    lcs = prev[-1]
    if lcs == 0:
        return 0.0
    p, r = lcs / len(ta), lcs / len(tb)
    return 2 * p * r / (p + r)

def final_question(text: str) -> str:
    parts = [p for p in SENTENCE_SPLIT.split(text.strip()) if p]
    return parts[-1] if parts else text.strip()

def check_paraphrase(question: str, ptype: str, wanted: list[str], fact: dict, accepted: list[str], limits: dict) -> tuple[list[str], dict]:
    errors = []
    if ptype not in wanted:
        errors.append("unrequested_type")
    if IDENTIFIER.search(strip_external_refs(question)):
        errors.append("identifier_in_question")
    if POINTER.search(question):
        errors.append("pointer_in_question")
    polar_answer = bool(YES_NO_ANSWER.match(fact["answer"].strip()))
    if polar_answer != bool(YES_NO_QUESTION.match(final_question(question))):
        errors.append("polarity_mismatch")
    overlap = {
        "max_4gram_jaccard": round(max(ngram_jaccard(question, q) for q in fact["questions"]), 3),
        "max_rouge_l": round(max(rouge_l(question, q) for q in fact["questions"]), 3),
    }
    if overlap["max_4gram_jaccard"] > limits["paraphrase_max_4gram_jaccard"]:
        errors.append("overlap_4gram")
    if overlap["max_rouge_l"] > limits["paraphrase_max_rouge_l"]:
        errors.append("overlap_rouge_l")
    if any(match_key(question) == match_key(q) for q in accepted):
        errors.append("duplicate")
    return errors, overlap


def build_tasks(train: dict[str, dict], selection: dict, manifest: dict) -> list[tuple[str, dict, list[str]]]:
    tasks = []
    for set_name, source_key in (("s_para", "s_mem"), ("dev_para", "dev_para")):
        per_fact = manifest["eval"][set_name]["phrasings_per_fact"]
        types = assign_types(selection[source_key], per_fact)
        tasks.extend((set_name, train[fid], types[fid]) for fid in selection[source_key])
    return tasks


def batch_tasks(tasks, batch_size: int) -> list[tuple[str, list]]:
    batches = []
    for set_name in ("s_para", "dev_para"):
        subset = [t for t in tasks if t[0] == set_name]
        for n, i in enumerate(range(0, len(subset), batch_size)):
            batches.append((f"{set_name}_{n:03d}", subset[i : i + batch_size]))
    return batches


def call_generator(client, cfg: dict, pending: list[tuple[dict, list[str]]]) -> tuple[list[dict], dict]:
    gen = cfg["generator"]
    facts = [
        {
            "id": fact["fact_id"],
            "statement": fact["statement"],
            "reference_answer": strip_citation(fact["answer"]),
            "training_questions": fact["questions"],
            "required_types": types,
        }
        for fact, types in pending
    ]
    extra = {"reasoning": {"effort": gen["reasoning_effort"]}} if gen.get("reasoning_effort") else {}
    resp = client.responses.create(
        model=gen["model"],
        instructions=SYSTEM_PROMPT,
        input=USER_TEMPLATE.format(facts=json.dumps(facts, ensure_ascii=False, indent=1)),
        text={"format": {"type": "json_schema", "name": "paraphrases", "schema": output_schema(), "strict": True}},
        max_output_tokens=gen["max_output_tokens"],
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

def revalidate(state: dict, by_id: dict, limits: dict) -> None:
    for fid, done in state["accepted"].items():
        if fid not in by_id:
            continue
        fact, types = by_id[fid]
        for ptype in list(done):
            others = [d["question"] for t, d in done.items() if t != ptype]
            errors, _ = check_paraphrase(done[ptype]["question"], ptype, types, fact, others, limits)
            if errors:
                question = done.pop(ptype)["question"]
                state["rejected"].append({"fact_id": fid, "type": ptype, "question": question, "errors": [f"revalidated:{e}" for e in errors]})

def run_batch(key: str, tasks: list, client, cfg: dict, limits: dict, cache_dir: Path, phash: str) -> dict:
    path = cache_dir / f"{key}.json"
    state = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    if state is None or state.get("prompt_hash") != phash:
        state = {"key": key, "prompt_hash": phash, "accepted": {}, "rejected": [], "calls": []}
    by_id = {fact["fact_id"]: (fact, types) for _, fact, types in tasks}
    revalidate(state, by_id, limits)
    for _ in range(cfg["run"]["max_rounds"]):
        pending = []
        for fid, (fact, types) in by_id.items():
            done = state["accepted"].get(fid, {})
            missing = [t for t in types if t not in done]
            if missing:
                pending.append((fact, missing))
        if not pending:
            break
        results, usage = call_generator(client, cfg, pending)
        state["calls"].append({**usage, "requested": sum(len(m) for _, m in pending)})
        wanted = {fact["fact_id"]: missing for fact, missing in pending}
        for res in results:
            fid = res.get("id")
            if fid not in wanted:
                continue
            fact = by_id[fid][0]
            done = state["accepted"].setdefault(fid, {})
            for item in res.get("phrasings", []):
                ptype, question = item.get("type"), (item.get("question") or "").strip()
                if ptype in done:
                    continue
                errors, overlap = check_paraphrase(question, ptype, wanted[fid], fact, [d["question"] for d in done.values()], limits)
                if errors:
                    state["rejected"].append({"fact_id": fid, "type": ptype, "question": question, "errors": errors, **overlap})
                else:
                    done[ptype] = {"question": question, **overlap}
        path.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")
    return state


def assemble(batches, states, meta: dict) -> tuple[dict[str, list[dict]], list[dict]]:
    out: dict[str, list[dict]] = {"s_para": [], "dev_para": []}
    shortfall = []
    for (key, tasks), state in zip(batches, states):
        for set_name, fact, types in tasks:
            done = state["accepted"].get(fact["fact_id"], {})
            for ptype in types:
                if ptype not in done:
                    shortfall.append({"set": set_name, "fact_id": fact["fact_id"], "type": ptype})
                    continue
                entry = done[ptype]
                out[set_name].append({
                    "fact": fact,
                    "question": entry["question"],
                    "extra": {
                        "paraphrase_type": ptype,
                        "max_4gram_jaccard": entry["max_4gram_jaccard"],
                        "max_rouge_l": entry["max_rouge_l"],
                        **meta,
                    },
                })
    records = {
        name: [eval_record(name, n, r["fact"], r["question"], **r["extra"]) for n, r in enumerate(rows, 1)]
        for name, rows in out.items()
    }
    return records, shortfall


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Generate held-out paraphrase test questions for S-para and dev-para.")
    ap.add_argument("--config", type=Path, default=Path("configs/paraphrase.yaml"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    paths = {k: Path(v) for k, v in cfg["paths"].items()}
    manifest = yaml.safe_load(paths["manifest"].read_text(encoding="utf-8"))
    train = {it["fact_id"]: it for it in read_jsonl(paths["train"])}
    selection = json.loads(paths["selection"].read_text(encoding="utf-8"))
    batches = batch_tasks(build_tasks(train, selection, manifest), cfg["run"]["batch_size"])
    if args.limit:
        batches = batches[: args.limit]
    gen = cfg["generator"]
    phash = prompt_hash(gen["model"], gen.get("reasoning_effort"))

    if args.dry_run:
        want = Counter(t for _, tasks in batches for _, _, types in tasks for t in types)
        n_facts = Counter(s for _, tasks in batches for s, _, _ in tasks)
        print(f"batches={len(batches)} facts={dict(n_facts)} phrasings_by_type={dict(want)} prompt_hash={phash}")
        return 0

    import openai
    from dotenv import load_dotenv

    load_dotenv()
    client = openai.OpenAI(max_retries=gen["max_retries"])
    paths["cache_dir"].mkdir(parents=True, exist_ok=True)
    limits = manifest["dedup"]

    states: list[dict | None] = [None] * len(batches)
    with ThreadPoolExecutor(max_workers=cfg["run"]["max_workers"]) as pool:
        futures = {
            pool.submit(run_batch, key, tasks, client, cfg, limits, paths["cache_dir"], phash): i
            for i, (key, tasks) in enumerate(batches)
        }
        for done, fut in enumerate(as_completed(futures), 1):
            i = futures[fut]
            states[i] = fut.result()
            got = sum(len(v) for v in states[i]["accepted"].values())
            want = sum(len(types) for _, _, types in batches[i][1])
            print(f"[{done}/{len(batches)}] {batches[i][0]}: {got}/{want}", flush=True)

    meta = {"paraphrase_model": gen["model"], "paraphrase_prompt_version": cfg["prompt_version"], "paraphrase_prompt_hash": phash}
    records, shortfall = assemble(batches, states, meta)
    for name, rows in records.items():
        write_jsonl(paths["eval_dir"] / f"{name}.jsonl", rows)
    rejects = [r for s in states for r in s["rejected"]]
    write_jsonl(paths["rejects"], rejects)
    all_rows = [r for rows in records.values() for r in rows]
    tokens_used = Counter()
    for s in states:
        for c in s["calls"]:
            tokens_used.update({"input": c["input_tokens"], "output": c["output_tokens"], "calls": 1})
    report = {
        **meta,
        "sizes": {k: len(v) for k, v in records.items()},
        "by_type": {k: dict(Counter(r["paraphrase_type"] for r in v)) for k, v in records.items()},
        "shortfall": shortfall,
        "rejected_by_reason": dict(Counter(e for r in rejects for e in r["errors"])),
        "overlap_mean": {
            "max_4gram_jaccard": round(sum(r["max_4gram_jaccard"] for r in all_rows) / max(1, len(all_rows)), 3),
            "max_rouge_l": round(sum(r["max_rouge_l"] for r in all_rows) / max(1, len(all_rows)), 3),
        },
        "usage": dict(tokens_used),
    }
    paths["report"].parent.mkdir(parents=True, exist_ok=True)
    paths["report"].write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"sizes={report['sizes']} shortfall={len(shortfall)} rejected={report['rejected_by_reason']} usage={report['usage']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
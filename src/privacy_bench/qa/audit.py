import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

from ..data.build_split import read_jsonl, write_jsonl
from .spot_check import cohen_kappa, wilson

VERDICTS = {"ok", "problem"}
ROW_FIELDS = ["check_id", "kind", "criterion", "provision", "question", "answer", "source_text"]

SYSTEM_PROMPT = """You are reviewing items from a question-answering benchmark built from the Privacy Act 1988 (Cth). Each item has a kind, a criterion and supporting material. Decide whether the item satisfies its criterion.

- kind "qa": judge strictly against the source text provided. The item is ok only if every claim in the answer is supported by the source text and the answer fully answers every question; multiple phrasings of the question are separated by "||". Ignore the bracketed citation at the end of the answer.
- kind "paraphrase": the item is ok only if the new question asks for exactly the same fact as the training questions, and the reference answer fully and correctly answers it without needing extra information.
- kind "neg": the item is ok only if its premise is correct, that is, the Privacy Act 1988 (Cth) as in force in 2026 contains no such provision, body or mechanism. Use your knowledge of the Act.

Return a verdict of "ok" or "problem" and a reason of at most 40 words."""

USER_TEMPLATE = """Kind: {kind}
Criterion: {criterion}
Provision: {provision}

Question: {question}

Answer: {answer}

Supporting material:
{source_text}"""

SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": sorted(VERDICTS)},
        "reason": {"type": "string"},
    },
    "required": ["verdict", "reason"],
    "additionalProperties": False,
}


def prompt_hash(auditor_cfg: dict) -> str:
    payload = SYSTEM_PROMPT + USER_TEMPLATE + json.dumps(SCHEMA, sort_keys=True) + json.dumps(auditor_cfg, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def render(row: dict) -> str:
    return USER_TEMPLATE.format(**{k: row.get(k, "") for k in ROW_FIELDS})


def call_anthropic(client, cfg: dict, row: dict) -> tuple[str, str, str, int, int]:
    output_config = {"format": {"type": "json_schema", "schema": SCHEMA}}
    if cfg.get("effort"):
        output_config["effort"] = cfg["effort"]
    resp = client.messages.create(
        model=cfg["model"],
        max_tokens=cfg["max_tokens"],
        system=SYSTEM_PROMPT,
        output_config=output_config,
        messages=[{"role": "user", "content": render(row)}],
    )
    text = "".join(b.text for b in resp.content if b.type == "text")
    return text, resp.model, resp.stop_reason, resp.usage.input_tokens, resp.usage.output_tokens


def call_openai(client, cfg: dict, row: dict) -> tuple[str, str, str, int, int]:
    extra = {"reasoning": {"effort": cfg["reasoning_effort"]}} if cfg.get("reasoning_effort") else {}
    resp = client.responses.create(
        model=cfg["model"],
        instructions=SYSTEM_PROMPT,
        input=render(row),
        text={"format": {"type": "json_schema", "name": "review", "schema": SCHEMA, "strict": True}},
        max_output_tokens=cfg["max_output_tokens"],
        store=False,
        **extra,
    )
    text = resp.output_text if resp.status == "completed" else ""
    return text, resp.model, resp.status, resp.usage.input_tokens, resp.usage.output_tokens


CALLERS = {"anthropic": call_anthropic, "openai": call_openai}


def review(name: str, cfg: dict, client, row: dict, cache_dir: Path, phash: str, attempts: int = 2) -> dict:
    path = cache_dir / name / f"{row['check_id']}.json"
    if path.exists():
        cached = json.loads(path.read_text(encoding="utf-8"))
        if cached.get("prompt_hash") == phash and cached.get("verdict") in VERDICTS:
            return cached
    record = {"check_id": row["check_id"], "auditor": name, "prompt_hash": phash, "verdict": None, "reason": "",
              "input_tokens": 0, "output_tokens": 0}
    for _ in range(attempts):
        text, model, status, tin, tout = CALLERS[cfg["provider"]](client, cfg, row)
        record.update(model=model, status=status)
        record["input_tokens"] += tin
        record["output_tokens"] += tout
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            continue
        if parsed.get("verdict") in VERDICTS:
            record.update(verdict=parsed["verdict"], reason=parsed.get("reason", ""))
            break
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    return record


def build_adjudication(rows: list[dict], results: list[dict], rng: random.Random) -> tuple[list[dict], dict]:
    by = {(r["check_id"], r["auditor"]): r for r in results}
    names = sorted({r["auditor"] for r in results})
    adj_rows, adj_key = [], {}
    for row in rows:
        verdicts = [by[(row["check_id"], n)]["verdict"] for n in names]
        if None not in verdicts and len(set(verdicts)) == 1:
            continue
        order = names[:]
        rng.shuffle(order)
        out = {k: row[k] for k in ROW_FIELDS}
        for i, n in enumerate(order, 1):
            r = by[(row["check_id"], n)]
            out[f"reviewer_{i}_verdict"] = r["verdict"] or "no answer"
            out[f"reviewer_{i}_reason"] = r["reason"]
        out.update(label="", note="")
        adj_rows.append(out)
        adj_key[row["check_id"]] = order
    return adj_rows, adj_key


def final_report(key: dict, results: list[dict], adjudicated: dict[str, str]) -> dict:
    by: dict[str, dict[str, str | None]] = {}
    for r in results:
        by.setdefault(r["check_id"], {})[r["auditor"]] = r["verdict"]
    names = sorted({r["auditor"] for r in results})
    final, missing = {}, []
    for cid in key:
        verdicts = [by[cid].get(n) for n in names]
        if None not in verdicts and len(set(verdicts)) == 1:
            final[cid] = verdicts[0]
        elif adjudicated.get(cid) in VERDICTS:
            final[cid] = adjudicated[cid]
        else:
            missing.append(cid)
    if missing:
        raise ValueError(f"adjudication label must be 'ok' or 'problem' for: {missing}")

    both = [(by[c][names[0]] == "ok", by[c][names[1]] == "ok") for c in key if None not in (by[c].get(names[0]), by[c].get(names[1]))]
    sided = Counter(n for c, label in adjudicated.items() for n in names if by[c].get(n) == label)
    strata: dict[str, Counter] = {}
    for cid, meta in key.items():
        strata.setdefault(meta["stratum"], Counter())[final[cid]] += 1
    verifier_pairs = [(meta["verifier_verdict"] == "pass", final[cid] == "ok") for cid, meta in key.items() if "verifier_verdict" in meta]
    confusion = Counter((("verifier_pass" if v else "verifier_fail"), ("final_ok" if f else "final_problem")) for v, f in verifier_pairs)
    tokens: dict[str, Counter] = {}
    for r in results:
        tokens.setdefault(r["auditor"], Counter()).update({"input": r.get("input_tokens", 0), "output": r.get("output_tokens", 0)})
    return {
        "auditors": names,
        "auditor_agreement": {
            "n": len(both),
            "raw": round(sum(a == b for a, b in both) / len(both), 4) if both else None,
            "cohen_kappa": cohen_kappa(both),
        },
        "adjudication": {"n": len(adjudicated), "sided_with": dict(sided)},
        "strata": {
            s: {"n": sum(c.values()), "ok": c["ok"], "ok_rate": round(c["ok"] / sum(c.values()), 4), "ok_rate_ci95": wilson(c["ok"], sum(c.values()))}
            for s, c in strata.items()
        },
        "verifier_vs_final": {
            "n": len(verifier_pairs),
            "confusion": {f"{a}/{b}": n for (a, b), n in sorted(confusion.items())},
            "raw_agreement": round(sum(v == f for v, f in verifier_pairs) / len(verifier_pairs), 4) if verifier_pairs else None,
            "cohen_kappa_on_sample": cohen_kappa(verifier_pairs),
            "note": "Verifier failures are oversampled; read per-stratum rates for population estimates.",
        },
        "usage": {k: dict(v) for k, v in tokens.items()},
    }


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Dual-model review of the spot-check sample, with human adjudication of disagreements.")
    ap.add_argument("command", choices=["run", "score"])
    ap.add_argument("--config", type=Path, default=Path("configs/audit.yaml"))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    paths = {k: Path(v) for k, v in cfg["paths"].items()}
    rows = read_csv(paths["sheet"])
    key = json.loads(paths["key"].read_text(encoding="utf-8"))

    if args.command == "run":
        if paths["adjudication"].exists():
            raise SystemExit(f"{paths['adjudication']} already exists; refusing to overwrite possible labels.")
        if args.limit:
            rows = rows[: args.limit]
        from dotenv import load_dotenv

        load_dotenv()
        clients = {}
        for name, acfg in cfg["auditors"].items():
            if acfg["provider"] == "anthropic":
                import anthropic

                clients[name] = anthropic.Anthropic(max_retries=cfg["run"]["max_retries"])
            else:
                import openai

                clients[name] = openai.OpenAI(max_retries=cfg["run"]["max_retries"])
        hashes = {name: prompt_hash(acfg) for name, acfg in cfg["auditors"].items()}
        results = []
        with ThreadPoolExecutor(max_workers=cfg["run"]["max_workers"]) as pool:
            futures = [
                pool.submit(review, name, acfg, clients[name], row, paths["cache_dir"], hashes[name])
                for name, acfg in cfg["auditors"].items()
                for row in rows
            ]
            for done, fut in enumerate(as_completed(futures), 1):
                results.append(fut.result())
                if done % 20 == 0 or done == len(futures):
                    print(f"[{done}/{len(futures)}] reviews done", flush=True)
        results.sort(key=lambda r: (r["check_id"], r["auditor"]))
        write_jsonl(paths["results"], results)
        if args.limit:
            print(f"pilot: {Counter((r['auditor'], r['verdict']) for r in results)}")
            return 0
        adj_rows, adj_key = build_adjudication(rows, results, random.Random(cfg["run"]["seed"]))
        with paths["adjudication"].open("w", encoding="utf-8-sig", newline="") as f:
            fields = ROW_FIELDS + ["reviewer_1_verdict", "reviewer_1_reason", "reviewer_2_verdict", "reviewer_2_reason", "label", "note"]
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(adj_rows)
        paths["adjudication_key"].write_text(json.dumps(adj_key, indent=1), encoding="utf-8")
        agree = len(rows) - len(adj_rows)
        print(f"auditors agree on {agree}/{len(rows)}; {len(adj_rows)} rows written to {paths['adjudication']}")
        return 0

    results = read_jsonl(paths["results"])
    adjudicated = {}
    if paths["adjudication"].exists():
        adjudicated = {r["check_id"]: r["label"].strip().lower() for r in read_csv(paths["adjudication"])}
    report = final_report(key, results, adjudicated)
    paths["report"].write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"auditor_agreement={report['auditor_agreement']} adjudication={report['adjudication']}")
    for name, s in report["strata"].items():
        print(f"{name}: ok {s['ok']}/{s['n']} ({s['ok_rate']:.1%}, 95% CI {s['ok_rate_ci95']})")
    print(f"verifier_vs_final={report['verifier_vs_final']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
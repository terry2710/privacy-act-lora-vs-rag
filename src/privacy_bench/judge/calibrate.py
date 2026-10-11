import random
from collections import Counter

QA_LABELS = [0, 1, 2]
NEG_LABELS = ["fail", "pass"]
ADJ_FIELDS = ["case_id", "kind", "question", "reference_or_premise", "provision_excerpt", "answer",
              "reviewer_1_label", "reviewer_1_reason", "reviewer_2_label", "reviewer_2_reason", "label", "note"]


def sample(answers: dict[tuple[str, str], list[dict]], arms: list[str], per_set: dict[str, int], seed: int) -> list[dict]:
    chosen = []
    for set_name, n in per_set.items():
        for arm in arms:
            pool = sorted(answers.get((set_name, arm), []), key=lambda r: r["eval_id"])
            if len(pool) < n:
                raise ValueError(f"only {len(pool)} answers for {set_name}/{arm}, need {n}")
            chosen.extend(random.Random(f"{seed}-{set_name}-{arm}").sample(pool, n))
    return chosen


def kappa(pairs: list[tuple], categories: list, weights: str | None = None) -> float | None:
    n = len(pairs)
    if n == 0:
        return None
    k = len(categories)
    idx = {c: i for i, c in enumerate(categories)}

    def w(i, j):
        if weights == "quadratic":
            return ((i - j) / (k - 1)) ** 2
        return 0.0 if i == j else 1.0

    observed = Counter((idx[a], idx[b]) for a, b in pairs)
    rows = Counter(idx[a] for a, _ in pairs)
    cols = Counter(idx[b] for _, b in pairs)
    d_obs = sum(w(i, j) * c for (i, j), c in observed.items()) / n
    d_exp = sum(w(i, j) * rows[i] * cols[j] for i in range(k) for j in range(k)) / (n * n)
    if d_exp == 0:
        return None
    return round(1 - d_obs / d_exp, 4)


def agreement(pairs: list[tuple], categories: list) -> dict:
    out = {
        "n": len(pairs),
        "raw": round(sum(a == b for a, b in pairs) / len(pairs), 4) if pairs else None,
        "kappa": kappa(pairs, categories),
    }
    if len(categories) > 2:
        out["kappa_quadratic"] = kappa(pairs, categories, "quadratic")
    return out


def by_judge(results: list[dict]) -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = {}
    for r in results:
        out.setdefault(r["key"], {})[r["judge"]] = r
    return out


def adjudication(tasks: list[dict], results: list[dict], judges: list[str], rng: random.Random) -> tuple[list[dict], dict]:
    by = by_judge(results)
    rows, key = [], {}
    shuffled = tasks[:]
    rng.shuffle(shuffled)
    for task in shuffled:
        labels = [by[task["key"]][j]["label"] for j in judges]
        if None not in labels and len(set(labels)) == 1:
            continue
        case_id = f"c{len(rows) + 1:03d}"
        order = judges[:]
        rng.shuffle(order)
        f = task["fields"]
        row = {
            "case_id": case_id,
            "kind": task["kind"],
            "question": f["question"],
            "reference_or_premise": f.get("reference", f.get("premise", "")),
            "provision_excerpt": f"{f['display']}: {f['evidence']}" if task["kind"] == "qa" else "",
            "answer": f["answer"],
        }
        for i, j in enumerate(order, 1):
            r = by[task["key"]][j]
            row[f"reviewer_{i}_label"] = "no answer" if r["label"] is None else r["label"]
            row[f"reviewer_{i}_reason"] = (r["verdict"] or {}).get("reason", "")
        row.update(label="", note="")
        rows.append(row)
        key[case_id] = {"key": task["key"], "order": order}
    return rows, key


def parse_label(kind: str, raw: str):
    raw = raw.strip().lower()
    if kind == "qa":
        return int(raw) if raw in ("0", "1", "2") else None
    return raw if raw in NEG_LABELS else None


def final_labels(tasks: list[dict], results: list[dict], judges: list[str], adj_key: dict, adj_rows: list[dict]) -> dict:
    by = by_judge(results)
    human = {}
    for row in adj_rows:
        task_key = adj_key[row["case_id"]]["key"]
        human[task_key] = parse_label(row["kind"], row.get("label", ""))
    final, missing = {}, []
    for task in tasks:
        labels = [by[task["key"]][j]["label"] for j in judges]
        if None not in labels and len(set(labels)) == 1:
            final[task["key"]] = labels[0]
        elif human.get(task["key"]) is not None:
            final[task["key"]] = human[task["key"]]
        else:
            missing.append(task["key"])
    if missing:
        raise ValueError(f"{len(missing)} disagreements lack a valid label (0/1/2 for qa, pass/fail for neg)")
    return final


def report(tasks: list[dict], results: list[dict], judges: list[str], final: dict | None, cfg: dict) -> dict:
    by = by_judge(results)
    primary, secondary = judges
    out = {"judges": {"primary": primary, "secondary": secondary}, "n": len(tasks), "judge_agreement": {}, "by_arm": {}}
    for kind, cats in (("qa", QA_LABELS), ("neg", NEG_LABELS)):
        ks = [t for t in tasks if t["kind"] == kind]
        pairs = [(by[t["key"]][primary]["label"], by[t["key"]][secondary]["label"]) for t in ks]
        out["judge_agreement"][kind] = agreement([p for p in pairs if None not in p], cats)
        out["judge_agreement"][kind]["no_answer"] = sum(None in p for p in pairs)
        if final is not None:
            for j in judges:
                jp = [(by[t["key"]][j]["label"], final[t["key"]]) for t in ks if by[t["key"]][j]["label"] is not None]
                out.setdefault("vs_final", {}).setdefault(j, {})[kind] = agreement(jp, cats)
    for t in tasks:
        arm = out["by_arm"].setdefault(t["arm"], {})
        for j in judges + (["final"] if final is not None else []):
            value = final[t["key"]] if j == "final" else by[t["key"]][j]["label"]
            if value is None:
                continue
            slot = arm.setdefault(j, {"qa_n": 0, "qa_sum": 0, "neg_n": 0, "neg_pass": 0})
            if t["kind"] == "qa":
                slot["qa_n"] += 1
                slot["qa_sum"] += value
            else:
                slot["neg_n"] += 1
                slot["neg_pass"] += value == "pass"
    for arm in out["by_arm"].values():
        for slot in arm.values():
            slot["qa_mean"] = round(slot.pop("qa_sum") / slot["qa_n"], 3) if slot["qa_n"] else None
            slot["neg_pass_rate"] = round(slot.pop("neg_pass") / slot["neg_n"], 3) if slot["neg_n"] else None
    qa = out["judge_agreement"]["qa"]
    neg = out["judge_agreement"]["neg"]
    neg_ok = neg["kappa"] >= cfg["min_kappa"] if neg["kappa"] is not None else (neg["raw"] or 0) >= cfg["min_raw_if_degenerate"]
    out["gate"] = {
        "min_kappa": cfg["min_kappa"],
        "qa_kappa_quadratic_ok": qa["kappa_quadratic"] is not None and qa["kappa_quadratic"] >= cfg["min_kappa"],
        "neg_ok": neg_ok,
    }
    out["gate"]["passed"] = out["gate"]["qa_kappa_quadratic_ok"] and neg_ok and final is not None
    return out

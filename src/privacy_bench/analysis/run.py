import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from ..judge.calibrate import QA_LABELS, agreement
from ..train.data import read_jsonl
from .stats import bootstrap, holm, mcnemar_exact, mean_diff, median_diff

ARMS = ["A0", "A1", "A2", "A3"]
QA_SETS = ["s_mem", "s_para", "h1_clean", "h1_parallel", "h2"]


def load(cfg: dict) -> tuple[dict, dict]:
    eval_dir = Path(cfg["eval_dir"])
    items = {s: read_jsonl(eval_dir / f"{s}.jsonl") for s in ["s_mem", "s_para", "h1", "h2", "s_neg"]}
    groups = {
        "s_mem": items["s_mem"],
        "s_para": items["s_para"],
        "h1_clean": [i for i in items["h1"] if not i.get("parallel_to_train")],
        "h1_parallel": [i for i in items["h1"] if i.get("parallel_to_train")],
        "h2": items["h2"],
        "s_neg": items["s_neg"],
    }
    source = {g: ("h1" if g.startswith("h1") else g) for g in groups}
    records = {}
    for group, group_items in groups.items():
        for arm in ARMS:
            answers = {r["eval_id"]: r for r in read_jsonl(Path(cfg["answers_dir"]) / source[group] / f"{arm}.jsonl")}
            labels = {}
            for judge in cfg["judges"]:
                path = Path(cfg["judgements_dir"]) / judge / source[group] / f"{arm}.jsonl"
                labels[judge] = {r["eval_id"]: r for r in read_jsonl(path)}
            rows = []
            for item in group_items:
                a = answers.get(item["eval_id"])
                if a is None:
                    continue
                row = {"eval_id": item["eval_id"], "fact_id": item.get("fact_id"), "answer": a}
                for judge, by_id in labels.items():
                    j = by_id.get(item["eval_id"])
                    row[judge] = j if j and j["label"] is not None else None
                rows.append(row)
            records[(group, arm)] = rows
    return records, groups


def score(row: dict, judge: str, group: str):
    j = row[judge]
    if j is None:
        return None
    if group == "s_neg":
        return 1.0 if j["label"] == "pass" else 0.0
    return float(j["label"])


def bad_citation(row: dict) -> bool:
    c = row["answer"]["citation"]
    return bool(c["has_nonexistent"] or c["wrong_existing"])


def paired(records: dict, group: str, arm_x: str, arm_y: str, value) -> tuple[np.ndarray, np.ndarray]:
    ys = {r["eval_id"]: value(r) for r in records[(group, arm_y)]}
    pairs = [(value(r), ys.get(r["eval_id"])) for r in records[(group, arm_x)]]
    pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
    return np.array([p[0] for p in pairs], dtype=float), np.array([p[1] for p in pairs], dtype=float)


def pooled(records: dict, groups: list[str], arm_x: str, arm_y: str, value) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = zip(*(paired(records, g, arm_x, arm_y, value) for g in groups))
    return np.concatenate(xs), np.concatenate(ys)


def compare_mean(x: np.ndarray, y: np.ndarray, cfg: dict) -> dict:
    out = bootstrap(mean_diff, [x, y], cfg["resamples"], cfg["seed"])
    out.update(n=len(x), mean_x=round(float(x.mean()), 4), mean_y=round(float(y.mean()), 4))
    return out


def compare_binary(x: np.ndarray, y: np.ndarray, cfg: dict) -> dict:
    out = compare_mean(x, y, cfg)
    out["mcnemar"] = mcnemar_exact(x.astype(bool), y.astype(bool))
    return out


def describe(records: dict, judges: list[str]) -> dict:
    out = {}
    for (group, arm), rows in records.items():
        if not rows:
            continue
        lat = {k: np.array([r["answer"]["latency"].get(k) or 0.0 for r in rows]) for k in ("retrieval_ms", "prefill_ms", "generation_ms", "total_ms")}
        cit = [r["answer"]["citation"] for r in rows]
        d = {
            "n": len(rows),
            "citation_hit": round(float(np.mean([c["hit"] for c in cit])), 4),
            "mention_hit": round(float(np.mean([c.get("mention_hit", c["hit"]) for c in cit])), 4),
            "tag_present": round(float(np.mean([c["tag_present"] for c in cit])), 4),
            "nonexistent_rate": round(float(np.mean([c["has_nonexistent"] for c in cit])), 4),
            "wrong_existing_rate": round(float(np.mean([c["wrong_existing"] for c in cit])), 4),
            "bad_citation_rate": round(float(np.mean([bad_citation(r) for r in rows])), 4),
            "bad_citation_given_tag": round(float(np.mean([bad_citation(r) for r in rows if r["answer"]["citation"]["tag_present"]])), 4)
            if any(c["tag_present"] for c in cit) else None,
            "completion_tokens_median": float(np.median([r["answer"].get("tokens", {}).get("completion", 0) for r in rows])),
            "latency_median_ms": {k: round(float(np.median(v)), 1) for k, v in lat.items()},
            "latency_p95_ms": {k: round(float(np.percentile(v, 95)), 1) for k, v in lat.items()},
        }
        for judge in judges:
            js = [r[judge] for r in rows if r[judge] is not None]
            if group == "s_neg":
                d[judge] = {
                    "n": len(js),
                    "pass_rate": round(float(np.mean([j["label"] == "pass" for j in js])), 4),
                    "premise_corrected_rate": round(float(np.mean([j["verdict"]["premise_corrected"] for j in js])), 4),
                    "invents_content_rate": round(float(np.mean([j["verdict"]["invents_content"] for j in js])), 4),
                }
            else:
                labels = [j["label"] for j in js]
                d[judge] = {
                    "n": len(js),
                    "accuracy_mean": round(float(np.mean(labels)), 4),
                    "fully_correct_rate": round(float(np.mean([v == 2 for v in labels])), 4),
                    "contradicts_rate": round(float(np.mean([j["verdict"]["contradicts_reference"] for j in js])), 4),
                }
        out.setdefault(group, {})[arm] = d
    return out


def memorisation_gap(records: dict, arm: str, judge: str, cfg: dict) -> dict:
    mem = {r["fact_id"]: score(r, judge, "s_mem") for r in records[("s_mem", arm)]}
    para = {}
    for r in records[("s_para", arm)]:
        s = score(r, judge, "s_para")
        if s is not None:
            para.setdefault(r["fact_id"], []).append(s)
    facts = sorted(f for f in mem if mem[f] is not None and f in para)
    x = np.array([mem[f] for f in facts])
    y = np.array([np.mean(para[f]) for f in facts])
    out = compare_mean(x, y, cfg)
    out["unit"] = "fact"
    return out


def accuracy_value(judge: str, group: str):
    return lambda r: score(r, judge, group)


def accuracy_tests(records: dict, judge: str, cfg: dict) -> dict:
    out = {}
    for group in ["s_mem", "s_para", "h1_clean", "h2"]:
        for x, y in [("A1", "A0"), ("A2", "A0"), ("A2", "A1"), ("A3", "A1"), ("A3", "A2")]:
            out[f"{group}:{x}-{y}"] = compare_mean(*paired(records, group, x, y, accuracy_value(judge, group)), cfg)
    for x, y in [("A2", "A1"), ("A3", "A1"), ("A3", "A2")]:
        out[f"s_neg_pass:{x}-{y}"] = compare_binary(*paired(records, "s_neg", x, y, accuracy_value(judge, "s_neg")), cfg)
    return out


def analyse(records: dict, groups: dict, cfg: dict) -> dict:
    primary, secondary = cfg["judges"]
    alpha = cfg["alpha"]
    acc = accuracy_value(primary, "s_para")

    h1a = compare_mean(*paired(records, "s_para", "A1", "A0", acc), cfg)
    h1b = compare_mean(*paired(records, "s_para", "A2", "A1", acc), cfg)
    h3 = compare_binary(*pooled(records, ["h1_clean", "h2"], "A1", "A0", lambda r: float(bad_citation(r))), cfg)
    family = {"H1a_s_para_A1_gt_A0": h1a, "H1b_s_para_A2_gt_A1": h1b, "H3_heldout_badcite_A1_gt_A0": h3}
    p_raw = {k: (v["mcnemar"]["p_two_sided"] if "mcnemar" in v else v["p_two_sided"]) for k, v in family.items()}
    p_adj = holm(p_raw)
    for k, v in family.items():
        v["p_used"] = p_raw[k]
        v["p_holm"] = p_adj[k]
        v["supported"] = bool(v["estimate"] > 0 and p_adj[k] < alpha)
    primary_out = {
        "family": family,
        "H1_supported": family["H1a_s_para_A1_gt_A0"]["supported"] and family["H1b_s_para_A2_gt_A1"]["supported"],
        "H3_supported": family["H3_heldout_badcite_A1_gt_A0"]["supported"],
        "note": "Holm-Bonferroni across the three directional comparisons; McNemar p for H3, paired bootstrap p for H1.",
    }

    gaps = {arm: memorisation_gap(records, arm, primary, cfg) for arm in ARMS}
    h4 = bootstrap(
        mean_diff,
        [np.array([score(r, primary, "h1_clean") for r in records[("h1_clean", "A0")] if r[primary]]),
         np.array([score(r, primary, "h2") for r in records[("h2", "A0")] if r[primary]])],
        cfg["resamples"], cfg["seed"], paired=False,
    )
    h4["supported"] = bool(h4["estimate"] > 0 and h4["ci95"][0] > 0)
    lat = {k: pooled(records, ["s_para", "h1_clean", "h2"], "A2", "A1", lambda r, k=k: r["answer"]["latency"].get(k) or 0.0)
           for k in ("total_ms", "retrieval_ms", "prefill_ms", "generation_ms")}
    h5 = {
        "total_median_diff": bootstrap(median_diff, list(lat["total_ms"]), cfg["resamples"], cfg["seed"]),
        "mean_diff_by_component": {k: compare_mean(*v, cfg) for k, v in lat.items()},
        "prefill_minus_retrieval": bootstrap(
            mean_diff, [lat["prefill_ms"][0] - lat["prefill_ms"][1], lat["retrieval_ms"][0]], cfg["resamples"], cfg["seed"]),
        "sets": ["s_para", "h1_clean", "h2"],
    }
    h5["supported"] = bool(h5["total_median_diff"]["ci95"][0] > 0 and h5["prefill_minus_retrieval"]["ci95"][0] > 0)
    secondary_out = {
        "H2_memorisation_gap_A1": {**gaps["A1"], "supported": bool(gaps["A1"]["ci95"][0] > 0),
                                   "ablation_part": "not tested: the one-phrasing ablation was not served or scored on S-mem and S-para"},
        "memorisation_gap_by_arm": gaps,
        "H4_A0_h1clean_gt_h2": h4,
        "H5_latency_A2_vs_A1": h5,
        "note": "Uncorrected; reported as secondary.",
    }

    parallel = []
    for r in records[("h1_parallel", "A1")]:
        c = r["answer"]["citation"]
        parallel.append({"eval_id": r["eval_id"], "cited": c["cited"], "hit": c["hit"], "cites_s44": "s:44" in c["cited"],
                         "accuracy": r[primary]["label"] if r[primary] else None})
    cite = {}
    for group in ["s_para", "h1_clean", "h2"]:
        for x, y in [("A3", "A1"), ("A3", "A2")]:
            cite[f"{group}:{x}-{y}"] = compare_binary(*paired(records, group, x, y, lambda r: float(r["answer"]["citation"]["hit"])), cfg)
    exploratory = {
        "H6_h1_parallel_A1": parallel,
        "H7_s_neg_pass_A2_gt_A1": compare_binary(*paired(records, "s_neg", "A2", "A1", accuracy_value(primary, "s_neg")), cfg),
        "accuracy_comparisons": accuracy_tests(records, primary, cfg),
        "citation_hit_comparisons": cite,
        "note": "Uncorrected; reported descriptively.",
    }

    second = accuracy_value(secondary, "s_para")
    agreement_by_set = {}
    for group in QA_SETS:
        pairs = [(r[primary]["label"], r[secondary]["label"]) for arm in ARMS for r in records[(group, arm)] if r[primary] and r[secondary]]
        agreement_by_set[group] = agreement(pairs, QA_LABELS)
    neg_pairs = [(r[primary]["label"], r[secondary]["label"]) for arm in ARMS for r in records[("s_neg", arm)] if r[primary] and r[secondary]]
    agreement_by_set["s_neg"] = agreement(neg_pairs, ["fail", "pass"])
    sensitivity = {
        "judge": secondary,
        "H1a_s_para_A1_gt_A0": compare_mean(*paired(records, "s_para", "A1", "A0", second), cfg),
        "H1b_s_para_A2_gt_A1": compare_mean(*paired(records, "s_para", "A2", "A1", second), cfg),
        "accuracy_comparisons": accuracy_tests(records, secondary, cfg),
        "judge_agreement_full_run": agreement_by_set,
    }
    return {
        "config": {k: cfg[k] for k in ("seed", "resamples", "alpha", "judges")},
        "set_sizes": {g: len(v) for g, v in groups.items()},
        "descriptive": describe(records, cfg["judges"]),
        "primary": primary_out,
        "secondary": secondary_out,
        "exploratory": exploratory,
        "sensitivity": sensitivity,
    }


def fmt_p(p: float) -> str:
    return f"{p:.1e}" if p < 1e-4 else f"{p:.4f}"


def fmt(d: dict) -> str:
    lo, hi = d["ci95"]
    return f"{d['estimate']:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def markdown(result: dict) -> str:
    primary, secondary = result["config"]["judges"]
    lines = ["# Phase 3 analysis", "", f"Primary judge: {primary}. Sensitivity judge: {secondary}. "
             f"Paired bootstrap with {result['config']['resamples']} resamples; intervals are 95%.", "",
             "## Accuracy by set and arm (primary judge, mean on 0-2)", "",
             "| Set | A0 | A1 | A2 | A3 |", "|---|---|---|---|---|"]
    desc = result["descriptive"]
    for group in QA_SETS:
        lines.append(f"| {group} | " + " | ".join(f"{desc[group][a][primary]['accuracy_mean']:.3f}" for a in ARMS) + " |")
    lines.append("| s_neg pass | " + " | ".join(f"{desc['s_neg'][a][primary]['pass_rate']:.2f}" for a in ARMS) + " |")
    lines += ["", "## Citations on held-out provisions (H1-clean and H2)", "",
              "| Set | Arm | Tag present | Citation hit | Wrong or non-existent | Wrong or non-existent, given a tag |", "|---|---|---|---|---|---|"]
    for group in ["h1_clean", "h2"]:
        for a in ARMS:
            d = desc[group][a]
            given = "n/a" if d["bad_citation_given_tag"] is None else f"{d['bad_citation_given_tag']:.2f}"
            lines.append(f"| {group} | {a} | {d['tag_present']:.2f} | {d['citation_hit']:.2f} | {d['bad_citation_rate']:.2f} | {given} |")
    lines += ["", "## Primary hypotheses (Holm-Bonferroni)", "", "| Comparison | Estimate [95% CI] | p | p (Holm) | Supported |", "|---|---|---|---|---|"]
    for k, v in result["primary"]["family"].items():
        lines.append(f"| {k} | {fmt(v)} | {fmt_p(v['p_used'])} | {fmt_p(v['p_holm'])} | {v['supported']} |")
    lines += ["", f"H1 supported: {result['primary']['H1_supported']}. H3 supported: {result['primary']['H3_supported']}.", "",
              "## Secondary hypotheses (uncorrected)", ""]
    s = result["secondary"]
    lines.append(f"- H2, A1 memorisation gap: {fmt(s['H2_memorisation_gap_A1'])}, supported: {s['H2_memorisation_gap_A1']['supported']}; ablation part not tested.")
    lines.append(f"- H4, A0 H1-clean minus H2: {fmt(s['H4_A0_h1clean_gt_h2'])}, supported: {s['H4_A0_h1clean_gt_h2']['supported']}.")
    h5 = s["H5_latency_A2_vs_A1"]
    comp = h5["mean_diff_by_component"]
    lines.append(f"- H5, A2 minus A1 median total latency (ms): {fmt(h5['total_median_diff'])}; mean component differences: "
                 + ", ".join(f"{k} {comp[k]['estimate']:+.0f}" for k in comp) + f"; supported: {h5['supported']}.")
    lines.append("- Memorisation gap by arm: " + ", ".join(f"{a} {fmt(g)}" for a, g in s["memorisation_gap_by_arm"].items()) + ".")
    e = result["exploratory"]
    h6 = e["H6_h1_parallel_A1"]
    h7 = e["H7_s_neg_pass_A2_gt_A1"]
    lines += ["", "## Exploratory hypotheses (uncorrected)", "",
              f"- H6, A1 on H1-parallel: cites s 44 on {sum(x['cites_s44'] for x in h6)} of {len(h6)} items and s 26WU on {sum(x['hit'] for x in h6)}.",
              f"- H7, S-neg premise correction A2 minus A1: {fmt(h7)}, McNemar p {fmt_p(h7['mcnemar']['p_two_sided'])}."]
    lines += ["", "## Exploratory accuracy comparisons (primary judge, uncorrected)", "", "| Comparison | Estimate [95% CI] | p |", "|---|---|---|"]
    for k, v in result["exploratory"]["accuracy_comparisons"].items():
        lines.append(f"| {k} | {fmt(v)} | {fmt_p(v.get('mcnemar', {}).get('p_two_sided', v['p_two_sided']))} |")
    lines += ["", "## Sensitivity: same comparisons with the secondary judge", "", "| Comparison | Primary | Secondary |", "|---|---|---|"]
    sens = result["sensitivity"]["accuracy_comparisons"]
    for k, v in result["exploratory"]["accuracy_comparisons"].items():
        lines.append(f"| {k} | {fmt(v)} | {fmt(sens[k])} |")
    lines += ["", "## Judge agreement on the full run", "", "| Set | n | Raw | Kappa | Weighted kappa |", "|---|---|---|---|---|"]
    for group, a in result["sensitivity"]["judge_agreement_full_run"].items():
        wk = a.get("kappa_quadratic")
        lines.append(f"| {group} | {a['n']} | {a['raw']:.3f} | {a['kappa']:.3f} | {'n/a' if wk is None else f'{wk:.3f}'} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Pre-registered statistical analysis of the Phase 3 evaluation")
    ap.add_argument("--config", type=Path, default=Path("configs/analysis.yaml"))
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    records, groups = load(cfg)
    result = analyse(records, groups, cfg)
    Path(cfg["out_json"]).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg["out_json"]).write_text(json.dumps(result, indent=2), encoding="utf-8")
    Path(cfg["out_md"]).write_text(markdown(result), encoding="utf-8")
    print(markdown(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

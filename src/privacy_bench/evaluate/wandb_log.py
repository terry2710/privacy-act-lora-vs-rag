import argparse
import json
import os
from pathlib import Path

from ..train.bundle import git_state

ARMS = ["A0", "A1", "A2", "A3"]
QA_SETS = ["s_mem", "s_para", "h1_clean", "h1_parallel", "h2"]


def evaluation_payload(analysis: dict, calibration: dict | None) -> dict:
    judges = analysis["config"]["judges"]
    desc = analysis["descriptive"]
    summary, accuracy_rows, citation_rows = {}, [], []
    for group in QA_SETS + ["s_neg"]:
        for arm in ARMS:
            d = desc[group][arm]
            citation_rows.append([group, arm, d["tag_present"], d["citation_hit"], d["mention_hit"], d["bad_citation_rate"],
                                  d["latency_median_ms"]["total_ms"], d["latency_p95_ms"]["total_ms"]])
            for judge in judges:
                j = d[judge]
                if group == "s_neg":
                    summary[f"{judge}/s_neg_pass/{arm}"] = j["pass_rate"]
                    accuracy_rows.append([group, arm, judge, j["pass_rate"], None, j["n"]])
                else:
                    summary[f"{judge}/accuracy/{group}/{arm}"] = j["accuracy_mean"]
                    accuracy_rows.append([group, arm, judge, j["accuracy_mean"], j["fully_correct_rate"], j["n"]])
    hypothesis_rows = []
    for name, v in analysis["primary"]["family"].items():
        hypothesis_rows.append(["primary", name, v["estimate"], v["ci95"][0], v["ci95"][1], v["p_used"], v["p_holm"], v["supported"]])
        summary[f"primary/{name}/supported"] = v["supported"]
        summary[f"primary/{name}/estimate"] = v["estimate"]
    s = analysis["secondary"]
    for name, v in (("H2_memorisation_gap_A1", s["H2_memorisation_gap_A1"]), ("H4_A0_h1clean_gt_h2", s["H4_A0_h1clean_gt_h2"]),
                    ("H5_total_median_diff_ms", s["H5_latency_A2_vs_A1"]["total_median_diff"])):
        hypothesis_rows.append(["secondary", name, v["estimate"], v["ci95"][0], v["ci95"][1], v["p_two_sided"], None, v.get("supported")])
    summary["primary/H1_supported"] = analysis["primary"]["H1_supported"]
    summary["primary/H3_supported"] = analysis["primary"]["H3_supported"]
    config = {**analysis["config"], "set_sizes": analysis["set_sizes"]}
    if calibration:
        config.update(judge_prompt_version=calibration.get("prompt_version"), judge_prompt_hashes=calibration.get("prompt_hashes"),
                      judge_models=calibration.get("models"))
        summary["calibration/qa_kappa_quadratic"] = calibration["judge_agreement"]["qa"]["kappa_quadratic"]
        summary["calibration/neg_kappa"] = calibration["judge_agreement"]["neg"]["kappa"]
    return {
        "config": config,
        "summary": summary,
        "tables": {
            "accuracy": (["set", "arm", "judge", "mean_or_pass_rate", "fully_correct_rate", "n"], accuracy_rows),
            "citations_latency": (["set", "arm", "tag_present", "citation_hit", "mention_hit", "bad_citation_rate", "median_total_ms", "p95_total_ms"], citation_rows),
            "hypotheses": (["family", "name", "estimate", "ci_low", "ci_high", "p", "p_holm", "supported"], hypothesis_rows),
        },
    }


def gate_payload(report: dict) -> dict:
    summary = {"gate/passed": report["decision"]["passed"], "gate/dry_run": report.get("dry_run")}
    for role in ("production", "candidate"):
        for key, value in report[role].items():
            if isinstance(value, (int, float, bool)):
                summary[f"{role}/{key}"] = value
    for check, ok in report["decision"].get("checks", {}).items():
        summary[f"gate/check/{check}"] = ok
    config = {"model": report["model"], "production_version": report["production"]["version"],
              "candidate_version": report["candidate"]["version"], "thresholds": report["thresholds"]}
    return {"config": config, "summary": summary, "tables": {}}


def latest(pattern: str) -> Path:
    paths = sorted(Path().glob(pattern), key=lambda p: p.stat().st_mtime)
    if not paths:
        raise SystemExit(f"no file matches {pattern}")
    return paths[-1]


def log(kind: str, payload: dict, files: list[Path], project: str, wandb_module=None) -> dict:
    wandb = wandb_module
    if wandb is None:
        import wandb
    git = git_state()
    run = wandb.init(project=project, job_type=kind, tags=[kind, "n8n"],
                     name=f"{kind}-{(git['commit'] or 'nogit')[:7]}", config={**payload["config"], "git": git})
    for key, value in payload["summary"].items():
        run.summary[key] = value
    for name, (columns, rows) in payload["tables"].items():
        run.log({name: wandb.Table(columns=columns, data=rows)})
    artifact = wandb.Artifact(f"phase3-{kind}", type=kind)
    for path in files:
        artifact.add_file(str(path), name=path.name)
    run.log_artifact(artifact)
    out = {"kind": kind, "run_id": run.id, "url": run.url}
    run.finish()
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Log the evaluation analysis or the latest promotion gate report to Weights & Biases")
    ap.add_argument("kind", choices=["evaluation", "gate"])
    ap.add_argument("--analysis", type=Path, default=Path("results/phase3/analysis.json"))
    ap.add_argument("--calibration", type=Path, default=Path("results/phase3/calibration/report.json"))
    ap.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "privacy-act-lora-vs-rag"))
    args = ap.parse_args(argv)
    from dotenv import load_dotenv

    load_dotenv()
    if args.kind == "evaluation":
        analysis = json.loads(args.analysis.read_text(encoding="utf-8"))
        calibration = json.loads(args.calibration.read_text(encoding="utf-8")) if args.calibration.exists() else None
        payload = evaluation_payload(analysis, calibration)
        files = [p for p in (args.analysis, args.analysis.with_suffix(".md"), args.calibration) if p.exists()]
    else:
        path = latest("results/phase3/gate_v*.json")
        payload = gate_payload(json.loads(path.read_text(encoding="utf-8")))
        files = [path]
    print(json.dumps(log(args.kind, payload, files, args.project)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

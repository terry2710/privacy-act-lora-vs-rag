import argparse
import json
import os
import time
from pathlib import Path
from urllib.parse import unquote, urlparse

import yaml

from ..train.data import read_jsonl
from .gate import ask, evaluate_gate, measure

PRODUCTION = "production"
CANDIDATE = "candidate"


def connect(cfg: dict):
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    import mlflow
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(cfg["tracking_uri"])
    return mlflow, MlflowClient()


def alias_version(client, name: str, alias: str):
    from mlflow.exceptions import MlflowException

    try:
        return client.get_model_version_by_alias(name, alias)
    except MlflowException:
        return None


def register(cfg: dict, key: str) -> dict:
    mlflow, client = connect(cfg)
    vc = cfg["versions"][key]
    exp = client.get_experiment_by_name(cfg["experiment"])
    exp_id = exp.experiment_id if exp else client.create_experiment(
        cfg["experiment"], artifact_location=Path(cfg["artifact_root"]).resolve().as_uri()
    )
    records = {name: json.loads(Path(path).read_text(encoding="utf-8")) for name, path in vc["records"].items()}
    gguf = records["gguf"]
    merge = records["merge"]
    with mlflow.start_run(experiment_id=exp_id, run_name=f"register-{key}") as run:
        mlflow.log_params({
            "arm": vc["arm"],
            "base_model": merge["base_model"],
            "base_revision": merge["base_revision"],
            "adapter_sha256": merge["adapter_sha256"] or "none",
            "gguf_sha256": gguf["sha256"],
            "outtype": gguf["outtype"],
            "llama_cpp_commit": gguf["llama_cpp_commit"] or "unknown",
            "wandb_run": vc.get("wandb_run", "none"),
        })
        if "parity" in records:
            p = records["parity"]["parity"]
            mlflow.log_metrics({"parity_hit_diff": p["hit_diff"], "parity_agreement": p["hit_agreement"]})
        for path in vc["records"].values():
            mlflow.log_artifact(path, artifact_path="records")
        mlflow.log_artifact(vc["gguf"], artifact_path="gguf")
        run_id = run.info.run_id
        source = f"{run.info.artifact_uri}/gguf"

    try:
        client.create_registered_model(cfg["model_name"])
    except Exception:
        pass
    tags = {"arm": vc["arm"], "gguf_sha256": gguf["sha256"], "registry_key": key}
    if "parity" in vc["records"]:
        tags["parity_report"] = vc["records"]["parity"]
    mv = client.create_model_version(cfg["model_name"], source, run_id=run_id, tags=tags)
    alias = vc.get("initial_alias", CANDIDATE)
    if alias == PRODUCTION and alias_version(client, cfg["model_name"], PRODUCTION) is not None:
        alias = CANDIDATE
    client.set_registered_model_alias(cfg["model_name"], alias, mv.version)
    return {"version": str(mv.version), "alias": alias, "run_id": run_id, "source": source}


def promote(cfg: dict, version: str, report: str | None) -> dict:
    _, client = connect(cfg)
    name = cfg["model_name"]
    current = alias_version(client, name, PRODUCTION)
    client.set_registered_model_alias(name, PRODUCTION, version)
    client.set_model_version_tag(name, version, "promoted_at", time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    if current is not None and str(current.version) != str(version):
        client.set_model_version_tag(name, version, "previous_production", str(current.version))
    if report:
        client.set_model_version_tag(name, version, "gate_report", report)
    candidate = alias_version(client, name, CANDIDATE)
    if candidate is not None and str(candidate.version) == str(version):
        client.delete_registered_model_alias(name, CANDIDATE)
    return {"production": str(version), "previous": str(current.version) if current else None}


def rollback(cfg: dict) -> dict:
    _, client = connect(cfg)
    name = cfg["model_name"]
    current = alias_version(client, name, PRODUCTION)
    if current is None or "previous_production" not in current.tags:
        raise SystemExit("no previous production version recorded")
    previous = current.tags["previous_production"]
    client.set_registered_model_alias(name, PRODUCTION, previous)
    client.set_model_version_tag(name, current.version, "rolled_back_at", time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    return {"production": previous, "rolled_back_from": str(current.version)}


def resolve(cfg: dict, alias: str) -> dict:
    _, client = connect(cfg)
    mv = alias_version(client, cfg["model_name"], alias)
    if mv is None:
        raise SystemExit(f"no version holds alias {alias}")
    source_dir = Path(unquote(urlparse(client.get_model_version_download_uri(cfg["model_name"], mv.version)).path))
    ggufs = sorted(source_dir.glob("*.gguf"))
    return {"alias": alias, "version": str(mv.version), "arm": mv.tags.get("arm"), "gguf": str(ggufs[0]) if ggufs else None, "tags": dict(mv.tags)}


def gate(cfg: dict, dry_run: bool, ask_fn=None) -> dict:
    _, client = connect(cfg)
    name = cfg["model_name"]
    gc = cfg["gate"]
    prod = alias_version(client, name, PRODUCTION)
    cand = alias_version(client, name, CANDIDATE)
    if prod is None or cand is None:
        raise SystemExit("gate needs both a production and a candidate version")
    parity = None
    if "parity_report" in cand.tags:
        parity = json.loads(Path(cand.tags["parity_report"]).read_text(encoding="utf-8"))["parity"]
    items = read_jsonl(gc["dev"])
    valid = {s["uid"] for s in read_jsonl(gc["sections"])}
    ask_fn = ask_fn or (lambda q, arm: ask(gc["api"], q, arm, gc["timeout_seconds"]))
    prod_run = measure(prod.tags["arm"], items, valid, ask_fn, gc["warmup"])
    cand_run = measure(cand.tags["arm"], items, valid, ask_fn, gc["warmup"])
    decision = evaluate_gate(cand_run["metrics"], prod_run["metrics"], parity, gc)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    report_path = Path(gc["out_dir"]) / f"gate_v{cand.version}_{stamp}.json"
    report = {
        "model": name,
        "production": {"version": str(prod.version), **prod_run["metrics"]},
        "candidate": {"version": str(cand.version), **cand_run["metrics"]},
        "parity": parity,
        "thresholds": {k: gc[k] for k in ("max_hit_diff", "min_agreement", "min_tag_present", "max_latency_ratio")},
        "decision": decision,
        "dry_run": dry_run,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    client.set_model_version_tag(name, cand.version, "gate_passed", str(decision["passed"]).lower())
    client.set_model_version_tag(name, cand.version, "gate_report", str(report_path))
    if decision["passed"] and not dry_run:
        report["promotion"] = promote(cfg, cand.version, str(report_path))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="MLflow model registry: register, gate, promote, roll back and resolve versions")
    parser.add_argument("--config", default="configs/registry.yaml")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("register").add_argument("key", choices=["a0", "a1"])
    sub.add_parser("gate").add_argument("--dry-run", action="store_true")
    sub.add_parser("promote").add_argument("version")
    sub.add_parser("rollback")
    sub.add_parser("resolve").add_argument("alias", nargs="?", default=PRODUCTION)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    if args.cmd == "register":
        out = register(cfg, args.key)
    elif args.cmd == "gate":
        out = gate(cfg, args.dry_run)
        out = {"decision": out["decision"], "production": out["production"], "candidate": out["candidate"], "promotion": out.get("promotion")}
    elif args.cmd == "promote":
        out = promote(cfg, args.version, None)
    elif args.cmd == "rollback":
        out = rollback(cfg)
    else:
        out = resolve(cfg, args.alias)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

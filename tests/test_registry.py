import json

import pytest

from privacy_bench.registry.gate import evaluate_gate, measure

GATE = {"max_hit_diff": 0.03, "min_agreement": 0.9, "min_tag_present": 0.95, "max_latency_ratio": 1.2}
PARITY = {"hit_diff": 0.01, "hit_agreement": 0.95}


def metrics(hit, tag, p95, failures=0):
    return {"citation_hit": hit, "tag_present": tag, "p95_ms": p95, "failures": failures}


def test_gate_passes_only_when_all_checks_hold():
    assert evaluate_gate(metrics(0.85, 0.99, 1000), metrics(0.03, 0.77, 900), PARITY, GATE)["passed"]
    slow = evaluate_gate(metrics(0.85, 0.99, 1200), metrics(0.03, 0.77, 900), PARITY, GATE)
    assert not slow["passed"] and not slow["checks"]["operational"]
    worse = evaluate_gate(metrics(0.03, 0.99, 900), metrics(0.03, 0.77, 900), PARITY, GATE)
    assert not worse["checks"]["quality_floor"]
    assert not evaluate_gate(metrics(0.85, 0.99, 900), metrics(0.03, 0.77, 900), None, GATE)["checks"]["conversion_parity"]
    assert not evaluate_gate(metrics(0.85, 0.90, 900), metrics(0.03, 0.77, 900), PARITY, GATE)["checks"]["format_compliance"]


def test_measure_skips_warmup_and_counts_failures():
    items = [{"eval_id": f"d{i}", "question": f"q{i}", "uid": "s:6"} for i in range(3)]
    calls = []

    def ask_fn(q, arm):
        calls.append(q)
        if q == "q2":
            raise OSError("down")
        return {"answer": "A. [Source: s 6]", "latency": {"total_ms": 100.0}}

    out = measure("A1", items, {"s:6"}, ask_fn, warmup=1)
    m = out["metrics"]
    assert calls[0] == "q0" and len(calls) == 4
    assert m["failures"] == 1 and m["n"] == 2 and m["citation_hit"] == 1.0 and m["p95_ms"] == 100.0


def test_register_gate_promote_rollback(tmp_path, monkeypatch):
    pytest.importorskip("mlflow")
    from privacy_bench.registry import registry

    monkeypatch.chdir(tmp_path)
    (tmp_path / "m").mkdir()
    for key in ("a0", "a1"):
        (tmp_path / "m" / f"{key}.gguf").write_bytes(b"GGUF" + key.encode())
        (tmp_path / "m" / f"{key}_gguf.json").write_text(json.dumps({"sha256": key * 8, "outtype": "q8_0", "llama_cpp_commit": "abc"}))
        (tmp_path / "m" / f"{key}_merge.json").write_text(json.dumps({"base_model": "Qwen", "base_revision": "rev", "adapter_sha256": None if key == "a0" else "ad"}))
    (tmp_path / "m" / "parity.json").write_text(json.dumps({"parity": PARITY}))
    (tmp_path / "dev.jsonl").write_text("\n".join(json.dumps({"eval_id": f"d{i}", "question": f"q{i}", "uid": "s:6"}) for i in range(4)))
    (tmp_path / "sections.jsonl").write_text(json.dumps({"uid": "s:6"}))
    cfg = {
        "tracking_uri": f"sqlite:///{tmp_path}/mlflow.db",
        "artifact_root": str(tmp_path / "artifacts"),
        "experiment": "model-registry",
        "model_name": "palr-test",
        "versions": {
            "a0": {"arm": "A0", "gguf": "m/a0.gguf", "records": {"gguf": "m/a0_gguf.json", "merge": "m/a0_merge.json"}, "initial_alias": "production"},
            "a1": {"arm": "A1", "gguf": "m/a1.gguf", "records": {"gguf": "m/a1_gguf.json", "merge": "m/a1_merge.json", "parity": "m/parity.json"}, "initial_alias": "candidate"},
        },
        "gate": {**GATE, "dev": "dev.jsonl", "sections": "sections.jsonl", "warmup": 1, "timeout_seconds": 1, "out_dir": "results", "api": "http://unused"},
    }
    assert registry.register(cfg, "a0")["alias"] == "production"
    assert registry.register(cfg, "a1")["alias"] == "candidate"
    resolved = registry.resolve(cfg, "production")["gguf"]
    assert resolved.endswith("a0.gguf") and not resolved.startswith("/")

    def ask_fn(q, arm):
        answer = "A. [Source: s 6]" if arm == "A1" else "A."
        return {"answer": answer, "latency": {"total_ms": 100.0}}

    report = registry.gate(cfg, dry_run=False, ask_fn=ask_fn)
    assert report["decision"]["passed"] and report["promotion"]["production"] == "2"
    assert registry.resolve(cfg, "production")["arm"] == "A1"
    assert registry.rollback(cfg) == {"production": "1", "rolled_back_from": "2"}
    assert registry.resolve(cfg, "production")["arm"] == "A0"

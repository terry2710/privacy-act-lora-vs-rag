import sys
import time

import pytest
from fastapi.testclient import TestClient

from privacy_bench.evaluate import wandb_log
from privacy_bench.ops import app as ops
from privacy_bench.ops.app import Runner, build_command, create_app, parse_output

TOKEN = {"Authorization": "Bearer t0ken"}


def test_build_command_validates_parameters():
    cmd = build_command("answers", {"sets": ["h1"], "arms": ["A2"]}, "py")
    assert cmd == ["py", "-m", "privacy_bench.evaluate.answers", "--sets", "h1", "--arms", "A2"]
    assert build_command("gate", {}, "py")[-1] == "--dry-run"
    assert "--dry-run" not in build_command("gate", {"dry_run": False}, "py")
    assert build_command("judge", {"judge": "claude"}, "py")[-2:] == ["--judge", "claude"]
    for name, params in [("answers", {"sets": ["x; rm"]}), ("judge", {"judge": "other"}), ("promote", {"version": "2 && x"}), ("shell", {})]:
        with pytest.raises(ValueError):
            build_command(name, params, "py")


def test_parse_output_handles_json_shapes():
    assert parse_output('{"a": 1}') == {"a": 1}
    assert parse_output('progress\n{\n  "b": 2\n}') == {"b": 2}
    assert parse_output('{"x": 1}\n{"x": 2}') == [{"x": 1}, {"x": 2}]
    assert parse_output("") is None


def wait(client, job_id):
    for _ in range(100):
        job = client.get(f"/jobs/{job_id}", headers=TOKEN).json()
        if job["done"]:
            return job
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_service_runs_jobs_and_guards_access(tmp_path, monkeypatch):
    scripts = {"analyze": "print('{\"ok\": 1}')", "gate": "import time; time.sleep(0.5); print('{}')", "rollback": "import sys; sys.exit(3)"}
    monkeypatch.setattr(ops, "build_command", lambda name, params, python: [sys.executable, "-c", scripts[name]] if name in scripts else (_ for _ in ()).throw(ValueError("unknown job")))
    client = TestClient(create_app(Runner(tmp_path, tmp_path / "logs", sys.executable, {}), "t0ken"))
    assert client.post("/jobs", json={"name": "analyze"}).status_code == 401
    assert client.post("/jobs", json={"name": "nope"}, headers=TOKEN).status_code == 400
    job = client.post("/jobs", json={"name": "analyze"}, headers=TOKEN).json()
    done = wait(client, job["id"])
    assert done["status"] == "succeeded" and done["result"] == {"ok": 1}
    slow = client.post("/jobs", json={"name": "gate"}, headers=TOKEN).json()
    assert client.post("/jobs", json={"name": "analyze"}, headers=TOKEN).status_code == 409
    assert client.get("/health").json()["running"] == slow["id"]
    wait(client, slow["id"])
    failed = wait(client, client.post("/jobs", json={"name": "rollback"}, headers=TOKEN).json()["id"])
    assert failed["status"] == "failed" and failed["returncode"] == 3
    assert len(client.get("/jobs", headers=TOKEN).json()) == 3


ANALYSIS = {
    "config": {"seed": 1, "resamples": 10, "alpha": 0.05, "judges": ["gpt", "claude"]},
    "set_sizes": {"s_para": 2},
    "descriptive": {
        g: {a: {"tag_present": 1.0, "citation_hit": 0.5, "mention_hit": 0.5, "bad_citation_rate": 0.1,
                "latency_median_ms": {"total_ms": 500.0}, "latency_p95_ms": {"total_ms": 900.0},
                "gpt": {"pass_rate": 0.1, "n": 2} if g == "s_neg" else {"accuracy_mean": 1.0, "fully_correct_rate": 0.5, "n": 2},
                "claude": {"pass_rate": 0.0, "n": 2} if g == "s_neg" else {"accuracy_mean": 0.9, "fully_correct_rate": 0.4, "n": 2}}
            for a in wandb_log.ARMS}
        for g in wandb_log.QA_SETS + ["s_neg"]
    },
    "primary": {"family": {"H1a": {"estimate": 0.9, "ci95": [0.8, 1.0], "p_used": 0.001, "p_holm": 0.002, "supported": True}},
                "H1_supported": False, "H3_supported": True},
    "secondary": {
        "H2_memorisation_gap_A1": {"estimate": 0.7, "ci95": [0.6, 0.8], "p_two_sided": 0.001, "supported": True},
        "H4_A0_h1clean_gt_h2": {"estimate": 0.0, "ci95": [-0.1, 0.1], "p_two_sided": 0.9, "supported": False},
        "H5_latency_A2_vs_A1": {"total_median_diff": {"estimate": 1200.0, "ci95": [1100.0, 1300.0], "p_two_sided": 0.001}},
    },
}


class FakeRun:
    id = "abc"
    url = "https://wandb.example/abc"

    def __init__(self):
        self.summary, self.logged, self.artifacts, self.finished = {}, [], [], False

    def log(self, data):
        self.logged.append(data)

    def log_artifact(self, artifact):
        self.artifacts.append(artifact)

    def finish(self):
        self.finished = True


class FakeWandb:
    def __init__(self):
        self.run = FakeRun()

    def init(self, **kwargs):
        self.kwargs = kwargs
        return self.run

    class Table:
        def __init__(self, columns, data):
            self.columns, self.data = columns, data

    class Artifact:
        def __init__(self, name, type):
            self.name, self.type, self.files = name, type, []

        def add_file(self, path, name):
            self.files.append(name)


def test_wandb_evaluation_payload_and_log(tmp_path):
    calibration = {"prompt_version": "judge-v1", "prompt_hashes": {"gpt": "x"}, "models": {"gpt": "m"},
                   "judge_agreement": {"qa": {"kappa_quadratic": 0.92}, "neg": {"kappa": 0.78}}}
    payload = wandb_log.evaluation_payload(ANALYSIS, calibration)
    assert payload["summary"]["gpt/accuracy/s_para/A1"] == 1.0
    assert payload["summary"]["claude/s_neg_pass/A0"] == 0.0
    assert payload["summary"]["primary/H3_supported"] is True
    assert payload["config"]["judge_prompt_version"] == "judge-v1"
    columns, rows = payload["tables"]["accuracy"]
    assert len(rows) == 6 * 4 * 2 and len(columns) == len(rows[0])
    file = tmp_path / "analysis.json"
    file.write_text("{}", encoding="utf-8")
    fake = FakeWandb()
    out = wandb_log.log("evaluation", payload, [file], "proj", wandb_module=fake)
    assert out["url"] == FakeRun.url and fake.run.finished
    assert fake.kwargs["job_type"] == "evaluation" and fake.kwargs["project"] == "proj"
    assert fake.run.artifacts[0].files == ["analysis.json"] and len(fake.run.logged) == 3


def test_wandb_gate_payload():
    report = {"model": "m", "production": {"version": "2", "citation_hit": 0.86, "p95_ms": 800.0},
              "candidate": {"version": "3", "citation_hit": 0.86, "p95_ms": 810.0},
              "thresholds": {"max_hit_diff": 0.03}, "decision": {"passed": False, "checks": {"quality_floor": False}}, "dry_run": True}
    payload = wandb_log.gate_payload(report)
    assert payload["summary"]["gate/passed"] is False and payload["summary"]["gate/check/quality_floor"] is False
    assert payload["summary"]["candidate/citation_hit"] == 0.86 and payload["config"]["candidate_version"] == "3"

import csv
import json
import random

import pytest
import yaml

from privacy_bench.evaluate.answers import collect
from privacy_bench.judge import llm, run
from privacy_bench.judge.calibrate import adjudication, final_labels, kappa, report, sample
from privacy_bench.judge.llm import judge_one
from privacy_bench.judge.rubric import build_task, label, strip_citations

QA_ITEM = {"eval_id": "h1-0001", "fact_id": "s:6#00-001", "uid": "s:6", "display": "s 6",
           "question": "What is personal information?", "reference_answer": "Information about an identified individual. [Source: s 6]"}
NEG_ITEM = {"eval_id": "s_neg-0001", "question": "What does APP 14 require?", "premise": "There are only 13 APPs."}


def answer(set_name, arm, eval_id, text):
    return {"eval_id": eval_id, "set": set_name, "arm": arm, "answer": text}


def fake_caller(responses):
    calls = []

    def caller(client, cfg, kind, user):
        calls.append(user)
        return responses.pop(0), cfg["model"], "completed", 100, 20

    caller.calls = calls
    return caller


def test_build_task_strips_citations_and_carries_evidence():
    task = build_task(QA_ITEM, answer("h1", "A1", "h1-0001", "About an identified person. [Source: s 6]"), {"s:6#00-001": "personal information means"})
    assert task["kind"] == "qa" and task["key"] == "h1/A1/h1-0001"
    assert "[Source" not in task["user"] and "personal information means" in task["user"]
    assert "Provision excerpt (s 6)" in task["user"]
    neg = build_task(NEG_ITEM, answer("s_neg", "A0", "s_neg-0001", "APP 14 requires X."), {})
    assert neg["kind"] == "neg" and "only 13 APPs" in neg["user"]
    assert strip_citations("A [Source: Sch 1, APP 6] b") == "A b"


def test_label_rules():
    assert label("qa", {"accuracy": 2, "contradicts_reference": False, "reason": ""}) == 2
    assert label("qa", {"accuracy": True, "contradicts_reference": False}) is None
    assert label("qa", {"accuracy": 3, "contradicts_reference": False}) is None
    assert label("neg", {"premise_corrected": True, "invents_content": False}) == "pass"
    assert label("neg", {"premise_corrected": True, "invents_content": True}) == "fail"
    assert label("neg", None) is None


def test_judge_one_retries_bad_json_and_uses_cache(tmp_path):
    task = build_task(QA_ITEM, answer("h1", "A1", "h1-0001", "x"), {"s:6#00-001": "e"})
    caller = fake_caller(["not json", json.dumps({"accuracy": 1, "contradicts_reference": False, "reason": "r"})])
    cfg = {"model": "m"}
    first = judge_one("gpt", cfg, None, task, tmp_path, "h", caller=caller)
    assert first["label"] == 1 and first["calls"] == 2 and first["input_tokens"] == 200
    again = judge_one("gpt", cfg, None, task, tmp_path, "h", caller=fake_caller([]))
    assert again["label"] == 1
    changed = dict(task, answer_hash="other")
    with pytest.raises(IndexError):
        judge_one("gpt", cfg, None, changed, tmp_path, "h", caller=fake_caller([]))


def test_kappa_values():
    assert kappa([(0, 0), (1, 1), (2, 2)], [0, 1, 2]) == 1.0
    assert kappa([("pass", "pass")] * 3, ["fail", "pass"]) is None
    pairs = [(2, 2)] * 6 + [(2, 1)] * 2 + [(0, 0)] * 2
    assert kappa(pairs, [0, 1, 2], "quadratic") > kappa(pairs, [0, 1, 2])
    assert kappa([(0, 2), (2, 0)], [0, 1, 2]) < 0


def test_sample_is_deterministic_and_checks_size():
    pool = {("h1", "A0"): [answer("h1", "A0", f"h1-{i:04d}", "a") for i in range(20)]}
    first = sample(pool, ["A0"], {"h1": 5}, 7)
    assert first == sample(dict(pool), ["A0"], {"h1": 5}, 7) and len(first) == 5
    with pytest.raises(ValueError):
        sample(pool, ["A0"], {"h1": 30}, 7)


def judged(task, judge, lab):
    verdict = {"reason": f"{judge} says"}
    return {"key": task["key"], "judge": judge, "label": lab, "verdict": verdict}


def test_adjudication_blind_and_final_labels():
    tasks = [build_task(QA_ITEM, answer("h1", arm, "h1-0001", "x"), {"s:6#00-001": "e"}) for arm in ("A0", "A1")]
    tasks.append(build_task(NEG_ITEM, answer("s_neg", "A0", "s_neg-0001", "y"), {}))
    results = [judged(tasks[0], "gpt", 2), judged(tasks[0], "claude", 2),
               judged(tasks[1], "gpt", 2), judged(tasks[1], "claude", 1),
               judged(tasks[2], "gpt", "pass"), judged(tasks[2], "claude", None)]
    rows, key = adjudication(tasks, results, ["gpt", "claude"], random.Random(1))
    assert len(rows) == 2
    assert all("A1" not in json.dumps(r) and "A0" not in json.dumps(r) for r in rows)
    with pytest.raises(ValueError):
        final_labels(tasks, results, ["gpt", "claude"], key, rows)
    for r in rows:
        r["label"] = "1" if r["kind"] == "qa" else "Pass"
    final = final_labels(tasks, results, ["gpt", "claude"], key, rows)
    assert final == {"h1/A0/h1-0001": 2, "h1/A1/h1-0001": 1, "s_neg/A0/s_neg-0001": "pass"}
    out = report(tasks, results, ["gpt", "claude"], final, {"min_kappa": 0.6, "min_raw_if_degenerate": 0.9})
    assert out["judge_agreement"]["neg"]["no_answer"] == 1
    assert out["by_arm"]["A1"]["final"]["qa_mean"] == 1.0
    assert out["vs_final"]["gpt"]["qa"]["raw"] == 0.5


def test_collect_resumes_and_records_failures(tmp_path):
    items = [{"eval_id": f"h1-{i}", "uid": "s:6", "question": f"q{i}"} for i in range(3)]
    out_path = tmp_path / "A1.jsonl"
    out_path.write_text(json.dumps({"eval_id": "h1-0"}) + "\n", encoding="utf-8")

    def ask_fn(q, arm):
        if q == "q2":
            raise OSError("down")
        return {"answer": "x [Source: s 6]", "retrieved": [], "latency": {"total_ms": 1}}

    summary = collect(items, "h1", "A1", ask_fn, out_path, {"s:6"}, warmup=0)
    assert summary["already_done"] == 1 and summary["new"] == 1 and len(summary["failures"]) == 1
    rows = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]
    assert rows[-1]["citation"]["hit"] and rows[-1]["retrieval_hit"] is None


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_cli_calibrate_score_and_gated_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    qa_items = [dict(QA_ITEM, eval_id=f"h1-{i:04d}") for i in range(4)]
    qa_items[0]["question"] = "odd question"
    neg_items = [dict(NEG_ITEM, eval_id=f"s_neg-{i:04d}") for i in range(4)]
    write_jsonl(tmp_path / "eval/h1.jsonl", qa_items)
    write_jsonl(tmp_path / "eval/s_neg.jsonl", neg_items)
    write_jsonl(tmp_path / "evidence.jsonl", [{"fact_id": "s:6#00-001", "evidence": "e"}])
    for arm in ("A0", "A1"):
        write_jsonl(tmp_path / f"answers/h1/{arm}.jsonl", [answer("h1", arm, i["eval_id"], f"{arm} ans") for i in qa_items])
        write_jsonl(tmp_path / f"answers/s_neg/{arm}.jsonl", [answer("s_neg", arm, i["eval_id"], f"{arm} ans") for i in neg_items])
    cfg = {
        "prompt_version": "t", "roles": {"primary": "gpt", "secondary": "claude"},
        "judges": {"gpt": {"provider": "openai", "model": "g"}, "claude": {"provider": "anthropic", "model": "c"}},
        "sets": ["h1", "s_neg"], "arms": ["A0", "A1"],
        "run": {"max_workers": 2, "max_retries": 0, "seed": 1},
        "calibration": {"arms": ["A0", "A1"], "per_set": {"h1": 4, "s_neg": 4}, "min_kappa": 0.6, "min_raw_if_degenerate": 0.9},
        "paths": {"answers_dir": "answers", "eval_dir": "eval", "evidence": ["evidence.jsonl"],
                  "cache_dir": "cache", "calibration_dir": "cal", "judgements_dir": "judgements"},
    }
    (tmp_path / "judge.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")

    def grader(client, jcfg, kind, user):
        if kind == "neg":
            return json.dumps({"premise_corrected": "A1 ans" in user, "invents_content": False, "reason": "r"}), jcfg["model"], "completed", 10, 5
        acc = 2 if "A1 ans" in user else 0
        if jcfg["model"] == "c" and user.startswith("Question: odd") and "A0 ans" in user:
            acc = 1
        return json.dumps({"accuracy": acc, "contradicts_reference": False, "reason": "r"}), jcfg["model"], "completed", 10, 5

    monkeypatch.setitem(llm.CALLERS, "openai", grader)
    monkeypatch.setitem(llm.CALLERS, "anthropic", grader)
    monkeypatch.setattr(run, "make_client", lambda c, r: None)
    argv = ["--config", "judge.yaml"]
    with pytest.raises(SystemExit):
        run.main(["run"] + argv)
    assert run.main(["calibrate"] + argv) == 0
    with pytest.raises(SystemExit):
        run.main(["calibrate"] + argv)
    with pytest.raises(ValueError):
        run.main(["calibrate-score"] + argv)
    path = tmp_path / "cal/adjudication.csv"
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["label"] = "0"
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    assert run.main(["calibrate-score"] + argv) == 0
    rep = json.loads((tmp_path / "cal/report.json").read_text(encoding="utf-8"))
    assert rep["judge_agreement"]["neg"]["raw"] == 1.0
    assert len(rows) == 1 and rep["gate"]["passed"]
    assert rep["vs_final"]["gpt"]["qa"]["raw"] == 1.0 and rep["vs_final"]["claude"]["qa"]["raw"] == 0.875
    assert run.main(["run"] + argv) == 0
    summary = json.loads((tmp_path / "judgements/summary.json").read_text(encoding="utf-8"))
    assert summary["h1"]["A1"]["mean_accuracy"] == 2.0 and summary["h1"]["A0"]["mean_accuracy"] == 0.0
    assert summary["s_neg"]["A1"]["pass_rate"] == 1.0 and summary["s_neg"]["A0"]["pass_rate"] == 0.0
    assert len((tmp_path / "judgements/h1/A0.jsonl").read_text(encoding="utf-8").splitlines()) == 4

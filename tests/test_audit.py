import json
import random
from types import SimpleNamespace

import pytest

from privacy_bench.qa.audit import build_adjudication, final_report, review

ROW = {"check_id": "chk-001", "kind": "qa", "criterion": "c", "provision": "s 1", "question": "q", "answer": "a", "source_text": "s"}


def _results(pairs):
    out = []
    for cid, (c, g) in pairs.items():
        out.append({"check_id": cid, "auditor": "claude", "verdict": c, "reason": f"claude says {c}"})
        out.append({"check_id": cid, "auditor": "gpt", "verdict": g, "reason": f"gpt says {g}"})
    return out


def test_build_adjudication_keeps_only_disagreements_and_unanswered():
    rows = [{**ROW, "check_id": c} for c in ("c1", "c2", "c3")]
    results = _results({"c1": ("ok", "ok"), "c2": ("ok", "problem"), "c3": ("problem", None)})
    adj_rows, adj_key = build_adjudication(rows, results, random.Random(0))
    assert [r["check_id"] for r in adj_rows] == ["c2", "c3"]
    assert sorted(adj_key["c2"]) == ["claude", "gpt"]
    first = adj_rows[0]
    assert {first["reviewer_1_verdict"], first["reviewer_2_verdict"]} == {"ok", "problem"}
    assert "no answer" in {adj_rows[1]["reviewer_1_verdict"], adj_rows[1]["reviewer_2_verdict"]}


def test_final_report_combines_agreement_and_adjudication():
    key = {
        "c1": {"stratum": "verified_pass_train", "verifier_verdict": "pass"},
        "c2": {"stratum": "verified_fail", "verifier_verdict": "fail"},
        "c3": {"stratum": "neg_unverifiable"},
    }
    results = _results({"c1": ("ok", "ok"), "c2": ("ok", "problem"), "c3": ("problem", "problem")})
    report = final_report(key, results, {"c2": "ok"})
    assert report["auditor_agreement"]["n"] == 3 and report["auditor_agreement"]["raw"] == pytest.approx(2 / 3, abs=1e-4)
    assert report["adjudication"]["sided_with"] == {"claude": 1}
    assert report["strata"]["verified_fail"]["ok"] == 1
    assert report["verifier_vs_final"]["confusion"] == {"verifier_fail/final_ok": 1, "verifier_pass/final_ok": 1}
    with pytest.raises(ValueError, match="c2"):
        final_report(key, results, {})


def test_review_retries_bad_json_and_caches(tmp_path):
    class FakeMessages:
        def __init__(self, texts):
            self.texts = list(texts)

        def create(self, **kwargs):
            usage = SimpleNamespace(input_tokens=10, output_tokens=5)
            return SimpleNamespace(content=[SimpleNamespace(type="text", text=self.texts.pop(0))], model="m", stop_reason="end_turn", usage=usage)

    cfg = {"provider": "anthropic", "model": "m", "max_tokens": 10, "effort": None}
    client = SimpleNamespace(messages=FakeMessages(["not json", json.dumps({"verdict": "problem", "reason": "r"})]))
    record = review("claude", cfg, client, ROW, tmp_path, "h")
    assert record["verdict"] == "problem" and record["input_tokens"] == 20
    again = review("claude", cfg, SimpleNamespace(messages=FakeMessages([])), ROW, tmp_path, "h")
    assert again == record
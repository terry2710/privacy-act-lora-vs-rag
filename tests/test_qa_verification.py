import json
from types import SimpleNamespace

import pytest

from privacy_bench.qa.generate import Job
from privacy_bench.qa.verify import assemble, chunk_key, group_items, normalise, run_job, strip_citation


def _job(key="s_20H__00"):
    row = {"uid": "s:20H", "display": "s 20H", "title": "t", "unit": "Part IIIA", "split": "train"}
    return Job(key=key, row=row, chunk=0, n_chunks=1, text="source", quota=2, n_questions=2, citation="[Source: s 20H]")


def _item(fact_id):
    return {"fact_id": fact_id, "uid": "s:20H", "unit": "Part IIIA", "split": "train", "questions": ["q1", "q2"], "answer": "a. [Source: s 20H]"}


class FakeClient:
    def __init__(self, batches):
        self.batches = list(batches)
        self.responses = self

    def create(self, **kwargs):
        results = self.batches.pop(0)
        usage = SimpleNamespace(input_tokens=100, output_tokens=20)
        return SimpleNamespace(status="completed", output_text=json.dumps({"results": results}), usage=usage, model="fake")


def test_chunk_key_and_citation_stripping():
    assert chunk_key("s:6#03-007") == "s_6__03"
    assert chunk_key("sch1:app6#00-012") == "sch1_app6__00"
    assert strip_citation("Only for direct marketing. [Source: s 20H]") == "Only for direct marketing."


def test_normalise_flags_inconsistent_verdicts():
    assert normalise({"verdict": "pass", "issue_type": "none"}) == ("pass", "none")
    assert normalise({"verdict": "fail", "issue_type": "incomplete"}) == ("fail", "incomplete")
    assert normalise({"verdict": "pass", "issue_type": "incomplete"}) == ("fail", "inconsistent")
    assert normalise({"verdict": "fail", "issue_type": "none"}) == ("fail", "inconsistent")


def test_group_items_rejects_unknown_chunks():
    with pytest.raises(ValueError, match="s_99__00"):
        group_items([_item("s:99#00-001")], [_job()])


def test_run_job_retries_missing_ids_and_ignores_unknown(tmp_path):
    items = [_item("s:20H#00-001"), _item("s:20H#00-002")]
    client = FakeClient([
        [{"id": "s:20H#00-001", "verdict": "pass", "issue_type": "none", "note": ""},
         {"id": "bogus", "verdict": "pass", "issue_type": "none", "note": ""}],
        [{"id": "s:20H#00-002", "verdict": "fail", "issue_type": "unsupported", "note": "extra claim"}],
    ])
    cfg = {"verifier": {"model": "m", "max_output_tokens": 10, "reasoning_effort": None}, "run": {"max_rounds": 2}}
    state = run_job(_job(), items, client, cfg, tmp_path, "h1")
    assert set(state["results"]) == {"s:20H#00-001", "s:20H#00-002"} and len(state["calls"]) == 2
    assert run_job(_job(), items, FakeClient([]), cfg, tmp_path, "h1")["calls"] == state["calls"]

    verified, results = assemble([(_job(), items)], [state], {"verifier_model": "m"})
    assert [r["fact_id"] for r in verified["train"]] == ["s:20H#00-001"]
    assert [r["verdict"] for r in results] == ["pass", "fail"]
import json
import re
from types import SimpleNamespace

from privacy_bench.qa.generate import Job, build_jobs, chunk_text, run_job
from privacy_bench.qa.validate import check_item

CITATION = "[Source: s 20H]"
SOURCE = "(1) A credit reporting body must not use or disclose a pre\u2011screening assessment unless it is for the purpose of direct marketing by a credit provider."
TYPES = ["definition", "obligation", "exception", "procedure", "enforcement", "scope"]


def _item(**overrides):
    item = {
        "statement": "A credit reporting body may use a pre-screening assessment only for direct marketing by a credit provider.",
        "question_type": "obligation",
        "questions": [
            "For what purpose may a credit reporting body use a pre-screening assessment?",
            "When is a credit reporting body permitted to disclose a pre-screening assessment?",
        ],
        "answer": f"Only for the purpose of direct marketing by a credit provider. {CITATION}",
        "evidence": "must not use or disclose a pre-screening assessment unless it is for the purpose of direct marketing",
    }
    item.update(overrides)
    return item


def _check(item, **kw):
    return check_item(item, source_text=SOURCE, citation=CITATION, n_questions=2, question_types=TYPES, **kw)


def test_valid_item_passes_with_hyphen_variants_in_evidence():
    assert _check(_item()) == []


def test_identifier_and_pointer_in_question_are_rejected():
    bad = _item(questions=["What does APP 6 require of a credit reporting body?", "What does this section require?"])
    assert _check(bad) == ["identifier_in_question", "pointer_in_question"]


def test_identifier_rule_ignores_lowercase_words():
    ok = _item(questions=["Which part in the process applies to a credit reporting body?", "What must a credit reporting body do?"])
    assert "identifier_in_question" not in _check(ok)


def test_part_numerals_are_detected():
    bad = _item(questions=["What does Part IIIA say about pre-screening?", "What must a credit reporting body do?"])
    assert "identifier_in_question" in _check(bad)


def test_citation_and_evidence_failures():
    bad = _item(answer="Only for direct marketing [Source: s 21G].", evidence="this text is not in the provision at all")
    assert _check(bad) == ["citation_format", "evidence_not_verbatim"]


def test_holdout_leak_rule_applies_to_answers():
    rules = [re.compile(r"\beligible data breach(?:e?s)?\b", re.IGNORECASE)]
    bad = _item(answer=f"It must also assess eligible data breaches. {CITATION}")
    assert "holdout_leak" in _check(bad, leak_rules=rules)


def test_chunk_text_respects_paragraph_boundaries():
    text = "\n".join(["a" * 40] * 5)
    chunks = chunk_text(text, 100)
    assert [len(c.split("\n")) for c in chunks] == [2, 2, 1]
    assert "\n".join(chunks) == text


def test_build_jobs_splits_quota_across_chunks():
    manifest = {"qa": {"train_phrasings_per_fact": 2, "citation_format": "[Source: {display}]"}}
    cfg = {"run": {"chunk_max_chars": 100}}
    row = {"uid": "s:6", "display": "s 6", "split": "train", "quota": 5, "text": "\n".join(["b" * 40] * 5)}
    jobs = build_jobs([row], manifest, cfg)
    assert [j.quota for j in jobs] == [2, 2, 1]
    assert jobs[0].key == "s_6__00" and jobs[0].citation == "[Source: s 6]" and jobs[0].n_questions == 2


class FakeClient:
    def __init__(self, batches):
        self.batches = list(batches)
        self.messages = self

    def create(self, **kwargs):
        facts = self.batches.pop(0)
        block = SimpleNamespace(type="text", text=json.dumps({"facts": facts}))
        usage = SimpleNamespace(input_tokens=100, output_tokens=50)
        return SimpleNamespace(content=[block], usage=usage, stop_reason="end_turn", model="fake-model")


def test_run_job_tops_up_after_rejection_and_caches(tmp_path):
    job = Job(key="s_20H__00", row={"uid": "s:20H", "display": "s 20H", "title": "t", "unit": "Part IIIA", "split": "train"},
              chunk=0, n_chunks=1, text=SOURCE, quota=2, n_questions=2, citation=CITATION)
    second = _item(statement="Pre-screening assessments are restricted.", questions=["What restricts pre-screening assessments?", "How are pre-screening assessments limited?"])
    client = FakeClient([[_item(), _item(evidence="fabricated quote that does not exist")], [second]])
    cfg = {"generator": {"model": "m", "max_tokens": 10, "effort": None}, "run": {"max_rounds": 3, "max_items_per_call": 15}}
    manifest = {"qa": {"question_types": TYPES}}
    state = run_job(job, client, cfg, manifest, None, tmp_path)
    assert len(state["accepted"]) == 2 and len(state["rejected"]) == 1 and len(state["calls"]) == 2
    cached = json.loads((tmp_path / "s_20H__00.json").read_text())
    assert len(cached["accepted"]) == 2
    assert run_job(job, FakeClient([]), cfg, manifest, None, tmp_path)["calls"] == state["calls"]


def test_yes_no_answer_requires_yes_no_phrasings():
    bad = _item(answer=f"Yes. Only for direct marketing by a credit provider. {CITATION}")
    assert "yes_no_mismatch" in _check(bad)
    ok = _item(
        questions=["May a credit reporting body use a pre-screening assessment for direct marketing?", "Is a credit reporting body allowed to disclose a pre-screening assessment for direct marketing?"],
        answer=f"Yes. Only for direct marketing by a credit provider. {CITATION}",
    )
    assert "yes_no_mismatch" not in _check(ok)


def test_run_job_discards_cache_from_other_prompt_version(tmp_path):
    job = Job(key="s_20H__00", row={"uid": "s:20H", "display": "s 20H", "title": "t", "unit": "Part IIIA", "split": "train"},
              chunk=0, n_chunks=1, text=SOURCE, quota=1, n_questions=2, citation=CITATION)
    stale = {"key": job.key, "prompt_hash": "old", "accepted": [_item()], "rejected": [], "calls": [{"input_tokens": 1, "output_tokens": 1}]}
    (tmp_path / "s_20H__00.json").write_text(json.dumps(stale))
    cfg = {"generator": {"model": "m", "max_tokens": 10, "effort": None}, "run": {"max_rounds": 1, "max_items_per_call": 15}}
    state = run_job(job, FakeClient([[_item()]]), cfg, {"qa": {"question_types": TYPES}}, None, tmp_path)
    assert state["prompt_hash"] != "old" and len(state["calls"]) == 1 and len(state["accepted"]) == 1
import json
from collections import Counter
from types import SimpleNamespace
from typing import Any

import pytest

from privacy_bench.qa.paraphrase import (
    assemble,
    assign_types,
    check_paraphrase,
    final_question,
    ngram_jaccard,
    rouge_l,
    run_batch
)

LIMITS = {"paraphrase_max_4gram_jaccard": 0.5, "paraphrase_max_rouge_l": 0.6}
FACT: dict[str, Any] = {
    "fact_id": "s:20H#00-001",
    "uid": "s:20H",
    "display": "s 20H",
    "unit": "Part IIIA",
    "question_type": "obligation",
    "statement": "A credit reporting body may use a pre-screening assessment only for direct marketing by a credit provider.",
    "questions": [
        "For what purpose may a credit reporting body use a pre-screening assessment?",
        "When is a credit reporting body permitted to disclose a pre-screening assessment?",
    ],
    "answer": "Only for direct marketing by a credit provider. [Source: s 20H]",
}


def test_assign_types_balances_types():
    types = assign_types([f"f{i}" for i in range(150)], 2)
    counts = Counter(t for ts in types.values() for t in ts)
    assert counts == {"lexical": 100, "syntactic": 100, "scenario": 100}
    assert all(len(set(ts)) == 2 for ts in types.values())


def test_overlap_metrics():
    assert ngram_jaccard("a b c d e", "a b c d e") == 1.0
    assert ngram_jaccard("a b c d", "w x y z") == 0.0
    assert rouge_l("the cat sat", "the cat sat") == pytest.approx(1.0)
    assert rouge_l("a b c d", "a x c y") == pytest.approx(0.5)


def test_check_paraphrase_accepts_novel_question():
    q = "A credit reporting body has prepared a pre-screening assessment. To whose marketing may it be put?"
    errors, overlap = check_paraphrase(q, "scenario", ["scenario"], FACT, [], LIMITS)
    assert errors == [] and overlap["max_rouge_l"] <= LIMITS["paraphrase_max_rouge_l"]


def test_check_paraphrase_rejects_copies_polarity_and_identifiers():
    errors, _ = check_paraphrase(FACT["questions"][0], "lexical", ["lexical"], FACT, [], LIMITS)
    assert {"overlap_4gram", "overlap_rouge_l"} <= set(errors)
    errors, _ = check_paraphrase("May a credit reporting body share such an assessment freely?", "lexical", ["lexical"], FACT, [], LIMITS)
    assert "polarity_mismatch" in errors
    errors, _ = check_paraphrase("What limit does APP 6 place on such assessments?", "syntactic", ["lexical"], FACT, [], LIMITS)
    assert {"identifier_in_question", "unrequested_type"} <= set(errors)


class FakeClient:
    def __init__(self, batches):
        self.batches = list(batches)
        self.responses = self

    def create(self, **kwargs):
        results = self.batches.pop(0)
        usage = SimpleNamespace(input_tokens=10, output_tokens=5)
        return SimpleNamespace(status="completed", output_text=json.dumps({"results": results}), usage=usage, model="fake")


def test_run_batch_tops_up_rejected_type_and_assembles(tmp_path):
    tasks = [("s_para", FACT, ["lexical", "scenario"])]
    good_lex = "Which kind of marketing can such an assessment from a credit reporting body serve?"
    good_scn = "A credit reporting body holds a pre-screening assessment and a lender asks about it. Whose marketing can it support?"
    client = FakeClient([
        [{"id": FACT["fact_id"], "phrasings": [{"type": "lexical", "question": good_lex}, {"type": "scenario", "question": FACT["questions"][0]}]}],
        [{"id": FACT["fact_id"], "phrasings": [{"type": "scenario", "question": good_scn}]}],
    ])
    cfg = {"generator": {"model": "m", "max_output_tokens": 10, "reasoning_effort": None}, "run": {"max_rounds": 3}}
    state = run_batch("s_para_000", tasks, client, cfg, LIMITS, tmp_path, "h")
    assert set(state["accepted"][FACT["fact_id"]]) == {"lexical", "scenario"}
    assert len(state["rejected"]) == 1 and len(state["calls"]) == 2

    records, shortfall = assemble([("s_para_000", tasks)], [state], {"paraphrase_model": "m"})
    assert shortfall == []
    assert [r["paraphrase_type"] for r in records["s_para"]] == ["lexical", "scenario"]
    assert records["s_para"][0]["eval_id"] == "s_para-0001" and records["s_para"][0]["reference_answer"] == FACT["answer"]



def test_polarity_uses_final_question_of_scenario():
    assert final_question("A body holds an assessment. Does it have to destroy it?") == "Does it have to destroy it?"
    polar_fact = {**FACT, "answer": "No, it need not. [Source: s 20H]"}
    scenario = "A credit reporting body receives an assessment from a lender. Does any marketing restriction apply to it?"
    errors, _ = check_paraphrase(scenario, "scenario", ["scenario"], polar_fact, [], LIMITS)
    assert "polarity_mismatch" not in errors
    errors, _ = check_paraphrase(scenario, "scenario", ["scenario"], FACT, [], LIMITS)
    assert "polarity_mismatch" in errors


def test_run_batch_revalidates_cached_entries(tmp_path):
    stale = "A credit reporting body receives an assessment from a lender. Does any marketing restriction apply to it?"
    state = {"key": "k", "prompt_hash": "h", "accepted": {FACT["fact_id"]: {"scenario": {"question": stale, "max_4gram_jaccard": 0.0, "max_rouge_l": 0.0}}}, "rejected": [], "calls": []}
    (tmp_path / "k.json").write_text(json.dumps(state))
    fresh = "A credit reporting body holds a pre-screening assessment and a lender asks about it. Whose marketing can it support?"
    client = FakeClient([[{"id": FACT["fact_id"], "phrasings": [{"type": "scenario", "question": fresh}]}]])
    cfg = {"generator": {"model": "m", "max_output_tokens": 10, "reasoning_effort": None}, "run": {"max_rounds": 2}}
    out = run_batch("k", [("s_para", FACT, ["scenario"])], client, cfg, LIMITS, tmp_path, "h")
    assert out["accepted"][FACT["fact_id"]]["scenario"]["question"] == fresh
    assert out["rejected"][0]["errors"] == ["revalidated:polarity_mismatch"]
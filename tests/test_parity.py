import pytest

from privacy_bench.serve.parity import compare, gate, percentile


def rec(i, hit, text="a"):
    return {"eval_id": f"d{i}", "hit": hit, "prediction": text}


def test_compare_counts_flips_and_agreement():
    ref = [rec(1, True), rec(2, True), rec(3, False), rec(4, True)]
    cand = [rec(1, True), rec(2, False, "b"), rec(3, True, "c"), rec(4, True)]
    m = compare(ref, cand)
    assert m["reference_citation_hit"] == 0.75 and m["candidate_citation_hit"] == 0.75
    assert m["hit_agreement"] == 0.5 and m["exact_text_match"] == 0.5
    assert m["flipped_to_miss"] == ["d2"] and m["flipped_to_hit"] == ["d3"]


def test_compare_rejects_mismatched_items():
    with pytest.raises(ValueError):
        compare([rec(1, True)], [rec(2, True)])


def test_gate_thresholds():
    ok = gate({"hit_diff": -0.03, "hit_agreement": 0.9}, 0.03, 0.9)
    assert ok["passed"]
    bad = gate({"hit_diff": -0.04, "hit_agreement": 0.95}, 0.03, 0.9)
    assert not bad["passed"] and not bad["checks"]["hit_diff_within_limit"]


def test_percentile():
    assert percentile([1, 2, 3, 4, 5], 0.5) == 3
    assert percentile([1, 2, 3, 4, 5], 0.95) == 5
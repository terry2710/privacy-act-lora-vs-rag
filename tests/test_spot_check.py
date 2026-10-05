import pytest

from privacy_bench.qa.spot_check import cohen_kappa, score, source_window, wilson


def test_wilson_interval_brackets_the_proportion():
    lo, hi = wilson(36, 40)
    assert lo < 0.9 < hi and 0.7 < lo and hi < 1.0
    assert wilson(0, 0) == (0.0, 0.0)


def test_cohen_kappa_extremes():
    assert cohen_kappa([(True, True), (False, False)] * 5) == 1.0
    assert cohen_kappa([(True, False), (False, True)] * 5) == -1.0
    assert cohen_kappa([]) is None


def test_source_window_centres_on_evidence():
    text = "a" * 5000 + "EVIDENCE HERE" + "b" * 5000
    window = source_window(text, "EVIDENCE HERE", width=1000)
    assert "EVIDENCE HERE" in window and window.startswith("...") and window.endswith("...")
    assert source_window("short text", "x") == "short text"


def test_score_reports_strata_and_verifier_agreement():
    rows = [
        {"check_id": "c1", "label": "ok"},
        {"check_id": "c2", "label": "Problem"},
        {"check_id": "c3", "label": "ok"},
        {"check_id": "c4", "label": "ok"},
    ]
    key = {
        "c1": {"stratum": "verified_pass_train", "verifier_verdict": "pass"},
        "c2": {"stratum": "verified_pass_train", "verifier_verdict": "pass"},
        "c3": {"stratum": "verified_fail", "verifier_verdict": "fail"},
        "c4": {"stratum": "neg_unverifiable"},
    }
    report = score(rows, key)
    assert report["strata"]["verified_pass_train"]["ok"] == 1 and report["strata"]["verified_pass_train"]["n"] == 2
    assert report["verifier_vs_human"]["n"] == 3
    assert report["verifier_vs_human"]["confusion"]["verifier_fail/human_ok"] == 1


def test_score_rejects_missing_labels():
    with pytest.raises(ValueError, match="c2"):
        score([{"check_id": "c1", "label": "ok"}, {"check_id": "c2", "label": ""}], {"c1": {"stratum": "s"}, "c2": {"stratum": "s"}})
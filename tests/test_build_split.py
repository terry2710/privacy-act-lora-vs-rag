import pytest

from privacy_bench.data.build_split import allocate, assign_splits, check_holdout_patterns, scan_leakage


def _manifest():
    return {
        "units": {
            "Part I": {"split": "train", "facts": 5},
            "Part IIIB": {"split": "train", "facts": 1},
            "Part IIIC": {"split": "holdout_h1", "questions": 4},
        },
        "section_overrides": [{"uid": "s:2", "split": "excluded", "reason": "test"}],
        "leakage": {"holdout_uid_patterns": ["^s:26W[A-Z]", "^s:26X"]},
    }


def _sections():
    return [
        {"uid": "s:1", "unit": "Part I"},
        {"uid": "s:2", "unit": "Part I"},
        {"uid": "s:26W", "unit": "Part IIIB"},
        {"uid": "s:26WA", "unit": "Part IIIC"},
    ]


def _leak_manifest():
    return {
        "leakage": {
            "holdout_citation_patterns": [r"\b26W[A-Z]+\b", r"\bSch(?:edule)?\s*2\b(?!\s+to\s+the\b)"],
            "holdout_terms": ["eligible data breach"],
            "source_review": {
                "s:96": {"action": "mask", "blocks": [{"from": "(2A)", "to": "(2B)"}]},
                "s:7": {"action": "false_positive"},
            },
        }
    }


def test_allocate_largest_remainder():
    assert allocate([100.0, 300.0, 600.0], total=10, minimum=1) == [2, 3, 5]


def test_allocate_rejects_infeasible_quota():
    with pytest.raises(ValueError):
        allocate([1.0, 1.0, 1.0], total=2, minimum=1)


def test_assign_splits_applies_overrides():
    sections = assign_splits(_sections(), _manifest())
    assert [s["split"] for s in sections] == ["train", "excluded", "train", "holdout_h1"]


def test_assign_splits_rejects_unconfigured_unit():
    sections = _sections() + [{"uid": "s:80F", "unit": "Part VIA"}]
    with pytest.raises(ValueError, match="unconfigured"):
        assign_splits(sections, _manifest())


def test_holdout_patterns_exclude_s26W():
    sections = assign_splits(_sections(), _manifest())
    check_holdout_patterns(sections, _manifest())


def test_holdout_patterns_flag_misassignment():
    m = _manifest()
    m["units"]["Part IIIC"] = {"split": "train", "facts": 1}
    sections = assign_splits(_sections(), m)
    with pytest.raises(ValueError, match="s:26WA"):
        check_holdout_patterns(sections, m)


def test_scan_leakage_plural_terms_blocks_and_false_positives():
    sections = [
        {"uid": "s:33C", "split": "train", "text": "(a) keep\n(i) assess suspected eligible data breaches; and\n(b) keep"},
        {"uid": "s:96", "split": "train", "text": "(1) review under subsection 26WQ(7)\n(2A) Only:\n(a) the entity\n(2B) Another\n(3) keep"},
        {"uid": "s:7A", "split": "train", "text": "Part I of Schedule 2 to the Freedom of Information Act 1982"},
        {"uid": "s:7", "split": "train", "text": "(B) Division 1 of Part I of Schedule 2;"},
    ]
    scan_leakage(sections, _leak_manifest())
    assert sections[0]["masked_lines"] == [1]
    assert sections[1]["masked_lines"] == [0, 1, 2, 3]
    assert sections[2]["leak_hits"] == []
    assert sections[3]["source_review"] == "false_positive" and sections[3]["masked_lines"] == []


def test_scan_leakage_rejects_missing_block_anchor():
    m = _leak_manifest()
    m["leakage"]["source_review"]["s:96"]["blocks"] = [{"from": "(9Z)", "to": "(2B)"}]
    sections = [{"uid": "s:96", "split": "train", "text": "(1) x 26WQ\n(2A) y"}, {"uid": "s:7", "split": "train", "text": ""}]
    with pytest.raises(ValueError, match="not found"):
        scan_leakage(sections, m)
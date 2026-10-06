from privacy_bench.train.citations import aggregate, parse_refs, score_citation

VALID = {"s:6", "s:7", "s:26WA", "s:100", "sch1:overview", "sch1:app6", "sch2:cl16A"}


def test_parse_refs_formats():
    assert parse_refs("s 6(1)") == ["s:6"]
    assert parse_refs("Schedule 1, APP 6.2") == ["sch1:app6"]
    assert parse_refs("Australian Privacy Principle 6") == ["sch1:app6"]
    assert parse_refs("Sch 2, clause 16a") == ["sch2:cl16A"]
    assert parse_refs("section 26wa and s 7") == ["s:26WA", "s:7"]
    assert parse_refs("Sch 1, Overview") == ["sch1:overview"]
    assert parse_refs("Part IIIC") == []


def test_hit_requires_single_gold_provision_in_last_tag():
    assert score_citation("Answer. [Source: s 6]", "s:6", VALID)["hit"]
    assert not score_citation("Answer. [Source: s 6, s 7]", "s:6", VALID)["hit"]
    assert score_citation("[Source: s 7] then [Source: s 6]", "s:6", VALID)["hit"]


def test_wrong_existing_and_nonexistent():
    wrong = score_citation("Answer. [Source: s 7]", "s:6", VALID)
    assert wrong["wrong_existing"] and not wrong["has_nonexistent"]
    fake = score_citation("Answer. [Source: s 999]", "s:6", VALID)
    assert fake["has_nonexistent"] and not fake["wrong_existing"]
    body = score_citation("See s 42Z. [Source: s 6]", "s:6", VALID)
    assert body["hit"] and body["nonexistent"] == ["s:42Z"]


def test_external_act_references_are_ignored():
    text = "Under section 188 of that Act and s 50 of the Bankruptcy Act 1966. [Source: s 6]"
    assert score_citation(text, "s:6", VALID)["nonexistent"] == []
    assert score_citation("See s 999 of the Privacy Act. [Source: s 6]", "s:6", VALID)["nonexistent"] == ["s:999"]


def test_missing_tag_and_aggregate():
    s = score_citation("No tag here.", "s:6", VALID)
    assert not s["tag_present"] and not s["hit"]
    agg = aggregate([score_citation("[Source: s 6]", "s:6", VALID), s])
    assert agg["n"] == 2 and agg["citation_hit"] == 0.5 and agg["tag_present"] == 0.5
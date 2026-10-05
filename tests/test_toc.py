from privacy_bench.data.text_norm import match_key
from privacy_bench.data.toc import parse_toc

SAMPLE = """Contents
Part I—Preliminary\t1
1\tShort title\t1
Part IIIC—Notification of eligible data breaches\t202
Division 3—Notification of eligible data breaches\t211
Subdivision A—Suspected eligible data breaches\t211
26WH\tAssessment of suspected eligible data breach\t211
Schedule 1—Australian Privacy Principles\t359
Overview of the Australian Privacy Principles\t359
Part 3—Dealing with personal information\t369
6\tAustralian Privacy Principle 6—use or disclosure of personal information\t369
Schedule 2—Statutory Tort for Serious Invasions of Privacy\t385
Part 2—Serious invasions of privacy\t391
7\tCause of action\t391
Endnotes\t404
Endnote 1—About the endnotes\t404
"""


def test_parse_toc_structure(tmp_path):
    path = tmp_path / "toc.txt"
    path.write_text(SAMPLE, encoding="utf-8")
    entries = parse_toc(path)
    sections = {e.uid: e for e in entries if e.uid}

    assert set(sections) == {"s:1", "s:26WH", "sch1:overview", "sch1:app6", "sch2:cl7"}
    assert (sections["s:26WH"].part, sections["s:26WH"].division, sections["s:26WH"].subdivision) == ("IIIC", "3", "A")
    assert sections["s:26WH"].unit == "Part IIIC"
    assert sections["sch1:app6"].part == "3"
    assert sections["sch1:app6"].unit == "Sch 1"
    assert sections["sch2:cl7"].display == "Sch 2, cl 7"
    assert entries[-1].kind == "stop"


def test_match_key_hyphen_and_dash_variants():
    assert match_key("20J\tDestruction of pre‑screening assessment") == match_key("20J Destruction of pre screening assessment")
    assert match_key("Part I—Preliminary") == match_key("Part I - Preliminary")

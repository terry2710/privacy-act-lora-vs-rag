import re
import unicodedata

TAG = re.compile(r"\[\s*Source\s*:\s*([^\]]*)\]", re.IGNORECASE)
REF = re.compile(
    r"(?P<overview>sch(?:edule)?\.?\s*1\s*,?\s*overview)"
    r"|(?P<app>(?:sch(?:edule)?\.?\s*1\s*,?\s*)?(?:APP|Australian\s+Privacy\s+Principle)\s*(?P<app_num>\d+))"
    r"|(?P<cl>sch(?:edule)?\.?\s*2\s*,?\s*(?:cl(?:ause)?\.?\s*)?(?P<cl_num>\d+[A-Z]*)\b)"
    r"|(?P<sec>\b(?:ss?|sub-?sections?|sections?)\.?\s*(?P<sec_num>\d+[A-Z]*)\b)",
    re.IGNORECASE,
)
EXTERNAL = re.compile(
    r"\b(?:ss?|sub-?sections?|sections?|clauses?|cl)\.?\s*\d+[A-Z]*(?:\([0-9A-Za-z]+\))*"
    r"\s+of\s+(?:that\s+Act\b|the\s+(?!Privacy\b)[A-Z][A-Za-z]*(?:\s+(?:[A-Z][A-Za-z]*|and|of|for))*\s+Act\b)"
)


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u2010", "-").replace("\u2011", "-").replace("\u00a0", " ")
    return re.sub(r"\s+", " ", text)


def parse_refs(text: str) -> list[str]:
    uids = []
    for m in REF.finditer(normalise(text)):
        if m.group("overview"):
            uid = "sch1:overview"
        elif m.group("app"):
            uid = f"sch1:app{int(m.group('app_num'))}"
        elif m.group("cl"):
            uid = f"sch2:cl{m.group('cl_num').upper()}"
        else:
            uid = f"s:{m.group('sec_num').upper()}"
        if uid not in uids:
            uids.append(uid)
    return uids


def split_answer(answer: str) -> tuple[str | None, str]:
    text = normalise(answer)
    tags = list(TAG.finditer(text))
    if not tags:
        return None, text
    body = TAG.sub(" ", text)
    return tags[-1].group(1).strip(), body


def score_citation(answer: str, gold_uid: str, valid_uids: set[str]) -> dict:
    tag, body = split_answer(answer)
    cited = parse_refs(tag) if tag is not None else []
    mentioned = parse_refs(EXTERNAL.sub(" ", body))
    hit = cited == [gold_uid]
    nonexistent = sorted({u for u in cited + mentioned if u not in valid_uids})
    return {
        "tag_present": tag is not None,
        "cited": cited,
        "hit": hit,
        "nonexistent": nonexistent,
        "has_nonexistent": bool(nonexistent),
        "wrong_existing": bool(cited) and not hit and all(u in valid_uids for u in cited),
    }


def aggregate(scores: list[dict]) -> dict:
    n = len(scores)
    if n == 0:
        return {"n": 0}
    return {
        "n": n,
        "citation_hit": sum(s["hit"] for s in scores) / n,
        "tag_present": sum(s["tag_present"] for s in scores) / n,
        "nonexistent_rate": sum(s["has_nonexistent"] for s in scores) / n,
        "wrong_existing_rate": sum(s["wrong_existing"] for s in scores) / n,
    }
import re

from ..data.text_norm import match_key

IDENTIFIER = re.compile(
    r"\b(?i:sections?|subsections?|paragraphs?|subparagraphs?|clauses?|parts?|divisions?|subdivisions?"
    r"|schedules?|APPs?|Australian Privacy Principles?)\s*\(?(?:\d+[A-Z]*|[IVX]+[A-Z]*)\b"
)
POINTER = re.compile(
    r"\bthis (?:section|subsection|paragraph|provision|part|division|subdivision|schedule|clause|principle|act)\b",
    re.IGNORECASE,
)
YES_NO_ANSWER = re.compile(r"^(?:Yes|No)\b")
YES_NO_QUESTION = re.compile(
    r"^(?:Is|Are|Was|Were|Can|Could|Does|Do|Did|May|Might|Must|Will|Would|Shall|Should|Has|Have|Had)\b"
)
MIN_EVIDENCE_CHARS = 20


def check_item(
    item: dict,
    *,
    source_text: str,
    citation: str,
    n_questions: int,
    question_types: list[str],
    leak_rules: list[re.Pattern] | None = None,
) -> list[str]:
    errors: set[str] = set()
    questions = [q.strip() for q in item.get("questions") or []]
    answer = (item.get("answer") or "").strip()
    evidence = (item.get("evidence") or "").strip()

    if len(questions) != n_questions:
        errors.add("question_count")
    if len({match_key(q) for q in questions}) != len(questions):
        errors.add("duplicate_phrasing")
    if any(IDENTIFIER.search(q) for q in questions):
        errors.add("identifier_in_question")
    if any(POINTER.search(q) for q in questions):
        errors.add("pointer_in_question")
    if YES_NO_ANSWER.match(answer) and not all(YES_NO_QUESTION.match(q) for q in questions):
        errors.add("yes_no_mismatch")
    if not answer.endswith(citation):
        errors.add("citation_format")
    elif answer.count("[Source:") != 1:
        errors.add("extra_citation")
    if len(evidence) < MIN_EVIDENCE_CHARS:
        errors.add("evidence_too_short")
    elif match_key(evidence) not in match_key(source_text):
        errors.add("evidence_not_verbatim")
    if item.get("question_type") not in question_types:
        errors.add("question_type")
    if not (item.get("statement") or "").strip():
        errors.add("missing_statement")
    if leak_rules:
        text = " ".join(questions + [answer, item.get("statement") or ""])
        if any(rx.search(text) for rx in leak_rules):
            errors.add("holdout_leak")
    return sorted(errors)
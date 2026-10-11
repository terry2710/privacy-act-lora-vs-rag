import hashlib
import json

from ..train.citations import TAG, normalise

QA_SYSTEM = """You grade answers to questions about the Privacy Act 1988 (Cth) of Australia. You are given the question, a reference answer, an excerpt of the provision the reference answer relies on, and the answer to grade. Citations have been removed from both answers; do not penalise their absence.

Score accuracy on this scale:
- 2: the answer states the substance of the reference answer correctly and completely, including every condition, exception, party, time limit and number it contains.
- 1: the core point is correct, but the answer omits or blurs a condition, exception, party, time limit or number that the reference answer contains.
- 0: the answer is wrong, contradicts the reference answer on its core point, answers a different question, refuses, or says the Act has no such provision.

Rules:
- Judge substance, not wording. Paraphrase is acceptable.
- Material beyond the reference answer does not raise the score. It lowers the score only if it contradicts the reference answer or the provision excerpt; a contradiction on a secondary point caps the score at 1.
- Use the provision excerpt only to check whether a claim is supported. Do not require the answer to quote it.
- Do not use outside knowledge to override the reference answer.

Set contradicts_reference to true if any statement in the answer conflicts with the reference answer or the provision excerpt. Give a reason of at most 40 words."""

QA_USER = """Question: {question}

Reference answer: {reference}

Provision excerpt ({display}): {evidence}

Answer to grade: {answer}"""

QA_SCHEMA = {
    "type": "object",
    "properties": {
        "accuracy": {"type": "integer", "enum": [0, 1, 2]},
        "contradicts_reference": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["accuracy", "contradicts_reference", "reason"],
    "additionalProperties": False,
}

NEG_SYSTEM = """You grade answers to questions built on a false premise about the Privacy Act 1988 (Cth) of Australia. Each question asks about a provision, body or mechanism that the Act does not contain, or treats part of another law as part of the Act. You are told why the premise is false. Citations have been removed from the answer.

Decide two things:
- premise_corrected: true if the answer clearly states that the Privacy Act 1988 (Cth) has no such provision, body or mechanism, or that the matter belongs to a different law. A vague hedge without a clear correction is false.
- invents_content: true if the answer describes obligations, powers, rules or other content of the nonexistent provision, body or mechanism as if it existed in the Act, even if it also corrects the premise elsewhere. Pointing to a real, related provision of the Act is not inventing content.

Give a reason of at most 40 words."""

NEG_USER = """Question: {question}

Why the premise is false: {premise}

Answer to grade: {answer}"""

NEG_SCHEMA = {
    "type": "object",
    "properties": {
        "premise_corrected": {"type": "boolean"},
        "invents_content": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["premise_corrected", "invents_content", "reason"],
    "additionalProperties": False,
}

RUBRICS = {
    "qa": {"system": QA_SYSTEM, "user": QA_USER, "schema": QA_SCHEMA},
    "neg": {"system": NEG_SYSTEM, "user": NEG_USER, "schema": NEG_SCHEMA},
}


def strip_citations(text: str) -> str:
    return normalise(TAG.sub(" ", text)).strip()


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def prompt_hash(judge_cfg: dict, prompt_version: str) -> str:
    payload = prompt_version + json.dumps(RUBRICS, sort_keys=True) + json.dumps(judge_cfg, sort_keys=True)
    return text_hash(payload)


def build_task(item: dict, answer: dict, evidence: dict[str, str]) -> dict:
    kind = "neg" if answer["set"] == "s_neg" else "qa"
    fields = {"question": item["question"], "answer": strip_citations(answer["answer"])}
    if kind == "qa":
        fields.update(
            reference=strip_citations(item["reference_answer"]),
            display=item["display"],
            evidence=evidence[item["fact_id"]],
        )
    else:
        fields["premise"] = item["premise"]
    return {
        "key": f"{answer['set']}/{answer['arm']}/{answer['eval_id']}",
        "kind": kind,
        "eval_id": answer["eval_id"],
        "set": answer["set"],
        "arm": answer["arm"],
        "answer_hash": text_hash(answer["answer"]),
        "fields": fields,
        "user": RUBRICS[kind]["user"].format(**fields),
    }


def valid_verdict(kind: str, verdict) -> bool:
    if not isinstance(verdict, dict):
        return False
    if kind == "qa":
        return type(verdict.get("accuracy")) is int and verdict["accuracy"] in (0, 1, 2) and isinstance(verdict.get("contradicts_reference"), bool)
    return isinstance(verdict.get("premise_corrected"), bool) and isinstance(verdict.get("invents_content"), bool)


def label(kind: str, verdict: dict | None):
    if not valid_verdict(kind, verdict):
        return None
    if kind == "qa":
        return verdict["accuracy"]
    return "pass" if verdict["premise_corrected"] and not verdict["invents_content"] else "fail"

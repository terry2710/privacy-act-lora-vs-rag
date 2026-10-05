import hashlib
import json

SYSTEM_PROMPT = """You are constructing a question-answering dataset about the Privacy Act 1988 (Cth) for a controlled machine-learning experiment. You will receive the text of one provision of the Act, or one part of a long provision. Extract atomic facts stated in that text and write questions and answers for them.

Rules:
1. Every fact must be stated explicitly in the provided text. Do not use outside knowledge, other provisions, or inferences beyond what the text says.
2. Each fact is atomic: a single obligation, definition, condition, exception, time limit, power or consequence.
3. Questions must be self-contained and answerable by someone who has not seen the text. Name the relevant actor or concept (for example "an APP entity", "the Commissioner", "a credit reporting body") instead of pointing at the text.
4. Questions must not contain any section, subsection, paragraph, clause, Part, Division, Schedule or Australian Privacy Principle number, and must not use phrases such as "this section", "this provision" or "this Act". Refer to the legislation as "the Privacy Act" where needed.
5. Questions must be direct questions. Do not frame them as hypothetical scenarios about a particular person or business.
6. Answers are one to three sentences, state the fact precisely using the Act's terminology, and end with exactly the citation string supplied, with nothing after it. Do not add any other citation.
7. The evidence field must be a verbatim, contiguous excerpt copied from the provided text, between 20 and 300 characters, that supports the answer.
8. Do not repeat facts listed as already covered, and do not produce two facts that differ only in wording.
9. Spread question_type across the allowed types where the text supports it; do not force a type the text does not support.
10. When several phrasings of a question are requested, they must ask for exactly the same fact but differ clearly in wording and sentence structure.
11. Every phrasing must be fully answered by the same answer text. Begin an answer with "Yes" or "No" only if every phrasing is a yes/no question; otherwise state the fact directly."""


USER_TEMPLATE = """Provision: {display} ({title})
Location in the Act: {context}
Citation string that must end every answer: {citation}

Provision text{chunk_note}:
<provision>
{text}
</provision>

Facts already covered for this provision (do not repeat them):
{covered}

Produce exactly {n} new facts. Give {k} phrasing(s) of the question for each fact."""


def build_output_schema(question_types: list[str], n_questions: int) -> dict:
    return {
        "type": "object",
        "properties": {
            "facts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "statement": {"type": "string", "description": "The atomic fact as one declarative sentence."},
                        "question_type": {"type": "string", "enum": list(question_types)},
                        "questions": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": f"Exactly {n_questions} phrasing(s) of the question.",
                        },
                        "answer": {"type": "string"},
                        "evidence": {"type": "string", "description": "Verbatim contiguous excerpt from the provision text."},
                    },
                    "required": ["statement", "question_type", "questions", "answer", "evidence"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["facts"],
        "additionalProperties": False,
    }


def build_user_prompt(job, n: int, covered: list[str]) -> str:
    row = job.row
    titles = [row.get(k) for k in ("schedule_title", "part_title", "division_title", "subdivision_title")]
    path = " > ".join(t for t in titles if t)
    return USER_TEMPLATE.format(
        display=row["display"],
        title=row["title"],
        context=f"{row['unit']}: {path}" if path else row["unit"],
        citation=job.citation,
        chunk_note=f" (part {job.chunk + 1} of {job.n_chunks})" if job.n_chunks > 1 else "",
        text=job.text,
        covered="\n".join(f"- {c}" for c in covered) or "(none)",
        n=n,
        k=job.n_questions,
    )


def prompt_hash() -> str:
    payload = SYSTEM_PROMPT + USER_TEMPLATE + json.dumps(build_output_schema(["type"], 1), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
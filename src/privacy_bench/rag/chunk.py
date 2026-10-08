import re

SENTENCE_END = re.compile(r"(?<=[.;:])\s+")


def split_long(paragraph: str, max_chars: int) -> list[str]:
    if len(paragraph) <= max_chars:
        return [paragraph]
    pieces, current = [], ""
    for sentence in SENTENCE_END.split(paragraph):
        while len(sentence) > max_chars:
            pieces.append(sentence[:max_chars])
            sentence = sentence[max_chars:]
        if current and len(current) + 1 + len(sentence) > max_chars:
            pieces.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        pieces.append(current)
    return pieces


def pack(paragraphs: list[str], max_chars: int, overlap_chars: int) -> list[str]:
    chunks, current = [], []
    for para in paragraphs:
        if current and len("\n".join(current + [para])) > max_chars:
            chunks.append("\n".join(current))
            tail = current[-1]
            current = [tail] if len(tail) <= overlap_chars and len(tail) + 1 + len(para) <= max_chars else []
        current.append(para)
    if current:
        chunks.append("\n".join(current))
    return chunks


def chunk_section(section: dict, max_chars: int, overlap_chars: int) -> list[dict]:
    paragraphs = []
    for line in section["text"].split("\n"):
        line = line.strip()
        if line:
            paragraphs.extend(split_long(line, max_chars))
    if not paragraphs:
        paragraphs = [section["title"]]
    header = f"{section['display']} {section['title']}"
    return [
        {
            "chunk_id": f"{section['uid']}#c{i:03d}",
            "uid": section["uid"],
            "display": section["display"],
            "title": section["title"],
            "unit": section["unit"],
            "text": text,
            "index_text": f"{header}\n{text}",
        }
        for i, text in enumerate(pack(paragraphs, max_chars, overlap_chars))
    ]


def chunk_corpus(sections: list[dict], max_chars: int, overlap_chars: int) -> list[dict]:
    return [c for s in sections for c in chunk_section(s, max_chars, overlap_chars)]

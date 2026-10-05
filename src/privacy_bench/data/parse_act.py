import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

from .text_norm import clean_text, match_key
from .toc import TocEntry, parse_toc

LOOKAHEAD = 4
LEVELS = ("schedule", "part", "division", "subdivision")


def iter_blocks(doc):
    for child in doc.element.body.iterchildren():
        if child.tag == qn("w:p"):
            para = Paragraph(child, doc)
            style = para.style.name if para.style is not None else ""
            yield style, para.text
        elif child.tag == qn("w:tbl"):
            for row in Table(child, doc).rows:
                cells: list[str] = []
                for cell in row.cells:
                    text = " ".join(cell.text.split())
                    if text and (not cells or cells[-1] != text):
                        cells.append(text)
                if cells:
                    yield "TableRow", " | ".join(cells)


def align(doc, entries: list[TocEntry]):
    keys = [match_key(e.heading) for e in entries]
    buffers: dict[int, list[str]] = {}
    skipped: list[int] = []
    ptr, current, stopped = 0, None, False
    for style, raw in iter_blocks(doc):
        if style.startswith("TOC"):
            continue
        text = clean_text(raw)
        if not text:
            continue
        key = match_key(text)
        window = range(ptr, min(ptr + LOOKAHEAD, len(entries)))
        hit = next((j for j in window if keys[j] == key), None)
        if hit is None:
            if current is not None:
                buffers[current].append(text)
            continue
        skipped.extend(range(ptr, hit))
        ptr = hit + 1
        if entries[hit].kind == "stop":
            stopped = True
            break
        current = hit if entries[hit].uid else None
        if current is not None:
            buffers[current] = []
    missing = [] if stopped else list(range(ptr, len(entries)))
    return buffers, skipped, missing


def build_records(entries: list[TocEntry], buffers: dict[int, list[str]]) -> list[dict]:
    titles: dict[str, str | None] = dict.fromkeys(LEVELS)
    records = []
    for i, e in enumerate(entries):
        if e.kind in LEVELS:
            titles[e.kind] = e.title
            for lower in LEVELS[LEVELS.index(e.kind) + 1:]:
                titles[lower] = None
            continue
        if not e.uid:
            continue
        lines = buffers.get(i)
        rec = e.to_dict()
        rec.update({f"{k}_title": v for k, v in titles.items()})
        rec["parsed"] = lines is not None
        rec["text"] = "\n".join(lines or [])
        rec["n_chars"] = len(rec["text"])
        records.append(rec)
    return records


def summarise(docx_path: Path, entries, records, skipped, missing) -> dict:
    def describe(idx):
        return [f"{entries[i].kind}: {entries[i].heading}" for i in idx]

    by_unit: dict[str, dict[str, int]] = defaultdict(lambda: {"sections": 0, "chars": 0})
    for r in records:
        by_unit[r["unit"]]["sections"] += 1
        by_unit[r["unit"]]["chars"] += r["n_chars"]

    return {
        "source_file": str(docx_path),
        "sha256": hashlib.sha256(docx_path.read_bytes()).hexdigest(),
        "toc_entries": len(entries),
        "sections_expected": len(records),
        "sections_parsed": sum(r["parsed"] for r in records),
        "skipped_headings": describe(skipped),
        "unreached_headings": describe(missing),
        "empty_sections": [r["display"] for r in records if r["parsed"] and r["n_chars"] == 0],
        "by_unit": dict(by_unit),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Parse the Privacy Act compilation into section records aligned to the TOC.")
    ap.add_argument("--toc", type=Path, required=True)
    ap.add_argument("--docx", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    args = ap.parse_args(argv)

    entries = parse_toc(args.toc)
    buffers, skipped, missing = align(Document(str(args.docx)), entries)
    records = build_records(entries, buffers)
    report = summarise(args.docx, entries, records, skipped, missing)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    ok = not (report["skipped_headings"] or report["unreached_headings"] or report["empty_sections"])
    print(f"parsed {report['sections_parsed']}/{report['sections_expected']} sections; ok={ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

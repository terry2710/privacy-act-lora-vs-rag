import re
from dataclasses import asdict, dataclass
from pathlib import Path

from .text_norm import clean_text

_DASH = r"\s*[—–-]\s*"
_PAGE = re.compile(r"^(?P<body>.*?)\s+(?P<page>\d+)$")
_PART_ROMAN = re.compile(rf"^Part (?P<num>[IVX]+[A-Z]*){_DASH}(?P<title>.+)$")
_PART_ARABIC = re.compile(rf"^Part (?P<num>\d+){_DASH}(?P<title>.+)$")
_DIVISION = re.compile(rf"^Division (?P<num>\d+[A-Z]*){_DASH}(?P<title>.+)$")
_SUBDIVISION = re.compile(rf"^Subdivision (?P<num>[A-Z]+){_DASH}(?P<title>.+)$")
_SCHEDULE = re.compile(rf"^Schedule (?P<num>\d+){_DASH}(?P<title>.+)$")
_SECTION = re.compile(r"^(?P<num>\d+[A-Z]*)\s+(?P<title>\S.*)$")

_PREFIX = {"schedule": "Schedule", "part": "Part", "division": "Division", "subdivision": "Subdivision"}


@dataclass
class TocEntry:
    kind: str
    num: str
    title: str
    page: int
    schedule: str | None = None
    part: str | None = None
    division: str | None = None
    subdivision: str | None = None
    uid: str | None = None
    display: str | None = None

    @property
    def heading(self) -> str:
        if self.kind in _PREFIX:
            return f"{_PREFIX[self.kind]} {self.num}—{self.title}"
        if self.kind == "section":
            return f"{self.num} {self.title}"
        return self.title

    @property
    def unit(self) -> str:
        return f"Sch {self.schedule}" if self.schedule else f"Part {self.part}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["heading"] = self.heading
        d["unit"] = self.unit
        return d


def _section_ids(schedule: str | None, num: str) -> tuple[str, str]:
    if schedule is None:
        return f"s:{num}", f"s {num}"
    if schedule == "1":
        return f"sch1:app{num}", f"Sch 1, APP {num}"
    return f"sch{schedule}:cl{num}", f"Sch {schedule}, cl {num}"


def parse_toc(path: Path) -> list[TocEntry]:
    entries: list[TocEntry] = []
    ctx: dict[str, str | None] = {"schedule": None, "part": None, "division": None, "subdivision": None}
    seen: set[str] = set()
    for lineno, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = clean_text(raw)
        if not line or line == "Contents":
            continue
        m = _PAGE.match(line)
        if not m:
            raise ValueError(f"line {lineno}: no page number: {raw!r}")
        body, page = m["body"].strip(), int(m["page"])

        if body == "Endnotes":
            entries.append(TocEntry("stop", "", "Endnotes", page))
            return entries

        if mm := _SCHEDULE.match(body):
            ctx = {"schedule": mm["num"], "part": None, "division": None, "subdivision": None}
            entries.append(TocEntry("schedule", mm["num"], mm["title"], page, **ctx))
            continue

        mm = _PART_ROMAN.match(body) if ctx["schedule"] is None else _PART_ARABIC.match(body)
        if mm:
            ctx.update(part=mm["num"], division=None, subdivision=None)
            entries.append(TocEntry("part", mm["num"], mm["title"], page, **ctx))
            continue

        if mm := _DIVISION.match(body):
            ctx.update(division=mm["num"], subdivision=None)
            entries.append(TocEntry("division", mm["num"], mm["title"], page, **ctx))
            continue

        if mm := _SUBDIVISION.match(body):
            ctx.update(subdivision=mm["num"])
            entries.append(TocEntry("subdivision", mm["num"], mm["title"], page, **ctx))
            continue

        if mm := _SECTION.match(body):
            kind, num, title = "section", mm["num"], mm["title"]
            uid, display = _section_ids(ctx["schedule"], num)
        elif ctx["schedule"] is not None:
            kind, num, title = "overview", "", body
            uid, display = f"sch{ctx['schedule']}:overview", f"Sch {ctx['schedule']}, Overview"
        else:
            raise ValueError(f"line {lineno}: unrecognised entry: {raw!r}")

        if uid in seen:
            raise ValueError(f"line {lineno}: duplicate uid {uid}")
        seen.add(uid)
        entries.append(TocEntry(kind, num, title, page, **ctx, uid=uid, display=display))

    raise ValueError("TOC has no 'Endnotes' line; cannot determine end of Act body")

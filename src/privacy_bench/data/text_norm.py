import re
import unicodedata

_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})
_DASHES = str.maketrans({c: " " for c in "‐‑‒–—―−-"})
_SPACES = re.compile(r"[ \t ]+")
_WS = re.compile(r"\s+")


def clean_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("‐", "-").translate(_QUOTES)
    return _SPACES.sub(" ", s).strip()


def match_key(s: str) -> str:
    return _WS.sub(" ", clean_text(s).translate(_DASHES)).strip().lower()

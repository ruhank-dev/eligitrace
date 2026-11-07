"""Turn a free-text clinical note into numbered sentences (the IDs the model cites)."""

import re

from dataprep.loader import parse_note

_NUMBERED = re.compile(r"^\s*0\.\s+\S")
_ABBREV = {"dr", "mr", "mrs", "ms", "vs", "e.g", "i.e", "etc", "approx", "no", "st", "fig", "pt", "hx",
           "b.i.d", "t.i.d", "q.d", "p.o", "a.m", "p.m"}
_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'])")


def _split_block(block: str) -> list[str]:
    parts, buf = [], ""
    for piece in _BOUNDARY.split(block):
        buf = f"{buf} {piece}".strip() if buf else piece
        last = buf.rstrip(".!?").rsplit(" ", 1)[-1].lower()
        if last in _ABBREV:  # "Dr. Smith" - keep joining
            continue
        parts.append(buf)
        buf = ""
    if buf:
        parts.append(buf)
    return parts


def segment(text: str) -> tuple[dict[int, str], bool]:
    """Returns ({id: sentence}, already_numbered). Dataset-style '0. ...' notes keep their IDs."""
    text = text.replace("\r\n", "\n").strip()
    if _NUMBERED.match(text):
        parsed = parse_note(text)
        if parsed and list(parsed) == list(range(len(parsed))):
            return parsed, True
    sentences: list[str] = []
    for block in re.split(r"\n\s*\n|\n(?=\s*[-*•])|\n", text):
        block = " ".join(block.split()).lstrip("-*• ").strip()
        if block:
            sentences += [s.strip() for s in _split_block(block) if s.strip()]
    return dict(enumerate(sentences)), False

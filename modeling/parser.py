"""Robust parsing + validation of model output into a typed prediction."""

import json
import re
from dataclasses import dataclass, field

from modeling.labels import INVALID, normalise_label

_OBJ_RE = re.compile(r"\{.*?\}", re.DOTALL)
_LABEL_RE = re.compile(r'"?label"?\s*[:=]\s*"([^"]+)"', re.IGNORECASE)
_EVID_RE = re.compile(r'"?evidence"?\s*[:=]\s*\[([^\]]*)\]', re.IGNORECASE)


@dataclass
class ParseResult:
    ok: bool
    label: str
    evidence: list[int] = field(default_factory=list)
    dropped_ids: list[int] = field(default_factory=list)  # cited IDs not present in the note
    error: str | None = None


def _ints(values) -> list[int]:
    out = []
    for v in values:
        try:
            out.append(int(str(v).strip()))
        except ValueError:
            continue
    return out


def parse_output(text: str, criterion_type: str, valid_ids: set[int]) -> ParseResult:
    raw_label, raw_evidence = None, []
    for m in _OBJ_RE.finditer(text):
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "label" in obj:
            raw_label = obj.get("label")
            ev = obj.get("evidence", [])
            raw_evidence = _ints(ev if isinstance(ev, list) else [ev])
            break
    if raw_label is None:  # lenient fallback for near-JSON output
        lm = _LABEL_RE.search(text)
        if lm:
            raw_label = lm.group(1)
            em = _EVID_RE.search(text)
            raw_evidence = _ints(em.group(1).split(",")) if em else []
    if raw_label is None:
        return ParseResult(False, INVALID, error="no label found")
    label = normalise_label(raw_label, criterion_type)
    if label is None:
        return ParseResult(False, INVALID, error=f"label {raw_label!r} not allowed for {criterion_type}")
    kept = sorted({i for i in raw_evidence if i in valid_ids})
    dropped = sorted({i for i in raw_evidence if i not in valid_ids})
    return ParseResult(True, label, kept, dropped)

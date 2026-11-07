"""Split pasted / ClinicalTrials.gov eligibility text into typed criteria."""

import re

_HEAD = re.compile(r"^\s*(?:key\s+)?(inclusion|exclusion)s?(?:\s+criteria)?\s*:?\s*$", re.IGNORECASE)
_BULLET = re.compile(r"^\s*(?:[-*•●]|\d+[.)]|[a-z][.)])\s+")


def parse_criteria(text: str) -> list[dict]:
    current, out = "inclusion", []
    for line in text.replace("\r\n", "\n").split("\n"):
        if not line.strip():
            continue
        m = _HEAD.match(line)
        if m:
            current = m.group(1).lower()
            continue
        bullet = bool(_BULLET.match(line))
        item = _BULLET.sub("", line).strip()
        if len(item) < 3:
            continue
        if bullet or not out or out[-1]["type"] != current or out[-1].get("_closed"):
            out.append({"type": current, "text": item[:2000]})
        else:  # wrapped continuation of the previous bullet
            out[-1]["text"] = f"{out[-1]['text']} {item}"[:2000]
    return [{"type": c["type"], "text": c["text"]} for c in out]

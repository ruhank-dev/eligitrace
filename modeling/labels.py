"""Label space. Each criterion type has its own four expert categories (no binary collapse)."""

LABELS: dict[str, tuple[str, ...]] = {
    "inclusion": ("included", "not included", "not enough information", "not applicable"),
    "exclusion": ("excluded", "not excluded", "not enough information", "not applicable"),
}
ALL_LABELS: tuple[str, ...] = (
    "included",
    "not included",
    "excluded",
    "not excluded",
    "not enough information",
    "not applicable",
)
INVALID = "invalid"

_SYNONYMS = {
    "not enough info": "not enough information",
    "insufficient information": "not enough information",
    "unknown": "not enough information",
    "n/a": "not applicable",
    "na": "not applicable",
    "eligible": None,  # ambiguous across types -> rejected
}


def normalise_label(raw: str, criterion_type: str) -> str | None:
    """Return a valid label for this criterion type, or None if not valid."""
    label = " ".join(str(raw).strip().lower().replace("_", " ").split())
    label = _SYNONYMS.get(label, label)
    if label is None:
        return None
    return label if label in LABELS[criterion_type] else None

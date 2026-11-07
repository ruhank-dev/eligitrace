"""Demo examples for the UI: VALIDATION-split patients only (the test split is never exposed)."""

from functools import lru_cache

from dataprep.formatting import read_jsonl
from modeling.prompts import format_note
from server.eligitrace_api.config import get_settings


@lru_cache
def validation_examples() -> list[dict]:
    rows = read_jsonl(get_settings().data_dir / "processed" / "val.jsonl")
    import json

    from dataprep.loader import load_clean

    df = load_clean().set_index("annotation_id")
    groups: dict[tuple, dict] = {}
    for r in rows:
        src = df.loc[r["annotation_id"]]
        key = (r["patient_id"], src.trial_id)
        g = groups.setdefault(key, {"key": f"{r['patient_id']}|{src.trial_id}", "patient_id": r["patient_id"],
                                    "trial_id": src.trial_id, "trial_title": src.trial_title,
                                    "note": format_note(src.sentences), "criteria": []})
        g["criteria"].append({"type": r["criterion_type"], "text": src.criterion_text})
    out = sorted(groups.values(), key=lambda g: -len(g["criteria"]))
    json.dumps(out)  # ensure serialisable
    return out

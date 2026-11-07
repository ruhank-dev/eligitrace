"""Load the TrialGPT criterion annotations and strip every model-generated column."""

import json
import re

import pandas as pd
from datasets import load_dataset

from settings.config import get_settings

# Model-generated or split-revealing columns: never used as input, feature or eval signal.
FORBIDDEN_COLUMNS = (
    "gpt4_explanation",
    "gpt4_sentences",
    "gpt4_eligibility",
    "explanation_correctness",
    "training",
)
KEEP_COLUMNS = (
    "annotation_id",
    "patient_id",
    "note",
    "trial_id",
    "trial_title",
    "criterion_type",
    "criterion_text",
    "expert_eligibility",
    "expert_sentences",
)
MISSING_CRITERION = "[criterion text missing in source dataset]"
_SENT_RE = re.compile(r"^(\d+)\.\s?(.*)$")


def load_raw() -> pd.DataFrame:
    s = get_settings()
    ds = load_dataset(
        s.dataset_id,
        split="train",
        revision=s.dataset_revision,
        cache_dir=str(s.data_dir / "raw"),
    )
    return ds.to_pandas()


def parse_note(note: str) -> dict[int, str]:
    """Split a dataset note ("0. text\\n1. text") into {sentence_id: text}."""
    sentences: dict[int, str] = {}
    last_id = None
    for line in note.split("\n"):
        line = line.strip()
        if not line:
            continue
        m = _SENT_RE.match(line)
        if m and (last_id is None or int(m.group(1)) == last_id + 1):
            last_id = int(m.group(1))
            sentences[last_id] = m.group(2).strip()
        elif last_id is not None:  # continuation line of the previous sentence
            sentences[last_id] = f"{sentences[last_id]} {line}"
    return sentences


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop forbidden columns, parse evidence lists, normalise labels."""
    out = df.drop(columns=[c for c in FORBIDDEN_COLUMNS if c in df.columns])[list(KEEP_COLUMNS)].copy()
    out["expert_sentences"] = out["expert_sentences"].map(lambda s: sorted(set(json.loads(s))))
    out["expert_eligibility"] = out["expert_eligibility"].str.strip().str.lower()
    out["criterion_type"] = out["criterion_type"].str.strip().str.lower()
    # Data-quality: annotation 883 has a null criterion_text. Kept (counts preserved) with explicit marker.
    out["criterion_text"] = out["criterion_text"].fillna("").astype(str).str.strip()
    out.loc[out["criterion_text"] == "", "criterion_text"] = MISSING_CRITERION
    out["sentences"] = out["note"].map(parse_note)
    return out.sort_values("annotation_id").reset_index(drop=True)


def load_clean() -> pd.DataFrame:
    return clean(load_raw())

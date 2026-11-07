"""Build and publish the split manifest (patient IDs + counts + hashes)."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from settings.config import get_settings

PROCEDURE = (
    "unique patient_id values sorted lexicographically into a NumPy array; "
    "numpy.random.default_rng(42).shuffle(array) in place; "
    "first 37 -> train, next 8 -> val, last 8 -> test; every annotation follows its patient."
)


def _hash_ids(ids) -> str:
    return hashlib.sha256(",".join(str(i) for i in sorted(ids)).encode()).hexdigest()


def build_manifest(df: pd.DataFrame, dataset_sha: str | None = None) -> dict:
    s = get_settings()
    splits = {}
    for name in ("train", "val", "test"):
        part = df[df["split"] == name]
        splits[name] = {
            "patient_ids": sorted(part["patient_id"].unique().tolist()),
            "n_patients": int(part["patient_id"].nunique()),
            "n_annotations": int(len(part)),
            "n_trials": int(part["trial_id"].nunique()),
            "type_counts": part["criterion_type"].value_counts().sort_index().to_dict(),
            "label_counts": part["expert_eligibility"].value_counts().sort_index().to_dict(),
            "annotation_ids_sha256": _hash_ids(part["annotation_id"]),
        }
    return {
        "dataset": s.dataset_id,
        "dataset_revision": dataset_sha or s.dataset_revision,
        "seed": s.seed,
        "procedure": PROCEDURE,
        "total_annotations": int(len(df)),
        "total_patients": int(df["patient_id"].nunique()),
        "excluded_columns": ["gpt4_explanation", "gpt4_sentences", "gpt4_eligibility",
                             "explanation_correctness", "training"],
        "splits": splits,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def write_manifest(df: pd.DataFrame, out_dir: Path, dataset_sha: str | None = None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(df, dataset_sha)
    path = out_dir / "split_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for name in ("train", "val", "test"):
        ids = sorted(df.loc[df["split"] == name, "annotation_id"].tolist())
        (out_dir / f"{name}_annotation_ids.txt").write_text(
            "\n".join(map(str, ids)) + "\n", encoding="utf-8"
        )
    return path

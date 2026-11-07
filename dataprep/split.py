"""Mandated patient-grouped split: sorted IDs -> default_rng(42).shuffle -> 37 / 8 / 8."""

import numpy as np
import pandas as pd

SPLIT_SIZES = {"train": 37, "val": 8, "test": 8}


def split_patients(patient_ids, seed: int = 42) -> dict[str, list[str]]:
    ids = np.array(sorted(set(patient_ids)))
    if len(ids) != sum(SPLIT_SIZES.values()):
        raise ValueError(f"expected 53 unique patients, got {len(ids)}")
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    a, b = SPLIT_SIZES["train"], SPLIT_SIZES["train"] + SPLIT_SIZES["val"]
    return {"train": ids[:a].tolist(), "val": ids[a:b].tolist(), "test": ids[b:].tolist()}


def assign_splits(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    splits = split_patients(df["patient_id"], seed)
    lookup = {pid: name for name, pids in splits.items() for pid in pids}
    out = df.copy()
    out["split"] = out["patient_id"].map(lookup)
    return out

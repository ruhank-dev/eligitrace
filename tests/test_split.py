"""B1: fixed split protocol, determinism and patient-leakage checks."""

import json

import pytest

from dataprep.loader import FORBIDDEN_COLUMNS
from dataprep.split import split_patients
from settings.config import get_settings

MANIFEST = get_settings().data_dir / "splits" / "split_manifest.json"


@pytest.fixture(scope="module")
def df():
    from dataprep.loader import load_clean
    from dataprep.split import assign_splits

    return assign_splits(load_clean())


def test_split_sizes_and_disjoint(df):
    s = split_patients(df.patient_id)
    assert [len(s[k]) for k in ("train", "val", "test")] == [37, 8, 8]
    assert not set(s["train"]) & set(s["val"])
    assert not set(s["train"]) & set(s["test"])
    assert not set(s["val"]) & set(s["test"])
    assert set(s["train"]) | set(s["val"]) | set(s["test"]) == set(df.patient_id)


def test_split_deterministic(df):
    assert split_patients(df.patient_id) == split_patients(list(reversed(df.patient_id.tolist())))


def test_every_annotation_in_exactly_one_split(df):
    assert df.split.notna().all()
    assert len(df) == 1015
    assert df.annotation_id.is_unique
    assert df.groupby("patient_id").split.nunique().max() == 1


def test_no_forbidden_columns(df):
    assert not set(FORBIDDEN_COLUMNS) & set(df.columns)


def test_manifest_matches_recomputation(df):
    m = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for name in ("train", "val", "test"):
        assert sorted(df.loc[df.split == name, "patient_id"].unique()) == m["splits"][name]["patient_ids"]
        assert (df.split == name).sum() == m["splits"][name]["n_annotations"]


def test_gold_evidence_ids_exist_in_note(df):
    for row in df.itertuples():
        assert set(row.expert_sentences) <= set(row.sentences), row.annotation_id

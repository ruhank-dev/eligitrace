"""B2: formatting correctness, GPT-4 leakage scan, parser robustness."""

import pytest

from modeling.labels import LABELS
from modeling.parser import parse_output


@pytest.fixture(scope="module")
def raw_and_clean():
    from dataprep.loader import clean, load_raw

    raw = load_raw()
    return raw, clean(raw)


def test_prompts_independent_of_gpt4_columns(raw_and_clean):
    """Overwrite every model-generated column with a sentinel: prompts must be byte-identical
    and the sentinel must never appear (proves GPT-4 outputs cannot leak into inputs)."""
    from dataprep.formatting import to_example
    from dataprep.loader import FORBIDDEN_COLUMNS, clean

    raw, df = raw_and_clean
    poisoned = raw.copy()
    for col in FORBIDDEN_COLUMNS:
        poisoned[col] = "SENTINEL_LEAK" if poisoned[col].dtype == object else poisoned[col]
    df2 = clean(poisoned)
    for a, b in zip(df.itertuples(), df2.itertuples(), strict=True):
        ea, eb = to_example(a), to_example(b)
        assert ea == eb
        assert "SENTINEL_LEAK" not in str(eb)


def test_targets_valid(raw_and_clean):
    from dataprep.formatting import to_example

    _, df = raw_and_clean
    for row in df.itertuples():
        ex = to_example(row)
        assert ex["label"] in LABELS[row.criterion_type]
        assert set(ex["evidence"]) <= set(row.sentences)


IDS = {0, 1, 2, 3}


@pytest.mark.parametrize(
    "text,ok,label,evidence",
    [
        ('{"label": "included", "evidence": [0, 2]}', True, "included", [0, 2]),
        ('```json\n{"label": "Included", "evidence": [2,0]}\n```', True, "included", [0, 2]),
        ('{"label": "excluded", "evidence": []}', False, "invalid", []),  # wrong type
        ('{"label": "not enough info", "evidence": [9]}', True, "not enough information", []),
        ("The patient is included.", False, "invalid", []),
        ('{"label": "not applicable"}', True, "not applicable", []),
        ('label: "included", evidence: [1]', True, "included", [1]),
    ],
)
def test_parser(text, ok, label, evidence):
    r = parse_output(text, "inclusion", IDS)
    assert (r.ok, r.label, r.evidence) == (ok, label, evidence)


def test_parser_drops_unknown_ids():
    r = parse_output('{"label": "included", "evidence": [1, 7]}', "inclusion", IDS)
    assert r.evidence == [1] and r.dropped_ids == [7]

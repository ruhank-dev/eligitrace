"""C1-C3: metric definitions on hand-computed toy cases (incl. every empty-set convention)."""

import pytest

from evaluation.metrics_cls import classification
from evaluation.metrics_evidence import evidence, record_prf


@pytest.mark.parametrize("g,p,exp", [
    (set(), set(), (1, 1, 1)),
    (set(), {1}, (0, 1, 0)),
    ({1}, set(), (1, 0, 0)),
    ({1, 2}, {2, 3}, (0.5, 0.5, 0.5)),
    ({1, 2}, {1, 2}, (1, 1, 1)),
])
def test_record_prf(g, p, exp):
    assert record_prf(g, p) == pytest.approx(exp)


def _r(gl, pl, ge, pe, t="inclusion"):
    return {"gold_label": gl, "pred_label": pl, "gold_evidence": ge, "pred_evidence": pe,
            "criterion_type": t, "parse_ok": pl != "invalid", "latency_s": 0}


def test_micro_vs_macro():
    recs = [_r("included", "included", [0, 1], [0]), _r("included", "included", [], [])]
    e = evidence(recs)
    assert e["micro"] == pytest.approx({"precision": 1.0, "recall": 0.5, "f1": 2 / 3})
    # macro: record1 (1, .5, .667), record2 (1,1,1)
    assert e["macro"]["recall"] == pytest.approx(0.75)
    assert e["nonempty_gold"]["n"] == 1 and e["empty_gold"]["correct_abstention"] == 1


def test_classification_subsets_and_invalid():
    recs = [_r("included", "included", [], []),
            _r("not included", "invalid", [], []),
            _r("excluded", "not excluded", [], [], "exclusion")]
    c = classification(recs, "inclusion")
    assert c["n"] == 2 and c["accuracy"] == 0.5 and c["n_invalid"] == 1
    assert c["confusion"]["matrix"][1][-1] == 1  # gold "not included" -> invalid column
    a = classification(recs, "all")
    assert a["accuracy"] == pytest.approx(1 / 3)
    assert "not applicable" in a["absent_gold_labels"]

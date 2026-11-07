"""Criterion classification metrics: accuracy, macro-F1, per-class P/R/F1, confusion matrix."""

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from modeling.labels import ALL_LABELS, INVALID, LABELS

SUBSETS = {"all": ALL_LABELS, "inclusion": LABELS["inclusion"], "exclusion": LABELS["exclusion"]}


def classification(records: list[dict], subset: str = "all") -> dict:
    """macro_f1 averages over the subset's full label list (classes with zero support score 0 and
    are flagged); macro_f1_present averages only over labels present in gold for this split."""
    recs = records if subset == "all" else [r for r in records if r["criterion_type"] == subset]
    labels = list(SUBSETS[subset])
    if not recs:
        return {"n": 0}
    y = [r["gold_label"] for r in recs]
    p = [r["pred_label"] for r in recs]
    prec, rec, f1, sup = precision_recall_fscore_support(y, p, labels=labels, zero_division=0)
    present = [i for i, s in enumerate(sup) if s > 0]
    cm = confusion_matrix(y, p, labels=labels + [INVALID])
    return {
        "n": len(recs),
        "accuracy": float(accuracy_score(y, p)),
        "macro_f1": float(np.mean(f1)),
        "macro_f1_present": float(np.mean(f1[present])) if present else 0.0,
        "labels": labels,
        "absent_gold_labels": [labels[i] for i, s in enumerate(sup) if s == 0],
        "per_class": {lab: {"precision": float(prec[i]), "recall": float(rec[i]), "f1": float(f1[i]),
                            "support": int(sup[i])} for i, lab in enumerate(labels)},
        "confusion": {"labels": labels + [INVALID], "matrix": cm[: len(labels)].tolist()},
        "n_invalid": sum(1 for x in p if x == INVALID),
    }

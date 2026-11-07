"""Evidence-sentence retrieval metrics with explicit empty-set conventions.

Per record (G = gold IDs, P = predicted IDs):
  G = {} and P = {}  -> P = R = F1 = 1   (correct abstention)
  G = {} and P != {} -> P = 0, R = 1, F1 = 0   (hallucinated evidence)
  G != {} and P = {} -> P = 1, R = 0, F1 = 0   (missed evidence)
  otherwise standard set precision / recall / F1.
micro  = pooled TP/FP/FN over all records (empty-empty records add nothing).
macro  = mean of per-record scores.
Also reported on the non-empty-gold subset, where no convention is needed for recall.
"""


def record_prf(gold: set[int], pred: set[int]) -> tuple[float, float, float]:
    if not gold and not pred:
        return 1.0, 1.0, 1.0
    if not gold:
        return 0.0, 1.0, 0.0
    if not pred:
        return 1.0, 0.0, 0.0
    tp = len(gold & pred)
    p, r = tp / len(pred), tp / len(gold)
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0}


def _aggregate(pairs) -> dict:
    tp = fp = fn = 0
    per = []
    for g, p in pairs:
        tp += len(g & p)
        fp += len(p - g)
        fn += len(g - p)
        per.append(record_prf(g, p))
    n = len(per)
    macro = {k: (sum(x[i] for x in per) / n if n else 0.0) for i, k in enumerate(("precision", "recall", "f1"))}
    return {"n": n, "micro": _prf(tp, fp, fn), "macro": macro, "tp": tp, "fp": fp, "fn": fn}


def evidence(records: list[dict]) -> dict:
    pairs = [(set(r["gold_evidence"]), set(r["pred_evidence"])) for r in records]
    out = _aggregate(pairs)
    out["nonempty_gold"] = _aggregate([(g, p) for g, p in pairs if g])
    out["empty_gold"] = {
        "n": sum(1 for g, _ in pairs if not g),
        "correct_abstention": sum(1 for g, p in pairs if not g and not p),
        "hallucinated": sum(1 for g, p in pairs if not g and p),
    }
    out["unsupported_citations"] = sum(len(r.get("dropped_ids", [])) for r in records)
    return out

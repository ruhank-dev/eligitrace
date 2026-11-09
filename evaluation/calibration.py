"""Validation-only study: can a per-label log-prior shift (class-aware calibration) improve the frozen model?

Step 1 (GPU, ~minutes): teacher-force every allowed label after the prompt prefix '{"label": "' and store the
log-probability of each label for every VALIDATION example (results/calibration/val_label_scores.json).
Step 2 (CPU): leave-one-patient-out cross-validation of additive per-label biases (coordinate search maximising
validation macro-F1 on the other seven patients), scored on the held-out patient. The test split is never read.

  uv run python -m evaluation.calibration score      # step 1
  uv run python -m evaluation.calibration cv         # step 2
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

from dataprep.formatting import read_jsonl
from modeling.labels import ALL_LABELS, LABELS
from modeling.loader import load_model
from modeling.prompts import render_prompt
from settings.config import get_settings

PREFIX = '{"label": "'


@torch.no_grad()
def label_scores(lm, ex: dict) -> dict[str, float]:
    msgs = ex["messages"][:-1] if ex["messages"][-1]["role"] == "assistant" else ex["messages"]
    base = render_prompt(lm.tokenizer, msgs) + PREFIX
    out = {}
    for lab in LABELS[ex["criterion_type"]]:
        full = base + lab + '"'
        enc = lm.tokenizer(full, return_tensors="pt", return_offsets_mapping=True)
        ids = enc.input_ids.to(lm.device)
        first = next(i for i, (_, e) in enumerate(enc.offset_mapping[0].tolist()) if e > len(base))
        logp = torch.log_softmax(lm.model(ids).logits[0, first - 1:-1].float(), dim=-1)
        out[lab] = float(logp.gather(1, ids[0, first:][:, None]).sum())
    return out


def score() -> None:
    s = get_settings()
    lm = load_model(None, adapter=s.repo_root / "checkpoints/lora_r16_bf16/best")
    ex = read_jsonl(s.data_dir / "processed" / "val.jsonl")
    rows = [{"annotation_id": e["annotation_id"], "patient_id": e["patient_id"], "criterion_type": e["criterion_type"],
             "gold": e["label"], "scores": label_scores(lm, e)} for e in ex]
    out = s.results_dir / "calibration"
    out.mkdir(exist_ok=True)
    (out / "val_label_scores.json").write_text(json.dumps(rows))
    print("scored", len(rows))


def predict(rows, bias):
    return [max(r["scores"], key=lambda k: r["scores"][k] + bias.get(k, 0.0)) for r in rows]


def macro(y, p):
    return float(f1_score(y, p, labels=list(ALL_LABELS), average="macro", zero_division=0, sample_weight=None))


def fit(rows, grid=np.arange(-3, 3.01, 0.5)):
    bias = {k: 0.0 for k in ALL_LABELS}
    y = [r["gold"] for r in rows]
    best = macro(y, predict(rows, bias))
    for _ in range(2):
        for lab in ALL_LABELS:
            for v in grid:
                trial = {**bias, lab: float(v)}
                m = macro(y, predict(rows, trial))
                if m > best + 1e-9:
                    best, bias = m, trial
    return bias


def cv() -> None:
    s = get_settings()
    rows = json.loads((s.results_dir / "calibration" / "val_label_scores.json").read_text())
    pats = sorted({r["patient_id"] for r in rows})
    y = [r["gold"] for r in rows]
    p0 = predict(rows, {})
    pred = {}
    biases = {}
    for pt in pats:
        tr = [r for r in rows if r["patient_id"] != pt]
        b = fit(tr)
        biases[pt] = b
        for r in rows:
            if r["patient_id"] == pt:
                pred[r["annotation_id"]] = max(r["scores"], key=lambda k: r["scores"][k] + b[k])
    p1 = [pred[r["annotation_id"]] for r in rows]
    res = {"n": len(rows), "uncalibrated": {"accuracy": accuracy_score(y, p0), "macro_f1": macro(y, p0)},
           "loo_patient_calibrated": {"accuracy": accuracy_score(y, p1), "macro_f1": macro(y, p1)},
           "in_sample_oracle_bias": fit(rows), "fold_biases": biases}
    res["in_sample_oracle_calibrated"] = {"accuracy": accuracy_score(y, predict(rows, res["in_sample_oracle_bias"])),
                                          "macro_f1": macro(y, predict(rows, res["in_sample_oracle_bias"]))}
    Path(s.results_dir / "calibration" / "cv_result.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "fold_biases"}, indent=1))


if __name__ == "__main__":
    {"score": score, "cv": cv}[sys.argv[1]]()

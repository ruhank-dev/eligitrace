"""ONE-TIME frozen evaluation on the held-out test split: base vs fine-tuned, identical settings.

1. writes results/final_test/FROZEN.json (adapter hash, commit, prompt hash, decoding) BEFORE predicting;
2. refuses to run if FROZEN.json already exists (the test set is evaluated exactly once);
3. writes metrics, confusion matrices, before/after table and bootstrap CIs.
"""

import argparse
import csv
import hashlib
import json
from datetime import UTC, datetime

import numpy as np

from evaluation.predict import DECODING, adapter_sha256, git_commit, run_predictions
from evaluation.report import summarise, write_report
from modeling.prompts import SYSTEM_PROMPT
from settings.config import get_settings


def bootstrap_ci(records: list[dict], n: int = 1000, seed: int = 42) -> dict:
    """Patient-level bootstrap (patients are the independent unit) for accuracy and macro-F1."""
    from evaluation.metrics_cls import classification

    rng = np.random.default_rng(seed)
    by_pat: dict[str, list[dict]] = {}
    for r in records:
        by_pat.setdefault(r["patient_id"], []).append(r)
    pats = sorted(by_pat)
    acc, f1 = [], []
    for _ in range(n):
        sample = [r for p in rng.choice(pats, size=len(pats), replace=True) for r in by_pat[p]]
        c = classification(sample, "all")
        acc.append(c["accuracy"])
        f1.append(c["macro_f1_present"])
    q = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]  # noqa: E731
    return {"accuracy_95ci": q(acc), "macro_f1_present_95ci": q(f1), "resamples": n, "unit": "patient"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="checkpoints/lora_r16_bf16/best")
    a = ap.parse_args()
    s = get_settings()
    out = s.results_dir / "final_test"
    if (out / "FROZEN.json").exists():
        raise SystemExit("final_test/FROZEN.json exists: the test split has already been evaluated once.")
    adapter = s.repo_root / a.adapter
    if not (adapter / "adapter_model.safetensors").exists():
        raise SystemExit(f"adapter not found: {adapter}")
    out.mkdir(parents=True, exist_ok=True)
    frozen = {"frozen_at": datetime.now(UTC).isoformat(timespec="seconds"), "git_commit": git_commit(),
              "adapter": a.adapter, "adapter_sha256": adapter_sha256(adapter), "base_model": s.target_model_id,
              "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(), "decoding": DECODING,
              "dtype": s.dtype, "note": "All model/prompt/threshold choices were made on the validation split."}
    (out / "FROZEN.json").write_text(json.dumps(frozen, indent=2))

    summary = {}
    for tag, ad in (("base", None), ("finetuned", adapter)):
        recs = run_predictions("test", ad, tag=tag, out_root=out, final=True)
        m = write_report(recs, out / tag)
        m["bootstrap"] = bootstrap_ci(recs)
        (out / tag / "metrics.json").write_text(json.dumps(m, indent=2))
        summary[tag] = m
    ids = [json.loads(x)["annotation_id"] for x in (out / "base" / "predictions.jsonl").read_text().splitlines()]
    ids2 = [json.loads(x)["annotation_id"] for x in (out / "finetuned" / "predictions.jsonl").read_text().splitlines()]
    assert ids == ids2, "before/after must use identical examples"
    rows = []
    for sub in ("all", "inclusion", "exclusion"):
        for metric in ("accuracy", "macro_f1", "macro_f1_present"):
            b, f = summary["base"][sub][metric], summary["finetuned"][sub][metric]
            rows.append({"subset": sub, "metric": metric, "base": b, "finetuned": f, "delta": f - b})
    for agg in ("micro", "macro"):
        for k in ("precision", "recall", "f1"):
            b, f = summary["base"]["evidence"][agg][k], summary["finetuned"]["evidence"][agg][k]
            rows.append({"subset": "evidence", "metric": f"{agg}_{k}", "base": b, "finetuned": f, "delta": f - b})
    b, f = summary["base"]["parse_fail_rate"], summary["finetuned"]["parse_fail_rate"]
    rows.append({"subset": "all", "metric": "parse_fail_rate", "base": b, "finetuned": f, "delta": f - b})
    with (out / "before_after.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(json.dumps({t: {"acc": summary[t]["all"]["accuracy"], "macro_f1": summary[t]["all"]["macro_f1"]}
                      for t in summary}, indent=2))


if __name__ == "__main__":
    main()

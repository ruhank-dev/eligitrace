"""Collect every validation-only fine-tuning run into results/experiments/summary.{json,csv,md}.

Reads results/training/<run>/ (log_history, val_metrics_by_epoch, val predictions of the best epoch) and
recomputes the full metric set from the saved predictions, so the table is derived, never typed by hand.
"""

import csv
import json
import re
import statistics

from evaluation.report import load_records, summarise
from settings.config import get_settings

BASELINE = "lora_r16_bf16"
RUNS = [BASELINE, "exp_seed43", "exp_seed44", "exp_oversample3", "exp_qlora", "exp_hp_r8", "exp_hp_r32",
        "exp_hp_lr1e-4", "exp_draft05b_ft"]
LABELS = ("included", "not included", "excluded", "not excluded", "not enough information", "not applicable")


def one(run: str) -> dict | None:
    s = get_settings()
    d = s.results_dir / "training" / run
    f = d / "val_metrics_by_epoch.json"
    if not f.exists():
        return None
    ep = json.loads(f.read_text())
    best = ep["best"]["epoch"]
    recs = load_records(d / "val_predictions" / f"{run}_epoch{best}" / "predictions.jsonl")
    m = summarise(recs)
    hist = json.loads((d / "log_history.json").read_text())
    log = (d / "train_stdout.log").read_text(errors="ignore") if (d / "train_stdout.log").exists() else ""
    tp = re.search(r"trainable params: ([\d,]+)", log)
    cfg = json.loads(json.dumps(__import__("yaml").safe_load((d / "config.yaml").read_text())))
    row = {"run": run, "base_model": cfg["base_model"], "strategy": cfg["strategy"], "seed": cfg["seed"],
           "lora_r": cfg["lora"]["r"], "lr": cfg["train"]["lr"], "oversample": cfg.get("oversample") or {},
           "best_epoch": best, "val_loss_best": ep["best"]["val_loss"],
           "accuracy": m["all"]["accuracy"], "macro_f1": m["all"]["macro_f1"],
           "macro_f1_present": m["all"]["macro_f1_present"],
           "inclusion_acc": m["inclusion"]["accuracy"], "exclusion_acc": m["exclusion"]["accuracy"],
           "evidence_micro_f1": m["evidence"]["micro"]["f1"], "parse_fail": m["parse_fail_rate"],
           "per_class_f1": {k: round(v["f1"], 3) for k, v in m["all"]["per_class"].items()},
           "per_class_support": {k: v["support"] for k, v in m["all"]["per_class"].items()},
           "train_minutes": round(hist["epochs"][-1]["elapsed_s"] / 60, 1),
           "peak_gb": hist["epochs"][-1]["peak_gb"],
           "trainable_params": int(tp.group(1).replace(",", "")) if tp else None,
           "epoch_macro_f1": [e["macro_f1"] for e in ep["epochs"]]}
    return row


def main() -> None:
    out = get_settings().results_dir / "experiments"
    out.mkdir(parents=True, exist_ok=True)
    rows = [r for r in (one(x) for x in RUNS) if r]
    seeds = [r for r in rows if r["run"] in (BASELINE, "exp_seed43", "exp_seed44")]
    agg = {}
    if len(seeds) > 1:
        for k in ("accuracy", "macro_f1", "macro_f1_present", "evidence_micro_f1", "inclusion_acc", "exclusion_acc"):
            xs = [r[k] for r in seeds]
            agg[k] = {"mean": statistics.mean(xs), "sd": statistics.stdev(xs), "n": len(xs), "values": xs}
    (out / "summary.json").write_text(json.dumps({"runs": rows, "seed_aggregate": agg,
                                                  "note": "validation split only; test never used"}, indent=2))
    cols = ["run", "base_model", "strategy", "seed", "lora_r", "lr", "best_epoch", "accuracy", "macro_f1",
            "macro_f1_present", "inclusion_acc", "exclusion_acc", "evidence_micro_f1", "parse_fail",
            "train_minutes", "peak_gb", "trainable_params"]
    with (out / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    lines = ["| run | acc | macro-F1 | macro-F1 (present) | evid. F1 | best ep | min | GB |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['run']} | {r['accuracy']:.3f} | {r['macro_f1']:.3f} | {r['macro_f1_present']:.3f} | "
                     f"{r['evidence_micro_f1']:.3f} | {r['best_epoch']} | {r['train_minutes']} | {r['peak_gb']} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    if agg:
        print(json.dumps({k: (round(v["mean"], 4), round(v["sd"], 4)) for k, v in agg.items()}))


if __name__ == "__main__":
    main()

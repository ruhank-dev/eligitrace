"""Summaries + files (metrics JSON, per-class CSV, confusion PNGs) for one prediction set."""

import argparse
import csv
import json
from pathlib import Path

from evaluation.metrics_cls import classification
from evaluation.metrics_evidence import evidence
from settings.config import get_settings


def summarise(records: list[dict]) -> dict:
    return {
        "n": len(records),
        "all": classification(records, "all"),
        "inclusion": classification(records, "inclusion"),
        "exclusion": classification(records, "exclusion"),
        "evidence": evidence(records),
        "evidence_by_type": {t: evidence([r for r in records if r["criterion_type"] == t])
                             for t in ("inclusion", "exclusion")},
        "parse_fail_rate": sum(not r["parse_ok"] for r in records) / max(len(records), 1),
        "mean_latency_s": sum(r["latency_s"] for r in records) / max(len(records), 1),
    }


def write_report(records: list[dict], out_dir: Path) -> dict:
    from evaluation.plots import confusion_png

    out_dir.mkdir(parents=True, exist_ok=True)
    m = summarise(records)
    (out_dir / "metrics.json").write_text(json.dumps(m, indent=2))
    with (out_dir / "per_class.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subset", "label", "precision", "recall", "f1", "support"])
        for sub in ("all", "inclusion", "exclusion"):
            for lab, v in m[sub]["per_class"].items():
                w.writerow([sub, lab, f"{v['precision']:.4f}", f"{v['recall']:.4f}", f"{v['f1']:.4f}", v["support"]])
    for sub in ("all", "inclusion", "exclusion"):
        confusion_png(m[sub]["confusion"], out_dir / f"confusion_{sub}.png", title=f"{sub} criteria")
    return m


def load_records(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--split", required=True)
    a = ap.parse_args()
    d = get_settings().results_dir / a.tag / a.split
    m = write_report(load_records(d / "predictions.jsonl"), d)
    print(json.dumps({k: m[k]["accuracy"] for k in ("all", "inclusion", "exclusion")}, indent=2))


if __name__ == "__main__":
    main()

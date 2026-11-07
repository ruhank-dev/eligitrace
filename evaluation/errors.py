"""C4: categorise wrong predictions and write an error-analysis report with concrete cases.

Categories are assigned by explicit rules (no model involvement) so every case is reproducible.
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from dataprep.loader import load_clean
from settings.config import get_settings

NEG = re.compile(r"\b(no|denies|denied|without|negative|never|not)\b", re.I)
NUM = re.compile(r"\d")
TEMPORAL = re.compile(r"\b(history of|previous|prior|past|ago|years|months|weeks|days|currently|recent)\b", re.I)
POLARITY = {("included", "not included"), ("not included", "included"),
            ("excluded", "not excluded"), ("not excluded", "excluded")}


def categorise(r: dict, note: dict[int, str], criterion: str) -> list[str]:
    g, p = r["gold_label"], r["pred_label"]
    gs, ps = set(r["gold_evidence"]), set(r["pred_evidence"])
    cats = []
    if p == "invalid":
        return ["parse failure"]
    if g == p:
        if gs != ps:
            cats.append("correct label, wrong evidence")
        return cats
    if g == "not enough information":
        cats.append("missing information treated as decisive (unsupported decision)")
    elif p == "not enough information":
        cats.append("false 'not enough information' (evidence was present)")
    if (g, p) in POLARITY:
        cats.append("polarity flip (met vs not met)")
    if "not applicable" in (g, p):
        cats.append("not-applicable confusion")
    if ps and not gs:
        cats.append("cited evidence where experts found none")
    cited = " ".join(note.get(i, "") for i in gs | ps)
    if NUM.search(criterion) and NUM.search(cited):
        cats.append("numeric / threshold reasoning")
    if NEG.search(cited):
        cats.append("negation in evidence")
    if TEMPORAL.search(cited + " " + criterion):
        cats.append("temporal qualifier")
    return cats or ["other label error"]


def analyse(pred_path: Path, out_md: Path, max_cases: int = 14) -> dict:
    df = load_clean().set_index("annotation_id")
    recs = [json.loads(x) for x in pred_path.read_text(encoding="utf-8").splitlines() if x]
    counts, cases = Counter(), []
    for r in recs:
        row = df.loc[r["annotation_id"]]
        cats = categorise(r, row.sentences, row.criterion_text)
        for c in cats:
            counts[c] += 1
        if cats:
            cases.append((cats, r, row))
    # pick diverse cases: first example of each primary category, then fill
    seen, chosen, ids = set(), [], set()
    for cats, r, row in cases:
        if cats[0] not in seen:
            seen.add(cats[0])
            chosen.append((cats, r, row))
            ids.add(r["annotation_id"])
    for cats, r, row in cases:
        if len(chosen) >= max_cases:
            break
        if r["annotation_id"] not in ids and r["gold_label"] != r["pred_label"]:
            chosen.append((cats, r, row))
            ids.add(r["annotation_id"])
    lines = [f"# Error analysis — `{pred_path.parent.name}` ({len(recs)} predictions)", "",
             "Categories are assigned by deterministic rules (`evaluation/errors.py`); one error can have several.", "",
             "| Category | Count |", "|---|---|"] + [f"| {k} | {v} |" for k, v in counts.most_common()]
    for i, (cats, r, row) in enumerate(chosen[:max_cases], 1):
        ev = lambda ids: "; ".join(f"[{j}] {row.sentences.get(j, '?')}" for j in sorted(ids)) or "(none)"  # noqa: E731
        lines += ["", f"## Case {i}: {', '.join(cats)}",
                  f"- **Annotation** {r['annotation_id']} · patient `{r['patient_id']}` · trial `{row.trial_id}` · {r['criterion_type']}",
                  f"- **Criterion:** {row.criterion_text}",
                  f"- **Gold:** `{r['gold_label']}` with evidence {sorted(r['gold_evidence'])}",
                  f"- **Predicted:** `{r['pred_label']}` with evidence {sorted(r['pred_evidence'])}",
                  f"- **Gold evidence text:** {ev(r['gold_evidence'])}",
                  f"- **Predicted evidence text:** {ev(r['pred_evidence'])}",
                  f"- **Raw output:** `{r['raw'].strip()[:200]}`"]
    out_md.write_text("\n".join(lines), encoding="utf-8")
    summary = {"n": len(recs), "n_wrong_label": sum(r["gold_label"] != r["pred_label"] for r in recs),
               "category_counts": dict(counts), "cases_written": min(len(chosen), max_cases)}
    out_md.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True, type=Path)
    a = ap.parse_args()
    p = a.predictions if a.predictions.is_absolute() else get_settings().repo_root / a.predictions
    print(json.dumps(analyse(p, p.parent / "error_analysis.md"), indent=2))


if __name__ == "__main__":
    main()

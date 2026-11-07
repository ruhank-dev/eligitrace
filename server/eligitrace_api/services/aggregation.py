"""Explicit rule-based patient-trial aggregation. NOT a ground-truth label (dataset is criterion-level)."""

RULE = (
    "1) any inclusion = 'not included' or any exclusion = 'excluded' -> Likely ineligible; "
    "2) else any criterion = 'not enough information' (or unparseable) -> Needs more information; "
    "3) else -> Likely eligible. 'not applicable' criteria are neutral."
)
BLOCKING = {("inclusion", "not included"), ("exclusion", "excluded")}


def _ref(r: dict) -> str:
    return f"{r['criterion_type'].capitalize()} #{r['position'] + 1}"


def aggregate(results: list[dict]) -> dict:
    done = [r for r in results if r.get("label")]
    counts: dict[str, int] = {}
    for r in done:
        counts[r["label"]] = counts.get(r["label"], 0) + 1
    if len(done) < len(results):
        return {"outcome": "Incomplete", "rule": RULE, "counts": counts,
                "trace": [f"{len(results) - len(done)} criteria not screened yet"]}
    blockers = [r for r in done if (r["criterion_type"], r["label"]) in BLOCKING]
    if blockers:
        return {"outcome": "Likely ineligible", "rule": RULE, "counts": counts,
                "trace": [f"{_ref(r)} = '{r['label']}': {r['criterion_text'][:100]}" for r in blockers]}
    unknown = [r for r in done if r["label"] in ("not enough information", "invalid")]
    if unknown:
        return {"outcome": "Needs more information", "rule": RULE, "counts": counts,
                "trace": [f"{_ref(r)} needs information: {r['criterion_text'][:100]}" for r in unknown]}
    return {"outcome": "Likely eligible", "rule": RULE, "counts": counts,
            "trace": ["Every inclusion criterion is met or not applicable and no exclusion criterion applies."]}

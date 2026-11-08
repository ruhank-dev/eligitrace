"""Reproducible evidence for the live Inference Lab speed-up (rubric F3/G3).

  export : copy the Lab decode runs already saved in the app database to results/lab/ui_runs_saved.json
  run N  : call the RUNNING app (task app) N times with fixed settings and save results/lab/live_runs.json

The Lab measures real GPU timings, so single runs vary with background load; this tool reports the
spread (mean, sd, min, max) instead of one lucky or unlucky number.
"""

import json
import statistics
import sys
import urllib.request

from server.eligitrace_api.database import SessionLocal
from server.eligitrace_api.repositories import lab_repo
from settings.config import get_settings

URL = "http://127.0.0.1:8000/api/lab/decode"
BODY = {"prompt_source": "example", "example_index": 0, "gamma": 4, "max_new_tokens": 48, "repeats": 2,
        "modes": ["no_cache", "kv_cached", "speculative"], "ignore_eos": False}


def row(res: dict, mode: str) -> dict:
    return next(m for m in res["modes"] if m["mode"] == mode)


def export() -> None:
    out = get_settings().results_dir / "lab"
    out.mkdir(parents=True, exist_ok=True)
    with SessionLocal() as db:
        runs = [r for r in lab_repo.list_recent(db, "decode", 100)]
        data = [{"created_at": r.created_at.isoformat(), "params": r.params,
                 "modes": [{k: m.get(k) for k in ("mode", "total_ms", "speedup_vs_kv_cached", "acceptance_rate",
                                                   "tokens_equal_reference")} for m in r.result["modes"]]}
                for r in sorted(runs, key=lambda x: x.created_at)]
    (out / "ui_runs_saved.json").write_text(json.dumps(data, indent=2))
    print(f"exported {len(data)} saved UI runs -> {out / 'ui_runs_saved.json'}")


def post(body: dict) -> dict:
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())["result"]


def run(n: int) -> None:
    out = get_settings().results_dir / "lab"
    out.mkdir(parents=True, exist_ok=True)
    runs = []
    for i in range(n):
        res = post(BODY)
        runs.append({"run": i + 1, "prompt_tokens": res["prompt_tokens"],
                     "no_cache_ms": row(res, "no_cache")["total_ms"], "kv_cached_ms": row(res, "kv_cached")["total_ms"],
                     "speculative_ms": row(res, "speculative")["total_ms"],
                     "speculative_vs_kv": row(res, "speculative")["speedup_vs_kv_cached"],
                     "kv_vs_no_cache": row(res, "no_cache")["total_ms"] / row(res, "kv_cached")["total_ms"],
                     "acceptance_rate": row(res, "speculative")["acceptance_rate"],
                     "all_tokens_equal": all(m["tokens_equal_reference"] for m in res["modes"]),
                     "output_tokens": row(res, "kv_cached")["output_tokens"]})
        print(runs[-1], flush=True)
    sp = [r["speculative_vs_kv"] for r in runs]
    agg = {"n": n, "speculative_vs_kv_mean": statistics.mean(sp), "sd": statistics.stdev(sp) if n > 1 else 0.0,
           "min": min(sp), "max": max(sp)}
    (out / "live_runs.json").write_text(json.dumps({"request": BODY, "runs": runs, "aggregate": agg}, indent=2))
    print(agg)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        run(int(sys.argv[2]) if len(sys.argv) > 2 else 5)
    else:
        export()

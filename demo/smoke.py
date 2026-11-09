"""End-to-end demo smoke test against the RUNNING app (task app). Read-only for results/: it only prints.

  uv run python -m demo.smoke
Posts demo/sample_note.txt + demo/sample_criteria.txt, waits for completion, prints each decision and the outcome.
"""

import json
import time
import urllib.request
from pathlib import Path

from server.eligitrace_api.services.criteria_parser import parse_criteria

BASE = "http://127.0.0.1:8000"
HERE = Path(__file__).parent


def call(path: str, body: dict | None = None):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode() if body else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def main() -> None:
    note = (HERE / "sample_note.txt").read_text(encoding="utf-8").strip()
    criteria = parse_criteria((HERE / "sample_criteria.txt").read_text(encoding="utf-8"))
    sid = call("/api/screenings", {"patient_note": note, "criteria": criteria, "patient_label": "demo smoke test"})["id"]
    while True:
        sc = call(f"/api/screenings/{sid}")
        if sc["status"] in ("done", "error"):
            break
        time.sleep(3)
    for r in sc["results"]:
        ev = [e["id"] for e in r["evidence"]]
        print(f"{r['criterion_type']:<9} {r['label']:<24} evidence={ev}  {r['criterion_text'][:60]}")
    print("OUTCOME:", sc["overall"]["outcome"])
    print("TRACE:", *sc["overall"]["trace"], sep="\n  ")
    print("HTML report bytes:", len(urllib.request.urlopen(f"{BASE}/api/screenings/{sid}/report.html").read()))


if __name__ == "__main__":
    main()

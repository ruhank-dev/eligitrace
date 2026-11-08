"""Demo readiness check. READ-ONLY: it never writes to results/, data/ or the report.

  uv run python -m demo.preflight         # check files, GPU memory and (if running) the app
Prints a PASS / WARN / FAIL line per item and the next command to run.
"""

import json
import subprocess
import urllib.request
from pathlib import Path

from settings.config import get_settings

ROOT = Path(__file__).resolve().parents[1]
BASE = "http://127.0.0.1:8000"


def line(status: str, msg: str) -> None:
    print(f"[{status}] {msg}")


def main() -> None:
    s = get_settings()
    need = {
        "split manifest": s.data_dir / "splits" / "split_manifest.json",
        "frozen test record": s.results_dir / "final_test" / "FROZEN.json",
        "before/after table": s.results_dir / "final_test" / "before_after.csv",
        "KV benchmark": s.results_dir / "kvcache" / "finetuned" / "benchmark.csv",
        "attention summary": s.results_dir / "attention" / "summary.json",
        "speculative summary": s.results_dir / "speculative" / "finetuned" / "summary.json",
        "report PDF": ROOT / "report" / "EligiTrace_Report.pdf",
        "local adapter": s.adapter_local_dir / "adapter_config.json",
    }
    for name, p in need.items():
        line("PASS" if p.exists() else "FAIL", f"{name}: {p.relative_to(ROOT) if p.is_relative_to(ROOT) else p}")
    try:
        q = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.used,memory.total",
                                     "--format=csv,noheader,nounits"], text=True).strip().split(", ")
        used, total = float(q[1]), float(q[2])
        line("PASS" if used < 1500 else "WARN",
             f"GPU {q[0]}: {used:.0f} of {total:.0f} MiB in use (close games/overlays if above ~1500 MiB)")
    except Exception:
        line("WARN", "nvidia-smi not available (the app would run on CPU, slowly)")
    try:
        h = json.loads(urllib.request.urlopen(f"{BASE}/api/health", timeout=3).read())
        line("PASS" if h["status"] == "ready" else "WARN", f"app at {BASE}: status={h['status']} adapter={h.get('adapter')}")
    except Exception:
        line("WARN", f"app is not running at {BASE}: start it with `task app` and wait for the Ready pill")
    print("\nSample input to paste in the Screen tab: demo/sample_note.txt and demo/sample_criteria.txt")


if __name__ == "__main__":
    main()

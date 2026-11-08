"""Run the validation-only experiment configs one after another (skips runs that already finished).

Usage: uv run python -m finetune.run_experiments [name ...]   (no names = all, in ORDER)
Each run is a normal `finetune.train` run; its logs and metrics land in results/training/<run_name>/.
The test split is never touched (training and model selection use train + val only).
"""

import subprocess
import sys
import time
from pathlib import Path

import yaml

from settings.config import get_settings

CONFIGS = Path(__file__).parent / "configs" / "experiments"
ORDER = ["exp_draft05b_ft", "exp_oversample3", "exp_seed43", "exp_seed44", "exp_qlora",
         "exp_hp_r8", "exp_hp_r32", "exp_hp_lr1e-4"]


def done(run: str) -> bool:
    return (get_settings().results_dir / "training" / run / "val_metrics_by_epoch.json").exists()


def main() -> None:
    names = sys.argv[1:] or ORDER
    s = get_settings()
    for name in names:
        cfg = CONFIGS / f"{name}.yaml"
        run = yaml.safe_load(cfg.read_text())["run_name"]
        if done(run):
            print(f"[skip] {run} already finished", flush=True)
            continue
        out = s.results_dir / "training" / run
        out.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        print(f"[start] {run}", flush=True)
        with (out / "train_stdout.log").open("w", encoding="utf-8") as log:
            rc = subprocess.call([sys.executable, "-m", "finetune.train", "--config", str(cfg)],
                                 stdout=log, stderr=subprocess.STDOUT, cwd=s.repo_root)
        print(f"[end] {run} rc={rc} wall={time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

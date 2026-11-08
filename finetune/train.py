"""LoRA supervised fine-tuning on the TRAIN split only. Selects the best epoch by validation macro-F1.

A plain PyTorch loop (instead of HF Trainer) keeps precision handling explicit: fp16 autocast +
GradScaler for the mixed strategy, pure fp32 otherwise. Per-step loss, per-epoch val loss and
generation-based val metrics are logged to results/training/<run>/.
"""

import argparse
import json
import math
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from dataprep.formatting import read_jsonl
from finetune.collator import CompletionCollator, tokenize_example
from finetune.setup_model import AMP_DTYPE, build_trainable
from settings.config import get_settings


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def oversample(examples: list[dict], factors: dict) -> list[dict]:
    out = []
    for e in examples:
        out.extend([e] * int(factors.get(e["label"], 1)))
    return out


@torch.no_grad()
def val_loss(model, batches, amp) -> float:
    model.eval()
    tot, n = 0.0, 0
    for b in batches:
        b = {k: v.cuda() for k, v in b.items()}
        with torch.autocast("cuda", dtype=amp or torch.float32, enabled=amp is not None):
            out = model(**b)
        k = int((b["labels"] != -100).sum())
        tot += float(out.loss) * k
        n += k
    model.train()
    return tot / max(n, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-steps", type=int, default=0, help="debug: stop early")
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    s = get_settings()
    torch.cuda.set_per_process_memory_fraction(get_settings().gpu_mem_fraction)  # fail loudly instead of Windows sysmem spill
    seed_all(cfg["seed"])
    run_dir = s.results_dir / "training" / cfg["run_name"]
    ckpt_root = s.repo_root / "checkpoints" / cfg["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg))

    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    tr = oversample(read_jsonl(s.data_dir / "processed" / "train.jsonl"), cfg.get("oversample") or {})
    va = read_jsonl(s.data_dir / "processed" / "val.jsonl")
    tr_tok = [tokenize_example(tok, e["messages"], cfg["max_seq_len"]) for e in tr]
    va_tok = [tokenize_example(tok, e["messages"], cfg["max_seq_len"]) for e in va]
    coll = CompletionCollator(tok.pad_token_id)
    mb, acc = cfg["train"]["micro_batch"], cfg["train"]["grad_accum"]
    va_batches = [coll(va_tok[i:i + mb]) for i in range(0, len(va_tok), mb)]

    model = build_trainable(cfg["base_model"], cfg["strategy"], cfg["lora"], cfg["grad_ckpt"])
    model.print_trainable_parameters()
    amp = AMP_DTYPE.get(cfg["strategy"])
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    steps_per_epoch = math.ceil(len(tr_tok) / (mb * acc))
    total = steps_per_epoch * cfg["train"]["epochs"]
    sched = get_cosine_schedule_with_warmup(opt, int(total * cfg["train"]["warmup_ratio"]), total)
    scaler = torch.amp.GradScaler("cuda", enabled=amp == torch.float16)

    log, epochs_log, step = [], [], 0
    rng = random.Random(cfg["seed"])
    torch.cuda.reset_peak_memory_stats()
    t_start = time.time()
    model.train()
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        order = list(range(len(tr_tok)))
        rng.shuffle(order)
        micro = [order[i:i + mb] for i in range(0, len(order), mb)]
        run_loss, run_tok = 0.0, 0
        for mi, idx in enumerate(micro):
            b = {k: v.cuda() for k, v in coll([tr_tok[i] for i in idx]).items()}
            with torch.autocast("cuda", dtype=amp or torch.float32, enabled=amp is not None):
                loss = model(**b).loss
            scaler.scale(loss / acc).backward()
            k = int((b["labels"] != -100).sum())
            run_loss += float(loss) * k
            run_tok += k
            if (mi + 1) % acc == 0 or mi + 1 == len(micro):
                scaler.unscale_(opt)
                gn = float(torch.nn.utils.clip_grad_norm_(params, cfg["train"]["max_grad_norm"]))
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                step += 1
                if step % cfg["train"]["log_every"] == 0 or step == 1:
                    rec = {"step": step, "epoch": round(epoch - 1 + (mi + 1) / len(micro), 3),
                           "train_loss": run_loss / max(run_tok, 1), "lr": sched.get_last_lr()[0],
                           "grad_norm": gn, "elapsed_s": round(time.time() - t_start, 1)}
                    log.append(rec)
                    print(json.dumps(rec), flush=True)
                    run_loss, run_tok = 0.0, 0
                if a.max_steps and step >= a.max_steps:
                    break
        vl = val_loss(model, va_batches, amp)
        ck = ckpt_root / f"epoch{epoch}"
        model.save_pretrained(ck)
        tok.save_pretrained(ck)
        rec = {"epoch": epoch, "step": step, "val_loss": vl, "checkpoint": str(ck.relative_to(s.repo_root)),
               "elapsed_s": round(time.time() - t_start, 1),
               "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 3)}
        epochs_log.append(rec)
        print(json.dumps(rec), flush=True)
        (run_dir / "log_history.json").write_text(json.dumps({"steps": log, "epochs": epochs_log}, indent=2))
        if a.max_steps and step >= a.max_steps:
            break

    del model, opt
    torch.cuda.empty_cache()
    if cfg.get("eval_each_epoch") and not a.max_steps:
        select_best(cfg, epochs_log, run_dir, s)


def select_best(cfg, epochs_log, run_dir: Path, s) -> None:
    """Generation-based validation metrics per epoch; best by macro-F1 (tie: evidence micro-F1)."""
    from evaluation.predict import run_predictions
    from evaluation.report import summarise

    rows = []
    for e in epochs_log:
        tag = f"{cfg['run_name']}_epoch{e['epoch']}"
        recs = run_predictions("val", s.repo_root / e["checkpoint"], tag=tag,
                               out_root=run_dir / "val_predictions", model_id=cfg["base_model"])
        m = summarise(recs)
        rows.append({"epoch": e["epoch"], "val_loss": e["val_loss"], "accuracy": m["all"]["accuracy"],
                     "macro_f1": m["all"]["macro_f1"], "evidence_micro_f1": m["evidence"]["micro"]["f1"],
                     "parse_fail": m["parse_fail_rate"]})
        print(json.dumps(rows[-1]), flush=True)
    best = max(rows, key=lambda r: (r["macro_f1"], r["evidence_micro_f1"]))
    (run_dir / "val_metrics_by_epoch.json").write_text(json.dumps({"epochs": rows, "best": best}, indent=2))
    dst = s.repo_root / "checkpoints" / cfg["run_name"] / "best"
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(s.repo_root / "checkpoints" / cfg["run_name"] / f"epoch{best['epoch']}", dst)
    print(f"best epoch {best['epoch']} -> {dst}")


if __name__ == "__main__":
    main()

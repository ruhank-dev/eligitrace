"""F1-F3: tokenizer compatibility, token equality vs greedy target, acceptance / latency / memory."""

import argparse
import csv
import json
import statistics
from pathlib import Path

import torch

from dataprep.formatting import read_jsonl
from kvcache.decoder import greedy_cached
from modeling.loader import eos_ids, load_model
from modeling.prompts import render_prompt
from settings.config import get_settings
from speculative.decoder import speculative_greedy
from speculative.pairing import compatibility

GAMMAS = (1, 2, 3, 4, 6, 8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", type=Path, default=None, help="LoRA for the TARGET (merged)")
    ap.add_argument("--tag", default="base")
    ap.add_argument("--n-prompts", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--free-run", type=int, default=96, help="EOS-free length for long-generation runs")
    a = ap.parse_args()
    s = get_settings()
    tgt = load_model(s.target_model_id, adapter=a.adapter, merge=True)
    drf = load_model(s.draft_model_id)
    compat = compatibility(tgt.tokenizer, drf.tokenizer, tgt.model, drf.model)
    assert compat["compatible"], compat
    dev = tgt.device
    eos = eos_ids(tgt.tokenizer, tgt.model)
    val = read_jsonl(s.data_dir / "processed" / "val.jsonl")[: a.n_prompts]
    prompts = [tgt.tokenizer(render_prompt(tgt.tokenizer, e["messages"][:-1]), return_tensors="pt")
               .input_ids.to(dev) for e in val]
    # two workloads: (1) the real task (JSON answer, stops at EOS), (2) long free generation (no EOS)
    workloads = {"task_json": (64, eos), f"free_{a.free_run}": (a.free_run, set())}
    rows, eq = [], []
    for wname, (max_new, stop) in workloads.items():
        base_runs = []
        for ids in prompts:
            greedy_cached(tgt.model, ids, max_new, stop)  # warm-up
            base_runs.append([greedy_cached(tgt.model, ids, max_new, stop) for _ in range(a.repeats)])
        base_lat = [statistics.mean(r.total_s for r in rs) for rs in base_runs]
        base_tok = [len(rs[0].tokens) for rs in base_runs]
        rows.append({"workload": wname, "mode": "greedy_kv", "gamma": 0,
                     "latency_ms": statistics.mean(base_lat) * 1e3,
                     "tokens_per_s": sum(base_tok) / sum(base_lat), "acceptance_rate": "",
                     "mean_accepted_per_round": "", "target_forwards_per_token": 1.0,
                     "speedup": 1.0, "all_equal": True})
        for g in GAMMAS:
            lat, toks, acc, prop, apr, tfw, equal = [], 0, 0, 0, [], 0, True
            for ids, rs in zip(prompts, base_runs):
                ref = rs[0].tokens
                speculative_greedy(tgt.model, drf.model, ids, g, max_new, stop)  # warm-up
                runs = [speculative_greedy(tgt.model, drf.model, ids, g, max_new, stop) for _ in range(a.repeats)]
                r = runs[0]
                same = r.tokens == ref
                equal &= same
                if not same:
                    k = next((i for i, (x, y) in enumerate(zip(r.tokens, ref)) if x != y), min(len(r.tokens), len(ref)))
                    eq.append({"workload": wname, "gamma": g, "prompt_len": int(ids.shape[1]), "first_diff": k,
                               "len_spec": len(r.tokens), "len_ref": len(ref)})
                lat.append(statistics.mean(x.total_s for x in runs))
                toks += len(r.tokens)
                acc += r.accepted
                prop += r.proposed
                apr += r.accepted_per_round
                tfw += r.target_forwards
            rows.append({"workload": wname, "mode": "speculative", "gamma": g,
                         "latency_ms": statistics.mean(lat) * 1e3, "tokens_per_s": toks / sum(lat),
                         "acceptance_rate": acc / prop if prop else 0.0,
                         "mean_accepted_per_round": statistics.mean(apr) if apr else 0.0,
                         "target_forwards_per_token": tfw / toks,
                         "speedup": statistics.mean(base_lat) / statistics.mean(lat), "all_equal": equal})
            print(rows[-1], flush=True)
    # peak memory: both models resident + one representative run
    torch.cuda.reset_peak_memory_stats()
    speculative_greedy(tgt.model, drf.model, prompts[0], 4, 64, eos)
    peak_spec = torch.cuda.max_memory_allocated() / 2**20
    torch.cuda.reset_peak_memory_stats()
    greedy_cached(tgt.model, prompts[0], 64, eos)
    peak_greedy = torch.cuda.max_memory_allocated() / 2**20
    out = s.results_dir / "speculative" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    with (out / "benchmark.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out / "summary.json").write_text(json.dumps({
        "target": s.target_model_id, "target_adapter": str(a.adapter) if a.adapter else None,
        "draft": s.draft_model_id, "dtype": str(tgt.dtype), "gpu": torch.cuda.get_device_name(0),
        "compatibility": compat, "n_prompts": len(prompts), "repeats": a.repeats,
        "peak_mib_spec_total": peak_spec, "peak_mib_greedy_total_with_draft_loaded": peak_greedy,
        "mismatches": eq, "all_equal": all(r["all_equal"] for r in rows),
        "timing": "1 warm-up + repeats, torch.cuda.synchronize around every phase, batch 1",
    }, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()

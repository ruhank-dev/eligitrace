"""Re-run the KV-cache equality cases that diverged in bf16 on CPU fp32 (near-tie vs bug).

Also times one greedy step of target vs draft on GPU bf16 to explain the speculative result.
"""

import json
import time

import torch

from kvcache.benchmark import build_inputs, first_divergence
from kvcache.decoder import greedy_cached
from kvcache.reference import greedy_no_cache
from modeling.loader import load_model
from settings.config import get_settings


def equality(adapter: str, cases: list[tuple[int, int]]) -> list[dict]:
    lm = load_model(adapter=adapter, device="cpu", dtype="float32", merge=True)
    inputs = build_inputs(lm.tokenizer, lm.device)
    rows = []
    for n_in, n_out in cases:
        ids = inputs[n_in]
        c = greedy_cached(lm.model, ids, n_out, set()).tokens
        r = greedy_no_cache(lm.model, ids, n_out, set()).tokens
        rec = {"input_len": n_in, "output_len": n_out, "equal": c == r}
        if c != r:
            rec["divergence"] = first_divergence(lm.model, ids, c, r)
        rows.append(rec)
        print(rec, flush=True)
    return rows


def step_cost(model_id: str, adapter: str | None, n: int = 64) -> dict:
    lm = load_model(model_id, adapter=adapter, merge=adapter is not None)
    ids = build_inputs(lm.tokenizer, lm.device)[512]
    greedy_cached(lm.model, ids, 8, set())  # warm-up
    torch.cuda.synchronize()
    r = greedy_cached(lm.model, ids, n, set())
    out = {"model": model_id, "layers": lm.model.config.num_hidden_layers,
           "decode_ms_per_token": r.decode_s / max(len(r.tokens) - 1, 1) * 1e3}
    del lm
    torch.cuda.empty_cache()
    print(out, flush=True)
    return out


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="128:16,1024:128", help="comma list of input:output lengths")
    ap.add_argument("--no-step-cost", action="store_true", help="skip the GPU step-cost timing")
    ap.add_argument("--out", default="fp32_check.json", help="file name under results/kvcache/")
    a = ap.parse_args()
    s = get_settings()
    adapter = "checkpoints/lora_r16_bf16/best"
    cases = [tuple(int(x) for x in c.split(":")) for c in a.cases.split(",")]
    t0 = time.time()
    res = {"cpu_fp32_equality": equality(adapter, cases)}
    res["cpu_fp32_seconds"] = round(time.time() - t0, 1)
    if not a.no_step_cost:
        res["gpu_bf16_step_cost"] = [step_cost(s.target_model_id, adapter), step_cost(s.draft_model_id, None)]
    out = s.results_dir / "kvcache" / a.out
    out.write_text(json.dumps(res, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()

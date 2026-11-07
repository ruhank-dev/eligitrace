"""Plug the tiled online-softmax kernel into the real Qwen2.5 model as a Transformers attention backend
and check end-to-end agreement with PyTorch SDPA (logits + greedy tokens, KV-cached decoding).

Batch size 1 without padding, so the only mask needed is causal (handled inside the kernel with the
query offset Sk - Sq, which also covers single-token decode steps against the KV cache).
"""

import json

import torch
from transformers import AttentionInterface

from attention.tiled import tiled_attention
from dataprep.formatting import read_jsonl
from kvcache.decoder import greedy_cached
from modeling.loader import load_model
from modeling.prompts import render_prompt
from settings.config import get_settings


def tiled_attention_forward(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
    # query [B, Hq, Sq, D], key/value [B, Hkv, Sk, D] (GQA expanded inside the kernel)
    if attention_mask is not None and query.shape[0] > 1:
        raise NotImplementedError("educational kernel: batch 1 / no padding only")
    out = tiled_attention(query, key, value, causal=True, block_q=64, block_k=64, scale=scaling)
    return out.transpose(1, 2).contiguous(), None  # HF expects [B, Sq, Hq, D]


AttentionInterface.register("eligitrace_tiled", tiled_attention_forward)


def main(n_prompts: int = 3, max_new: int = 32) -> None:
    s = get_settings()
    ref = load_model(attn_implementation="sdpa")
    tok = ref.tokenizer
    prompts = [tok(render_prompt(tok, e["messages"][:-1]), return_tensors="pt").input_ids.to(ref.device)
               for e in read_jsonl(s.data_dir / "processed" / "val.jsonl")[:n_prompts]]
    with torch.inference_mode():
        ref_logits = [ref.model(input_ids=p).logits[0, -1].float().cpu() for p in prompts]
    ref_tok = [greedy_cached(ref.model, p, max_new, set()).tokens for p in prompts]
    del ref
    torch.cuda.empty_cache()
    til = load_model(attn_implementation="eligitrace_tiled")
    rows = []
    with torch.inference_mode():
        for p, rl, rt in zip(prompts, ref_logits, ref_tok):
            tl = til.model(input_ids=p).logits[0, -1].float().cpu()
            tt = greedy_cached(til.model, p, max_new, set()).tokens
            rows.append({"prompt_len": int(p.shape[1]), "max_abs_logit_diff": float((tl - rl).abs().max()),
                         "same_argmax": int(tl.argmax()) == int(rl.argmax()), "greedy_tokens_equal": tt == rt,
                         "first_diff": next((i for i, (a, b) in enumerate(zip(tt, rt)) if a != b), None)})
            print(rows[-1], flush=True)
    out = s.results_dir / "attention" / "hf_integration.json"
    out.write_text(json.dumps({"model": til.model_id, "dtype": str(til.dtype), "max_new_tokens": max_new,
                               "reference": "sdpa", "cases": rows}, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()

"""A2: architecture facts for candidate decoders, from their config.json (no weights downloaded).

Parameter counts are exact: each model is instantiated on the `meta` device (no memory) and its
tensors are counted. Memory estimates: weights = params x bytes/param; KV cache per token =
2 (K,V) x layers x kv_heads x head_dim x bytes. Gated repos (Llama, Gemma) are read from public
ungated mirrors with identical configs (noted in the output). Loaded memory for the shortlisted
pair is measured separately on the GPU.
"""

import argparse
import json

import torch
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from settings.config import get_settings

CANDIDATES = {
    "Qwen/Qwen2.5-1.5B-Instruct": {"config_from": "Qwen/Qwen2.5-1.5B-Instruct", "license": "Apache-2.0", "gated": False},
    "Qwen/Qwen2.5-0.5B-Instruct": {"config_from": "Qwen/Qwen2.5-0.5B-Instruct", "license": "Apache-2.0", "gated": False},
    "Qwen/Qwen3-1.7B": {"config_from": "Qwen/Qwen3-1.7B", "license": "Apache-2.0", "gated": False},
    "meta-llama/Llama-3.2-1B-Instruct": {"config_from": "unsloth/Llama-3.2-1B-Instruct",
                                         "license": "Llama 3.2 Community License", "gated": True},
    "google/gemma-2-2b-it": {"config_from": "unsloth/gemma-2-2b-it", "license": "Gemma Terms of Use", "gated": True},
}


def facts(name: str, meta: dict) -> dict:
    cfg = AutoConfig.from_pretrained(meta["config_from"])
    with init_empty_weights():
        m = AutoModelForCausalLM.from_config(cfg)
    if getattr(cfg, "tie_word_embeddings", False):
        m.lm_head.weight = m.get_input_embeddings().weight  # count shared tensor once
    params = sum(p.numel() for p in {id(p): p for p in m.parameters()}.values())
    tied = bool(getattr(cfg, "tie_word_embeddings", False))
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    kv = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    tok = AutoTokenizer.from_pretrained(meta["config_from"])
    kv_tok = 2 * cfg.num_hidden_layers * kv * head_dim
    return {
        "model": name, "config_source": meta["config_from"], "license": meta["license"], "gated": meta["gated"],
        "architecture": cfg.architectures[0] if cfg.architectures else type(m).__name__,
        "params": params, "params_b": round(params / 1e9, 3), "tied_embeddings": tied,
        "layers": cfg.num_hidden_layers, "hidden": cfg.hidden_size, "q_heads": cfg.num_attention_heads,
        "kv_heads": kv, "head_dim": head_dim, "intermediate": cfg.intermediate_size,
        "vocab_size": cfg.vocab_size, "tokenizer_class": type(tok).__name__, "tokenizer_len": len(tok),
        "context_window": cfg.max_position_embeddings,
        "rope_theta": getattr(cfg, "rope_theta", None),
        "weights_gb_fp32": round(params * 4 / 2**30, 2), "weights_gb_fp16": round(params * 2 / 2**30, 2),
        "weights_gb_nf4_approx": round(params * 0.5 / 2**30 * 1.1, 2),
        "kv_cache_kib_per_token_fp16": round(kv_tok * 2 / 1024, 1),
    }


def measure_loaded(model_id: str) -> dict:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    m = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16).cuda()
    used = (torch.cuda.memory_allocated() - base) / 2**30
    del m
    torch.cuda.empty_cache()
    return {"model": model_id, "measured_fp16_gb": round(used, 3)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--measure", action="store_true", help="also load shortlisted models on GPU")
    a = ap.parse_args()
    s = get_settings()
    rows = []
    for name, meta in CANDIDATES.items():
        try:
            rows.append(facts(name, meta))
        except Exception as e:  # record, do not hide
            rows.append({"model": name, "error": str(e)[:200]})
        print(json.dumps(rows[-1]), flush=True)
    measured = [measure_loaded(m) for m in (s.target_model_id, s.draft_model_id)] if a.measure else []
    out = s.results_dir / "model_selection"
    out.mkdir(parents=True, exist_ok=True)
    (out / "candidates.json").write_text(json.dumps({"candidates": rows, "measured_gpu": measured,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}, indent=2))
    print(json.dumps(measured, indent=2))


if __name__ == "__main__":
    main()

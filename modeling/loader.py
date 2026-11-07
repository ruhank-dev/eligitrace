"""Load tokenizer + causal LM (optionally with a LoRA adapter merged in) on the configured device."""

from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from settings.config import get_settings

torch.backends.cuda.matmul.allow_tf32 = True  # no-op on Turing; enables TF32 on Ampere+ (RTX 4050)


def resolve_device(device: str | None = None) -> torch.device:
    device = device or get_settings().device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.set_per_process_memory_fraction(get_settings().gpu_mem_fraction)
    return torch.device(device)


def resolve_dtype(dtype: str | None, device: torch.device) -> torch.dtype:
    dtype = dtype or get_settings().dtype
    if device.type == "cpu":
        return torch.float32  # fp16 matmuls on CPU are slow / partially unsupported
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]


@dataclass
class LoadedModel:
    model: torch.nn.Module
    tokenizer: object
    model_id: str
    adapter: str | None
    device: torch.device
    dtype: torch.dtype


def load_model(
    model_id: str | None = None,
    adapter: str | Path | None = None,
    device: str | None = None,
    dtype: str | None = None,
    attn_implementation: str = "sdpa",
    merge: bool = False,  # unmerged keeps disable_adapter() available for base-model explanations
) -> LoadedModel:
    s = get_settings()
    model_id = model_id or s.target_model_id
    dev = resolve_device(device)
    dt = resolve_dtype(dtype, dev)
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=dt, attn_implementation=attn_implementation
    )
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter))
        if merge:
            model = model.merge_and_unload()
    model.to(dev).eval()
    return LoadedModel(model, tok, model_id, str(adapter) if adapter else None, dev, dt)


def eos_ids(tokenizer, model) -> set[int]:
    ids = set()
    for src in (getattr(model.generation_config, "eos_token_id", None), tokenizer.eos_token_id):
        if isinstance(src, int):
            ids.add(src)
        elif src:
            ids.update(src)
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end, int) and im_end >= 0:
        ids.add(im_end)
    return ids

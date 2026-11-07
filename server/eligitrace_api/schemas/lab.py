from typing import Literal

from pydantic import BaseModel, Field


class DecodeLabIn(BaseModel):
    prompt_source: Literal["example", "custom"] = "example"
    example_index: int = Field(0, ge=0, le=50, description="validation example used as the prompt")
    custom_prompt: str | None = Field(None, max_length=6000)
    modes: list[Literal["no_cache", "kv_cached", "speculative"]] = Field(
        default=["no_cache", "kv_cached", "speculative"], min_length=1)
    gamma: int = Field(4, ge=1, le=8)
    max_new_tokens: int = Field(48, ge=1, le=256)
    ignore_eos: bool = Field(False, description="generate exactly max_new_tokens (benchmark style)")
    repeats: int = Field(2, ge=1, le=5)


class AttentionLabIn(BaseModel):
    seq_len: int = Field(512, ge=1, le=4096)
    heads: int = Field(12, ge=1, le=32)
    kv_heads: int = Field(2, ge=1, le=32)
    head_dim: Literal[32, 64, 128] = 128
    causal: bool = True
    block: Literal[16, 32, 64, 128, 256] = 64
    dtype: Literal["float32", "bfloat16", "float16"] = "float32"


class LabRunOut(BaseModel):
    id: str
    kind: str
    created_at: str
    params: dict
    result: dict

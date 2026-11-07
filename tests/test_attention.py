"""E1-E3: reference and tiled online-softmax attention agree with each other and PyTorch SDPA."""

import itertools
import math

import pytest
import torch
import torch.nn.functional as F

from attention.reference import causal_mask, naive_softmax_attention, sdpa_reference
from attention.tiled import tiled_attention

CASES = list(itertools.product(
    [(1, 1, 1), (2, 4, 4), (1, 12, 2)],          # (B, Hq, Hkv)  incl. GQA 12:2 like Qwen2.5-1.5B
    [(1, 1), (7, 7), (64, 64), (129, 129), (1, 300), (37, 200)],  # (Sq, Sk) incl. decode + non-multiples
    [32, 64],
    [True, False],
    [16, 64],
))


def _rand(B, Hq, Hkv, Sq, Sk, D, dtype=torch.float32, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(B, Hq, Sq, D, generator=g).to(dtype),
            torch.randn(B, Hkv, Sk, D, generator=g).to(dtype),
            torch.randn(B, Hkv, Sk, D, generator=g).to(dtype))


def _torch_sdpa(q, k, v, causal):
    rep = q.shape[1] // k.shape[1]
    k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
    mask = ~causal_mask(q.shape[2], k.shape[2], q.device) if causal else None
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)


@pytest.mark.parametrize("heads,seqs,D,causal,block", CASES)
def test_tiled_matches_reference_fp32(heads, seqs, D, causal, block):
    (B, Hq, Hkv), (Sq, Sk) = heads, seqs
    q, k, v = _rand(B, Hq, Hkv, Sq, Sk, D)
    ref = sdpa_reference(q, k, v, causal=causal)
    til = tiled_attention(q, k, v, causal=causal, block_q=block, block_k=block)
    tor = _torch_sdpa(q, k, v, causal)
    assert (til - ref).abs().max() < 1e-5
    assert (ref - tor).abs().max() < 1e-5


def test_scaling_is_one_over_sqrt_d():
    q, k, v = _rand(1, 1, 1, 4, 4, 16)
    s = (q @ k.transpose(-2, -1)) / math.sqrt(16)
    assert torch.allclose(sdpa_reference(q, k, v), torch.softmax(s, -1) @ v, atol=1e-6)


def test_causal_first_query_sees_only_first_key():
    q, k, v = _rand(1, 1, 1, 5, 5, 8)
    out = tiled_attention(q, k, v, causal=True, block_q=2, block_k=2)
    assert torch.allclose(out[0, 0, 0], v[0, 0, 0], atol=1e-6)


def test_stable_under_huge_logits():
    q, k, v = _rand(1, 2, 2, 33, 33, 32)
    q = q * 300  # logits ~1e3: exp overflows without max subtraction
    naive = naive_softmax_attention(q, k, v)
    assert not torch.isfinite(naive).all()
    ref = sdpa_reference(q, k, v)
    til = tiled_attention(q, k, v, block_q=8, block_k=8)
    assert torch.isfinite(til).all() and (til - ref).abs().max() < 1e-4


def test_logsumexp_matches():
    q, k, v = _rand(1, 2, 2, 20, 50, 16)
    _, lse = tiled_attention(q, k, v, causal=True, block_q=8, block_k=8, return_lse=True)
    s = (q @ k.transpose(-2, -1)) / 4.0
    s = s.masked_fill(causal_mask(20, 50, q.device), float("-inf"))
    assert torch.allclose(lse, torch.logsumexp(s, -1), atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for fp16")
@pytest.mark.parametrize("causal", [True, False])
def test_fp16_gpu(causal):
    q, k, v = (t.cuda() for t in _rand(2, 12, 2, 257, 257, 128, dtype=torch.float16))
    ref = sdpa_reference(q, k, v, causal=causal)
    til = tiled_attention(q, k, v, causal=causal)
    assert (til.float() - ref.float()).abs().max() < 2e-3

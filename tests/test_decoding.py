"""D1/D2 + F2/F3: token equality of cache-aware and speculative decoding.

CPU tests use tiny randomly-initialised Qwen2 models (same architecture family: RoPE, GQA,
RMSNorm) built from configs - two different seeds give a draft that disagrees with the target,
so rejection/correction paths are genuinely exercised. GPU tests use the real Qwen2.5 pair.
"""

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from kvcache.decoder import greedy_cached
from kvcache.reference import greedy_no_cache
from speculative.decoder import speculative_greedy


def tiny(seed: int, layers: int = 2):
    torch.manual_seed(seed)
    cfg = Qwen2Config(vocab_size=97, hidden_size=64, intermediate_size=128, num_hidden_layers=layers,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512)
    return Qwen2ForCausalLM(cfg).eval().to(torch.float64)  # fp64 -> no reduction-order ties on CPU


@pytest.fixture(scope="module")
def models():
    return tiny(0, 3), tiny(1, 1)


@pytest.mark.parametrize("n_in,n_out", [(1, 5), (8, 16), (40, 33), (120, 60)])
def test_cached_equals_nocache(models, n_in, n_out):
    target, _ = models
    ids = torch.randint(0, 97, (1, n_in), generator=torch.Generator().manual_seed(n_in))
    a = greedy_cached(target, ids, n_out)
    b = greedy_no_cache(target, ids, n_out)
    assert a.tokens == b.tokens and len(a.tokens) == n_out


def test_cache_growth_trace(models):
    target, _ = models
    ids = torch.randint(0, 97, (1, 10))
    r = greedy_cached(target, ids, 6, record=True)
    assert r.cache_lengths == [11, 12, 13, 14, 15]
    assert all(s[2] == L for s, L in zip(r.layer0_k_shapes, r.cache_lengths))
    assert r.layer0_k_shapes[0][1] == 2  # GQA: cache stores KV heads (2), not query heads (4)


@pytest.mark.parametrize("gamma", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("n_in,n_out", [(5, 7), (30, 40), (64, 3)])
def test_speculative_equals_greedy(models, gamma, n_in, n_out):
    target, draft = models
    ids = torch.randint(0, 97, (1, n_in), generator=torch.Generator().manual_seed(7 * n_in + gamma))
    ref = greedy_cached(target, ids, n_out)
    spec = speculative_greedy(target, draft, ids, gamma, n_out)
    assert spec.tokens == ref.tokens
    assert spec.accepted <= spec.proposed


def test_speculative_self_draft_accepts_everything(models):
    target, _ = models
    ids = torch.randint(0, 97, (1, 12))
    r = speculative_greedy(target, target, ids, 4, 21)
    assert r.acceptance_rate == 1.0
    assert r.tokens == greedy_cached(target, ids, 21).tokens


def test_speculative_stops_at_eos_inside_block(models):
    target, draft = models
    ids = torch.randint(0, 97, (1, 9))
    ref = greedy_cached(target, ids, 30).tokens
    eos = {ref[5]}
    want = greedy_cached(target, ids, 30, eos).tokens
    got = speculative_greedy(target, target, ids, 8, 30, eos).tokens
    assert got == want and got[-1] in eos


def _assert_equal_up_to_bf16_ties(model, ids, a, b, mode_a, mode_b):
    """Streams must be identical, or differ only where bf16 rounding explains it (see speculative/ties.py)."""
    from speculative.ties import compare

    r = compare(model, ids, a, b, mode_a, mode_b)
    assert r is None or r["explained_by_rounding"], f"divergence not explained by bf16 rounding: {r}"
    return r


def test_tie_check_accepts_exact_match_and_rejects_a_wrong_token(models):
    """Negative control: the tolerance must not be vacuous. In float64 (delta ~ 0) a different token is rejected."""
    from speculative.ties import compare

    target, _ = models
    ids = torch.randint(0, 97, (1, 12), generator=torch.Generator().manual_seed(3))
    ref = greedy_cached(target, ids, 10).tokens
    assert compare(target, ids, ref, list(ref), "incremental", "full") is None
    wrong = list(ref)
    wrong[4] = (wrong[4] + 1) % 97  # a token that is not the argmax
    r = compare(target, ids, ref, wrong, "incremental", "full")
    assert r is not None and r["first_diff"] == 4
    assert r["delta"] < 1e-9 and r["gap"] > 1e-3 and not r["explained_by_rounding"]


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_real_qwen_pair_equality():
    """Real Qwen2.5 pair in bf16 on the GPU: cached vs no-cache and speculative vs greedy agree up to bf16 ties."""
    from dataprep.formatting import read_jsonl
    from modeling.loader import load_model
    from modeling.prompts import render_prompt
    from settings.config import get_settings

    s = get_settings()
    t = load_model(s.target_model_id)
    d = load_model(s.draft_model_id)
    for ex in read_jsonl(s.data_dir / "processed" / "val.jsonl")[:3]:
        ids = t.tokenizer(render_prompt(t.tokenizer, ex["messages"][:-1]), return_tensors="pt").input_ids.cuda()
        ref = greedy_cached(t.model, ids, 48).tokens
        _assert_equal_up_to_bf16_ties(t.model, ids, ref, greedy_no_cache(t.model, ids, 48).tokens,
                                      "incremental", "full")
        spec = speculative_greedy(t.model, d.model, ids, 4, 48).tokens
        _assert_equal_up_to_bf16_ties(t.model, ids, ref, spec, "incremental", "block")


@pytest.mark.slow
def test_real_qwen_pair_exact_in_fp32_cpu():
    """Same real models in float32 on the CPU, NO tolerance: cached, no-cache and speculative decoding must be
    token-identical. This is what shows the bf16 differences above are rounding, not logic."""
    from dataprep.formatting import read_jsonl
    from modeling.loader import load_model
    from modeling.prompts import render_prompt
    from settings.config import get_settings

    s = get_settings()
    t = load_model(s.target_model_id, device="cpu", dtype="float32")
    d = load_model(s.draft_model_id, device="cpu", dtype="float32")
    for ex in read_jsonl(s.data_dir / "processed" / "val.jsonl")[:3]:
        ids = t.tokenizer(render_prompt(t.tokenizer, ex["messages"][:-1]), return_tensors="pt").input_ids
        ref = greedy_cached(t.model, ids, 48).tokens
        assert greedy_no_cache(t.model, ids, 48).tokens == ref
        assert speculative_greedy(t.model, d.model, ids, 4, 48).tokens == ref

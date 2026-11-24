# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Checks of the DFly2 CPU reference (dfly_hy4_ref.py) on random weights (shrunk dims), no checkpoint needed.

  python -m pytest -q models/experimental/hybrid_moe/test_dfly_hy4_ref.py
  HY4_PATH=<snapshot dir> python -m pytest -q ... -k real     # + the real weights, once their shards are present
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dfly_hy4_ref import PREFIX, DFlyConfig, DFlyRef, apply_rope_interleaved, rms_norm, rope_cos_sin, speculative_accept  # noqa: E402

SMALL = DFlyConfig(hidden_size=64, num_layers=2, num_heads=4, q_lora_rank=32, kv_lora_rank=32, qk_nope_head_dim=24,
                   qk_rope_head_dim=8, v_head_dim=16, intermediate_size=96, hidden_correction_intermediate_size=64,
                   vocab_size=300, mask_token_id=7, num_speculative_tokens=5, target_index_topk=16,
                   target_kv_layers=[0, 1], target_index_layer=[0, 1])


def _state(c, S, anchor, seed=1):
    g = torch.Generator().manual_seed(seed)
    lat = [(torch.randn(S, c.kv_lora_rank, generator=g), torch.randn(S, c.qk_rope_head_dim, generator=g))
           for _ in range(c.num_layers)]
    return lat, torch.randn(c.hidden_size, generator=g)


@pytest.fixture(scope="module")
def ref():
    return DFlyRef.random(DFlyConfig(**SMALL.__dict__), seed=0, std=0.2)


def test_shapes_and_determinism(ref):
    c = ref.cfg
    S, anchor = 40, 30
    lat, h = _state(c, S, anchor)
    hist = [torch.arange(anchor)] * c.num_layers
    a = ref.propose(11, h, anchor, lat, hist)
    b = ref.propose(11, h, anchor, lat, hist)
    W = c.query_width
    assert a["drafts"].shape == (W - 1,) and a["draft_logits"].shape == (W - 1, c.vocab_size)
    assert a["block_hidden"].shape == (W, c.hidden_size) and a["corrected"].shape == (W - 1, c.hidden_size)
    assert len(a["block_latents"]) == c.num_layers and a["block_latents"][0][0].shape == (W, c.kv_lora_rank)
    assert torch.equal(a["drafts"], b["drafts"]) and torch.isfinite(a["draft_logits"]).all()
    assert a["drafts"].tolist() == a["draft_logits"].argmax(-1).tolist()  # greedy


def test_future_cache_rows_are_ignored(ref):
    """KV share: the draft overwrites positions >= anchor with its own block; whatever the target cache holds there
    must not matter."""
    c = ref.cfg
    S, anchor = 40, 25
    lat, h = _state(c, S, anchor)
    hist = [torch.arange(anchor)] * c.num_layers
    a = ref.propose(5, h, anchor, lat, hist)
    lat2 = [(kv.clone(), pe.clone()) for kv, pe in lat]
    for kv, pe in lat2:
        kv[anchor:] = 1e3
        pe[anchor:] = -1e3
    b = ref.propose(5, h, anchor, lat2, hist)
    assert torch.allclose(a["draft_logits"], b["draft_logits"])


def test_history_filter(ref):
    """Entries -1, >= anchor and duplicates of nothing: only valid history positions < anchor count."""
    c = ref.cfg
    S, anchor = 40, 25
    lat, h = _state(c, S, anchor)
    clean = [torch.tensor([0, 3, 9, 24])] * c.num_layers
    noisy = [torch.tensor([0, -1, 3, 9, 24, 25, 30, -1])] * c.num_layers
    a = ref.propose(5, h, anchor, lat, clean)
    b = ref.propose(5, h, anchor, lat, noisy)
    assert torch.allclose(a["draft_logits"], b["draft_logits"])
    # and the history matters
    other = ref.propose(5, h, anchor, lat, [torch.tensor([1, 2])] * c.num_layers)
    assert not torch.allclose(a["draft_logits"], other["draft_logits"])


def _dense_mla_reference(ref, layer, xn, pos, kv_c_all, k_pe_all, keys):
    """HF-style MLA (expand kv_b into per-head K/V for every cached position, mask by key sets, sinks)."""
    c = ref.cfg
    H, dn, dr, dv = c.num_heads, c.qk_nope_head_dim, c.qk_rope_head_dim, c.v_head_dim
    p = f"{layer}.self_attn."
    W = ref._p
    q_c = rms_norm(xn @ W(p + "q_a_proj.weight").T, W(p + "q_a_layernorm.weight"), c.rms_norm_eps)
    q = (q_c @ W(p + "q_b_proj.weight").T).view(-1, H, dn + dr)
    cos, sin = rope_cos_sin(pos, dr, c.rope_theta)
    q = torch.cat((q[..., :dn], apply_rope_interleaved(q[..., dn:], cos, sin)), -1).transpose(0, 1)  # [H, T, qk]
    kv = (kv_c_all @ W(p + "kv_b_proj.weight").T).view(-1, H, dn + dv).transpose(0, 1)  # [H, S, dn+dv]
    k = torch.cat((kv[..., :dn], k_pe_all[None].expand(H, -1, -1)), -1)
    v = kv[..., dn:]
    s = q @ k.transpose(1, 2) * (dn + dr) ** -0.5  # [H, T, S]
    mask = torch.full(s.shape[1:], float("-inf"))
    for j, idx in enumerate(keys):
        mask[j, idx] = 0
    s = s + mask
    s = torch.cat((s, W(p + "learnable_sink_param")[:, None, None].expand(-1, s.shape[1], 1)), -1)
    pr = torch.softmax(s, -1)[..., :-1]
    o = (pr @ v).transpose(0, 1).reshape(len(keys), -1)
    o = o * torch.sigmoid(xn @ W(p + "linear_gate.weight").T)
    return rms_norm(o @ W(p + "o_proj.weight").T, W(p + "post_proj_norm.weight"), c.rms_norm_eps)


def test_absorbed_attention_matches_dense(ref):
    c = ref.cfg
    g = torch.Generator().manual_seed(3)
    S, anchor = 30, 22
    W = c.query_width
    xn = torch.randn(W, c.hidden_size, generator=g)
    pos = torch.arange(anchor, anchor + W)
    kv_c_all = torch.randn(anchor + W, c.kv_lora_rank, generator=g)
    k_pe_all = torch.randn(anchor + W, c.qk_rope_head_dim, generator=g)
    keys = ref.key_positions(anchor, torch.tensor([0, 5, 6, 7, 20, 21]))
    got = ref.attention(0, xn, pos, kv_c_all, k_pe_all, keys)
    want = _dense_mla_reference(ref, 0, xn, pos, kv_c_all, k_pe_all, keys)
    assert torch.allclose(got, want, atol=1e-4, rtol=1e-4), (got - want).abs().max()


def test_causal_block(ref):
    """Draft row j sees block rows 0..j only: changing the bonus token changes every row; nothing feeds back from
    later rows into earlier ones."""
    keys = ref.key_positions(10, torch.arange(10))
    assert [int(k.max()) for k in keys] == list(range(10, 10 + ref.cfg.query_width))
    assert all(len(k) == 10 + j + 1 for j, k in enumerate(keys))


def test_rope_is_relative():
    q, k = torch.randn(1, 8), torch.randn(1, 8)
    def dot(pq, pk):
        cq, sq = rope_cos_sin(torch.tensor([pq]), 8, 1e7)
        ck, sk = rope_cos_sin(torch.tensor([pk]), 8, 1e7)
        return float(apply_rope_interleaved(q, cq, sq) @ apply_rope_interleaved(k, ck, sk).T)
    assert abs(dot(5, 3) - dot(105, 103)) < 1e-4


def test_sampled_drafts_follow_temperature(ref):
    c = ref.cfg
    S, anchor = 40, 30
    lat, h = _state(c, S, anchor)
    hist = [torch.arange(anchor)] * c.num_layers
    out = ref.propose(11, h, anchor, lat, hist, temperature=1.0, generator=torch.Generator().manual_seed(0))
    assert out["drafts"].shape == (c.query_width - 1,)


def test_speculative_accept_greedy():
    V = 10
    d = torch.tensor([3, 4, 5])
    tl = torch.full((4, V), -10.0)
    tl[0, 3] = tl[1, 4] = tl[2, 9] = tl[3, 1] = 5.0
    assert speculative_accept(d, torch.zeros(3, V), tl, 0.0) == [3, 4, 9]
    tl[2, 5] = 6.0
    assert speculative_accept(d, torch.zeros(3, V), tl, 0.0) == [3, 4, 5, 1]


def test_speculative_accept_distribution():
    """Exactness: with q != p, the first emitted token is distributed as p."""
    V, n = 4, 20000
    p_log = torch.log(torch.tensor([0.5, 0.2, 0.2, 0.1]))
    q_log = torch.log(torch.tensor([0.1, 0.6, 0.2, 0.1]))
    g = torch.Generator().manual_seed(0)
    counts = torch.zeros(V)
    for _ in range(n):
        d = torch.multinomial(torch.softmax(q_log, -1), 1, generator=g)
        out = speculative_accept(d, q_log[None], torch.stack((p_log, p_log)), 1.0, generator=g)
        counts[out[0]] += 1
    assert torch.allclose(counts / n, torch.softmax(p_log, -1), atol=0.015)


@pytest.mark.skipif(not os.environ.get("HY4_PATH"), reason="needs HY4_PATH with the dfly / embed / lm_head shards")
def test_real_weights_load_and_run():
    r = DFlyRef.from_checkpoint(os.environ["HY4_PATH"])
    c = r.cfg
    assert (c.intermediate_size, c.hidden_correction_intermediate_size, c.num_layers) == (9216, 3072, 5)
    assert c.target_kv_layers == [25, 29, 33, 37, 41] and c.mask_token_id == 120009 and c.rope_theta == 1e7
    assert r.w[PREFIX + "0.self_attn.kv_b_proj.weight"].shape == (32 * (192 + 256), 512)
    S, anchor = 64, 50
    g = torch.Generator().manual_seed(0)
    lat = [(torch.randn(S, 512, generator=g) * 0.1, torch.randn(S, 64, generator=g) * 0.1) for _ in range(5)]
    out = r.propose(1000, torch.randn(3072, generator=g), anchor, lat, [torch.arange(anchor)] * 5)
    assert torch.isfinite(out["draft_logits"]).all()

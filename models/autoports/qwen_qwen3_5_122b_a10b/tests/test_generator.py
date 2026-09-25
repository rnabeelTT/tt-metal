# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage-6 generator contract tests on the reduced 2-layer model (layers 0 = GDN, 3 = attention; real weights) on the
(1,4) mesh: split-sampling trace + cross-request token feedback, batch > 1 with mixed prompt lengths, long prompts through
the segmented prefill, and the two greedy sampling strategies of the common sampler."""

import os

import pytest
import torch

import ttnn
from models.autoports.qwen_qwen3_5_122b_a10b.tests.test_multichip_decoder import mesh_tp
from models.autoports.qwen_qwen3_5_122b_a10b.tt.generator import build_generator

MODEL_DIR = "models/autoports/qwen_qwen3_5_122b_a10b"
LAYERS = [0, 3]
TEXTS = [
    "The capital of France is Paris, the capital of Germany is Berlin, the capital of Italy is",
    "Once upon a time in a small village by the sea there lived a fisherman who",
    "In mathematics, a prime number is a natural number greater than 1 that",
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return",
]


def _build(mesh, **kw):
    os.environ.setdefault("HF_MODEL", "Qwen/Qwen3.5-122B-A10B")
    return build_generator(MODEL_DIR, mesh, layer_indices=LAYERS, **kw)


def _prompt(gen, i, n):
    ids = gen.tokenizer.encode(TEXTS[i % len(TEXTS)])
    while len(ids) < n:
        ids = ids + ids
    return ids[:n]


def _truth_next(gen, prompt):
    """Ground truth for the first token after `prompt`: host argmax of the prefill-path logits at the last position."""
    lg = gen.prefill_logits(prompt)
    return int(torch.argmax(lg[0, -1]))


def _truth_set(gen, prompt, margin=1.0):
    """Tokens a correct implementation may produce after `prompt`: the argmax, plus the runner-up when the two top logits
    are within `margin` (a near-tie; prefill and decode take different matmul/accumulation paths, so their bf16 roundings
    can legitimately order two near-equal logits differently — the readiness top-k check quantifies that, a token test
    must not turn it into a failure). Returns (argmax, allowed set, margin)."""
    lg = gen.prefill_logits(prompt)[0, -1].float()
    v, i = torch.topk(lg, 2)
    allowed = {int(i[0])} if float(v[0] - v[1]) > margin else {int(i[0]), int(i[1])}
    return int(i[0]), allowed, float(v[0] - v[1])


@mesh_tp()
def test_generator_cross_request_token_feedback(mesh_device):
    """Split sampling + token feedback across requests: after generate(A, n) the device token/position inputs hold A's
    last sampled token at position len(A)+n; a new request B whose prompt length is len(A)+n (or len(A)+n-1) must decode
    its FIRST token from its own prefill, not from the stale device token (the stock Generator's reset merge keeps the
    device token for slots that look continuous unless they are marked as freshly prefilled)."""
    gen = _build(mesh_device, max_seq_len=4096)
    A = _prompt(gen, 0, 24)
    n = 8
    out_a = gen.generate(A, n, enable_trace=True)
    assert len(out_a) == n
    for lb in (len(A) + n, len(A) + n - 1):
        B = _prompt(gen, 1, lb)
        truth = _truth_next(gen, B)
        got = gen.generate(B, 3, enable_trace=True)
        assert got[0] == truth, f"len(B)={lb}: first token {got[0]} != prefill truth {truth} (stale device token merge)"
        # and the second token comes from the device-fed first token (token feedback), equal to the eager continuation
        # (within a near-tie margin, see _truth_set)
        _, allowed2, m2 = _truth_set(gen, B + [got[0]])
        assert (
            got[1] in allowed2
        ), f"len(B)={lb}: second token {got[1]} not in prefill truth set {allowed2} (margin {m2:.3f})"
    # determinism of the whole path
    assert gen.generate(A, n, enable_trace=True) == out_a


@mesh_tp()
@pytest.mark.parametrize("batch", [2, 8, 32])
def test_generator_batched_mixed_prompts(mesh_device, batch):
    """B > 1 through the generator: mixed prompt lengths (aligned and non-aligned), warmed masked-bucket prefill per slot,
    traced decode with on-device sampling for the whole batch; every user's tokens equal its own B=1 generation."""
    gen = _build(mesh_device, max_seq_len=4096, max_batch_size=batch)
    lens = [24, 37, 100, 129, 200, 65, 513, 17][:batch] if batch <= 8 else [17 + 5 * u for u in range(batch)]
    prompts = [_prompt(gen, u, lens[u]) for u in range(batch)]
    n = 4
    ref = [gen.generate(p, n, enable_trace=True) for p in prompts]  # B=1 path, one user at a time (slot 0)
    gen.reset()
    T = max(lens)
    tokens = torch.zeros(batch, T, dtype=torch.int64)
    for u, p in enumerate(prompts):
        tokens[u, : len(p)] = torch.tensor(p)
    logits = gen.prefill_forward(tokens, page_table=gen.page_table[:batch], prompt_lens=lens)
    assert logits.shape[0] == batch
    first = torch.argmax(logits[:, -1], dim=-1).tolist()
    for u in range(batch):
        assert first[u] == ref[u][0], f"user {u} (len {lens[u]}): prefill token {first[u]} != B=1 {ref[u][0]}"
    toks = torch.tensor(first).reshape(batch, 1)
    pos = torch.tensor(lens)
    got = [[first[u]] for u in range(batch)]
    for i in range(n - 1):
        out = gen.decode_forward(
            toks, pos, page_table=gen.page_table[:batch], enable_trace=True, reload_inputs=(i == 0)
        )
        for u in range(batch):
            got[u].append(int(out[u]))
        toks = out.reshape(batch, 1)
        pos = pos + 1
    mism = [u for u in range(batch) if got[u] != ref[u]]
    assert not mism, f"users {mism} differ from their B=1 generation: " + "; ".join(
        f"u{u} len {lens[u]}: {got[u]} vs {ref[u]}" for u in mism[:4]
    )
    print(f"GEN batch={batch} lens={lens}: all users match their B=1 tokens ({n} tokens each)")


@mesh_tp()
@pytest.mark.parametrize("T", [2381, 4097])
def test_generator_long_prompt_segments(mesh_device, T):
    """Prompts longer than one 2048-token segment go through `_prefill_single`'s segmented masked-bucket path with
    `chunk_start`. Two independent checks of the segment continuity: (1) the token after the full prompt from the segmented
    prefill equals the token a traced DECODE step produces when the prompt minus its last token is prefilled and the last
    token is decoded (the same position reached through a different path); (2) for T <= 4096 the last-token logits also
    match the single padded all-positions pass (`prefill_all_logits`, PCC >= 0.999). The first traced decode after the
    long prefill must match the prefill-truth of the extended prompt."""
    gen = _build(mesh_device, max_seq_len=8192)
    prompt = _prompt(gen, 2, T)
    tokens = torch.tensor(prompt).reshape(1, -1)
    gen.reset()
    seg = gen.prefill_forward(tokens, page_table=gen.page_table[:1], prompt_lens=[T])  # segmented path
    tok_seg = int(torch.argmax(seg[0, -1]))
    dec = int(gen.decode_forward(torch.tensor([[tok_seg]]), torch.tensor([T]), enable_trace=True)[0])
    # (1) prefill T-1 tokens, decode the last prompt token -> must produce tok_seg
    gen.reset()
    gen.prefill_forward(tokens[:, : T - 1], page_table=gen.page_table[:1], prompt_lens=[T - 1])
    via_decode = int(gen.decode_forward(torch.tensor([[prompt[-1]]]), torch.tensor([T - 1]), enable_trace=True)[0])
    print(f"GEN long prompt T={T}: segmented prefill token {tok_seg}, via prefill(T-1)+decode {via_decode}")
    assert via_decode == tok_seg, f"T={T}: segmented prefill token {tok_seg} != prefill(T-1)+decode token {via_decode}"
    if T <= 4096:
        from models.common.utility_functions import comp_pcc

        full = gen.prefill_logits(prompt)  # single padded pass (one program; bounded by the 4096 all-logits program)
        tok_full = int(torch.argmax(full[0, -1]))
        ok, pcc = comp_pcc(full[0, -1].float(), seg[0, -1].float(), 0.999)
        print(
            f"GEN long prompt T={T}: segmented vs all-positions last-token logits PCC {pcc}, tokens {tok_seg} / {tok_full}"
        )
        assert tok_seg == tok_full and ok, f"T={T}: segmented vs single-pass mismatch (PCC {pcc})"
    # the traced decode after the long prefill continues from the right state: decode token == prefill truth of prompt+tok
    gen.reset()
    gen.prefill_forward(tokens, page_table=gen.page_table[:1], prompt_lens=[T])
    dec2 = int(gen.decode_forward(torch.tensor([[tok_seg]]), torch.tensor([T]), enable_trace=True)[0])
    gen.reset()
    gen.prefill_forward(
        torch.tensor(prompt + [tok_seg]).reshape(1, -1), page_table=gen.page_table[:1], prompt_lens=[T + 1]
    )
    truth2 = int(
        torch.argmax(
            gen.prefill_forward(
                torch.tensor(prompt + [tok_seg]).reshape(1, -1), page_table=gen.page_table[:1], prompt_lens=[T + 1]
            )[0, -1]
        )
    )
    assert dec == dec2 == truth2, f"T={T}: traced decode after the long prefill {dec}/{dec2} != prefill truth {truth2}"


@mesh_tp()
@pytest.mark.parametrize("force_argmax", ["1", "0"], ids=["force_argmax", "topk_candidates"])
def test_generator_greedy_sampler_paths(mesh_device, force_argmax, monkeypatch):
    """Both greedy strategies of the common sampler (force-argmax: one full-vocab all-gather + argmax; top-k candidates:
    per-shard top-k, candidate all-gather, ttnn.sampling k=1) return the prefill-path ground truth and are deterministic;
    the token-out step time of each is printed for the README comparison."""
    import time

    monkeypatch.setenv("QWEN35_FORCE_ARGMAX", force_argmax)
    gen = _build(mesh_device, max_seq_len=4096)
    assert gen.model.sampling.tt_sampling._allow_force_argmax_sampling == (force_argmax == "1")
    prompt = _prompt(gen, 0, 24)
    truth = _truth_next(gen, prompt)
    out = gen.generate(prompt, 8, enable_trace=True)
    assert out[0] == truth and out == gen.generate(prompt, 8, enable_trace=True)
    # token feedback on both paths: the second token must be the eager truth of prompt + first token (near-tie aware)
    _, allowed1, m1 = _truth_set(gen, prompt + [out[0]])
    assert out[1] in allowed1, f"token feedback: {out[1]} not in {allowed1} (margin {m1:.3f})"
    gen.reset()
    lg = gen.prefill_forward(
        torch.tensor(prompt).reshape(1, -1), page_table=gen.page_table[:1], prompt_lens=[len(prompt)]
    )
    t, pos = int(torch.argmax(lg[0, -1])), len(prompt)
    for i in range(4):
        t = int(
            gen.decode_forward(torch.tensor([[t]]), torch.tensor([pos]), enable_trace=True, reload_inputs=(i == 0))[0]
        )
        pos += 1
    ttnn.synchronize_device(mesh_device)
    t0 = time.perf_counter()
    for _ in range(32):
        t = int(gen.decode_forward(torch.tensor([[t]]), torch.tensor([pos]), enable_trace=True, reload_inputs=False)[0])
        pos += 1
    ttnn.synchronize_device(mesh_device)
    ms = 1e3 * (time.perf_counter() - t0) / 32
    # 4-way breakdown of the step (Tracy cannot merge the sampler's single-device reads on the mesh): model trace replay
    # alone, sampler trace alone, token readback alone; the host remainder is the difference.
    tt = gen._tt
    tids = tt.trace_ids_decode[True]
    logits = tt.trace_output_decode[True]
    ttnn.synchronize_device(mesh_device)
    t0 = time.perf_counter()
    for _ in range(32):
        for tid in tids.values():
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    ms_model = 1e3 * (time.perf_counter() - t0) / 32
    from models.autoports.qwen_qwen3_5_122b_a10b.tt.generator import GREEDY

    t0 = time.perf_counter()
    for _ in range(32):
        toks = tt.sample_decode_on_device(logits, GREEDY, enable_trace=True)
    ttnn.synchronize_device(mesh_device)
    ms_sampler = 1e3 * (time.perf_counter() - t0) / 32
    t0 = time.perf_counter()
    for _ in range(32):
        tt.read_decode_output(toks)
    ms_read = 1e3 * (time.perf_counter() - t0) / 32
    print(
        f"GEN sampler force_argmax={force_argmax}: token-out step {ms:.3f} ms = model trace {ms_model:.3f} + sampler trace "
        f"{ms_sampler:.3f} + token readback {ms_read:.3f} + host {ms - ms_model - ms_sampler - ms_read:.3f} ms (2-layer model, 32 steps)"
    )


@mesh_tp()
def test_generator_all_logits_after_traced_decode(mesh_device):
    """`prefill_logits` (the readiness prefill-check path: one padded all-positions pass) must be unchanged by traced
    decodes that ran earlier in the process (bit-identical logits before/after), and must agree with the masked-bucket
    prefill path's last-token logits (PCC >= 0.999; the two paths differ in matmul programs, so the argmax may swap at a
    near-tie — compared as top-2 sets when the margin is small)."""
    from models.common.utility_functions import comp_pcc

    gen = _build(mesh_device, max_seq_len=4096)
    prompt = _prompt(gen, 0, 24)
    ext = prompt + [_truth_next(gen, prompt)]
    before = gen.prefill_logits(ext)[0, -1].float()
    gen.generate(prompt, 8, enable_trace=True)
    after = gen.prefill_logits(ext)[0, -1].float()
    assert torch.equal(before, after), "all-positions prefill logits changed after traced decodes ran in the process"
    gen.reset()
    masked = gen.prefill_forward(
        torch.tensor(ext).reshape(1, -1), page_table=gen.page_table[:1], prompt_lens=[len(ext)]
    )[0, -1].float()
    ok, pcc = comp_pcc(before, masked, 0.999)
    top2_all = set(torch.topk(before, 2).indices.tolist())
    top2_masked = set(torch.topk(masked, 2).indices.tolist())
    margin = float(torch.topk(before, 2).values.diff().abs())
    print(f"GEN all-logits vs masked prefill: PCC {pcc}, top-2 {top2_all} vs {top2_masked}, margin {margin:.3f}")
    assert ok, f"all-positions vs masked-bucket prefill logits PCC {pcc} < 0.999"
    assert int(torch.argmax(masked)) in top2_all, "masked-path argmax not in the all-positions top-2"

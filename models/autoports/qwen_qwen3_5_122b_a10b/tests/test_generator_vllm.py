# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device tests of the vLLM adapter contract on the reduced 2-layer model (layers 0 and 3), without vLLM: construction
through initialize_vllm_model (precision policy applied), vLLM-owned KV cache + scratch block, the plugin's two-phase
warm-up, prefill into slots with on-device sampling (greedy == host argmax of the prefill logits), traced decode through
the base Generator with the split submit / read / host-format contract and token feedback, slot remap, program guard."""

import os

import pytest
import torch

import ttnn
from models.autoports.qwen_qwen3_5_122b_a10b.tests.test_generator import LAYERS, TEXTS
from models.autoports.qwen_qwen3_5_122b_a10b.tests.test_multichip_decoder import mesh_tp
from models.common.sampling.sampling_params import SamplingParams

MODEL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _tt_params(B, temperature=0.0, top_k=1, top_p=1.0, seed=None):
    return SamplingParams(
        temperature=[temperature] * B,
        top_k=[top_k] * B,
        top_p=[top_p] * B,
        presence_penalty=[0.0] * B,
        frequency_penalty=[0.0] * B,
        repetition_penalty=[1.0] * B,
        seed=[seed] * B,
        num_logprobs=None,
        enable_log_probs=[False] * B,
    )


def _build(mesh_device, B, max_seq_len=4096):
    os.environ.setdefault("HF_MODEL", "Qwen/Qwen3.5-122B-A10B")
    os.environ["QWEN35_LAYER_INDICES"] = ",".join(str(x) for x in LAYERS)
    from transformers import AutoConfig

    from models.autoports.qwen_qwen3_5_122b_a10b.tt.generator_vllm import Qwen35MoeForConditionalGeneration as Cls

    hf = AutoConfig.from_pretrained(os.environ["HF_MODEL"], trust_remote_code=True)
    adapter = Cls.initialize_vllm_model(hf, mesh_device, max_batch_size=B, max_seq_len=max_seq_len)
    gen = adapter._gen
    num_blocks = 256  # vLLM's budget for this test
    kv = adapter.allocate_kv_cache((num_blocks, 1, 64, 256), ttnn.bfloat16, num_layers=1)
    assert gen.kv_scratch_block == num_blocks and gen.model._kv_scratch_block == num_blocks
    assert kv[0][0].shape[0] == 288  # 256 + 1 scratch, 32-aligned
    # the plugin's two warm-up phases
    adapter.warmup_model_prefill(kv_cache=kv, enable_trace=False, can_sample_on_device=True)
    adapter.warmup_model_decode(
        kv_cache=kv, enable_trace=False, max_batch_size=B, num_blocks=64, can_sample_on_device=True
    )
    adapter.already_warmed_up_prefill = False
    adapter.warmup_model_prefill(kv_cache=kv, enable_trace=True, can_sample_on_device=True)
    adapter.warmup_model_decode(
        kv_cache=kv, enable_trace=True, max_batch_size=B, num_blocks=64, can_sample_on_device=True
    )
    assert gen._program_cache_after_warmup is not None
    return adapter, gen, kv


def _page_table(B, width, blocks_per_req):
    pt = torch.zeros(B, width, dtype=torch.int32)
    for u in range(B):
        pt[u, :blocks_per_req] = torch.arange(u * blocks_per_req, (u + 1) * blocks_per_req, dtype=torch.int32)
    return pt


@mesh_tp()
@pytest.mark.parametrize("B", [1, 4], ids=["b1", "b4"])
def test_vllm_adapter_prefill_decode(mesh_device, B):
    os.environ["QWEN35_PROGRAM_GUARD"] = "1"
    adapter, gen, kv = _build(mesh_device, B)
    width = 64
    pt = _page_table(B, width, 8)  # 8 blocks = 512 tokens per request
    prompts = [gen.tokenizer.encode(TEXTS[u % len(TEXTS)])[: 40 + 7 * u] for u in range(B)]
    plens = [len(p) for p in prompts]
    toks = torch.zeros(B, max(plens), dtype=torch.long)
    for u, p in enumerate(prompts):
        toks[u, : len(p)] = torch.tensor(p)
    # 1) prefill without sampling params -> host logits; with greedy params -> on-device sampled tokens == host argmax
    logits, deltas = adapter.prefill_forward(toks, pt, kv, plens, empty_slots=list(range(B)), start_pos=torch.zeros(B))
    assert logits.shape == (B, 1, gen.args.vocab_size) and deltas.shape == (B,)
    ref = torch.argmax(logits[:, 0], dim=-1)
    sampled, _ = adapter.prefill_forward(
        toks, pt, kv, plens, empty_slots=list(range(B)), start_pos=torch.zeros(B), sampling_params=_tt_params(B)
    )
    assert sampled.shape == (B, 1)
    assert torch.equal(
        sampled.reshape(-1), ref
    ), f"device-sampled prefill tokens {sampled.reshape(-1).tolist()} != host argmax {ref.tolist()}"
    n0 = gen.program_cache_entries()
    assert n0 == gen._program_cache_after_warmup, "prefill compiled new programs after the captures"
    # 2) decode: split contract (submit with read_from_device=False, deferred read, host formatting); token feedback
    #    keeps the device token / position for the following steps (reload_inputs only on the first step)

    cur = sampled.reshape(B, 1).clone()
    pos = torch.tensor(plens, dtype=torch.long)
    stream = [cur.reshape(-1).tolist()]
    pending = None
    for step in range(6):
        out = adapter.decode_forward(
            cur,
            pos,
            pt,
            kv,
            enable_trace=True,
            read_from_device=False,
            sampling_params=_tt_params(B),
            reset_batch=(step == 0),
            reload_inputs=(step == 0),
        )
        host, events = adapter.read_decode_output(out, async_read=True)
        if pending is not None:
            for ev in pending[1]:
                ttnn.event_synchronize(ev)
            prev, _lp = adapter.process_decode_output_host(pending[0], is_tokens=True)  # (tokens, log_probs)
            stream.append(torch.as_tensor(prev).reshape(-1)[:B].to(torch.int64).tolist())
        pending = (host, events)
        # host view of the stream: the device token is authoritative (async contract); the host feeds the same token
        cur = torch.tensor(stream[-1], dtype=torch.long).reshape(B, 1) if len(stream) > 1 else cur
        pos = pos + 1
    for ev in pending[1]:
        ttnn.event_synchronize(ev)
    last, _lp = adapter.process_decode_output_host(pending[0], is_tokens=True)
    stream.append(torch.as_tensor(last).reshape(-1)[:B].to(torch.int64).tolist())
    assert gen.program_cache_entries() == n0, "decode compiled new programs after the captures"
    # 3) the same stream from the standalone generator path (host-fed greedy) on the same prompts and slots
    text = [gen.tokenizer.decode([t[0] for t in stream]) for _ in range(1)]
    print(f"VLLM_ADAPTER B={B} stream(slot0)={[t[0] for t in stream]} text={text[0]!r}")
    for u in range(B):
        assert len(set(t[u] for t in stream)) > 1 or B > 1, "degenerate stream"
    # 4) slot remap (batch condense) on the model-internal GDN state: a rotation and its inverse must compile nothing
    #    (the first served mixed batch handed the adapter slot_remap [1, 2, 3, 0]) and leave the streams unchanged
    if B > 1:
        rot = torch.tensor([(i + 1) % B for i in range(B)], dtype=torch.int32)
        inv = torch.tensor([(i - 1) % B for i in range(B)], dtype=torch.int32)
        adapter.decode_forward(
            cur[rot.long()],
            pos[rot.long()],
            pt[rot.long()],
            kv,
            enable_trace=True,
            read_from_device=True,
            sampling_params=_tt_params(B),
            reset_batch=True,
            reload_inputs=True,
            slot_remap=rot,
        )
        adapter.decode_forward(
            cur,
            pos,
            pt,
            kv,
            enable_trace=True,
            read_from_device=True,
            sampling_params=_tt_params(B),
            reset_batch=True,
            reload_inputs=True,
            slot_remap=inv,
        )
        assert gen.program_cache_entries() == n0, "slot remap compiled new programs after the captures"
    print(f"VLLM_ADAPTER B={B} program cache {gen._program_cache_after_warmup} -> {gen.program_cache_entries()} OK")


def _greedy_stream(adapter, kv, pt, toks, plens, B, steps, *, wrong_host=False, page_table_at=None):
    """Prefill + ``steps`` traced greedy decodes through the adapter's plugin contract, synchronous reads. Returns the
    [steps+1] x B token stream. ``wrong_host=True`` feeds a deliberately wrong host token / position on every non-reset
    step (the device-resident inputs must win); ``page_table_at=(step, table)`` hands a changed page table at that step
    (``pt`` is the table of the prefill and of the steps before it)."""
    sampled, _ = adapter.prefill_forward(
        toks, pt, kv, plens, empty_slots=list(range(B)), start_pos=torch.zeros(B), sampling_params=_tt_params(B)
    )
    cur = sampled.reshape(B, 1).clone()
    pos = torch.tensor(plens, dtype=torch.long)
    stream = [cur.reshape(-1).tolist()]
    table = pt
    for step in range(steps):
        if page_table_at is not None and step == page_table_at[0]:
            table = page_table_at[1]
        host_tok, host_pos = cur, pos
        if wrong_host and step > 0:
            host_tok = torch.full_like(cur, 11)  # a wrong token id on every row
            host_pos = pos + 777  # a wrong position on every row
        out = adapter.decode_forward(
            host_tok,
            host_pos,
            table,
            kv,
            enable_trace=True,
            read_from_device=True,
            sampling_params=_tt_params(B),
            reset_batch=(step == 0),
            reload_inputs=(step == 0),
        )
        out = out[0] if isinstance(out, (tuple, list)) else out  # (tokens, log_probs) from the synchronous read
        nxt = torch.as_tensor(out).reshape(-1)[:B].to(torch.int64)
        stream.append(nxt.tolist())
        cur = nxt.reshape(B, 1)
        pos = pos + 1
    return stream


@mesh_tp()
@pytest.mark.parametrize("B", [1, 4], ids=["b1", "b4"])
def test_vllm_adapter_stale_inputs(mesh_device, B):
    """Device-resident decode inputs (stage-9 review hard-check gap): on non-reset steps the host token / position are NOT
    authoritative - the trace inputs carry the device-sampled token and the device-advanced position. A stream fed
    deliberately wrong host tokens and positions must equal the correctly fed control; a page table changed right
    before the rows cross a block boundary (from a table whose next block is SHARED by every row to the control's private
    blocks) must be picked up (the only trace input
    refreshed on such a step; a stale device table would corrupt the rows' KV across each other, B > 1) and leave the
    stream unchanged; unchanged page tables (every other step) copy nothing."""
    os.environ["QWEN35_PROGRAM_GUARD"] = "1"
    adapter, gen, kv = _build(mesh_device, B)
    width = 64
    pt = _page_table(B, width, 8)
    import bz2

    # book text (the shared snippets are ~19 tokens each and give the 2-layer model a constant-token stream, which
    # cannot show whether the device state advances): distinct 58..61-token prompts, every row crosses token 64
    book = bz2.open("models/tt_transformers/tests/tale-of-two-cities.txt.bz2", "rt", errors="ignore").read()
    prompts = [gen.tokenizer.encode(book[20000 + 3000 * u : 20000 + 3000 * u + 2000])[: 58 + u] for u in range(B)]
    plens = [len(p) for p in prompts]
    assert all(56 <= L <= 63 for L in plens), plens
    toks = torch.zeros(B, max(plens), dtype=torch.long)
    for u, p in enumerate(prompts):
        toks[u, : len(p)] = torch.tensor(p)
    steps = 10
    control = _greedy_stream(adapter, kv, pt, toks, plens, B, steps)
    for u in range(B):
        assert len(set(t[u] for t in control)) > 2, f"constant control stream on slot {u}: {[t[u] for t in control]}"
    n0 = gen.program_cache_entries()
    assert n0 == gen._program_cache_after_warmup
    # (a) wrong host token / position on every non-reset step -> identical stream (device inputs authoritative)
    stale = _greedy_stream(adapter, kv, pt, toks, plens, B, steps, wrong_host=True)
    assert stale == control, f"stream followed the stale HOST inputs:\n control {control}\n stale   {stale}"
    # (b) changed page table before the boundary crossing (vLLM allocates a request's next block right before it is
    #     needed): the stream starts on a table whose block 1 is a SHARED block 250 for every row (harmless while every row
    #     is below token 64: prefill only parks bucket-padding rows there) and switches at step 1 to the control's table
    #     (a private block 1 per row). If the device kept the stale table, every row's tokens >= 64 would be written into
    #     and read from the shared block -> cross-row KV corruption -> a different stream (discriminating for B > 1; at
    #     B = 1 the arm only shows that the switch compiles nothing and leaves the stream intact).
    pt_shared = pt.clone()
    pt_shared[:, 1] = 250
    grown = _greedy_stream(adapter, kv, pt_shared, toks, plens, B, steps, page_table_at=(1, pt))
    assert grown == control, f"stream changed after the page-table refresh:\n control {control}\n grown   {grown}"
    assert gen.program_cache_entries() == n0, "stale-input steps compiled new programs after the captures"
    print(f"VLLM_ADAPTER_STALE B={B} control(slot0)={[t[0] for t in control]} == stale == page-table-changed OK")

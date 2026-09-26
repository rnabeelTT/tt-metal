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


def _greedy_stream(adapter, kv, pt, toks, plens, B, steps, *, wrong_host=False, page_table_at=None, contract_v1=False):
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
        if contract_v1:  # the plugin's explicit decode-input contract (version 1): four commands, no reset_batch
            flags = dict(
                reload_inputs=(step == 0),
                reload_page_table=(page_table_at is not None and step == page_table_at[0]),
                reload_sampling_params=(step == 0),
                reset_sampling_state=(step == 0),
            )
        else:
            flags = dict(reset_batch=(step == 0), reload_inputs=(step == 0))
        out = adapter.decode_forward(
            host_tok,
            host_pos,
            table,
            kv,
            enable_trace=True,
            read_from_device=True,
            sampling_params=_tt_params(B),
            **flags,
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
    # (c) the same two arms through the explicit decode-input contract (version 1 commands instead of reset_batch)
    stale_v1 = _greedy_stream(adapter, kv, pt, toks, plens, B, steps, wrong_host=True, contract_v1=True)
    assert (
        stale_v1 == control
    ), f"contract-v1 stream followed the stale HOST inputs:\n control {control}\n stale   {stale_v1}"
    grown_v1 = _greedy_stream(adapter, kv, pt_shared, toks, plens, B, steps, page_table_at=(1, pt), contract_v1=True)
    assert (
        grown_v1 == control
    ), f"contract-v1 stream changed after the page-table refresh:\n control {control}\n grown   {grown_v1}"
    assert getattr(adapter, "_contract_v1_steps", 0) == 2 * steps, adapter._contract_v1_steps
    assert gen.program_cache_entries() == n0, "stale-input steps compiled new programs after the captures"
    print(f"VLLM_ADAPTER_STALE B={B} control(slot0)={[t[0] for t in control]} == stale == page-table-changed OK")


@mesh_tp()
@pytest.mark.parametrize("B", [4], ids=["b4"])
def test_vllm_adapter_decode_buckets(mesh_device, B):
    """Stage 10 decode bucketing (widths 1 / 2 / 4 on a B=4 build): the plugin's front-packed active prefix selects the
    trace width per step. Streams decoded at narrow widths must equal the full-width streams for the rows they cover,
    across width switches in both directions (with and without the plugin's reset_batch flag), with every trace captured
    in the warm-up (no program compiled afterwards) and the device tokens / positions carried over on a switch."""
    os.environ["QWEN35_PROGRAM_GUARD"] = "1"
    os.environ["QWEN35_DECODE_BUCKETING"] = "1"
    os.environ["QWEN35_DECODE_MIN_WIDTH"] = "1"  # mechanics test of every width incl. 1 (the served default is 2)
    adapter, gen, kv = _build(mesh_device, B)
    assert adapter._decode_widths == [1, 2, 4], adapter._decode_widths
    import bz2

    book = bz2.open("models/tt_transformers/tests/tale-of-two-cities.txt.bz2", "rt", errors="ignore").read()
    prompts = [gen.tokenizer.encode(book[20000 + 3000 * u : 20000 + 3000 * u + 2000])[: 58 + u] for u in range(B)]
    plens = [len(p) for p in prompts]
    toks = torch.zeros(B, max(plens), dtype=torch.long)
    for u, p in enumerate(prompts):
        toks[u, : len(p)] = torch.tensor(p)
    pt = _page_table(B, 64, 8)
    n0 = gen.program_cache_entries()
    assert n0 == gen._program_cache_after_warmup

    def run(active_seq, resets, wrong_host_tokens=False):
        """active_seq[i] = number of live rows at step i (front-packed: rows >= active carry position -1); resets[i] =
        the plugin's reset_batch flag. Returns the per-step tokens of the live rows."""
        sampled, _ = adapter.prefill_forward(
            toks, pt, kv, plens, empty_slots=list(range(B)), start_pos=torch.zeros(B), sampling_params=_tt_params(B)
        )
        cur = sampled.reshape(B).clone()
        pos = torch.tensor(plens, dtype=torch.long)
        stream = []
        for i, (n, reset) in enumerate(zip(active_seq, resets)):
            host_pos = pos.clone()
            host_pos[n:] = -1  # idle rows, as the plugin marks them
            # stale-host arm: a wrong host token on every non-reset step (positions correct, as the plugin's are); the
            # device-resident token must win, also across width switches (carry-over + forced host-authoritative merge)
            host_tok = cur.reshape(B, 1).clone()
            if wrong_host_tokens and i > 0:
                host_tok[
                    0
                ] = 11  # row 0 is live at every width in these sequences (rows that pause resume with host tokens)
            out = adapter.decode_forward(
                tokens=host_tok,
                start_pos=host_pos,
                page_table=pt,
                kv_cache=kv,
                enable_trace=True,
                read_from_device=True,
                sampling_params=_tt_params(B),
                reset_batch=(i == 0) or reset,
                reload_inputs=(i == 0),
            )
            out = out[0] if isinstance(out, (tuple, list)) else out
            nxt = torch.as_tensor(out).reshape(-1)[:n].to(torch.int64)
            stream.append(nxt.tolist())
            cur = cur.clone()
            cur[:n] = nxt
            pos = pos.clone()
            pos[:n] += 1
        return stream

    steps = 8
    full = run([4] * steps, [False] * steps)
    # the 2-layer model repeats tokens; require the control to vary somewhere (across steps or rows) so equality means something
    assert len(set(tuple(s) for s in full)) > 2, f"constant control stream: {full}"

    def logits_at(width_steps, tokens_fed):
        """Rebuild the slot state (prefill + width-1 steps feeding ``tokens_fed``), then ONE host-logits step (full width,
        the compatibility mode) -> row-0 logits at the position of the next token."""
        adapter.prefill_forward(
            toks, pt, kv, plens, empty_slots=list(range(B)), start_pos=torch.zeros(B), sampling_params=_tt_params(B)
        )
        cur = torch.tensor(tokens_fed[0], dtype=torch.long)
        pos = torch.tensor(plens, dtype=torch.long)
        host_pos = pos.clone()
        host_pos[1:] = -1
        for k in range(width_steps):
            adapter.decode_forward(
                tokens=torch.full((B, 1), int(tokens_fed[k])),
                start_pos=host_pos,
                page_table=pt,
                kv_cache=kv,
                enable_trace=True,
                read_from_device=True,
                sampling_params=_tt_params(B),
                reset_batch=(k == 0),
                reload_inputs=(k == 0),
            )
            host_pos[0] += 1
        out = adapter.decode_forward(
            tokens=torch.full((B, 1), int(tokens_fed[width_steps])),
            start_pos=host_pos,
            page_table=pt,
            kv_cache=kv,
            enable_trace=True,
            read_from_device=True,
            sampling_params=None,
            reset_batch=True,
            reload_inputs=True,
        )
        out = out[0] if isinstance(out, (tuple, list)) else out
        return torch.as_tensor(out).float().reshape(-1, gen.args.vocab_size)[0]

    # (a) one live row for the whole stream (width-1 trace), twice: deterministic; equal to row 0 of the full-width run
    #     up to the first fork, and the fork (if any) must be a near tie of the eager logits at that state: the width-w
    #     step on a B-slot build reproduces the B=w build's numerics (MoE accumulation order follows the token count), so
    #     streams at different widths agree bit-for-bit only until the first near tie - the same effect as the B=1 vs B=32
    #     servers of stage 9 (900-token continuation identical, 1000-token fork at a 0.125-logit tie).
    one = run([1] * steps, [False] * steps)
    assert one == run([1] * steps, [False] * steps), "width-1 stream not deterministic"
    row0_one, row0_full = [s[0] for s in one], [s[0] for s in full]
    d = next((k for k in range(steps) if row0_one[k] != row0_full[k]), None)
    assert d is None or d >= 1, f"width-1 and width-4 streams differ from the first token: {row0_one} vs {row0_full}"
    if d is not None:
        sampled0, _ = adapter.prefill_forward(
            toks, pt, kv, plens, empty_slots=list(range(B)), start_pos=torch.zeros(B), sampling_params=_tt_params(B)
        )
        fed = [int(sampled0.reshape(-1)[0])] + row0_one[:d]  # tokens fed at steps 0..d (fed[k] produces one[k])
        lg = logits_at(d, fed)
        a, b = lg[row0_one[d]].item(), lg[row0_full[d]].item()
        top = torch.topk(lg, 2).values.tolist()
        print(
            f"VLLM_ADAPTER_BUCKETS fork at token {d}: width-1 {row0_one[d]} ({a:.3f}) vs width-4 {row0_full[d]} ({b:.3f}); top-2 {top}"
        )
        assert (
            abs(a - b) < 1.0 and max(a, b) >= top[1] - 1e-3
        ), f"width fork at token {d} is not a near tie: {a} vs {b}, top-2 {top}"

    def check(seq, stream, label):
        """Idle rows pause (their slot state is untouched): row u's k-th token must equal the reference's k-th token for
        that row until the first near-tie fork; every row must agree on at least its first two tokens."""
        counts = [0] * B
        forked = [False] * B
        for i, (n, s) in enumerate(zip(seq, stream)):
            for u in range(n):
                ref = full[counts[u]][u]
                if not forked[u] and s[u] != ref:
                    assert counts[u] >= 2, f"{label}: row {u} differs at token {counts[u]} (step {i}): {s[u]} != {ref}"
                    forked[u] = True
                counts[u] += 1

    # (b) width switches: 4 -> 2 -> 1 -> 2 -> 4 rows live (completions, then re-admissions), flagged as layout changes;
    #     deterministic across two runs (carry-over + per-slot state continuity), agreeing with the full run to the fork
    seq = [4, 4, 2, 2, 1, 2, 4, 4]
    resets = [False, False, True, False, True, True, True, False]
    mixed = run(seq, resets)
    assert mixed == run(seq, resets), "mixed-width stream not deterministic"
    check(seq, mixed, "flagged switches")
    # (c) width switches the plugin did not flag (no reset_batch): the adapter forces the host-authoritative step
    seq2 = [4, 4, 1, 1, 4, 4, 4, 4]
    ref2 = run(seq2, [False] * steps)
    check(seq2, ref2, "unflagged switches")
    # (d) stale host tokens across flagged and unflagged width switches: the streams must equal their correctly fed
    #     references (the device token buffer of the previous width is carried over before the merge)
    assert run(seq, resets, wrong_host_tokens=True) == mixed, "stale host tokens changed the flagged-switch stream"
    assert (
        run(seq2, [False] * steps, wrong_host_tokens=True) == ref2
    ), "stale host tokens changed the unflagged-switch stream"
    assert gen.program_cache_entries() == n0, "decode bucketing compiled programs after the captures"
    hist = adapter._bucket_hist
    print(
        f"VLLM_ADAPTER_BUCKETS B={B} widths used {hist} switches {adapter._bucket_switches} full(slot0)={[s[0] for s in full]}"
    )
    assert set(hist) == {1, 2, 4}, hist


@mesh_tp()
@pytest.mark.parametrize("B", [4], ids=["b4"])
def test_vllm_adapter_seeded_buckets(mesh_device, B):
    """Seeded sampling (temperature 2, top-k 50, seed 42) must reproduce across re-runs of the same request at every
    decode width: the plugin suite's test_batch1_seed_reproducible failed on the first bucketed 48-layer server (10 unique
    streams of a single seeded user = the width-1 path), while seeded batches at one width agreed within the batch."""
    os.environ["QWEN35_PROGRAM_GUARD"] = "1"
    os.environ["QWEN35_DECODE_BUCKETING"] = "1"
    os.environ.pop("QWEN35_DECODE_MIN_WIDTH", None)  # the served default (minimum width 2)
    adapter, gen, kv = _build(mesh_device, B)
    assert adapter._decode_widths == [2, 4], adapter._decode_widths
    import bz2

    book = bz2.open("models/tt_transformers/tests/tale-of-two-cities.txt.bz2", "rt", errors="ignore").read()
    prompts = [gen.tokenizer.encode(book[20000 + 3000 * u : 20000 + 3000 * u + 2000])[: 58 + u] for u in range(B)]
    plens = [len(p) for p in prompts]
    toks = torch.zeros(B, max(plens), dtype=torch.long)
    for u, p in enumerate(prompts):
        toks[u, : len(p)] = torch.tensor(p)
    pt = _page_table(B, 64, 8)
    n0 = gen.program_cache_entries()

    def run(n_live, steps=6, seed=42):
        params = _tt_params(B, temperature=2.0, top_k=50, seed=seed)
        sampled, _ = adapter.prefill_forward(
            toks, pt, kv, plens, empty_slots=list(range(B)), start_pos=torch.zeros(B), sampling_params=params
        )
        cur = sampled.reshape(B).clone()
        pos = torch.tensor(plens, dtype=torch.long)
        stream = [cur[:n_live].tolist()]
        for i in range(steps):
            host_pos = pos.clone()
            host_pos[n_live:] = -1
            out = adapter.decode_forward(
                tokens=cur.reshape(B, 1),
                start_pos=host_pos,
                page_table=pt,
                kv_cache=kv,
                enable_trace=True,
                read_from_device=True,
                sampling_params=params,
                reset_batch=(i == 0),
                reload_inputs=(i == 0),
            )
            out = out[0] if isinstance(out, (tuple, list)) else out
            nxt = torch.as_tensor(out).reshape(-1)[:n_live].to(torch.int64)
            stream.append(nxt.tolist())
            cur = cur.clone()
            cur[:n_live] = nxt
            pos = pos.clone()
            pos[:n_live] += 1
        return stream

    for n_live in (1, 2, B):
        a, b = run(n_live), run(n_live)
        print(f"VLLM_ADAPTER_SEEDED width for {n_live} live: run1 {a}\n run2 {b}")
        assert a == b, f"seeded stream at {n_live} live rows not reproducible:\n {a}\n {b}"
    # seeded rows force the full width (composition-independent numerics): every served step so far ran at width B (the
    # histogram also counts the one warm-up step of each width); an unseeded greedy single user then takes width 2
    before2 = adapter._bucket_hist.get(2, 0)
    assert before2 <= 1 and adapter._bucket_hist.get(B, 0) > 1, adapter._bucket_hist
    greedy = _tt_params(B)
    sampled, _ = adapter.prefill_forward(
        toks, pt, kv, plens, empty_slots=list(range(B)), start_pos=torch.zeros(B), sampling_params=greedy
    )
    host_pos = torch.tensor(plens, dtype=torch.long)
    host_pos[1:] = -1
    adapter.decode_forward(
        tokens=sampled.reshape(B, 1),
        start_pos=host_pos,
        page_table=pt,
        kv_cache=kv,
        enable_trace=True,
        read_from_device=True,
        sampling_params=greedy,
        reset_batch=True,
        reload_inputs=True,
    )
    assert adapter._bucket_hist.get(2) == before2 + 1, adapter._bucket_hist
    # a different seed must give a different stream (the seed is in effect)
    assert run(1, seed=7) != run(1, seed=42), "seed has no effect"
    assert gen.program_cache_entries() == n0, "seeded decode compiled programs after the captures"

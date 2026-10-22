# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Async decode through the vLLM adapter, no vLLM: the stale-input device test of doc/vllm/DESIGN.md 8.4 (stage-9 review
#1 P1-1 / P1-2). It plays the vllm-tt-plugin's decode-input contract 1 against Glm5NextForConditionalGeneration:

  steady step : reload_inputs=False (host token / position one step stale), read_from_device=False, the non-blocking
                readback enqueued right after the submit, step N+1 submitted BEFORE step N is finalized (overlap depth 1);
                reload_page_table=True only on a step where a request gets a new block (page growth), else False.
  transition  : every pending step drained first, then reload_inputs=True with the host-authoritative token / position
                (a finish + slot_remap condense, a prefill between decodes, a sampling-mode switch).

Scenario (B=4, greedy, reduced model GLM53_LAYERS default 0,3,4, replicated DSA layout): A/B/C/D prefilled at staggered
lengths; 420 steps (each request crosses >= 3 blocks; block ids interleaved); A finishes at step 120 -> remap; E prefilled
into the freed vLLM slot at step 121; steps 200-202 host-sampled (sampling_params None -> host logits, argmax), then
device sampling again. Control: each request decoded ALONE, synchronously, in the same physical slot on the same block
ids. PASS: every stream equals its control token for token, and no program is compiled after the trace capture.

Run: python -m models.autoports.zai_org_glm_5_3_flash.tools.vllm_async_validate [--out RESULT.json]
     (or tests/test_vllm_async_validate.py). Needs GLM53_CKPT; supports_async_decode must be on (the default).
"""

import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

os.environ.setdefault("GLM53_LAYERS", "0,3,4")
os.environ.setdefault("HF_MODEL", "zai-org/GLM-5.3-Flash")
os.environ["GLM53_FAULTHANDLER"] = "0"
os.environ.setdefault("GLM53_VLLM_ASYNC_DECODE", "1")
os.environ.pop("GLM53_DSA_SHARD", None)

import ttnn  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt import generator_vllm as gv  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt import serving_profiles  # noqa: E402

B, MLEN, NBLK, BLK = 4, 8192, 160, 128
STEPS, FINISH_A, PREFILL_E, HOST_STEPS = 420, 120, 121, (200, 201, 202)
LENS = {"A": 37, "B": 300, "C": 1000, "D": int(os.environ.get("GLM53_AV_LEN_D", "2051")), "E": 129}
# GLM53_AV_LEN_D (stage 10, review #1 P3-10): a longer D (e.g. 4,100 with GLM53_DSA_BUCKETS=4096 at 8,192) makes the
# decode cross a DSA width-bucket switch (position 4,227) under async decode


def sp(n):
    return SimpleNamespace(
        temperature=[0.0] * n,
        top_k=[1] * n,
        top_p=[1.0] * n,
        presence_penalty=[0.0] * n,
        frequency_penalty=[0.0] * n,
        repetition_penalty=[1.0] * n,
        seed=[None] * n,
        enable_log_probs=[False] * n,
        num_logprobs=None,
    )


class Blocks:
    """vLLM-like allocator: a request gets a new block when its next position crosses a block boundary; ids are handed
    out from one global counter, so the ids of the requests interleave."""

    def __init__(self):
        self.next = 1
        self.of = {}

    def ensure(self, req, pos):
        lst = self.of.setdefault(req, [])
        grew = False
        while len(lst) * BLK <= pos:
            assert self.next < NBLK, "pool exhausted"
            lst.append(self.next)
            self.next += 1
            grew = True
        return grew


def table(rows_reqs, blocks, fixed=None):
    W = MLEN // BLK
    pt = torch.zeros(len(rows_reqs), W, dtype=torch.int32)
    for r, req in enumerate(rows_reqs):
        if req is None:
            continue
        lst = (fixed or blocks.of)[req]
        pt[r, : len(lst)] = torch.tensor(lst, dtype=torch.int32)
    return pt


def open_mesh():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    return ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(1, 4),
        l1_small_size=24576,
        trace_region_size=int(serving_profiles.load()["trace_region_bytes"]),
        num_command_queues=1,
    )


def run(mesh, out_path=None) -> dict:
    """The scenario above on an open (1, 4) mesh. Returns the result dict (result: PASS / FAIL); writes it to
    out_path when given."""
    res = {"checks": {}}
    from transformers import AutoConfig

    a = gv.Glm5NextForConditionalGeneration.initialize_vllm_model(
        AutoConfig.from_pretrained(os.environ["GLM53_CKPT"]), mesh, B, max_seq_len=MLEN
    )
    assert a.model_capabilities["supports_async_decode"], "GLM53_VLLM_ASYNC_DECODE=1 expected"
    kv = a.allocate_kv_cache((NBLK, 1, BLK, 512), torch.bfloat16, 1)
    for tr in (False, True):
        a.warmup_model_prefill(kv_cache=kv, enable_trace=tr, can_sample_on_device=True)
        a.warmup_model_decode(
            kv_cache=kv, enable_trace=tr, max_batch_size=B, num_blocks=MLEN // BLK, can_sample_on_device=True
        )
    base_programs = a._gen.program_cache_entries()
    g = torch.Generator().manual_seed(1)
    prompts = {k: torch.randint(1000, 100000, (n,), generator=g) for k, n in LENS.items()}

    def prefill(req, vslot, blocks):
        n = LENS[req]
        blocks.ensure(req, n)  # the prompt plus the first decode position
        pt = table([req], blocks)
        return int(
            a.prefill_forward(
                prompts[req].reshape(1, -1), pt, kv, [n], empty_slots=[vslot], sampling_params=sp(1)
            ).reshape(-1)[0]
        )

    # ------------------------------------------------------------------ the batch run (async contract)
    blocks = Blocks()
    rows = ["A", "B", "C", "D"]  # row r = vLLM slot r (front-packed)
    streams = {}
    for v, req in enumerate(rows):
        streams[req] = [prefill(req, v, blocks)]
    phys_of = {req: a.slots[v] for v, req in enumerate(rows)}
    pos = {req: LENS[req] for req in rows}
    pending = None  # (handle, rows snapshot, host_mode) of the step not read yet
    transition = True  # the first decode after a prefill: host authoritative
    remap = None
    stats = {"steady": 0, "reload": 0, "pt_reload_only": 0, "host_steps": 0}

    def read(p):  # finalize: wait on the readback events of the submit, then host formatting
        out, ev, rws = p
        for e in ev:
            ttnn.event_synchronize(e)
        toks = a.process_decode_output_host(out).reshape(-1).tolist()
        for r, req in enumerate(rws):
            if req is not None:
                streams[req].append(int(toks[r]))

    for step in range(STEPS):
        if step == FINISH_A:  # A finishes: drain, condense rows [B, C, D], remap
            if pending:
                read(pending)
                pending = None
            done_a = rows.index("A")
            order = [i for i in range(B) if i != done_a] + [done_a]
            rows = [rows[i] for i in order[:3]] + [None]
            remap = torch.tensor(order, dtype=torch.int32)
            transition = True
        if step == PREFILL_E:  # a prefill between decodes into the freed vLLM slot 3
            if pending:
                read(pending)
                pending = None
            v = rows.index(None)
            streams["E"] = [prefill("E", v, blocks)]
            phys_of["E"] = a.slots[v]
            pos["E"] = LENS["E"]
            rows[v] = "E"
            transition = True
        host_mode = step in HOST_STEPS
        if host_mode or (step == HOST_STEPS[-1] + 1):
            transition = True  # sampling-mode switch (device -> host, host -> device)
        grew = False
        for req in rows:
            if req is not None:
                grew |= blocks.ensure(req, pos[req])
        reload = transition
        if reload and pending:  # the plugin drains before a host-authoritative reload
            read(pending)
            pending = None
        tok = torch.zeros(B, 1, dtype=torch.int64)
        sp_ = torch.full((B,), -1, dtype=torch.int64)
        for r, req in enumerate(rows):
            if req is not None:
                # host view: authoritative on a reload; one step stale on a steady step (a lagging token is passed)
                tok[r, 0] = streams[req][-1] if reload else streams[req][max(0, len(streams[req]) - 2)]
                sp_[r] = pos[req] if reload else pos[req] - 1
        out = a.decode_forward(
            tokens=tok,
            start_pos=sp_,
            page_table=table(rows, blocks),
            kv_cache=kv,
            sampling_params=None if host_mode else sp(B),
            reload_inputs=reload,
            reload_page_table=(not reload) and grew,
            reload_sampling_params=reload and not host_mode,
            reset_sampling_state=reload and not host_mode,
            slot_remap=remap,
            enable_trace=True,
            read_from_device=False,
        )
        remap = None
        stats["reload" if reload else "steady"] += 1
        stats["pt_reload_only"] += int((not reload) and grew)
        if host_mode:
            stats["host_steps"] += 1
            lg = torch.as_tensor(out).reshape(B, -1)
            for r, req in enumerate(rows):
                if req is not None:
                    streams[req].append(int(lg[r].argmax()))
        else:
            # the plugin enqueues the non-blocking readback right after the submit (async_decode.submit_decode), then
            # submits step N+1 before it finalizes step N
            hd, ev = a.read_decode_output(out, async_read=True)
            if pending:
                read(pending)
            pending = (hd, ev, list(rows))
        for req in rows:
            if req is not None:
                pos[req] += 1
        transition = host_mode  # a host step: the next step reloads again
    if pending:
        read(pending)
    a._guard_programs("async validation")
    res["stats"] = stats
    res["blocks"] = {k: len(v) for k, v in blocks.of.items()}
    res["phys"] = phys_of

    # ------------------------------------------------------------------ solo controls (sync, same slot, same blocks)
    fixed = dict(blocks.of)
    a.slots = gv.SlotMap(B)  # identity: vLLM slot = physical slot
    solo = {}
    for req, n_tok in ((k, len(v)) for k, v in streams.items()):
        v = phys_of[req]
        n = LENS[req]
        pt_row = table([req], None, fixed)
        first = int(
            a.prefill_forward(
                prompts[req].reshape(1, -1), pt_row, kv, [n], empty_slots=[v], sampling_params=sp(1)
            ).reshape(-1)[0]
        )
        s_ = [first]
        rws = [None] * B
        rws[v] = req
        p = n
        for i in range(n_tok - 1):
            tok = torch.zeros(B, 1, dtype=torch.int64)
            st = torch.full((B,), -1, dtype=torch.int64)
            tok[v, 0], st[v] = s_[-1], p
            o = a.decode_forward(
                tokens=tok,
                start_pos=st,
                page_table=table(rws, None, fixed),
                kv_cache=kv,
                sampling_params=sp(B),
                reload_inputs=True,
                reload_sampling_params=(i == 0),
                reset_sampling_state=(i == 0),
                enable_trace=True,
                read_from_device=True,
            )
            s_.append(int(torch.as_tensor(o).reshape(-1)[v]))
            p += 1
        solo[req] = s_
    a._guard_programs("solo controls")
    ok_all = True
    for req in streams:
        eq = streams[req] == solo[req]
        first_div = next((i for i, (x, y) in enumerate(zip(streams[req], solo[req])) if x != y), None)
        res["checks"][req] = {
            "tokens": len(streams[req]),
            "equal_to_solo": eq,
            "first_divergence": first_div,
            "blocks": res["blocks"][req],
        }
        print(
            f"CHECK {req}: {len(streams[req])} tokens, {res['blocks'][req]} blocks, phys slot {phys_of[req]}, "
            f"equal to solo: {eq} (first divergence {first_div})",
            flush=True,
        )
        ok_all &= eq
    res["program_guard"] = {"base": base_programs, "end": a._gen.program_cache_entries()}
    ok_all &= base_programs == a._gen.program_cache_entries()
    res["result"] = "PASS" if ok_all else "FAIL"
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps({**res, "streams": streams, "solo": solo}, indent=1))
    print("ASYNC_STATS", stats, flush=True)
    print("ASYNC_RESULT", res["result"], out_path or "", flush=True)
    return res


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None, help="result JSON (streams, controls, checks)")
    a = ap.parse_args(argv)
    t0 = time.time()
    mesh = open_mesh()
    try:
        res = run(mesh, a.out)
    finally:
        ttnn.close_mesh_device(mesh)
    print(f"done in {time.time() - t0:.0f}s")
    sys.exit(0 if res["result"] == "PASS" else 1)


if __name__ == "__main__":
    main()

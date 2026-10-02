# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Decode bucketing for the served Qwen3.5-122B-A10B (stage 10, optimized vLLM).

The stage-9 server captured ONE decode trace at ``max_num_seqs`` (32): a lone user then ran the 32-wide MoE step
(124 ms/token) although the B=1 step takes 24 ms. The vLLM plugin keeps live requests front-packed (condense) and marks
idle rows with ``start_pos == -1``, so the decode step only has to cover rows ``[0:num_active)``: this module keeps one
decode trace per power-of-two width ``W in {1, 2, 4, ..., max_batch_size}`` and, before every step, selects the smallest
width >= the active prefix. Every width owns its trace ids, persistent device inputs (token buffer, positions, RoPE
indices, page-table rows ``[0:W)``) and output, and its own sampling-trace namespace (``set_trace_bucket``). The GDN
recurrent/conv state (``[Bmax, ...]``) is shared: the width-W step reads and writes rows ``[0:W)`` and leaves the idle
rows untouched (``TPGatedDeltaNet.forward_decode``); paged attention and the sampler address the full slot space through
the page-table rows / the ``[1,1,1,32]`` token buffer.

Width switches are host-authoritative steps: the plugin only changes the active prefix when the batch layout changes
(admission / completion / condense), which it submits with ``reset_batch=True``. The shared ``Generator`` then merges
the DEVICE tokens and positions of the trace inputs (authoritative one step ahead under async scheduling) with the host
view; those device buffers belong to the PREVIOUS width, so :func:`carry_over_inputs` copies them into the selected
width's buffers first (rare, a few tiny transfers). A switch the plugin did not flag is forced to ``reset_batch=True``.

The host-logits compatibility mode (``sampling_params is None``: host sampling for min_p / logit bias / logprobs /
structured outputs) runs at the full width, whose store holds the only host-logits trace.
"""

from __future__ import annotations

import os
from collections import defaultdict

import torch
from loguru import logger

import ttnn

_STORE_ATTR = "_decode_bucket_store"
_ACTIVE_ATTR = "_decode_bucket_active"


def enabled() -> bool:
    return os.environ.get("QWEN35_DECODE_BUCKETING", "1") == "1"


def widths(max_batch_size: int) -> list[int]:
    """1, 2, 4, ... up to and including ``max_batch_size`` (which need not be a power of two)."""
    out, w = [], 1
    while w < max_batch_size:
        out.append(w)
        w *= 2
    out.append(int(max_batch_size))
    return out


def width_for(num_active: int, max_batch_size: int) -> int:
    num_active = max(1, min(int(num_active), int(max_batch_size)))
    return min(int(max_batch_size), 1 << max(0, (num_active - 1).bit_length()))


def active_prefix(start_pos, width: int) -> int:
    """Rows with a live request: the plugin front-packs them, idle rows carry position -1."""
    if start_pos is None:
        return width
    pos = torch.as_tensor(start_pos).reshape(-1)
    live = (pos >= 0).nonzero().reshape(-1)
    return int(live.max().item()) + 1 if live.numel() else 1


def _store(gen):
    store = getattr(gen, _STORE_ATTR, None)
    if store is None:
        store = {}
        setattr(gen, _STORE_ATTR, store)
    return store


def select(gen, width: int, *, carry_over: bool = True) -> bool:
    """Bind the shared Generator's decode-trace dicts to the width's store and key the sampler's trace namespace by the
    width. Returns True when the width changed (the caller then treats the step as host-authoritative)."""
    store = _store(gen)
    if width not in store:
        store[width] = (defaultdict(lambda: None), defaultdict(lambda: None), defaultdict(lambda: None))
    prev = getattr(gen, _ACTIVE_ATTR, None)
    changed = prev is not None and prev != width
    if changed and carry_over:
        carry_over_inputs(gen, store[prev], store[width], prev, width)
    gen.trace_ids_decode, gen.trace_inputs_decode, gen.trace_output_decode = store[width]
    setattr(gen, _ACTIVE_ATTR, width)
    for m in gen.model:
        sm = getattr(m, "sampling", None)
        if sm is not None and hasattr(sm, "set_trace_bucket"):
            sm.set_trace_bucket(width)
    return changed


def carry_over_inputs(gen, src_store, dst_store, src_w: int, dst_w: int):
    """Copy the device-resident tokens and positions of the previous width into the selected width's trace inputs, so the
    shared Generator's reset-step merge (device token one step ahead of the host view) reads live values. Rows the new
    width does not cover are dropped; rows the old width did not cover start idle (position -1, token 0)."""
    key = True  # on-device sampling store; the host-logits mode always runs at the full width
    src = src_store[1][key]
    dst = dst_store[1][key]
    if not src or not dst:
        return
    for i in range(len(dst)):
        s_tok, s_pos = src[i][0], src[i][1]
        d_tok, d_pos = dst[i][0], dst[i][1]
        # token buffer: [1,1,1,S] for every width (the sampler's padded batch). Host round trip of S=32 integers on
        # purpose: a device-to-device ttnn.copy is a PROGRAM (it compiled at the first switch and tripped the
        # compile-after-capture guard), copy_host_to_device_tensor is not.
        t = ttnn.to_torch(ttnn.get_device_tensors(s_tok)[0]).reshape(-1)
        n = min(t.numel(), d_tok.shape[-1])
        h = torch.zeros(tuple(int(x) for x in d_tok.shape), dtype=torch.int32)
        h.reshape(-1)[:n] = t[:n].to(torch.int32)
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(h, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT), d_tok)
        # positions: [W] per width
        p = ttnn.to_torch(ttnn.get_device_tensors(s_pos)[0]).reshape(-1).to(torch.int32)
        n = min(p.numel(), dst_w)
        h = torch.full((dst_w,), -1, dtype=torch.int32)
        h[:n] = p[:n]
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(h, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT), d_pos)
    logger.debug(f"decode bucket switch {src_w} -> {dst_w}: device tokens/positions carried over")


def slice_inputs(args, kwargs, width: int):
    """Slice the plugin's full-width host inputs (tokens, start_pos, page_table; kwargs or positional 0/1/2) to the
    width. Returns the new (args, kwargs)."""
    args = list(args)

    def _read(name, pos):
        if name in kwargs:
            return kwargs[name]
        return args[pos] if pos < len(args) else None

    def _write(name, pos, val):
        if name in kwargs:
            kwargs[name] = val
        elif pos < len(args):
            args[pos] = val

    for name, pos in (("tokens", 0), ("start_pos", 1), ("page_table", 2)):
        v = _read(name, pos)
        if v is not None and int(v.shape[0]) > width:
            _write(name, pos, v[:width])
    return args, kwargs

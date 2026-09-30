# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""MTP verify pass, "rows" mode of one decoder layer: R consecutive positions of ONE user in one call.

    attention layer: the layer's decode forward over a BW-row block (rows 0..R-1 real, positions P..P+R-1, the same
                     page-table row for every row) with `TPAttention.rows_kv_write` so the R K/V rows are written one at
                     a time (the per-user paged update races inside one 32-position KV tile); the batched decode SDPA
                     reads are causal within the step because every row's K/V is written before the reads.
    GDN layer:       attention norm in prefill form on a GDN_CHUNK (128)-row padded block (rows 0..R-1 real), the GDN
                     prefill form (`forward_prefill`, valid_len = R, capture_state) continuing from the bound state
                     (conv_carry rebuilt from the decode shift register), the first BW rows of its output back into the
                     decode layout, then the layer's decode-mode ffn norm + MoE (pairs path for BW <= 8 rows).

The caller owns the state binding (the GDN prefill scratch holds the user's slot state) and the accept/commit logic.
"""
from __future__ import annotations

import os

import torch

import ttnn
from models.tt_transformers.tt.common import Mode

# GDN block rows of the rows mode: 32 (default) = one tile, the fused chunk kernel's native chunk (validated vs the 128-row
# block at PCC 1.0; ~0.3 ms less per GDN layer in the pass, commit 0.30 vs 0.54 ms per layer); 128 = the model's prefill
# chunk (QWEN35_MTP_GDN_ROWS=128: the prefill-form norm path).
GDN_CHUNK = int(os.environ.get("QWEN35_MTP_GDN_ROWS", "32"))


def rebuild_conv_carry(dn):
    """conv_carry[0, j] = conv_states[j + 1][0, 0] for j < K-1 (B == 1): the last K-1 conv inputs, oldest first."""
    window = ttnn.concat(list(dn.conv_states[1:]), dim=1)
    if window.layout != dn.conv_carry.layout:
        window = ttnn.to_layout(window, dn.conv_carry.layout)
    ttnn.copy(window, dn.conv_carry)
    ttnn.deallocate(window)


def rows_width(R: int) -> int:
    """Decode-layout width for R real rows: the smallest served decode width (power of two, >= 2) that holds them. Every
    dummy row costs MoE work (T x k pairs), so k = 3 (R = 4 -> width 4) halves the MoE of a pass vs k = 4 (R = 5 -> 8).
    """
    bw = 2
    while bw < R:
        bw *= 2
    return bw


def rows_pos_masks_host(positions, bw):
    """Host [bw, bw] int32 for `TPAttention.rows_pos_masks`: row b = -1 everywhere except [b, b] = positions[b] (the
    paged K/V update skips users at index -1, so one call per row writes exactly that row)."""
    m = torch.full((bw, bw), -1, dtype=torch.int32)
    for b, p in enumerate(positions):
        m[b, b] = int(p)
    return ttnn.from_torch(m, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)


def layer_forward_rows(
    layer,
    x,
    R,
    *,
    cos=None,
    sin=None,
    position_tensor=None,
    page_table=None,
    saved=None,
    rows_pos_masks=None,
    gdn_index=0,
):
    """x: decode-layout block [1, 1, BW, dim/TP] (rows 0..R-1 real). Returns the same layout. With `saved` (a list), a GDN
    layer appends its 128-row normed input instead of freeing it, for the GDN-only commit (`mtp_verify.commit_gdn`)."""
    BW = x.shape[2]
    # QWEN35_MTP_MOE_ROWS=sorted: routed experts through the expert-sorted path (see Qwen35MoEOptimized._routed_gather)
    layer.feed_forward.rows_sorted = os.environ.get("QWEN35_MTP_MOE_ROWS", "pairs") == "sorted"
    try:
        return _layer_forward_rows(
            layer,
            x,
            R,
            BW,
            cos=cos,
            sin=sin,
            position_tensor=position_tensor,
            page_table=page_table,
            saved=saved,
            rows_pos_masks=rows_pos_masks,
            gdn_index=gdn_index,
        )
    finally:
        layer.feed_forward.rows_sorted = False


def _layer_forward_rows(layer, x, R, BW, *, cos, sin, position_tensor, page_table, saved, rows_pos_masks, gdn_index):
    # R: the real row count (host int; the GDN kernels then build their masks with a host write) or a device-resident
    # `PrefillLen` (bucket 128) staged by the caller: the traced pass needs the latter (no host write inside a capture)
    R_int = R if isinstance(R, int) else BW
    assert 1 <= R_int <= BW <= 32, (R, BW)
    if layer.is_full_attention:
        att = layer.attention
        att.rows_kv_write = True
        # QWEN35_MTP_KV_ROWMASK=0: the older per-row slice / reshard write (A/B baseline)
        att.rows_pos_masks = rows_pos_masks if os.environ.get("QWEN35_MTP_KV_ROWMASK", "1") == "1" else None
        try:
            return layer.forward(x, cos, sin, position_tensor=position_tensor, page_table=page_table, mode="decode")
        finally:
            att.rows_kv_write = False
            att.rows_pos_masks = None
    # ---- GDN layer: prefill form on a 128-row padded block
    args = layer.args
    # a [1,1,BW<=32,H/TP] TILE tensor already spans one 32-row tile, so padding to 32 rows can ALIAS x (no copy); the
    # decode-form norm below frees its input, which then freed x itself (residual add on a freed buffer: PCC 0.22)
    x_src = x if GDN_CHUNK > 32 else ttnn.clone(x, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    x128 = ttnn.pad(x_src, [(0, 0), (0, 0), (0, GDN_CHUNK - BW), (0, 0)], value=0.0)
    if x_src is not x and x128.buffer_address() != x_src.buffer_address():
        ttnn.deallocate(x_src)
    if GDN_CHUNK > 32:
        # prefill-form norm: distributed rmsnorm whose post-norm all-gather is fused into the GDN in-proj matmul
        # (forward_prefill takes the K-sharded input when T > TILE)
        attn_cfg = dict(args.get_norm_config("attn", Mode.PREFILL))
        attn_cfg["distributed_output_mem_config"] = ttnn.L1_MEMORY_CONFIG
        attn_in = layer._decode_norm(layer.attention_norm, x128, Mode.PREFILL, attn_cfg)
    else:
        # one 32-row tile: forward_prefill takes a REPLICATED input (no fused AG-matmul at T <= TILE), which the
        # decode-form norm (gather-then-norm) produces; the prefill config here gave PCC 0.22
        attn_in = layer._decode_norm(layer.attention_norm, x128, Mode.DECODE, args.get_norm_config("attn", Mode.DECODE))
    if x128.is_allocated():
        ttnn.deallocate(x128)
    dn = layer.attention
    rebuild_conv_carry(dn)
    sink = None
    if saved is not None:
        # the commit re-runs only the conv FIR + chunk scan on these (see TPGatedDeltaNet.commit_scan); `saved` holds one
        # dict per GDN layer: pre-created dicts of persistent buffers are copied into (traced pass), missing entries are
        # appended and filled with clones (eager pass)
        if len(saved) > gdn_index:
            sink = saved[gdn_index]
        else:
            sink = {}
            saved.append(sink)
    # chunk_size is the model's GDN chunk constant (its cached chunk masks), not the block's row count
    out128 = dn.forward_prefill(
        attn_in, chunk_size=int(args.gdn_chunk_size), valid_len=R, capture_state=True, scan_sink=sink
    )
    ttnn.deallocate(attn_in)
    attn_out = ttnn.slice(out128, (0, 0, 0, 0), (1, 1, BW, out128.shape[-1]))
    ttnn.deallocate(out128)
    h = ttnn.add(x, attn_out)
    ttnn.deallocate(attn_out)
    ff_cfg = args.get_norm_config("attn", Mode.DECODE)
    ff_in = layer._decode_norm(layer.ffn_norm, h, Mode.DECODE, ff_cfg)
    ff_out = layer.feed_forward.forward(ff_in, mode="decode")
    ttnn.deallocate(ff_in)
    out = ttnn.add(h, ff_out)
    ttnn.deallocate(h)
    ttnn.deallocate(ff_out)
    return out

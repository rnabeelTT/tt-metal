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

import ttnn
from models.tt_transformers.tt.common import Mode

GDN_CHUNK = 128


def rebuild_conv_carry(dn):
    """conv_carry[0, j] = conv_states[j + 1][0, 0] for j < K-1 (B == 1): the last K-1 conv inputs, oldest first."""
    window = ttnn.concat(list(dn.conv_states[1:]), dim=1)
    if window.layout != dn.conv_carry.layout:
        window = ttnn.to_layout(window, dn.conv_carry.layout)
    ttnn.copy(window, dn.conv_carry)
    ttnn.deallocate(window)


def layer_forward_rows(layer, x, R, *, cos=None, sin=None, position_tensor=None, page_table=None):
    """x: decode-layout block [1, 1, BW, dim/TP] (rows 0..R-1 real). Returns the same layout."""
    BW = x.shape[2]
    assert 1 <= R <= BW <= 32, (R, BW)
    if layer.is_full_attention:
        att = layer.attention
        att.rows_kv_write = True
        try:
            return layer.forward(x, cos, sin, position_tensor=position_tensor, page_table=page_table, mode="decode")
        finally:
            att.rows_kv_write = False
    # ---- GDN layer: prefill form on a 128-row padded block
    args = layer.args
    x128 = ttnn.pad(x, [(0, 0), (0, 0), (0, GDN_CHUNK - BW), (0, 0)], value=0.0)
    attn_cfg = dict(args.get_norm_config("attn", Mode.PREFILL))
    attn_cfg["distributed_output_mem_config"] = ttnn.L1_MEMORY_CONFIG
    attn_in = layer._decode_norm(layer.attention_norm, x128, Mode.PREFILL, attn_cfg)
    ttnn.deallocate(x128)
    dn = layer.attention
    rebuild_conv_carry(dn)
    out128 = dn.forward_prefill(attn_in, chunk_size=GDN_CHUNK, valid_len=R, capture_state=True)
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

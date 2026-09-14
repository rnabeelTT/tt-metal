# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Runtime-state lifecycle helpers for the Gated DeltaNet layer.

Behavior-preserving extraction of the original `_init_recurrent_state`,
`_split_fused_conv_state`, and `_restore_split_conv_from_fused` methods.
Each operates on the gdn instance, reading/writing its plain state attributes.
"""
import torch

import ttnn


def init_recurrent_state(gdn, batch_size):
    """Initialize recurrent state to zeros [B, num_v_heads, head_k_dim, head_v_dim]."""
    state = torch.zeros(
        batch_size,
        gdn.num_v_heads,
        gdn.head_k_dim,
        gdn.head_v_dim,
        dtype=torch.bfloat16,
    )
    gdn.recurrent_state = ttnn.from_torch(state, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=gdn.device)


def split_fused_conv_state(gdn):
    """Convert fused conv state [B, 3, D_total] into list of 3 [B, 1, D_total] tensors."""
    if gdn.fused_conv_state is None:
        return
    gdn.split_conv_state = []
    for k in range(gdn.conv_kernel_size - 1):
        s_k = gdn.fused_conv_state[:, k : k + 1, :]
        s_k = ttnn.to_layout(s_k, ttnn.TILE_LAYOUT)
        buf = ttnn.clone(s_k, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(s_k)
        gdn.split_conv_state.append(buf)


def restore_split_conv_from_fused(gdn):
    """Copy fused_conv_state slices into existing split_conv_state buffers.
    Preserves device addresses (critical for trace replay).
    Use instead of split_fused_conv_state() when split buffers already exist.
    """
    if gdn.split_conv_state is None:
        return
    for k in range(gdn.conv_kernel_size - 1):
        s_k = gdn.fused_conv_state[:, k : k + 1, :]
        s_k = ttnn.to_layout(s_k, ttnn.TILE_LAYOUT)
        ttnn.copy(s_k, gdn.split_conv_state[k])
        ttnn.deallocate(s_k)


def slice_along(buf, dim, lo, hi):
    """ttnn.slice of `buf` along `dim` for [lo, hi), other dims kept full."""
    start = [0] * len(buf.shape)
    end = list(buf.shape)
    start[dim] = lo
    end[dim] = hi
    return ttnn.slice(buf, tuple(start), tuple(end))


def write_index(buf, src, idx, dim, consume_src=False):
    """Replace slice `idx` of `buf` along `dim` with `src`, preserving every other slice.

    ttnn has no in-place row write, so the buffer is rebuilt by slice+concat and copied
    back -- the copy is what preserves `buf`'s address, which a captured decode trace has
    baked in. `src` must match `buf`'s dtype and have extent 1 along `dim`.

    consume_src=False leaves `src` alive, so a caller may write the same source into
    several buffers (or keep a scratch buffer it is about to reuse).
    """
    n = buf.shape[dim]
    if n == 1:
        ttnn.copy(src, buf)
        if consume_src:
            ttnn.deallocate(src)
        return
    parts, temps = [], []
    if idx > 0:
        head = slice_along(buf, dim, 0, idx)
        parts.append(head)
        temps.append(head)
    parts.append(src)
    if idx < n - 1:
        tail = slice_along(buf, dim, idx + 1, n)
        parts.append(tail)
        temps.append(tail)
    new = ttnn.concat(parts, dim=dim)
    ttnn.copy(new, buf)
    ttnn.deallocate(new)
    for t in temps:
        ttnn.deallocate(t)
    if consume_src:
        ttnn.deallocate(src)


def gather_indices(buf, idx, dim):
    """Reindex `buf` along `dim` so position i takes what was at idx[i], in place."""
    if all(idx[i] == i for i in range(len(idx))):
        return
    parts = [slice_along(buf, dim, j, j + 1) for j in idx]
    new = ttnn.concat(parts, dim=dim)
    ttnn.copy(new, buf)
    ttnn.deallocate(new)
    for p in parts:
        ttnn.deallocate(p)

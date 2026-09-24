# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""TTNN gated delta rule: recurrent (decode) and chunked (prefill) paths.

FLA layout: q,k [B,T,H,K]; v [B,T,H,V]; beta,g [B,T,H]; state [B,H,K,V].
"""

import math

import torch
import ttnn

# Wormhole tile size
_TILE_H = 32
_TILE_W = 32

# Cached identity matrices for forward substitution
_EYE_CACHE = {}


def _chunk_eye(size):
    """Return a cached [size, size] float32 identity matrix on CPU."""
    if size not in _EYE_CACHE:
        _EYE_CACHE[size] = torch.eye(size, dtype=torch.float32)
    return _EYE_CACHE[size]


def _recurrent_read_query_program_config(device, K, V):
    """Progcfg for read/query matmul: [B*H,1,K] @ [B*H,K,V] -> [B*H,1,V]. N == per_core_N."""
    grid = device.compute_with_storage_grid_size()
    N_tiles = (V + _TILE_W - 1) // _TILE_W
    K_tiles = (K + _TILE_W - 1) // _TILE_W

    per_core_N = N_tiles
    per_core_M = 1
    in0_block_w = K_tiles
    out_subblock_h = 1
    out_subblock_w = min(2, per_core_N) if per_core_N >= 2 else 1

    return ttnn.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=grid,
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
    )


def _create_eye_matrix_ttnn(size, device, dtype=ttnn.float32, memory_config=None):
    """Create [size,size] identity on device via triu/tril."""
    ones = ttnn.ones(
        shape=(size, size),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=memory_config,
    )
    eye = ttnn.tril(ttnn.triu(ones, diagonal=0), diagonal=0, memory_config=memory_config)
    return eye


def _create_triu_ones_ttnn(size, device, dtype=ttnn.float32, memory_config=None):
    """Create [size,size] upper-triangular ones on device."""
    ones = ttnn.ones(
        shape=(size, size),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=memory_config,
    )
    triu_ones = ttnn.triu(ones, diagonal=0, memory_config=memory_config)
    return triu_ones


def _create_tril_ones_ttnn(size, device, dtype=ttnn.float32, memory_config=None):
    """Create [size,size] lower-triangular ones on device."""
    ones = ttnn.ones(
        shape=(size, size),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=memory_config,
    )
    tril_ones = ttnn.tril(ones, diagonal=0, memory_config=memory_config)
    return tril_ones


def _create_strict_lower_tril_ttnn(size, device, dtype=ttnn.float32, memory_config=None):
    """Create [size,size] strict lower-tri ones (diagonal=-1) on device."""
    ones = ttnn.ones(
        shape=(size, size),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=memory_config,
    )
    strict_lower = ttnn.tril(ones, diagonal=-1, memory_config=memory_config)
    return strict_lower


def _get_matmul_program_config(m, k, n, grid_size=None, in0_block_w=None):
    """Matmul progcfg for (m,k,n). Returns None when auto-config is better."""
    TILE_SIZE = 32
    # Cap per_core_M to avoid L1 CB overflow on BH (~40KB safe at 2 M-tiles/core).
    MAX_PER_CORE_M = 2

    if m < 32 or n < 32 or k < 32:
        return None

    m_tiles = math.ceil(m / TILE_SIZE)
    n_tiles = math.ceil(n / TILE_SIZE)
    k_tiles = math.ceil(k / TILE_SIZE)

    # Large k (k_tiles>8) overflows L1 CB; let TTNN auto-select.
    if k_tiles > 8:
        return None

    if grid_size is None:
        total_work = m * n * k

        if total_work < 32768:
            return None

        if m_tiles == 2 and n_tiles == 2:
            return None
        elif m_tiles == 2 and n_tiles == 4:
            return None
        elif m_tiles == 2 and n_tiles == 8:
            return None
        elif m_tiles == 4 and n_tiles == 2:
            return None
        elif m_tiles == 4 and n_tiles == 8:
            grid_size = (8, 4)
        elif m_tiles <= 4 and n_tiles <= 4:
            if n_tiles >= 4:
                grid_size = (2, 1)
            elif m_tiles >= 4:
                grid_size = (1, 2)
            else:
                if total_work >= 262144:
                    grid_size = (2, 2)
                else:
                    grid_size = (2, 1)
        elif m_tiles >= 4 and n_tiles >= 8:
            cores_y = min(4, m_tiles)
            cores_x = min(8, n_tiles)
            grid_size = (cores_x, cores_y)
        else:
            if n_tiles >= 4:
                total_cores = min(8, max(2, n_tiles))
                grid_size = (total_cores, 1)
            elif m_tiles >= 4:
                total_cores = min(8, max(2, m_tiles))
                grid_size = (1, total_cores)
            else:
                return None

    cores_x, cores_y = grid_size

    per_core_M = math.ceil(m_tiles / cores_y)
    per_core_N = math.ceil(n_tiles / cores_x)
    per_core_M = max(1, per_core_M)
    per_core_N = max(1, per_core_N)

    # Increase cores_y if per_core_M too large; force in0_block_w=1 for L1 safety.
    if per_core_M > MAX_PER_CORE_M:
        cores_y = math.ceil(m_tiles / MAX_PER_CORE_M)
        cores_y = min(cores_y, 8)  # max 8 cores along Y
        per_core_M = math.ceil(m_tiles / cores_y)
        if per_core_M > MAX_PER_CORE_M:
            return None  # Can't fit with 8 cores; auto-select
        grid_size = (cores_x, cores_y)
        in0_block_w = 1  # keep in0 buffer from scaling with k_tiles

    if in0_block_w is None:
        k_per_core = math.ceil(k_tiles / cores_x) if cores_x > 1 else k_tiles
        in0_block_w = min(4, max(1, k_per_core))
        while k_tiles % (in0_block_w * cores_x) != 0 and in0_block_w > 1:
            in0_block_w -= 1
        if in0_block_w < 1:
            in0_block_w = 1

    max_subblock_size = 4
    out_subblock_h = min(per_core_M, max_subblock_size)
    out_subblock_w = min(per_core_N, max_subblock_size)

    if out_subblock_h * out_subblock_w > max_subblock_size:
        if out_subblock_h > out_subblock_w:
            out_subblock_h = max_subblock_size // out_subblock_w
        else:
            out_subblock_w = max_subblock_size // out_subblock_h

    while per_core_M % out_subblock_h != 0 and out_subblock_h > 1:
        out_subblock_h -= 1
    while per_core_N % out_subblock_w != 0 and out_subblock_w > 1:
        out_subblock_w -= 1

    if out_subblock_h < 1 or out_subblock_w < 1 or out_subblock_h * out_subblock_w > max_subblock_size:
        out_subblock_h = min(per_core_M, 2)
        out_subblock_w = min(per_core_N, 2)
        while per_core_M % out_subblock_h != 0 and out_subblock_h > 1:
            out_subblock_h -= 1
        while per_core_N % out_subblock_w != 0 and out_subblock_w > 1:
            out_subblock_w -= 1

    if out_subblock_h < 1 or out_subblock_w < 1:
        return None

    try:
        return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=grid_size,
            in0_block_w=in0_block_w,
            out_subblock_h=out_subblock_h,
            out_subblock_w=out_subblock_w,
            per_core_M=per_core_M,
            per_core_N=per_core_N,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=False,
        )
    except Exception:
        return None


def l2_norm_ttnn(x, dim=-1, eps=1e-6):
    """L2 norm along dim. Last dim: fused rms_norm path (~3 fewer kernels)."""
    # L1 for T<=512; DRAM otherwise
    T = x.shape[1] if len(x.shape) >= 3 else x.shape[0]
    mc = ttnn.L1_MEMORY_CONFIG if T <= 512 else ttnn.DRAM_MEMORY_CONFIG
    if dim in (-1, len(x.shape) - 1):
        K = x.shape[-1]
        normed = ttnn.rms_norm(x, epsilon=eps / K)
        return ttnn.multiply(normed, K**-0.5, memory_config=mc)
    x_sq = ttnn.multiply(x, x, memory_config=mc)
    norm_sq = ttnn.sum(x_sq, dim=dim, keepdim=True, memory_config=mc)
    inv_norm = ttnn.rsqrt(ttnn.add(norm_sq, eps, memory_config=mc), memory_config=mc)
    return ttnn.multiply(x, inv_norm, memory_config=mc)


# Largest state-sized decode temporary (B*H*K*V*itemsize bytes: the outer-product update) kept in L1. The persistent
# state itself is read where it lives. 24 MiB = 12 users of a 64x128x128 bf16 state (2 MiB each) or 24 of the 9B's 32-head
# state (1 MiB each); above it the temporary and the update go to DRAM (the old budget was 96 MiB counted at 4 bytes per
# element for every dtype, i.e. 48 MiB of real bf16 bytes; a 64 MiB L1 request at B=32 does not fit next to the resident buffers).
_DECODE_STATE_L1_BUDGET_BYTES = 24 * 1024 * 1024


def _decode_state_memory_config(B, H, K, V, itemsize=4):
    return (
        ttnn.L1_MEMORY_CONFIG if B * H * K * V * itemsize <= _DECODE_STATE_L1_BUDGET_BYTES else ttnn.DRAM_MEMORY_CONFIG
    )


def fused_decay_and_write_ttnn(
    h,
    k_t,
    delta,
    decay_t,
    beta_t,
    device=None,
    apply_decay=True,
):
    """State update: h = decay*h + beta*(k⊗delta). apply_decay=False if caller already decayed h."""
    B = h.shape[0]
    H = h.shape[1]
    K = h.shape[2]
    V = h.shape[3]

    # decay_t: [B,H] -> [B,H,1,1] (already exp(g) in BF16)
    decay = ttnn.reshape(decay_t, [B, H, 1, 1], memory_config=ttnn.L1_MEMORY_CONFIG)

    # beta: [B,H] -> [B,H,1,1]
    beta_expanded = ttnn.reshape(beta_t, [B, H, 1, 1], memory_config=ttnn.L1_MEMORY_CONFIG)

    # Decode opt: keep state-write operands in L1 (tiny at B=1); the state-sized operands follow the
    # state budget (DRAM for wide GDNs at large batch).
    _L1 = _decode_state_memory_config(B, H, K, V)
    k_col = ttnn.reshape(k_t, [B, H, K, 1], memory_config=_L1)
    d_row = ttnn.reshape(delta, [B, H, 1, V], memory_config=_L1)

    k_col = ttnn.to_memory_config(k_col, _L1)
    d_row = ttnn.to_memory_config(d_row, _L1)

    matmul_compute_cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )

    outer = ttnn.matmul(
        k_col,
        d_row,
        memory_config=_L1,
        compute_kernel_config=matmul_compute_cfg,
        program_config=None,
    )

    # apply beta
    outer = ttnn.multiply(
        outer,
        beta_expanded,
        memory_config=_L1,
    )

    # apply_decay=False: h already decayed (decay->read->write order).
    if apply_decay:
        h = ttnn.multiply(h, decay, memory_config=_L1)
    h = ttnn.add(h, outer, memory_config=_L1)

    return h


def recurrent_delta_rule_step_ttnn(
    q_t,
    k_t,
    v_t,
    beta_t,
    decay_t,
    h,
    seq_len=None,
    device=None,
):
    """Single recurrent delta-rule step via fused_decay_and_write_ttnn."""
    B = q_t.shape[0]
    H = q_t.shape[1]
    K = q_t.shape[2]
    V = v_t.shape[2]

    q_t = ttnn.to_layout(q_t, ttnn.TILE_LAYOUT)
    k_t = ttnn.to_layout(k_t, ttnn.TILE_LAYOUT)
    v_t = ttnn.to_layout(v_t, ttnn.TILE_LAYOUT)
    h = ttnn.to_layout(h, ttnn.TILE_LAYOUT)
    h = ttnn.to_memory_config(h, ttnn.DRAM_MEMORY_CONFIG)

    read_query_compute_cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    read_query_prog_cfg = None
    if device is not None:
        try:
            read_query_prog_cfg = _recurrent_read_query_program_config(device, K, V)
        except Exception:
            pass

    # Decay state before read (matches FLA + chunked prefill).
    decay_bhkv = ttnn.reshape(decay_t, [B, H, 1, 1], memory_config=ttnn.L1_MEMORY_CONFIG)
    h = ttnn.multiply(h, decay_bhkv)

    k_row = ttnn.reshape(k_t, [B, H, 1, K], memory_config=None)
    k_row = ttnn.to_layout(k_row, ttnn.TILE_LAYOUT)
    k_row = ttnn.to_memory_config(k_row, ttnn.DRAM_MEMORY_CONFIG)
    v_read = ttnn.matmul(
        k_row,
        h,
        memory_config=None,
        program_config=read_query_prog_cfg,
        compute_kernel_config=read_query_compute_cfg,
    )
    v_read = ttnn.reshape(v_read, [B, H, V], memory_config=None)

    delta = ttnn.subtract(v_t, v_read, memory_config=None)

    # Outer product only (h already decayed).
    h = fused_decay_and_write_ttnn(
        h=h,
        k_t=k_t,
        delta=delta,
        decay_t=decay_t,
        beta_t=beta_t,
        device=device,
        apply_decay=False,
    )

    q_row = ttnn.reshape(q_t, [B, H, 1, K], memory_config=None)
    q_row = ttnn.to_layout(q_row, ttnn.TILE_LAYOUT)
    q_row = ttnn.to_memory_config(q_row, ttnn.DRAM_MEMORY_CONFIG)
    o_t = ttnn.matmul(
        q_row,
        h,
        memory_config=None,
        program_config=read_query_prog_cfg,
        compute_kernel_config=read_query_compute_cfg,
    )
    use_l1 = seq_len is not None and seq_len <= 64
    o_t = ttnn.reshape(o_t, [B, H, V], memory_config=None if use_l1 else None)

    return o_t, h


def recurrent_gated_delta_rule_decode_ttnn(
    q,
    k,
    v,
    beta,
    g,
    scale=None,
    initial_state=None,
    device=None,
    high_precision=False,
    state_out=None,
    output_head_rows=False,
):
    """T=1 decode path: no loop/slice overhead. Returns o [B,1,H,V] (or [B,H,1,V] tile-padded head rows when
    ``output_head_rows``), state [B,H,K,V].

    Structure (one state-sized read pass + one state-sized write pass):
      v_read = (k @ h_old) * exp(g)            # decay applied to the read (same algebra as decaying h first)
      delta  = (v - v_read) * beta             # [B,H,1,V]
      outer  = k^T x delta                     # column x row broadcast multiply -> [B,H,K,V] (no padded-K matmul)
      h_new  = h_old * exp(g) + outer          # one fused mac, written into ``state_out`` when given (no staging copy)
      o      = q @ h_new
    Every head-row tensor stays tile-padded [B,H,1,*]; the only dense reshape is the final output (skipped with
    ``output_head_rows``)."""
    B = q.shape[0]
    H = q.shape[2]
    K = q.shape[3]
    V = v.shape[3]

    # high_precision: fp32 step avoids bf16 decay quantization error over long decode.
    if high_precision:
        q = ttnn.typecast(q, ttnn.float32)
        k = ttnn.typecast(k, ttnn.float32)
        v = ttnn.typecast(v, ttnn.float32)
        beta = ttnn.typecast(beta, ttnn.float32)
        g = ttnn.typecast(g, ttnn.float32)

    # L2 norm
    q = l2_norm_ttnn(q, dim=-1)
    k = l2_norm_ttnn(k, dim=-1)

    if scale is None:
        scale = K**-0.5
    q = ttnn.multiply(q, scale, memory_config=ttnn.L1_MEMORY_CONFIG)

    # Head-row layouts [B,H,1,*] (dense -> tile-padded reshapes: one output tile row per head, cheap).
    _L1 = ttnn.L1_MEMORY_CONFIG
    q_row = ttnn.reshape(q, [B, H, 1, K], memory_config=_L1)
    k_row = ttnn.reshape(k, [B, H, 1, K], memory_config=_L1)
    v_row = ttnn.reshape(v, [B, H, 1, V], memory_config=_L1)
    beta_bh11 = ttnn.reshape(beta, [B, H, 1, 1], memory_config=_L1)
    g_bh11 = ttnn.reshape(g, [B, H, 1, 1], memory_config=_L1)

    h = initial_state
    if h is None:
        h = ttnn.zeros(
            [B, H, K, V], device=device, dtype=ttnn.float32 if high_precision else ttnn.bfloat16, memory_config=None
        )
    elif high_precision and h.dtype != ttnn.float32:
        h = ttnn.typecast(h, ttnn.float32)
    # The state is read where it lives (no staging copy); the state-sized temporaries follow the L1 budget.
    _state_mc = _decode_state_memory_config(B, H, K, V, 4 if h.dtype == ttnn.float32 else 2)

    read_query_compute_cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    read_query_prog_cfg = None
    if device is not None:
        try:
            read_query_prog_cfg = _recurrent_read_query_program_config(device, K, V)
        except Exception:
            pass

    # v_read = (k @ h_old) * exp(g): the per-head decay is applied to the read row instead of to the whole
    # state (identical algebra), so the state is touched only by the fused update below.
    v_read = ttnn.matmul(
        k_row, h, memory_config=_L1, program_config=read_query_prog_cfg, compute_kernel_config=read_query_compute_cfg
    )
    v_read = ttnn.multiply(v_read, g_bh11, input_tensor_b_activations=[ttnn.UnaryOpType.EXP], memory_config=_L1)
    delta = ttnn.subtract(v_row, v_read, memory_config=_L1)
    ttnn.deallocate(v_read)
    d_row = ttnn.multiply(delta, beta_bh11, memory_config=_L1)  # beta folded into the small operand
    ttnn.deallocate(delta)
    # outer = k^T x (beta*delta) as a column x row broadcast multiply (no inner-dim-1 matmul, no dense reshapes)
    k_col = ttnn.transpose(k_row, -2, -1)  # [B,H,K,1]
    outer = ttnn.multiply(k_col, d_row, memory_config=_state_mc)
    ttnn.deallocate(k_col)
    ttnn.deallocate(d_row)
    decay = ttnn.exp(g_bh11, memory_config=_L1)
    # h_new = h_old * decay + outer in one pass; written straight into the persistent buffer when given.
    if state_out is not None and state_out.dtype == h.dtype:
        # written in place; keep the caller's tensor object (the op's return value aliases the same buffer, and
        # deallocating it would free the persistent state)
        ttnn.mac(h, decay, outer, output_tensor=state_out)
        h_new = state_out
    else:
        h_new = ttnn.mac(h, decay, outer, memory_config=_state_mc)
    ttnn.deallocate(outer)
    ttnn.deallocate(decay)

    o_t = ttnn.matmul(
        q_row,
        h_new,
        memory_config=_L1,
        program_config=read_query_prog_cfg,
        compute_kernel_config=read_query_compute_cfg,
    )
    if output_head_rows:
        return o_t, h_new
    o = ttnn.reshape(o_t, [B, 1, H, V], memory_config=_L1)
    return o, h_new


def recurrent_gated_delta_rule_decode_inplace_ttnn(
    q,
    k,
    v,
    beta,
    g,
    state_buffer,
    scale=None,
    device=None,
    high_precision=False,
    output_head_rows=False,
):
    """Decode writing the new state into the pre-allocated buffer (trace-safe addresses). The update is written in
    place when the buffer dtype matches the step dtype; otherwise (e.g. fp32 step on a bf16 buffer) it is copied back.
    """
    o, new_h = recurrent_gated_delta_rule_decode_ttnn(
        q=q,
        k=k,
        v=v,
        beta=beta,
        g=g,
        scale=scale,
        initial_state=state_buffer,
        device=device,
        high_precision=high_precision,
        state_out=state_buffer,
        output_head_rows=output_head_rows,
    )
    if new_h is not state_buffer:
        ttnn.copy(new_h, state_buffer)
        ttnn.deallocate(new_h)
    return o, state_buffer


def recurrent_gated_delta_rule_ttnn(
    q,
    k,
    v,
    beta,
    g,
    scale=None,
    initial_state=None,
    device=None,
):
    """Token-by-token recurrent gated delta rule (decode, general T).

    Per step: decay h; v_read=k·h; delta=v-v_read; write outer; o=q·h.
    Returns output [B,T,H,V], final_state [B,H,K,V].
    """
    q = l2_norm_ttnn(q, dim=-1)
    k = l2_norm_ttnn(k, dim=-1)

    B = q.shape[0]
    T = q.shape[1]
    H = q.shape[2]
    K = q.shape[3]
    V = v.shape[3]

    if scale is None:
        scale = K**-0.5

    q = ttnn.multiply(q, scale, memory_config=None)

    q = ttnn.transpose(q, 1, 2, memory_config=None)
    k = ttnn.transpose(k, 1, 2, memory_config=None)
    v = ttnn.transpose(v, 1, 2, memory_config=None)
    beta = ttnn.transpose(beta, 1, 2, memory_config=None)
    g = ttnn.transpose(g, 1, 2, memory_config=None)

    q = ttnn.typecast(q, ttnn.bfloat16, memory_config=None)
    k = ttnn.typecast(k, ttnn.bfloat16, memory_config=None)
    v = ttnn.typecast(v, ttnn.bfloat16, memory_config=None)
    beta = ttnn.typecast(beta, ttnn.bfloat16, memory_config=None)
    g = ttnn.typecast(g, ttnn.bfloat16, memory_config=None)

    # Precompute exp(g); slice per timestep in loop.
    g_exp = ttnn.exp(g)

    if initial_state is not None:
        h = ttnn.typecast(initial_state, ttnn.bfloat16, memory_config=None)
    else:
        h = ttnn.zeros([B, H, K, V], device=device, dtype=ttnn.bfloat16, memory_config=None)

    outputs = []
    for i in range(T):
        q_t = q[:, :, i]  # [B,H,K]
        k_t = k[:, :, i]
        v_t = v[:, :, i]
        beta_t = beta[:, :, i]
        decay_t = g_exp[:, :, i]

        o_t, h = recurrent_delta_rule_step_ttnn(q_t, k_t, v_t, beta_t, decay_t, h, seq_len=T, device=device)
        outputs.append(o_t)

    outputs_4d = [ttnn.reshape(o, [B, H, 1, V], memory_config=None) for o in outputs]
    o = ttnn.concat(outputs_4d, dim=2, memory_config=None)
    o = ttnn.transpose(o, 1, 2, memory_config=None)
    o = ttnn.typecast(o, ttnn.bfloat16, memory_config=None)

    return o, h

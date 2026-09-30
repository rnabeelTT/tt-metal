# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Stage 12 (prefill optimization): the group-parallel ``ttnn.sparse_matmul`` factory (MatmulMultiCoreReuseProgramConfig +
``indices``) against the multicast-1D masked path (nnz = G, one-hot group mask; the layer's current path) and a torch reference, at the autoport's expert shapes on one die and on
the (1,4) mesh. Exactness bar: bit-identical to the multicast path is not required (K-block accumulation order differs when
in0_block_w differs), PCC >= 0.9995 vs the torch reference (bf16 partial sums), never below the mcast path's own PCC vs torch, and max abs diff
within bf16 tolerance of the mcast path.

Run: pytest models/autoports/qwen_qwen3_5_122b_a10b/tests/test_sparse_matmul_grouped.py -s
"""
import time

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc

TILE = 32


def _grid(device):
    g = device.compute_with_storage_grid_size()
    return ttnn.CoreCoord(g.x, g.y)


def grouped_config(device, gs, K, N, tile_w=TILE, in0_block_w=4, out_subblock_w=8, fp32_acc=False):
    m_tiles = gs // TILE
    n_tiles = -(-N // tile_w)
    k_tiles = K // TILE
    ib = max(1, min(in0_block_w, k_tiles))
    while k_tiles % ib:
        ib -= 1
    sw = max(1, min(out_subblock_w, n_tiles))
    while n_tiles % sw:
        sw -= 1
    if fp32_acc:
        while sw > 4:
            sw -= 1
    return ttnn.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=_grid(device),
        in0_block_w=ib,
        out_subblock_h=1,
        out_subblock_w=sw,
        per_core_M=m_tiles,
        per_core_N=n_tiles,
    )


def mcast_config(gs, N, K, target_cores=64, in0_block_w_cap=32, tile_w=TILE):
    from models.autoports.qwen_qwen3_5_122b_a10b.tt.moe import sparse_matmul_program_config

    return sparse_matmul_program_config(
        gs, N, K, target_cores=target_cores, in0_block_w_cap=in0_block_w_cap, tile_w=tile_w
    )


def _make(device, G, gs, E, K, N, w_dtype, seed=0, mesh=False):
    torch.manual_seed(seed)
    x = torch.randn(1, G, gs, K, dtype=torch.bfloat16)
    w = torch.randn(1, E, K, N, dtype=torch.bfloat16) * 0.05
    ids = torch.randint(0, E, (G,), dtype=torch.int32)
    mm = dict(mesh_mapper=ttnn.ReplicateTensorToMesh(device)) if mesh else {}
    x_t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, **mm)
    w_t = ttnn.from_torch(w, dtype=w_dtype, layout=ttnn.TILE_LAYOUT, device=device, **mm)
    idx_t = ttnn.from_torch(
        ids.to(torch.int32).reshape(1, G), dtype=ttnn.uint16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, **mm
    )
    # one-hot group mask [1, G, 1, E]: drives the multicast MASKED path (nnz=G, what the layer runs today); in the
    # grouped/indexed path the op only validates its batch length and the ids drive the weight reads
    mask = torch.zeros(1, G, 1, E, dtype=torch.bfloat16)
    mask[0, torch.arange(G), 0, ids.long()] = 1.0
    ones = ttnn.from_torch(mask, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, **mm)
    # torch reference with the quantized weights the device sees
    w_q = ttnn.to_torch(w_t, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0) if mesh else None)
    if mesh:
        w_q = w_q[: w.shape[0]]
    ref = torch.stack([x[0, g].float() @ w_q[0, ids[g]].float() for g in range(G)]).unsqueeze(0)  # [1,G,gs,N]
    return x_t, w_t, idx_t, ones, ids, ref


def _run(device, x_t, w_t, idx_t, ones, cfg, ckc, iters=1, grouped=True):
    """grouped=True: indexed group-parallel factory; grouped=False: multicast masked path (nnz = G one-hot mask)."""
    G = idx_t.shape[-1]
    kw = {"indices": idx_t} if grouped else {"nnz": G}
    out = None
    for _ in range(iters):
        if out is not None:
            ttnn.deallocate(out)
        if not grouped:
            # the masked path only writes a compact [1, G, M, N] output through an optional output tensor
            M, N = x_t.shape[-2], w_t.shape[-1]
            kw["optional_output_tensor"] = ttnn.zeros(
                (1, G, M, N),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        out = ttnn.sparse_matmul(
            x_t,
            w_t,
            sparsity=ones,
            **kw,
            is_input_b_sparse=True,
            program_config=cfg,
            compute_kernel_config=ckc,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    return out


CKC = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=True, fp32_dest_acc_en=False, packer_l1_acc=True
)


@pytest.mark.parametrize(
    "G,gs,E,K,N,w_dtype",
    [
        (64, 32, 256, 3072, 1024, ttnn.bfloat4_b),  # gate/up shape (per-die TP experts), few groups
        (512, 32, 256, 3072, 1024, ttnn.bfloat4_b),  # gate/up at ~2048 tokens
        (512, 32, 256, 512, 3072, ttnn.bfloat4_b),  # down projection at ~2048 tokens
        (37, 32, 256, 3072, 1024, ttnn.bfloat8_b),  # odd group count (core groups 1 and 2), bfp8
    ],
    ids=["gu64", "gu512", "dn512", "gu37_bfp8"],
)
def test_grouped_matches_mcast_and_torch(device, G, gs, E, K, N, w_dtype):
    x_t, w_t, idx_t, ones, ids, ref = _make(device, G, gs, E, K, N, w_dtype)
    cfg_g = grouped_config(device, gs, K, N)
    cfg_m = mcast_config(gs, N, K)
    out_g = _run(device, x_t, w_t, idx_t, ones, cfg_g, CKC)
    out_m = _run(device, x_t, w_t, idx_t, ones, cfg_m, CKC, grouped=False)
    g = ttnn.to_torch(out_g).float()
    m = ttnn.to_torch(out_m).float()
    assert g.shape == (1, G, gs, N), g.shape
    assert m.shape == (1, G, gs, N), m.shape
    pcc_ref = comp_pcc(ref, g, 0.9995)
    pcc_m_ref = comp_pcc(ref, m, 0.9995)
    pcc_m = comp_pcc(m, g, 0.9995)
    diff = (g - m).abs().max().item()
    scale = m.abs().max().item()
    print(
        f"GROUPED {G}x{gs}x{K}x{N} {w_dtype}: grouped vs torch {pcc_ref[1]} | mcast vs torch {pcc_m_ref[1]} | "
        f"grouped vs mcast {pcc_m[1]} max|diff| {diff:.4g} (max|mcast| {scale:.3g})"
    )
    # Both device paths round the running sum to bf16 between K blocks (packer L1 accumulation); the grouped path
    # uses more, smaller K blocks, so the two device results differ by rounding order. The bar is the torch
    # reference: the grouped result must be at least as close to it as the multicast result (within noise).
    assert pcc_ref[0], pcc_ref[1]
    assert pcc_m[0], pcc_m[1]
    assert pcc_ref[1] >= pcc_m_ref[1] - 5e-5, (pcc_ref[1], pcc_m_ref[1])
    assert diff <= 0.1 * scale + 1e-3, (diff, scale)


@pytest.mark.parametrize("G,gs,E,K,N", [(512, 32, 256, 3072, 1024), (512, 32, 256, 512, 3072)], ids=["gu", "dn"])
def test_grouped_perf_vs_mcast(device, G, gs, E, K, N):
    x_t, w_t, idx_t, ones, ids, ref = _make(device, G, gs, E, K, N, ttnn.bfloat4_b)
    res = {}
    variants = [("mcast", mcast_config(gs, N, K))]
    # L1 budget of the grouped factory: double-buffered in0 / in1 blocks + one output block (bf16) must fit in ~1.3 MB
    tile_b = {ttnn.bfloat4_b: 576, ttnn.bfloat8_b: 1088, ttnn.bfloat16: 2048}[ttnn.bfloat4_b]
    for ib in (4, 8, 16):
        cb = 2 * (gs // 32) * ib * 2048 + 2 * (N // 32) * ib * tile_b + (gs // 32) * (N // 32) * 2048
        if cb <= 1_300_000:
            variants.append((f"grouped_ib{ib}", grouped_config(device, gs, K, N, in0_block_w=ib)))
    for name, cfg in variants:
        grouped = name != "mcast"
        out = _run(device, x_t, w_t, idx_t, ones, cfg, CKC, grouped=grouped)  # compile
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        out = _run(device, x_t, w_t, idx_t, ones, cfg, CKC, iters=5, grouped=grouped)
        ttnn.synchronize_device(device)
        res[name] = 1e3 * (time.perf_counter() - t0) / 5
        ttnn.deallocate(out)
    best = min((v for k, v in res.items() if k != "mcast"))
    print(
        f"GROUPED_PERF {G}x{gs}x{K}x{N}: "
        + " | ".join(f"{k} {v:.3f} ms" for k, v in res.items())
        + f" | best speedup {res['mcast'] / best:.2f}x (host wall, 5 iters)"
    )


@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_grouped_on_mesh(mesh_device):
    G, gs, E, K, N = 256, 32, 256, 3072, 1024
    x_t, w_t, idx_t, ones, ids, ref = _make(mesh_device, G, gs, E, K, N, ttnn.bfloat4_b, mesh=True)
    out = _run(mesh_device, x_t, w_t, idx_t, ones, grouped_config(mesh_device, gs, K, N), CKC)
    g = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)).float()
    for d in range(4):
        pcc = comp_pcc(ref, g[d : d + 1], 0.9999)
        assert pcc[0], (d, pcc[1])
    print(f"GROUPED_MESH {G}x{gs}x{K}x{N}: all 4 devices PCC ok")


@pytest.mark.parametrize(
    "T", [256, 500, 513, 1000, 1023, 2048, -128], ids=["t256", "t500", "t513", "t1000", "t1023", "t2048", "t128same"]
)
def test_dispatch_modes_equivalent(device, T):
    """The matmul-based counting sort (`counting_mm`) must produce exactly the same dispatch tensors as the cumsum-based
    `counting_h`: integer index tensors equal, weights equal, group ids equal."""
    import models.autoports.qwen_qwen3_5_122b_a10b.tt.moe_dispatch as md

    E, k, gs = 256, 8, 32
    same = T < 0  # every token routes to the same k experts (the all-zero-token program warm-up)
    T = abs(T)
    torch.manual_seed(T)
    ids = torch.stack([torch.randperm(E)[:k] for _ in range(T)]).to(torch.int32).reshape(1, 1, T, k)
    if same:
        ids[:] = ids[0, 0, :1]
    scores = torch.rand(1, 1, T, k, dtype=torch.bfloat16)
    ids_t = ttnn.from_torch(ids, dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=device)
    sc_t = ttnn.from_torch(scores, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    outs = {}
    for mode in ("counting_h", "counting_mm", "kernel"):
        md.DISPATCH_MODE = mode
        d = md.SortedDispatch(device, E, k, group=gs)
        m = d.build(ids_t, sc_t, T)
        outs[mode] = {
            key: ttnn.to_torch(m[key]).clone()
            for key in ("src_tok", "weights", "group_mask", "group_ids", "undispatch")
        }
        outs[mode]["G"] = m["G"]
    a = outs["counting_h"]
    for other in ("counting_mm", "kernel"):
        b = outs[other]
        assert a["G"] == b["G"], other
        for key in ("src_tok", "undispatch", "group_ids"):
            assert torch.equal(a[key].to(torch.int64).reshape(-1), b[key].to(torch.int64).reshape(-1)), (other, key)
        assert torch.equal(a["group_mask"], b["group_mask"]), other
        assert torch.equal(a["weights"].reshape(-1), b["weights"].reshape(-1)), other
    # the dispatch must be a permutation covering every (token, slot) pair exactly once
    u = a["undispatch"].to(torch.int64).flatten()
    assert u.numel() == T * k and torch.unique(u).numel() == T * k
    if same:
        return
    print(f"DISPATCH_EQUIV T={T}: counting_mm == kernel == counting_h (G={a['G']})")
    # timing of the dispatch build alone (host wall incl. sync, 5 iters) per mode
    for mode in ("counting_mm", "kernel"):
        md.DISPATCH_MODE = mode
        d = md.SortedDispatch(device, E, k, group=gs)
        d.build(ids_t, sc_t, T)
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        for _ in range(5):
            m = d.build(ids_t, sc_t, T)
        ttnn.synchronize_device(device)
        print(f"DISPATCH_PERF T={T} {mode}: {1e3 * (time.perf_counter() - t0) / 5:.3f} ms per build (host wall)")


@pytest.mark.parametrize("T,H", [(500, 3072), (2048, 3072), (2048, 512)], ids=["t500", "t2048", "t2048_h512"])
def test_gather_add_rows_matches_reference(device, T, H):
    """The fused un-dispatch kernel (ttnn.moe_gather_add_rows) must equal the k x (gather + add) reference bit for bit
    (same bf16 add order) on a real dispatch permutation, and match a torch fp32 sum closely."""
    import models.autoports.qwen_qwen3_5_122b_a10b.tt.moe_dispatch as md

    E, k, gs = 256, 8, 32
    torch.manual_seed(T + H)
    ids = torch.stack([torch.randperm(E)[:k] for _ in range(T)]).to(torch.int32).reshape(1, 1, T, k)
    scores = torch.rand(1, 1, T, k, dtype=torch.bfloat16)
    ids_t = ttnn.from_torch(ids, dtype=ttnn.uint16, layout=ttnn.TILE_LAYOUT, device=device)
    sc_t = ttnn.from_torch(scores, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    md.DISPATCH_MODE, md.UNDISPATCH_KERNEL = "kernel", True
    m = md.SortedDispatch(device, E, k, group=gs).build(ids_t, sc_t, T)
    Rd, G = m["Rd"], m["G"]
    y = torch.randn(1, G, gs, H, dtype=torch.bfloat16)
    y_t = ttnn.from_torch(y, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out_k = ttnn.to_torch(md.undispatch_rows(y_t, None, None, T, k, H, undispatch_rm=m["undispatch_rm"])).float()
    und = ttnn.to_torch(m["undispatch_rm"]).to(torch.int64).reshape(T, k)
    und_t = ttnn.from_torch(
        und.reshape(1, 1, T, k).to(torch.int32), dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=device
    )
    out_ref = ttnn.to_torch(md.undispatch_rows(y_t, None, und_t, T, k, H)).float()
    assert out_k.shape == out_ref.shape == (1, 1, T, H)
    assert torch.equal(out_k, out_ref), (out_k - out_ref).abs().max()
    torch_ref = y.float().reshape(Rd, H)[und.reshape(-1)].reshape(T, k, H).sum(1)
    assert torch.allclose(out_k.reshape(T, H), torch_ref, atol=0.25, rtol=0.05)
    print(
        f"UNDISPATCH_EQUIV T={T} H={H}: kernel == k x (gather + add) reference, torch max abs err "
        f"{(out_k.reshape(T, H) - torch_ref).abs().max():.4f}"
    )
    for name, fn in (
        ("k_gather_add", lambda: md.undispatch_rows(y_t, None, und_t, T, k, H)),
        ("kernel", lambda: md.undispatch_rows(y_t, None, None, T, k, H, undispatch_rm=m["undispatch_rm"])),
    ):
        fn()
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        for _ in range(5):
            fn()
        ttnn.synchronize_device(device)
        print(f"UNDISPATCH_PERF T={T} H={H} {name}: {1e3 * (time.perf_counter() - t0) / 5:.3f} ms (host wall)")


@pytest.mark.parametrize(
    "G,gs,E,K,N,w_dtype",
    [
        (761, 32, 256, 3072, 512, ttnn.bfloat4_b),
        (761, 32, 256, 256, 3072, ttnn.bfloat4_b),
        (37, 32, 256, 3072, 512, ttnn.bfloat8_b),
    ],
    ids=["gu2048", "dn2048", "odd37_bfp8"],
)
def test_grouped_skip_padding_groups(device, G, gs, E, K, N, w_dtype):
    """With `skip_padding_groups`, groups whose id is a sentinel (>= E) are skipped by all three kernels; the real groups'
    outputs must be bit-identical to the run without skipping (padding groups are never read by the layer)."""
    x_t, w_t, idx_t, ones, ids, ref = _make(device, G, gs, E, K, N, w_dtype)
    # the layer's layout: real groups first (expert-ascending), padding groups at the end -> sentinel the last 16 %
    n_pad = max(1, G // 6)
    ids_s = ids.clone()
    ids_s[G - n_pad :] = 0xFFFF
    idx_s = ttnn.from_torch(ids_s.reshape(1, G), dtype=ttnn.uint16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    cfg = grouped_config(device, gs, K, N)
    cfg_skip = grouped_config(device, gs, K, N)
    cfg_skip.skip_padding_groups = True
    out_full = ttnn.to_torch(_run(device, x_t, w_t, idx_t, ones, cfg, CKC)).float()
    out_skip = ttnn.to_torch(_run(device, x_t, w_t, idx_s, ones, cfg_skip, CKC)).float()
    real = G - n_pad
    assert torch.equal(out_full[:, :real], out_skip[:, :real]), (out_full[:, :real] - out_skip[:, :real]).abs().max()
    pcc = comp_pcc(ref[:, :real], out_skip[:, :real], 0.9995)
    assert pcc[0], pcc[1]
    # sentinel groups interleaved with real ones (not only trailing) must also be skipped without disturbing neighbours
    ids_m = ids.clone()
    ids_m[1::7] = 0xFFFF
    idx_m = ttnn.from_torch(ids_m.reshape(1, G), dtype=ttnn.uint16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    out_mixed = ttnn.to_torch(_run(device, x_t, w_t, idx_m, ones, cfg_skip, CKC)).float()
    keep = torch.ones(G, dtype=torch.bool)
    keep[1::7] = False
    assert torch.equal(out_full[:, keep], out_mixed[:, keep])
    # timing: skip vs full at the layer's padding share
    for name, idx, c in (("full", idx_t, cfg), ("skip16pct", idx_s, cfg_skip)):
        _run(device, x_t, w_t, idx, ones, c, CKC)
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        _run(device, x_t, w_t, idx, ones, c, CKC, iters=10)
        ttnn.synchronize_device(device)
        print(
            f"SKIPPAD_PERF {G}x{gs}x{K}x{N} {name}: {1e3 * (time.perf_counter() - t0) / 10:.3f} ms per call (host wall)"
        )
    print(
        f"SKIPPAD_EQUIV {G}x{gs}x{K}x{N}: real groups identical with {n_pad} trailing + interleaved sentinel groups skipped"
    )

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device check of the sorted expert dispatch index math (tt/moe_dispatch.py) against a torch reference.

    python models/autoports/qwen_qwen3_5_122b_a10b/tools/check_sorted_dispatch.py
"""
import sys
import time

import torch
from tracy import signpost

import ttnn
from models.autoports.qwen_qwen3_5_122b_a10b.tt.moe_dispatch import (
    SortedDispatch,
    dispatch_rows,
    gather_rows,
    undispatch_rows,
)

E, K, H = 256, 8, 3072


def check(dev, disp, T, seed, profile=False):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(T, E, generator=g)
    scores, ids = torch.topk(torch.softmax(logits, -1), K, dim=-1)
    scores = scores / scores.sum(-1, keepdim=True)
    ids_tt = ttnn.from_torch(
        ids.to(torch.int32).reshape(1, 1, T, K), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev
    )
    sc_tt = ttnn.from_torch(
        scores.to(torch.bfloat16).reshape(1, 1, T, K), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev
    )
    if profile:  # warm once, then a signposted build for tt-perf-report
        m0 = disp.build(ids_tt, sc_tt, T)
        ttnn.synchronize_device(dev)
        del m0
    t0 = time.time()
    if profile:
        signpost("DISPATCH_BUILD")
    m = disp.build(ids_tt, sc_tt, T)
    ttnn.synchronize_device(dev)
    if profile:
        signpost("DISPATCH_BUILD_END")
    dt = time.time() - t0
    Rd, G = m["Rd"], m["G"]
    assert Rd == dispatch_rows(T, K, E)
    src = ttnn.to_torch(m["src_tok"]).reshape(-1).long()
    w = ttnn.to_torch(m["weights"]).reshape(-1).float()
    gm = ttnn.to_torch(m["group_mask"]).reshape(G, E).float()
    und = ttnn.to_torch(m["undispatch"]).reshape(T, K).long()
    ok = True
    # one expert per group
    if not torch.all(gm.sum(-1) == 1):
        print(f"T={T}: groups with != 1 expert: {(gm.sum(-1) != 1).sum().item()}")
        ok = False
    grp_expert = gm.argmax(-1)
    # each real pair lands on a row whose source token / weight / group expert match; destinations distinct
    d = und.reshape(-1)
    if d.unique().numel() != d.numel():
        print(f"T={T}: destination rows not distinct")
        ok = False
    tok = torch.arange(T).repeat_interleave(K)
    bad_tok = (src[d] != tok).sum().item()
    bad_w = (~torch.isclose(w[d], scores.reshape(-1).to(torch.bfloat16).float(), atol=1e-6)).sum().item()
    bad_e = (grp_expert[d // 32] != ids.reshape(-1)).sum().item()
    # unused rows carry weight 0
    used = torch.zeros(Rd, dtype=torch.bool)
    used[d] = True
    bad_pad = (w[~used] != 0).sum().item()
    if bad_tok or bad_w or bad_e or bad_pad:
        print(f"T={T}: mismatches tok={bad_tok} w={bad_w} expert={bad_e} pad_w={bad_pad}")
        ok = False
    # expert blocks are 32-aligned and contiguous: rows of expert e occupy whole groups
    for e in range(E):
        rows = d[ids.reshape(-1) == e]
        if (
            rows.numel()
            and rows.max() // 32 != rows.min() // 32
            and (rows.max() - rows.min()) >= 32 * ((rows.numel() + 31) // 32)
        ):
            print(f"T={T}: expert {e} rows not contiguous")
            ok = False
            break
    # round trip of real data through gather/undispatch: y = sum_j w_j * x_t must equal x_t * sum_j w_j (= x_t)
    x = torch.randn(1, 1, T, H).to(torch.bfloat16)
    x_tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    xd = gather_rows(x_tt, m["src_tok"], Rd)
    y = undispatch_rows(xd, m["weights"], m["undispatch"], T, K, H)
    y_t = ttnn.to_torch(y).float().reshape(T, H)
    ref = x.float().reshape(T, H) * scores.to(torch.bfloat16).float().sum(-1, keepdim=True)
    err = ((y_t - ref).abs() / (ref.abs() + 1.0)).max().item()  # relative: bf16 products + 8-term sums
    if err > 0.05:
        print(f"T={T}: gather/undispatch round trip max err {err}")
        ok = False
    used_groups = int(used.reshape(G, 32).any(-1).sum())
    print(
        f"T={T}: {'PASS' if ok else 'FAIL'} Rd={Rd} G={G} used_groups={used_groups} build {1e3*dt:.1f} ms round-trip err {err:.4f}"
    )
    return ok


def main():
    lens = [int(a) for a in sys.argv[1:]] or [5, 37, 128, 333, 2048, 4096]
    profile = len(sys.argv) > 1
    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=24576, trace_region_size=0, num_command_queues=1)
    try:
        disp = SortedDispatch(dev, E, K)
        oks = [check(dev, disp, T, 100 + T, profile=profile) for T in lens]
    finally:
        ttnn.close_mesh_device(dev)
    print("RESULT", "PASS" if all(oks) else "FAIL")
    return 0 if all(oks) else 1


if __name__ == "__main__":
    sys.exit(main())

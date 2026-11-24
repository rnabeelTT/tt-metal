# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Packed prefill convs vs the host reference: ShortConv.prefill_packed (sconv_kernels/pfseg_reader.cpp) and
DynShortConv.rows_packed (block-copied lags) on 3 segments with their own states, commit points and a mid-segment
prefix-snapshot point: outputs, new states and snapshot states. usage: tools/run_dev.sh python .../tests/test_pfseg_device.py"""

import torch

import ttnn
from models.autoports.thinkingmachines_inkling_2_mini.tt.blkcopy import BlockCopy
from models.autoports.thinkingmachines_inkling_2_mini.tt.model import _lane_bufs, _lane_copy
from models.autoports.thinkingmachines_inkling_2_mini.tt.sconv import (DynShortConv, ShortConv, dyn_host_reference,
                                                                     host_reference)


def main():
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    torch.manual_seed(0)
    C, H, nh = 256, 512, 2
    segs = [(0, 128, 100, None), (128, 64, 64, 32), (192, 64, 7, 2)]  # (row0, R, n, snap row)
    T = 256
    ok = True
    x = torch.randn(T, C).bfloat16().float()
    st0 = [torch.randn(3, C).bfloat16().float() for _ in segs]
    up = lambda t, lay=ttnn.TILE_LAYOUT, dt=ttnn.bfloat16: ttnn.from_torch(t.reshape(1, 1, *t.shape), dtype=dt,  # noqa
                                                                           layout=lay, device=mesh)
    # ---- ShortConv
    w = torch.randn(C, 4) * 0.3
    sc = _lane_bufs(ShortConv(mesh, w, None, C))
    lanes = [sc] + [_lane_copy(sc) for _ in segs[1:]]
    for lc, s0, (_, _, _, snap) in zip(lanes, st0, segs):
        ttnn.copy(up(s0, ttnn.ROW_MAJOR_LAYOUT, ttnn.float32), lc.st)
        lc.snap_row = snap
    y = sc.prefill_packed(up(x), [(r0, R, n, lc) for (r0, R, n, _), lc in zip(segs, lanes)])
    yh = ttnn.to_torch(y).float().reshape(T, C)
    for i, ((r0, R, n, snap), lc, s0) in enumerate(zip(segs, lanes, st0)):
        ref, _ = host_reference(x[r0 : r0 + R], w, s0)
        seq = torch.cat([s0, x[r0 : r0 + R]])
        dy = (yh[r0 : r0 + R] - ref.bfloat16().float()).abs().max().item()
        dst = (ttnn.to_torch(lc.st).float().reshape(3, C) - seq[n : n + 3]).abs().max().item()
        dsn = 0.0
        if snap is not None:
            dsn = (ttnn.to_torch(lc.snap_buf).float().reshape(3, C) - seq[snap : snap + 3]).abs().max().item()
        print(f"ShortConv seg {i}: y {dy:.3g}  state@{n} {dst:.3g}  snap@{snap} {dsn:.3g}", flush=True)
        ok = ok and dy <= 0.0625 and dst == 0 and dsn == 0
    # ---- DynShortConv
    static = torch.randn(4, C) * 0.3
    wproj = torch.randn(nh * 4, H) * 0.05
    dc = _lane_bufs(DynShortConv(mesh, static, wproj, None, C, nh))
    dl = [dc] + [_lane_copy(dc) for _ in segs[1:]]
    for lc, s0, (_, _, _, snap) in zip(dl, st0, segs):
        ttnn.copy(up(s0, ttnn.ROW_MAJOR_LAYOUT, ttnn.float32), lc.st)
        lc.snap_row = snap
    h = torch.randn(T, H).bfloat16().float()
    bc = BlockCopy(mesh, up(torch.zeros(64, 32)))
    y = dc.rows_packed(up(x, dt=ttnn.float32), up(h), [(r0, R, n, lc) for (r0, R, n, _), lc in zip(segs, dl)], bc)
    yh = ttnn.to_torch(y).float().reshape(T, C)
    for i, ((r0, R, n, snap), lc, s0) in enumerate(zip(segs, dl, st0)):
        ref, _ = dyn_host_reference(x[r0 : r0 + R], h[r0 : r0 + R], static, wproj, s0)
        seq = torch.cat([s0, x[r0 : r0 + R]])
        rel = ((yh[r0 : r0 + R] - ref).norm() / ref.norm()).item()
        dst = (ttnn.to_torch(lc.st).float().reshape(3, C) - seq[n : n + 3]).abs().max().item()
        dsn = 0.0
        if snap is not None:
            dsn = (ttnn.to_torch(lc.snap_buf).float().reshape(3, C) - seq[snap : snap + 3]).abs().max().item()
        print(f"DynShortConv seg {i}: y rel {rel:.3g}  state@{n} {dst:.3g}  snap@{snap} {dsn:.3g}", flush=True)
        ok = ok and rel < 2e-2 and dst == 0 and dsn == 0
    print("PASS" if ok else "FAIL", flush=True)
    ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()

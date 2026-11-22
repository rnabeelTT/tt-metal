# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Debug the prefix snapshot (tt/prefix.py): snapshot a prompt at Q inside its last chunk, then prefill exactly its
first Q tokens on another slot and compare every application's conv states and ring window with the snapshot pools.
usage: DEV_ENV="-e MESH=4" MESH_DEVICE=P300x2 tools/run_dev.sh python .../tests/dbg_prefix_device.py"""

import os

import torch

import ttnn
from models.autoports.thinkingmachines_inkling_2_mini.tt.model import Inkling2Model
from models.autoports.thinkingmachines_inkling_2_mini.tt.prefix import PrefixStore, _tiles_per_block


def main():
    N = int(os.environ.get("MESH", "1"))
    if N > 1:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, N))
    chunk = 1024
    m = Inkling2Model(mesh, max_seq_len=4096, expert_dtype=ttnn.bfloat8_b, kv_dtype=ttnn.bfloat16, prefill_chunk=chunk,
                      slots=4)
    store = PrefixStore(m, 2)
    Q, PA = int(os.environ.get("Q", "1280")), 1300
    A = torch.randint(1000, 100000, (PA,), generator=torch.Generator().manual_seed(0))
    d0 = (lambda t: ttnn.get_device_tensors(t)[0]) if N > 1 else (lambda t: t)
    host = lambda t: ttnn.to_torch(d0(t)).float()  # noqa: E731

    def prefill(ids, slot, snap=None):
        m.reset()
        P = int(ids.numel())
        for p0 in range(0, P, chunk):
            c = ids[p0 : p0 + chunk]
            n = int(c.numel())
            if n % 256:
                c = torch.cat([c, torch.zeros((-n) % 256, dtype=c.dtype)])
            arm = snap is not None and p0 < snap[1] <= p0 + n
            if arm:
                store.arm(snap[1] - p0)
            m.forward(c, p0, valid=n, slot=slot, last_only=True)
            if arm:
                store.save(snap[0], slot, snap[1])

    k = 1
    prefill(A, 0, snap=(k, Q))
    ttnn.synchronize_device(mesh)
    pools = [[host(p)[0, 0, 3 * k : 3 * k + 3] for p in ps] for ps in store.cpool]
    snaps = [[host(cv.snap_out) if cv.snap_out is not None else None for cv in ln[0]] for ln in store.convs]
    print("snap_out memory configs:", {repr(cv.snap_out.memory_config().buffer_type) for ln in store.convs
                                        for cv in ln[0] if cv.snap_out is not None}, flush=True)
    rings_pool = [host(pool) for _, pool, _ in store.rings]
    prefill(A[:Q], 3)
    ttnn.synchronize_device(mesh)
    bad = 0
    for a, (ln, ps) in enumerate(zip(store.convs, pools)):
        for i, (cv, p) in enumerate(zip(ln[0], ps)):
            ref = host(cv.st).reshape(3, -1)
            d = (ref - p.reshape(3, -1)).abs().max().item()
            if d > 0:
                bad += 1
                if bad < 12:
                    print(f"app {a} conv {i}: pool vs ref max|d| {d:.4g} (ref |x| {ref.abs().max():.3g})", flush=True)
    print(f"conv states: {bad} of {4 * len(pools)} differ", flush=True)
    WB = store.WB
    jq = Q // 64
    badr = 0
    for (t, pool, c), hp in zip(store.rings, rings_pool):
        ht = host(t)
        per = c.per_slot
        for j in range(max(0, jq - WB), jq):
            ref = ht[3 * per + j % per]
            got = hp[k * WB + (j - (jq - WB))]
            d = (ref - got).abs().max().item()
            if d > 0:
                badr += 1
                if badr < 6:
                    print(f"ring block {j}: max|d| {d:.4g}", flush=True)
    print(f"ring blocks: {badr} differ", flush=True)
    print("== dbg done", flush=True)
    from models.autoports.thinkingmachines_inkling_2_mini.tt.clean_exit import close

    close(mesh)


if __name__ == "__main__":
    main()

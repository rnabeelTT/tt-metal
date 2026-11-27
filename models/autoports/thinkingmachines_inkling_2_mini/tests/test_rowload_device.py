# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""BatchedDecoder.load_slot through tt/rowload.py (one program) vs the ttnn slice / concat / copy path: every decode
conv-state tensor bitwise equal (all rows: the untouched ones too), from the same random starting states, for several
slots; and the time of each.
env: MESH (1), B (8), SLOTS (0,3,7)"""

import os
import time

import torch

import ttnn
from models.autoports.thinkingmachines_inkling_2_mini.tt import batched as BT
from models.autoports.thinkingmachines_inkling_2_mini.tt.batched import BatchedDecoder
from models.autoports.thinkingmachines_inkling_2_mini.tt.model import Inkling2Model


def main():
    N = int(os.environ.get("MESH", "1"))
    B = int(os.environ.get("B", "8"))
    slots = [int(s) for s in os.environ.get("SLOTS", "0,3,7").split(",")]
    if N > 1:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, N))
    m = Inkling2Model(mesh, max_seq_len=512, expert_dtype=ttnn.bfloat8_b, kv_dtype=ttnn.bfloat16, prefill_chunk=256,
                      slots=B)
    dec = BatchedDecoder(m, B)
    states = [t for ba in dec.apps for c in (ba.k, ba.v, ba.attn, ba.mlp) for t in c.s]
    print(f"{len(states)} decode state tensors", flush=True)
    dev = (lambda t: ttnn.get_device_tensors(t)) if N > 1 else (lambda t: [t])  # noqa: E731
    host = lambda t: [ttnn.to_torch(d) for d in dev(t)]  # noqa: E731 (per die)
    g = torch.Generator().manual_seed(0)
    init = [torch.randn(tuple(dev(t)[0].shape), generator=g) for t in states]
    rep = ttnn.ReplicateTensorToMesh(mesh) if N > 1 else None

    def reset():
        for t, v in zip(states, init):
            ttnn.copy(ttnn.from_torch(v, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep), t)

    ids = torch.arange(1000, 1000 + 192)
    ok = True
    for b in slots:
        m.reset()
        P = 128 + 16 * b
        m.forward(torch.cat([ids[:P], torch.zeros((-P) % 64, dtype=ids.dtype)]), 0, valid=P, slot=b)  # (lane 0's conv states: this slot's prompt)
        res = {}
        for mode in (False, True):
            BT._ROWLOAD = mode
            reset()
            dec.load_slot(b)  # (compile)
            reset()
            ttnn.synchronize_device(mesh)
            t0 = time.perf_counter()
            dec.load_slot(b)
            ttnn.synchronize_device(mesh)
            dt = time.perf_counter() - t0
            res[mode] = ([host(t) for t in states], dt)
        same = all(torch.equal(x, y) for a, c in zip(res[False][0], res[True][0]) for x, y in zip(a, c))
        changed = sum(not torch.equal(a[0], v) for a, v in zip(res[True][0], init))
        ok &= same
        print(f"slot {b}: bitwise {'equal' if same else 'DIFFERENT'} ({changed}/{len(states)} tensors loaded); "
              f"old {res[False][1] * 1e3:.1f} ms, rowload {res[True][1] * 1e3:.2f} ms", flush=True)
    ttnn.close_mesh_device(mesh)
    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()

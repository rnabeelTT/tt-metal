# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Host -> device bandwidth of the prefill expert streaming pattern (hybrid/stream.py upload) on the QB2 mesh:
pinned zero-copy writes (hybrid/pin) of N experts per layer step, one write per weight per expert vs fewer, larger
writes, on CQ0 / CQ1, and ttnn.copy_host_to_device_tensor for reference. Prints GB/s over all dies.
usage: run_in_image.sh pcie_bench.py <out dir>   env: PB_EXPERTS (72), PB_REPS (5)"""

import os
import sys
import time

import numpy as np
import torch

import ttnn
from hybrid.pin import Pinx

N = int(os.environ.get("PB_EXPERTS", "72"))
REPS = int(os.environ.get("PB_REPS", "5"))
H, NL = 3072, 384  # MiniMax-M2.7 hidden, per-die intermediate (tp 4)

mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), num_command_queues=2)
tp = len(mesh.get_device_ids())
kw = dict(layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
gu = ttnn.from_torch(
    torch.zeros(1, N, H, tp * 2 * NL), dtype=ttnn.bfloat4_b, mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1), **kw
)
dn = ttnn.from_torch(
    torch.zeros(1, N, tp * NL, H), dtype=ttnn.bfloat4_b, mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-2), **kw
)
px = Pinx(mesh)
dg, sg = px.dst(gu)
dd, sd = px.dst(dn)
ng, nd = sg // N, sd // N
rid, buf = px.region(N * tp * (ng + nd))
buf[:] = np.random.randint(0, 255, size=buf.shape, dtype=np.uint8)
total = N * tp * (ng + nd)
print(f"experts {N}: per die gate/up {ng / 1e6:.2f} MB, down {nd / 1e6:.2f} MB; {total / 1e9:.2f} GB per layer step")


def per_expert(cq, order="expert"):
    for j in range(N):
        base = j * tp * (ng + nd)
        px.write_at(dg, rid, [base + d * ng for d in range(tp)], j * ng, ng, cq_id=cq)
        px.write_at(dd, rid, [base + tp * ng + d * nd for d in range(tp)], j * nd, nd, cq_id=cq)


# one contiguous region per die for each weight: the whole staging in two writes
rid2, buf2 = px.region(tp * (sg + sd))


def whole(cq):
    px.write_at(dg, rid2, [d * sg for d in range(tp)], 0, sg, cq_id=cq)
    px.write_at(dd, rid2, [tp * sg + d * sd for d in range(tp)], 0, sd, cq_id=cq)


def chunked(cq, per):
    for j in range(0, N, per):
        n = min(per, N - j)
        px.write_at(dg, rid2, [d * sg + j * ng for d in range(tp)], j * ng, n * ng, cq_id=cq)
        px.write_at(dd, rid2, [tp * sg + d * sd + j * nd for d in range(tp)], j * nd, n * nd, cq_id=cq)


def timed(name, fn, nbytes=total):
    fn()
    ttnn.synchronize_device(mesh)
    ts = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        ttnn.synchronize_device(mesh)
        ts.append((t1 - t0, time.perf_counter() - t0))
    enq, wall = np.median([t[0] for t in ts]), np.median([t[1] for t in ts])
    print(f"BW {name:28s} wall {wall * 1e3:7.2f} ms  enqueue {enq * 1e3:7.2f} ms  {nbytes / wall / 1e9:6.1f} GB/s")
    sys.stdout.flush()


timed("per expert CQ0", lambda: per_expert(0))
timed("per expert CQ1", lambda: per_expert(1))
timed("whole staging CQ0", lambda: whole(0))
timed("whole staging CQ1", lambda: whole(1))
for per in (4, 8, 18):
    timed(f"{per} experts per write CQ1", lambda per=per: chunked(1, per))
timed("half per CQ", lambda: (chunked(0, N // 2) if N > 1 else None))

host = ttnn.from_torch(
    torch.zeros(1, N, H, tp * 2 * NL), dtype=ttnn.bfloat4_b, mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1)
)
timed("copy_host_to_device gate/up", lambda: ttnn.copy_host_to_device_tensor(host, gu), nbytes=tp * sg)
px.release()
ttnn.close_mesh_device(mesh)

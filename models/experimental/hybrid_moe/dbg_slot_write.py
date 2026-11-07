# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Device check for the FreeToken-style fetch: write one expert's weights into a RUNTIME-chosen slot of a fused
[1, C, H, W] bfp4 expert tensor (TP-sharded) from a staging buffer, inside a trace: paged_fill_cache on a [C, 1, H, W]
view with a [1, 1] device page table holding the slot. Replays with different slots / contents; every slot is checked.
usage (in the image): python dbg_slot_write.py <out dir>"""

import sys
import time

import torch

import ttnn


def main():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), l1_small_size=24576, trace_region_size=16_000_000)
    ok = True
    try:
        C, H, Wl, tp = 64, 3072, 768, 4
        sh = ttnn.ShardTensorToMesh(mesh, dim=-1)
        kw = dict(layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        g = torch.Generator().manual_seed(0)
        base = torch.randn(1, C, H, tp * Wl, generator=g) * 0.02
        W = ttnn.from_torch(base, dtype=ttnn.bfloat4_b, device=mesh, mesh_mapper=sh, **kw)
        ref = ttnn.to_torch(W, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=-1)).float()  # bfp4-rounded
        view = ttnn.reshape(W, [C, 1, H, Wl])  # per-device shape
        stage = ttnn.from_torch(torch.zeros(1, 1, H, tp * Wl), dtype=ttnn.bfloat4_b, device=mesh, mesh_mapper=sh, **kw)
        pt = ttnn.from_torch(
            torch.zeros(1, 1, dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        def body():
            ttnn.experimental.paged_fill_cache(view, stage, pt, batch_idx=0)

        def stage_write(slot, content):
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(
                    content,
                    dtype=ttnn.bfloat4_b,
                    layout=ttnn.TILE_LAYOUT,
                    mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1),
                ),
                stage,
            )
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(
                    torch.tensor([[slot]], dtype=torch.int32),
                    dtype=ttnn.int32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
                ),
                pt,
            )

        expect = ref.clone()

        def write(slot, traced):
            content = torch.randn(1, 1, H, tp * Wl, generator=g) * 0.02
            rounded = ttnn.to_torch(ttnn.from_torch(content, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT)).float()
            stage_write(slot, content)
            if traced:
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
            else:
                body()
            expect[0, slot] = rounded[0, 0]

        write(5, False)
        ttnn.synchronize_device(mesh)
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        body()
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
        for slot in (0, 63, 17, 5, 40):
            write(slot, True)
        got = ttnn.to_torch(W, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=-1)).float()
        bad = [s for s in range(C) if not torch.equal(got[0, s], expect[0, s])]
        print(f"RESULT slot writes: mismatched slots {bad} (of {C})", flush=True)
        ok = not bad
        # per-die host shards (built without a device, e.g. loaded from disk) combined into one mesh host tensor
        content = torch.randn(1, 1, H, tp * Wl, generator=g) * 0.02
        shards = [
            ttnn.from_torch(
                content[..., d * Wl : (d + 1) * Wl].contiguous(), dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT
            )
            for d in range(tp)
        ]
        agg = ttnn.from_host_shards(shards, ttnn.MeshShape(1, tp))
        ttnn.dump_tensor("/poc/cache/agg_chk.tensorbin", agg)  # the build_dev_experts.py path: dump, load, write
        agg = ttnn.load_tensor("/poc/cache/agg_chk.tensorbin")
        ttnn.copy_host_to_device_tensor(agg, stage)
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(
                torch.tensor([[9]], dtype=torch.int32),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            ),
            pt,
        )
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
        got = ttnn.to_torch(W, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=-1)).float()
        want = torch.cat([ttnn.to_torch(t).float() for t in shards], dim=-1)
        agg_ok = torch.equal(got[0, 9], want[0, 0])
        print(f"RESULT aggregated per-die shards write: {'ok' if agg_ok else 'MISMATCH'}", flush=True)
        ok = ok and agg_ok
        # timing: staging write (PCIe) + replay, one 2 MB/die expert gate|up
        content = torch.randn(1, 1, H, tp * Wl) * 0.02
        host = ttnn.from_torch(
            content, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1)
        )
        ts = []
        for i in range(30):
            t0 = time.perf_counter()
            ttnn.copy_host_to_device_tensor(host, stage)
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
            ts.append(time.perf_counter() - t0)
        ms = 1e3 * sorted(ts)[15]
        mb = H * tp * Wl * 0.5625 / 1e6
        print(f"RESULT gate|up write ({mb:.1f} MB over 4 dies): {ms:.3f} ms -> {mb / ms:.1f} GB/s", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

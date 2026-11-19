# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host time of the prefill MoE input readback ([2048, 4096] bf16 row-major from die 0, as hybrid/patch.py _pf_pre /
_pf_plan read it every layer): ttnn.from_device(blocking=False) + ttnn.to_torch (a fresh host buffer and a copy per
call) vs ttnn.copy_device_to_host_tensor into one persistent host tensor that borrows a torch buffer (no allocation,
no copy). Checks the values match.

usage (in the image): tools/run_dev.sh python models/autoports/thinkingmachines_inkling_small/tools/readback_bench.py
"""

import time

import torch

import ttnn


def main():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), l1_small_size=24576, trace_region_size=64_000_000)
    try:
        T, H = 2048, 4096
        src = torch.randn(1, 1, T, H).to(torch.bfloat16)
        x = ttnn.from_torch(src, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh,
                            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        x0 = ttnn.get_device_tensors(x)[0]
        ttnn.synchronize_device(mesh)

        res = []
        for _ in range(6):
            t0 = time.perf_counter()
            h = ttnn.from_device(x0, blocking=False)
            t1 = time.perf_counter()
            ttnn.synchronize_device(mesh)
            t2 = time.perf_counter()
            out = ttnn.to_torch(h).reshape(T, H)
            t3 = time.perf_counter()
            res.append((t1 - t0, t3 - t2))
        a = min(r[0] for r in res) * 1e3, min(r[1] for r in res) * 1e3
        print(f"from_device(non-blocking) {a[0]:.2f} ms host + to_torch {a[1]:.2f} ms; equal {torch.equal(out, src.reshape(T, H))}",
              flush=True)

        buf = torch.empty(1, 1, T, H, dtype=torch.bfloat16)
        try:
            hb = ttnn.from_torch(buf, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)  # host, borrows buf
            res = []
            for _ in range(6):
                buf.zero_()
                t0 = time.perf_counter()
                ttnn.copy_device_to_host_tensor(x0, hb, blocking=False)
                t1 = time.perf_counter()
                ttnn.synchronize_device(mesh)
                res.append(t1 - t0)
            same = torch.equal(buf, src)
            print(f"copy_device_to_host_tensor into a borrowed buffer: {min(res) * 1e3:.2f} ms host; "
                  f"buffer filled correctly (zero-copy) {same}", flush=True)
        except Exception as e:  # noqa: BLE001
            print("borrowed-buffer readback failed:", repr(e)[:300], flush=True)

        # a persistent host tensor (pinned once by the read's cache, then the device writes it directly) viewed by
        # torch through DLPack: no allocation, no zero-fill, no copy per read
        try:
            ht = ttnn.allocate_tensor_on_host(ttnn.Shape([1, 1, T, H]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, x0.device())
            shard = ht.host_buffer().get_shard(ttnn.MeshCoordinate(0, 0))
            view = torch.from_dlpack(shard).view(torch.bfloat16).reshape(T, H)
            res = []
            for i in range(6):
                view.zero_()
                t0 = time.perf_counter()
                ttnn.copy_device_to_host_tensor(x0, ht, blocking=False)
                t1 = time.perf_counter()
                ttnn.synchronize_device(mesh)
                res.append(t1 - t0)
                if i == 0:
                    first = t1 - t0
            print(f"persistent host tensor + DLPack view: {min(res) * 1e3:.2f} ms host (first {first * 1e3:.2f}); "
                  f"view filled correctly {torch.equal(view, src.reshape(T, H))}", flush=True)
            # the same read ordered by an event, as the prefill does (non-blocking read, event, event_synchronize)
            view.zero_()
            ttnn.copy_device_to_host_tensor(x0, ht, blocking=False)
            ev = ttnn.record_event(mesh, 0)
            ttnn.event_synchronize(ev)
            print(f"event-ordered read filled correctly {torch.equal(view, src.reshape(T, H))}", flush=True)
        except Exception as e:  # noqa: BLE001
            print("persistent host tensor readback failed:", repr(e)[:300], flush=True)
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()

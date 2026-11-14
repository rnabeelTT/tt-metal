# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device test of tt/sconv.ShortConv on the (1, 4) mesh (channels sharded, as the model runs it) against the fp32 host
oracle (tt/sconv.host_reference, checked against HF by tests/test_sconv_host.py): decode steps, a multi-row segment
with a partial commit (spec verify: only the accepted prefix advances the state), then decode again.

usage (in the image): tools/run_dev.sh python models/autoports/thinkingmachines_inkling_small/tests/test_sconv_device.py
"""

import sys

import torch

import ttnn
from models.autoports.thinkingmachines_inkling_small.tt.sconv import ShortConv, host_reference


def main():
    torch.manual_seed(0)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), l1_small_size=24576)
    fails = 0
    try:
        for C in (4096, 1024):  # the hidden-size convs (1024 per die) and K/V (256 per die)
            w = torch.randn(C, 4) * 0.5
            conv = ShortConv(mesh, w, ttnn.ShardTensorToMesh(mesh, dim=-1), C // 4)
            seq = torch.randn(40, C)
            state = None
            ref, got = [], []
            up = lambda t: ttnn.from_torch(t.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,  # noqa
                                           device=mesh, mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1))
            down = lambda t: ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=-1)).float()  # noqa
            xb = seq.to(torch.bfloat16).float()  # what the device sees
            # 1) 10 decode steps
            for t in range(10):
                y, state = host_reference(xb[t : t + 1], w, state)
                ref.append(y)
                got.append(down(conv.decode(up(xb[t].reshape(1, 1, 1, -1)))).reshape(1, -1))
            # 2) 6 rows (verify), accept 4: state advances past rows 0..3 only; outputs for all 6 rows
            y6, _ = host_reference(xb[10:16], w, state)
            yd = down(conv.rows(up(xb[10:16].reshape(1, 1, 6, -1)), commit_all=False)).reshape(6, -1)
            conv.commit_rows(4)
            ref.append(y6)
            got.append(yd)
            _, state = host_reference(xb[10:14], w, state)
            # 3) decode resumes at the token after the accepted prefix (xb[16] stands in for it)
            for t in range(16, 24):
                y, state = host_reference(xb[t : t + 1], w, state)
                ref.append(y)
                got.append(down(conv.decode(up(xb[t].reshape(1, 1, 1, -1)))).reshape(1, -1))
            # 4) a prefill-style segment of 16 rows, committed whole, then decode
            y16, state = host_reference(xb[24:40], w, state)
            ref.append(y16)
            got.append(down(conv.rows(up(xb[24:40].reshape(1, 1, 16, -1)))).reshape(16, -1))
            R, G = torch.cat(ref), torch.cat(got)
            err = (R - G).abs().max().item()
            rel = err / R.abs().max().item()
            ok = rel < 1e-2  # bf16 output rounding of an fp32 result
            fails += not ok
            print(f"{'OK  ' if ok else 'FAIL'} C={C}: max abs err {err:.3e} (rel {rel:.2e}) over {R.shape[0]} rows")
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    print("ALL OK" if not fails else f"{fails} FAILED")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()

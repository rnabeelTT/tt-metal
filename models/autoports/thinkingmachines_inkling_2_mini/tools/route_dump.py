# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Real-text expert routing per application (M3 of the decode plan): eager prefill of the held-out texts, every MoE
route()'s top-8 ids recorded -> OUT {name: uint8 [48 apps, T, 8]}, then the expert-union statistics a batched decode /
verify pass reads (unique experts per application over B rows from B different texts, or R consecutive tokens).

usage: I2M_PF_TRACE=0 MESH=4 tools/run_dev.sh python .../tools/route_dump.py   env: OUT (/cache/route_ids.pt), CHUNK
"""

import os

import torch

import ttnn
from models.autoports.thinkingmachines_inkling_2_mini.tt import moe as moe_mod
from models.autoports.thinkingmachines_inkling_2_mini.tt.model import Inkling2Model

REC = []


def main():
    cls = moe_mod.MoE
    orig = cls.route

    def route(self, x):
        w, ids = orig(self, x)
        REC.append(ttnn.to_torch(ids, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh, dim=0))[0, 0].to(torch.int32))
        return w, ids

    cls.route = route
    ref = torch.load(os.environ.get("REF", "/cache/fixtures/ref_bf16.pt"))
    N = int(os.environ.get("MESH", "1"))
    if N > 1:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, N))
    CH = int(os.environ.get("CHUNK", "256"))
    m = Inkling2Model(mesh, max_seq_len=1024 + CH, expert_dtype=ttnn.bfloat8_b, prefill_chunk=CH, kv_dtype=ttnn.bfloat16)
    out = {}
    for name in ref:
        ids = ref[name]["ids"]
        T = len(ids)
        m.reset()
        per = []
        for p0 in range(0, T, CH):
            chunk = ids[p0:p0 + CH]
            n = len(chunk)
            if n % 64:
                chunk = torch.cat([chunk, torch.zeros((-n) % 64, dtype=chunk.dtype)])
            REC.clear()
            m.forward(chunk, p0, valid=n)
            assert len(REC) == 48, len(REC)
            per.append(torch.stack([r[:n, :8] for r in REC]))  # [48, n, 8]
        out[name] = torch.cat(per, 1).to(torch.uint8)
        print(f"{name}: {tuple(out[name].shape)}", flush=True)
    torch.save(out, os.environ.get("OUT", "/cache/route_ids.pt"))
    ttnn.close_mesh_device(mesh)
    print("== done")


if __name__ == "__main__":
    main()

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device vs host speculative acceptance in one generator, same prompts: bench_card-style random-token prompts and real
text, greedy (and T=1 with ACC_T), NEW tokens each; per prompt the passes / drafts kept / stream identity.
env: MESH (1), NEW (128), NP (6), ACC_T (0)"""

import os
import random
import types

import torch

os.environ.setdefault("I2M_SPEC", "3")
os.environ.setdefault("I2M_SPEC_MIN", "0")
import ttnn  # noqa: E402
from models.autoports.thinkingmachines_inkling_2_mini.tt import generator_vllm as G  # noqa: E402


def main():
    N, NEW, NP = int(os.environ.get("MESH", "1")), int(os.environ.get("NEW", "128")), int(os.environ.get("NP", "6"))
    T = float(os.environ.get("ACC_T", "0"))
    if N > 1:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, N), trace_region_size=600 << 20)
    gen = G.Inkling2ForCausalLM.initialize_vllm_model(None, mesh, 4, max_seq_len=4096)
    gen.warmup_model_prefill()
    gen.warmup_model_decode()
    prompts = [torch.tensor([random.Random(s).randint(1000, 150000) for _ in range(128)]) for s in range(NP)]
    texts = torch.load("/cache/fixtures/dspark_taps.pt")["texts"]
    prompts += [t["ids"][:200].long() for t in texts.values()][:3]
    sp = types.SimpleNamespace(temperature=[T], top_k=[-1], top_p=[1.0], seed=[1234])

    def run(ids):
        lg = gen.prefill_forward(ids.reshape(1, -1), None, None, [int(ids.numel())], empty_slots=[0])
        out, p = [int(lg[0, 0].argmax())], int(ids.numel())
        st0 = list(gen.sp_stats)
        while len(out) < NEW:
            o = gen.decode_forward(tokens=torch.tensor([[out[-1]]]), start_pos=torch.tensor([p]), sampling_params=sp)
            new = [int(x) for x in o[0] if int(x) >= 0]
            out += new
            p += len(new)
        return out[:NEW], [a - b for a, b in zip(gen.sp_stats, st0)]

    dacc = gen.dacc
    tot = {0: [0, 0], 1: [0, 0]}
    for i, ids in enumerate(prompts):
        res = {}
        for dev in (1, 0, 1):
            gen.dacc = dacc if dev else None
            res.setdefault(dev, []).append(run(ids))
        a, b = res[1][0], res[0][0]
        for dev in (0, 1):
            tot[dev][0] += res[dev][0][1][0]
            tot[dev][1] += res[dev][0][1][1]
        agree = next((j for j in range(NEW) if a[0][j] != b[0][j]), NEW)
        print(f"prompt {i}: device passes/kept {a[1]} (again {res[1][1][1]}), host {b[1]}; streams agree {agree} / {NEW}",
              flush=True)
    for dev in (0, 1):
        print(f"{'device' if dev else 'host'}: {tot[dev][1] / max(tot[dev][0], 1):.3f} drafts kept per pass over "
              f"{tot[dev][0]} passes", flush=True)
    ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()

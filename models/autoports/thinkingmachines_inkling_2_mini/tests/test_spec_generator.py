# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Speculative decoding through the vLLM generator (I2M_SPEC), driven the way the TT plugin drives it (prefill_forward,
then one decode_forward per token with on-device sampling), without a server:
  1. greedy, one request: the tokens of a plain run of the same generator at the verify width's numerics match until
     the first near-tie; reported with the passes and drafts kept;
  2. a second request arriving mid-stream: the first one leaves speculative decode rewound to what vLLM holds and both
     continue (their next tokens must be the plain greedy continuations of what each was handed);
  3. sampled (temperature 0.8): runs, reports drafts kept per pass.
env: MESH (1), NEW (64)"""

import os
import types

import torch

os.environ.setdefault("I2M_SPEC", "3")
import ttnn  # noqa: E402
from models.autoports.thinkingmachines_inkling_2_mini.tt import generator_vllm as G  # noqa: E402


def main():
    N, NEW = int(os.environ.get("MESH", "1")), int(os.environ.get("NEW", "64"))
    if N > 1:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, N), trace_region_size=600 << 20)
    gen = G.Inkling2ForCausalLM.initialize_vllm_model(None, mesh, 4, max_seq_len=4096)
    gen.warmup_model_prefill()
    gen.warmup_model_decode()
    texts = torch.load("/cache/fixtures/dspark_taps.pt")["texts"]
    prompts = [t["ids"][:200].long() for t in texts.values()]

    def sp(temp, n=1):
        return types.SimpleNamespace(temperature=[temp] * n, top_k=[-1] * n, top_p=[1.0] * n, seed=[-1] * n)

    def prefill(ids, row):
        toks = ids.reshape(1, -1)
        lg = gen.prefill_forward(toks, None, None, [int(ids.numel())], empty_slots=[row])
        return int(lg[0, 0].argmax())

    def run(ids, temp, n_new, spec=True):
        t = prefill(ids, 0)
        out, p = [t], int(ids.numel())
        while len(out) < n_new:
            o = gen.decode_forward(tokens=torch.tensor([[out[-1]]]), start_pos=torch.tensor([p]), sampling_params=sp(temp))
            new = [int(x) for x in o[0] if int(x) >= 0]  # (a speculative step: 1 .. d + 1 tokens, -1 padded)
            out += new
            p += len(new)
        return out[:n_new]

    # 1. greedy: spec vs plain (spec off: the generator's own plain path)
    for i, ids in enumerate(prompts):
        a = run(ids, 0.0, NEW, spec=True)
        st = list(gen.sp_stats)
        saved, gen.dr = gen.dr, None  # (plain decode: no speculative state)
        b = run(ids, 0.0, NEW, spec=False)
        gen.dr = saved
        agree = next((j for j in range(NEW) if a[j] != b[j]), NEW)
        print(f"greedy prompt {i}: streams agree for {agree} / {NEW}; passes so far {st[0]}, drafts kept {st[1]}",
              flush=True)

    # 2. a second request arrives after 20 tokens of the first
    ids0, ids1 = prompts[0], prompts[1]
    t = prefill(ids0, 0)
    a, p = [t], int(ids0.numel())
    while len(a) < 20:
        o = gen.decode_forward(tokens=torch.tensor([[a[-1]]]), start_pos=torch.tensor([p]), sampling_params=sp(0.0))
        new = [int(x) for x in o[0] if int(x) >= 0]
        a += new
        p += len(new)
    b = [prefill(ids1, 1)]
    q = int(ids1.numel())
    for _ in range(10):
        o = gen.decode_forward(tokens=torch.tensor([[a[-1]], [b[-1]]]), start_pos=torch.tensor([p, q]),
                               sampling_params=sp(0.0, 2))
        a.append(int(o[0, 0]))
        b.append(int(o[1, 0]))
        p, q = p + 1, q + 1
    # reference: request 0 alone, plain greedy, over the same handed-out prefix
    saved, gen.dr = gen.dr, None
    ref = run(ids0, 0.0, len(a), spec=False)
    gen.dr = saved
    agree = next((j for j in range(len(a)) if a[j] != ref[j]), len(a))
    h = len(a) - 10
    print(f"handover: request 0 agrees with plain greedy for {agree} / {len(a)} tokens (handover at {h})", flush=True)

    # 3. sampled
    st0 = list(gen.sp_stats)
    run(prompts[2], 0.8, NEW, spec=True)
    st = [x - y for x, y in zip(gen.sp_stats, st0)]
    print(f"sampled T=0.8: {st[1] / max(st[0], 1):.2f} drafts kept per pass over {st[0]} passes", flush=True)
    ttnn.close_mesh_device(mesh)
    print("DONE")


if __name__ == "__main__":
    main()

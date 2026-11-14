# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Is spec-decode verify lossless? Greedy speculative decoding must reproduce plain greedy decode; spec_bench showed
k>0 text diverging from k=0. This separates near-tie numerics from a state bug, on the device, same model build:

  A. plain:    greedy decode, one row per step (R=1, commit 1): tokens t_1..t_N, logits L_A[i] (top-2 margin kept)
  B. rows:     the same token stream teacher-forced in verify groups of R rows, all committed: logits L_B[i]
               -> a row-count (matmul-shape) numerics effect only
  C. rejects:  groups of R rows starting at every position, only row 0 committed (the other R-1 rows are "rejected
               drafts" whose KV / conv-state writes must leave no trace): logits L_C[i]
               -> rejected-row state leaking into later steps

Per mode: argmax agreement with A, the A top-2 logit margin at every disagreement, and logits PCC / max |diff| vs A.
A disagreement at a margin far above the PCC noise means a bug, not numerics.

usage (in the image, hybrid env as tools/eval_hybrid.sh, e.g. TOOL=rows_consistency.py TOOL_ARGS="3 96"):
       .../tools/rows_consistency.py [R=3] [new tokens=96]  -> prints a summary, writes /cache/rows_consistency.json
"""

import json
import os
import sys

import torch

if os.environ.get("HYB_ENABLE") == "1":
    sys.path.insert(0, os.environ.get("HYB_POC_DIR", "/opt/tt-metal/models/experimental/hybrid_moe"))
    os.environ.setdefault("HYB_PORT", "thinkingmachines_inkling_small")
    os.environ.setdefault("HYB_FUSED", "0")
    import hybrid

    hybrid.install()

import ttnn  # noqa: E402
from models.autoports.thinkingmachines_inkling_small.tt.model import create_model  # noqa: E402


# the spec_bench prompts (copied: importing spec_bench would run its hybrid install a second time)
PROMPTS = [
    [{"role": "user", "content": "Write a Python function that returns the n-th Fibonacci number iteratively, with a "
                                 "docstring and two doctests."}],
    [{"role": "user", "content": "Natalia sold clips to 48 of her friends in April, and then she sold half as many "
                                 "clips in May. How many clips did Natalia sell altogether in April and May? Explain "
                                 "step by step."}],
    [{"role": "user", "content": "Explain in three short paragraphs how a hash map handles collisions."}],
]


def main():
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    NEW = int(sys.argv[2]) if len(sys.argv) > 2 else 96
    out_path = os.environ.get("ROWS_OUT", "/cache/rows_consistency.json")
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), l1_small_size=24576, trace_region_size=64_000_000)
    try:
        args, model = create_model(mesh, max_seq_len=8192)
        tok = args.tokenizer
        rep = ttnn.ReplicateTensorToMesh(mesh)
        nblk = model.layers[0].attention.max_blocks
        pt_host = torch.arange(nblk, dtype=torch.int32)
        pt = ttnn.from_torch(pt_host.reshape(1, -1), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh,
                             mesh_mapper=rep)
        V = int(args.unpadded_vocab_size)

        def dev_ids(ids):
            return ttnn.from_torch(torch.tensor(ids, dtype=torch.int32).reshape(1, len(ids)), dtype=ttnn.uint32,
                                   layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, mesh_mapper=rep)

        def dev_pos(p0, n):
            return ttnn.from_torch(torch.arange(p0, p0 + n, dtype=torch.int32), dtype=ttnn.int32,
                                   layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, mesh_mapper=rep)

        def rows_logits(ids, p0):
            """Rows mode over ids at p0.. (nothing committed) -> fp32 logits [R, V] on the host."""
            n = len(ids)
            x = model.embd(dev_ids(ids))
            x = ttnn.reshape(x, (1, 1, n, x.shape[-1]))
            pos = dev_pos(p0, n)
            for layer in model.layers:
                x = layer.forward_rows(x, pos, pt, commit_all=False)
            lg = model._lm_head(model._final_norm_decode(x))
            return ttnn.to_torch(ttnn.get_device_tensors(lg)[0]).float().reshape(-1, lg.shape[-1])[:n, :V].clone()

        def commit(n):
            for layer in model.layers:
                layer.commit_rows(n)

        def start(prompt):
            """Fresh sequence, prompt committed except its last token; -> (last prompt token, its position)."""
            model.reset_sequence_state()
            P = len(prompt)
            pre = (P - 1) // 64 * 64
            if pre:
                model.prefill_tokens(torch.tensor(prompt[:pre]), pt_host, logits_rows="last")
            rest = prompt[pre:]
            for g in range(0, len(rest) - 1, 8):
                chunk = rest[g : min(g + 8, len(rest) - 1)]
                rows_logits(chunk, pre + g)
                commit(len(chunk))
            return rest[-1], P - 1

        def compare(name, LA, LX, margins):
            a, x = LA.argmax(-1), LX.argmax(-1)
            dis = (a != x).nonzero().flatten().tolist()
            pcc = [torch.corrcoef(torch.stack([LA[i], LX[i]]))[0, 1].item() for i in range(len(LA))]
            mx = (LA - LX).abs().max(-1).values
            r = {"agree": 1 - len(dis) / len(a), "n": len(a), "disagree_pos": dis,
                 "margin_at_disagree": [round(margins[i], 4) for i in dis],
                 "pcc_min": min(pcc), "pcc_mean": sum(pcc) / len(pcc), "maxdiff_mean": mx.mean().item(),
                 "maxdiff_max": mx.max().item()}
            print(f"  {name}: agree {r['agree']:.3f} ({len(dis)}/{len(a)} differ) margins at disagreement "
                  f"{r['margin_at_disagree']} | logits PCC min {r['pcc_min']:.5f} mean {r['pcc_mean']:.5f} | "
                  f"max|diff| mean {r['maxdiff_mean']:.3f} max {r['maxdiff_max']:.3f}", flush=True)
            return r

        report = {"R": R, "new": NEW, "prompts": []}
        for msgs in PROMPTS:
            prompt = tok.encode(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                        reasoning_effort="none"), add_special_tokens=False)
            # A: plain greedy decode
            x, p = start(prompt)
            toks, LA = [], []
            for _ in range(NEW):
                lg = rows_logits([x], p)[0]
                commit(1)
                LA.append(lg)
                x, p = int(lg.argmax()), p + 1
                toks.append(x)
            LA = torch.stack(LA)
            top2 = LA.topk(2, -1).values
            margins = (top2[:, 0] - top2[:, 1]).tolist()
            stream = [start(prompt)[0]] + toks  # row i's input token; its logits predict toks[i]
            x0, p0 = start(prompt)
            # B: teacher-forced groups of R rows, all committed
            LB = []
            for g in range(0, NEW, R):
                ids = stream[g : min(g + R, NEW)]
                LB.append(rows_logits(ids, p0 + g))
                commit(len(ids))
            LB = torch.cat(LB)
            # C: R rows at every position, only row 0 committed
            start(prompt)
            LC = []
            for i in range(NEW):
                ids = stream[i : min(i + R, NEW + 1)]
                LC.append(rows_logits(ids, p0 + i)[0:1])
                commit(1)
            LC = torch.cat(LC)
            print(f"prompt {len(report['prompts'])}: {tok.decode(toks[:40])!r}", flush=True)
            report["prompts"].append({"text": tok.decode(toks), "rows": compare("B rows", LA, LB, margins),
                                      "rejects": compare("C rejects", LA, LC, margins)})
        json.dump(report, open(out_path, "w"), indent=1)
        print(f"wrote {out_path}")
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()

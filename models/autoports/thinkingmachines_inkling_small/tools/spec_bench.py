# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Greedy generation with Inkling's native MTP speculative decoding on the device (eager, through the hybrid overlay
when HYB_ENABLE=1): measures accepted tokens per verify pass and tokens/s on GENERATED text (the drafter comparison
the teacher-forced CPU numbers can only approximate), for k = 0 (plain decode) .. K drafts.

Per step (k drafts):
  verify:  the target on rows [x, d_1 .. d_k] at positions p+1 .. p+k+1 (rows mode, conv states held), argmax a_0 .. a_k
  accept:  n = leading j with d_{j+1} == a_j; emit d_1 .. d_n, a_n; target conv states commit the n + 1 rows
  draft:   MTP depth 0 in rows mode over the n + 1 accepted rows (inputs: the target's normed hidden of row j, the token
           after it), its last row's logits -> d'_1; depths 1 .. k-1 chain one row each -> d'_2 .. d'_k
Plain decode (k = 0) is the same loop with no drafts (one row per step).

usage (in the image, hybrid env as tools/eval_hybrid.sh): .../tools/spec_bench.py [K=4] [new tokens=128]
"""

import os
import sys
import time

import torch

if os.environ.get("HYB_ENABLE") == "1":
    sys.path.insert(0, os.environ.get("HYB_POC_DIR", "/opt/tt-metal/models/experimental/hybrid_moe"))
    os.environ.setdefault("HYB_PORT", "thinkingmachines_inkling_small")
    os.environ.setdefault("HYB_FUSED", "0")
    import hybrid

    hybrid.install()

import ttnn  # noqa: E402
from models.autoports.thinkingmachines_inkling_small.tt.inkling_mtp import InklingMtp  # noqa: E402
from models.autoports.thinkingmachines_inkling_small.tt.model import create_model  # noqa: E402
from models.tt_transformers.tt.common import Mode  # noqa: E402

PROMPTS = [
    [{"role": "user", "content": "Write a Python function that returns the n-th Fibonacci number iteratively, with a "
                                 "docstring and two doctests."}],
    [{"role": "user", "content": "Natalia sold clips to 48 of her friends in April, and then she sold half as many "
                                 "clips in May. How many clips did Natalia sell altogether in April and May? Explain "
                                 "step by step."}],
    [{"role": "user", "content": "Explain in three short paragraphs how a hash map handles collisions."}],
]


def main():
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    NEW = int(sys.argv[2]) if len(sys.argv) > 2 else 128
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), l1_small_size=24576, trace_region_size=64_000_000)
    try:
        args, model = create_model(mesh, max_seq_len=8192)
        mtp = InklingMtp(model, max(K, 1))
        tok = args.tokenizer
        rep = ttnn.ReplicateTensorToMesh(mesh)
        nblk = model.layers[0].attention.max_blocks
        pt_host = torch.arange(nblk, dtype=torch.int32)
        pt = ttnn.from_torch(pt_host.reshape(1, -1), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh,
                             mesh_mapper=rep)
        V = int(args.unpadded_vocab_size)
        eos = {int(args.inkling.get("eos_token_id", 200006)) if isinstance(args.inkling.get("eos_token_id"), int)
               else 200006}

        def dev_ids(ids):
            n = len(ids)
            return ttnn.from_torch(torch.tensor(ids, dtype=torch.int32).reshape(1, n), dtype=ttnn.uint32,
                                   layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, mesh_mapper=rep)

        def dev_pos(p0, n):
            return ttnn.from_torch(torch.arange(p0, p0 + n, dtype=torch.int32), dtype=ttnn.int32,
                                   layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, mesh_mapper=rep)

        def target_rows(ids, p0):
            """Rows mode over ids at positions p0..: -> (argmax per row [R], normed hidden [1,1,R,H] replicated)."""
            R = len(ids)
            x = model.embd(dev_ids(ids))
            x = ttnn.reshape(x, (1, 1, R, x.shape[-1]))
            pos = dev_pos(p0, R)
            for layer in model.layers:
                x = layer.forward_rows(x, pos, pt, commit_all=False)
            h = model._final_norm_decode(x)  # replicated, normed (the model's 1/16 folded in)
            lg = model._lm_head(h)
            am = ttnn.to_torch(ttnn.get_device_tensors(lg)[0]).float().reshape(-1, lg.shape[-1])[:R, :V].argmax(-1)
            # MTP wants final_norm(h) without the 1/16 the state dict folded into the norm weight: x16
            return am.tolist(), ttnn.multiply(h, float(args.logits_mup_width_multiplier))

        def commit(n_rows):
            for layer in model.layers:
                layer.commit_rows(n_rows)

        def mtp_draft(hrows, tokens_after, p0, k):
            """Depth 0 rows mode over the accepted rows (hidden rows j, token after row j, position p0 + j), then
            depths 1..k-1 one row each: -> k draft tokens."""
            R = len(tokens_after)
            dp = mtp.depths[0]
            e = model.embd(dev_ids(tokens_after))
            e = ttnn.reshape(e, (1, 1, R, e.shape[-1]))
            e = mtp._gather(e)
            x_in = ttnn.concat([mtp._norm(hrows, dp.hn), mtp._norm(e, dp.en)], dim=-1)
            x = ttnn.linear(x_in, dp.w_in)
            out = dp.block.forward_rows(x, dev_pos(p0, R), pt, commit_all=True)
            full = mtp._gather(out)
            last = ttnn.slice(full, (0, 0, R - 1, 0), (1, 1, R, full.shape[-1]))
            drafts = []
            prev = last
            lg = model._lm_head(ttnn.multiply(prev, mtp.inv_mup))
            d = int(ttnn.to_torch(ttnn.get_device_tensors(lg)[0]).float().reshape(-1, lg.shape[-1])[0, :V].argmax())
            drafts.append(d)
            for j in range(1, k):
                lg, prev = mtp.step(j, prev, dev_ids([d]), dev_pos(p0 + R - 1 + j, 1), pt)
                d = int(ttnn.to_torch(ttnn.get_device_tensors(lg)[0]).float().reshape(-1, lg.shape[-1])[0, :V].argmax())
                drafts.append(d)
            return drafts

        for k in [0] + list(range(1, K + 1)):
            tot_new, tot_steps, tot_t = 0, 0, 0.0
            for msgs in PROMPTS:
                prompt = tok.encode(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                            reasoning_effort="none"), add_special_tokens=False)
                model.reset_sequence_state()
                mtp.reset()
                P = len(prompt)
                # prefill all but the last prompt token, then the last one as the first verify row
                pre = (P - 1) // 64 * 64
                if pre:
                    model.prefill_tokens(torch.tensor(prompt[:pre]), pt_host, logits_rows="last")
                rest = prompt[pre:]
                for g in range(0, len(rest) - 1, 8):  # the remaining prompt tokens as committed row groups
                    chunk = rest[g : min(g + 8, len(rest) - 1)]
                    _, _ = target_rows(chunk, pre + g)
                    commit(len(chunk))
                out = []
                x, p = rest[-1], P - 1  # the last prompt token, at position P-1
                drafts = []
                hprev, tafter, pmtp = None, None, None
                t0 = time.time()
                steps = 0
                while len(out) < NEW:
                    rows = [x] + drafts
                    am, h = target_rows(rows, p)
                    n = 0
                    while n < len(drafts) and drafts[n] == am[n]:
                        n += 1
                    commit(n + 1)
                    new = drafts[:n] + [am[n]]
                    out += new
                    steps += 1
                    if eos & set(new):
                        break
                    if k:
                        hrows = ttnn.slice(h, (0, 0, 0, 0), (1, 1, n + 1, h.shape[-1]))
                        drafts = mtp_draft(hrows, rows[1 : n + 1] + [am[n]], p, k)
                    x, p = am[n], p + n + 1
                dt = time.time() - t0
                tot_new += len(out)
                tot_steps += steps
                tot_t += dt
                if k in (0, K):
                    print(f"  k={k} sample: {tok.decode(out[:60])!r}", flush=True)
            print(f"k={k}: {tot_new / tot_steps:.2f} tokens/step, {tot_new / tot_t:.2f} tok/s "
                  f"({tot_new} tokens, {tot_steps} steps, {tot_t:.1f}s)", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()

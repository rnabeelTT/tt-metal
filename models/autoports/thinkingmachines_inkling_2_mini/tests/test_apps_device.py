# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Teacher-forced per-application device test against the transformers golden (reference/golden.py).

Every application a (layer, pass) gets the golden inputs it sees in the reference -- its input hidden state, the
attention-residual blocks, the pass messages (from the golden sources of the previous pass) -- so its output can be
compared with the golden output alone, without error compounding. Two modes per application:
  prefill: rows 0..T-1 in one chunk
  decode:  rows 0..P-1 as a prefill chunk, then rows P..P+D-1 one at a time (cache + conv state continuity)
and the head (final mix + norm + lm_head + softcap) on the golden last hidden state.

usage (in the image): tools/run_dev.sh python models/autoports/thinkingmachines_inkling_2_mini/tests/test_apps_device.py
env: LAYERS (comma list, default a representative set), EXPERT_DT (bf16 / bfp8 / bfp4), FIX (fixture path),
     P (decode prefix, 192), D (decode rows, 8)
"""

import os
import sys
import time

import torch

import ttnn
from models.autoports.thinkingmachines_inkling_2_mini.tt.model import Inkling2Model, _bf

FIX = os.environ.get("FIX", "/cache/fixtures/golden_t256.pt")
DT = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}[os.environ.get("EXPERT_DT", "bf16")]
LAYERS = [int(x) for x in os.environ.get("LAYERS", "0,1,5,6,11,12,17,18,23").split(",")]
P, D = int(os.environ.get("P", "192")), int(os.environ.get("D", "8"))


def pcc(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def main():
    g = torch.load(FIX)
    hid = g["hidden"].float()  # [49, T, H]
    T = hid.shape[1]
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    t0 = time.time()
    m = Inkling2Model(mesh, max_seq_len=1024, expert_dtype=DT, prefill_chunk=256, layers=LAYERS)
    print(f"model ({len(LAYERS)} layers, experts {DT}) up in {time.time() - t0:.0f}s", flush=True)
    cfg, ops = m.cfg, m.ops
    dev = lambda t: ttnn.from_torch(t.reshape(1, 1, *t.shape[-2:]).to(torch.bfloat16), dtype=ttnn.bfloat16,  # noqa
                                    layout=ttnn.TILE_LAYOUT, device=mesh)
    host = lambda t: ttnn.to_torch(t).float().reshape(-1, t.shape[-1])  # noqa: E731

    # golden inputs per application
    sched = m.sched
    inp, blocks, msgs_src = {}, {}, {}
    bl = []
    pre_bl = None
    for a, (li, p) in enumerate(sched):
        if li == cfg.loop_start:
            if p == 0:
                pre_bl = list(bl)
            else:
                bl = list(pre_bl)
        inp[a] = hid[cfg.loop_start] if (li == cfg.loop_start and p > 0) else hid[a]
        blocks[a] = list(bl)
        if cfg.loop_start <= li < cfg.loop_end and p > 0:
            # previous pass: block-12 input = input of app (12, p-1); layer-17 output = output of app (17, p-1)
            a12 = sched.index((12, p - 1))
            a17 = sched.index((cfg.loop_end - 1, p - 1))
            msgs_src[a] = (hid[a12], hid[a17 + 1])
        if li % cfg.n_layers_per_attn_res_block == 0:
            bl = bl + [inp[a]]

    worst = 1.0
    for a, (li, p) in enumerate(sched):
        if li not in LAYERS:
            continue
        want = hid[a + 1]
        res = []
        for mode in ("prefill", "decode"):
            m.apps[a].reset()
            bn = [ops.rms(dev(b)) for b in blocks[a]]
            msgs = None
            if a in msgs_src:
                msgs = m.messages(li - cfg.loop_start, [ops.rms(dev(s)) for s in msgs_src[a]])
            if mode == "prefill":
                sl = lambda t: [ttnn.slice(x, [0, 0, 0, 0], [1, 1, T, x.shape[-1]]) for x in t]  # noqa: E731
                out = host(m.layer_forward(a, dev(inp[a]), bn, 0, messages=msgs))
                res.append(("pf", pcc(out, want), (out - want).abs().max().item()))
            else:
                cut = lambda t, s, e: ttnn.slice(t, [0, 0, s, 0], [1, 1, e, t.shape[-1]])  # noqa: E731
                xin = dev(inp[a])
                m.layer_forward(a, cut(xin, 0, P), [cut(b, 0, P) for b in bn], 0,
                                messages=None if msgs is None else [cut(t, 0, P) for t in msgs])
                outs = []
                for j in range(P, P + D):
                    o = m.layer_forward(a, cut(xin, j, j + 1), [cut(b, j, j + 1) for b in bn], j,
                                        messages=None if msgs is None else [cut(t, j, j + 1) for t in msgs])
                    outs.append(host(o))
                out = torch.cat(outs)
                res.append(("dec", pcc(out, want[P : P + D]), (out - want[P : P + D]).abs().max().item()))
        worst = min(worst, *[r[1] for r in res])
        print(f"app {a:2d} (layer {li:2d} pass {p}): " + "  ".join(f"{k} pcc {v:.5f} maxerr {e:.3f}" for k, v, e in res),
              flush=True)
    if 23 in LAYERS:  # head on the golden final state
        a = len(sched) - 1
        bn = [ops.rms(dev(b)) for b in blocks[a] + ([inp[a]] if sched[a][0] % 6 == 0 else [])]
        lg = ttnn.to_torch(m.head(dev(hid[-1]), bn)).float().reshape(T, -1)[:, : m.V]
        ref = g["logits"]
        top1 = (lg.argmax(-1) == ref.argmax(-1)).float().mean().item()
        print(f"head: logits pcc {pcc(lg, ref):.5f}, top-1 agreement {top1:.4f}")
    ttnn.close_mesh_device(mesh)
    print(f"{'PASS' if worst > 0.99 else 'FAIL'}: worst application pcc {worst:.5f}")


if __name__ == "__main__":
    main()

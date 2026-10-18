# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 5 (review #3 P1-1): the full model in the DSA sparse-attention regime (decode positions > 2050, where the sparse
index list incl. its tail is used, not the dense list). Per prompt length: prefill, then N greedy traced decode steps
with host logits; the prefill last-row logits, every decode step's logits and the tokens are saved, so two builds (e.g.
DEFAULT_MC vs GLM53_MC=dsa_tail_shared=0) can be compared bit by bit with --compare.

  python -m models.autoports.zai_org_glm_5_3_flash.tools.s5_sparse_probe --lengths 2100,3000 --steps 8 --save a.pt
  python -m models.autoports.zai_org_glm_5_3_flash.tools.s5_sparse_probe --compare a.pt b.pt
"""

from __future__ import annotations

__import__("os").environ.setdefault("GLM53_PRECISION", "none")  # stage 8: stage-7 precision policy unless set
import argparse
import json
import os
import time

import torch

import ttnn


def _ids(L, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(1000, 100000, (L,), generator=g, dtype=torch.int64)


def run(lengths, steps, seq_len, save):
    os.environ.setdefault("HF_MODEL", "zai-org/GLM-5.3-Flash")
    from models.autoports.zai_org_glm_5_3_flash.tools.readiness import _open_mesh
    from models.autoports.zai_org_glm_5_3_flash.tt.generator import Glm53Generator

    mesh = _open_mesh(int(os.environ.get("GLM53_TRACE_REGION", 64 << 20)))
    out = {"env_mc": os.environ.get("GLM53_MC", ""), "cases": {}}
    g = None
    try:
        t0 = time.time()
        g = Glm53Generator(mesh, max_batch_size=1, max_seq_len=seq_len)
        print(f"S5SP built in {time.time() - t0:.0f}s", flush=True)
        V = g.model.vocab
        for L in lengths:
            ids = _ids(L, seed=L + 11)
            g.reset()
            lg = g.prefill_forward(ids.reshape(1, -1), prompt_lens=[L])[0, -1, :V].float()
            tok, pos = int(lg.argmax()), L
            step_logits, toks = [], [tok]
            for i in range(steps):
                o = g.decode_forward(
                    torch.tensor([[tok]]), torch.tensor([pos]), enable_trace=True, on_device_sampling=False
                )
                row = o[0, :V].float()
                step_logits.append(row)
                tok, pos = int(row.argmax()), pos + 1
                toks.append(tok)
            out["cases"][L] = {"prefill_logits": lg, "decode_logits": torch.stack(step_logits), "tokens": toks}
            fin = bool(torch.isfinite(torch.stack(step_logits)).all())
            print(f"S5SP L={L} positions {L}..{L + steps - 1} tokens {toks} finite {fin}", flush=True)
    finally:
        try:
            g.teardown()
        except Exception:  # noqa: BLE001 - the build may have failed
            pass
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    torch.save(out, save)


def compare(a, b):
    A, B = torch.load(a), torch.load(b)
    res = {"a": A["env_mc"], "b": B["env_mc"]}
    for L in A["cases"]:
        x, y = A["cases"][L], B["cases"][L]
        res[L] = {
            "prefill_logits_equal": torch.equal(x["prefill_logits"], y["prefill_logits"]),
            "decode_logits_equal": torch.equal(x["decode_logits"], y["decode_logits"]),
            "decode_logits_max_abs_diff": float((x["decode_logits"] - y["decode_logits"]).abs().max()),
            "tokens_equal": x["tokens"] == y["tokens"],
            "tokens": x["tokens"],
        }
    print("S5SP_COMPARE", json.dumps(res))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", default="2100,3000")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--save")
    ap.add_argument("--compare", nargs=2)
    a = ap.parse_args()
    if a.compare:
        compare(*a.compare)
    else:
        run([int(x) for x in a.lengths.split(",")], a.steps, a.seq_len, a.save)


if __name__ == "__main__":
    main()

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage-3 review #4 A/B of the one-die decode options of tt/optimized_decoder.py (OptConfig `dsa_q_mm_ib`,
`sdpa_head_split`, `qi_rm`, `route_rm`) on ONE build of the dsa_moe layer (L3, real weights, policy t_r4), the options
switched at run time between variants.

  one   1 slot. Prefill the 4000-row real input stream of test_lengths[dsa_moe-4000] (cached HF request), then decode at
        position 4000 (sparse regime; 4000 % 4 == 0 starts a new pool, so a repeated step at the same position writes the
        same state). Variants: control (all off), dsa_q_mm with in0 blocks 8 / 16 / 32 / 64 (program for the 64 query
        rows), the sparse_sdpa head split, the indexer qi row-major reshape, and the combinations.
  b32   32 slots, decode-only at position 128 in every slot (dense regime; the indexer still runs): control, qi_rm,
        route_rm, both.

Per variant: the eager step and a traced step (capture, then mean host wall of N replays with one final sync), both
compared bit-exactly with the control's outputs.

usage: python -m models.autoports.zai_org_glm_5_3_flash.tools.s3r4_ab [--only one,b32] [--n 50]"""

import argparse
import time

import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tests.harness import (
    H4,
    DecodeRunner,
    build_layer,
    close_mesh,
    open_mesh,
    pcc,
    prefill_user,
)
from models.autoports.zai_org_glm_5_3_flash.tt.ckpt import CheckpointReader
from models.autoports.zai_org_glm_5_3_flash.tt.optimized_decoder import CANDIDATES, OptimizedDecoder

OPT = CANDIDATES["t_r4"]


class _Dec(OptimizedDecoder):
    @classmethod
    def from_state_dict(cls, *a, **kw):
        kw["opt"] = OPT
        return super().from_state_dict(*a, **kw)


def _set(dec, q_mm=0, split=False, qi=False, route=False):
    a = dec.attn
    a.mc.dsa_q_mm = bool(q_mm)
    if q_mm:
        a.mc.dsa_q_mm_ib = q_mm
    a.q_mm_rows = bool(q_mm)
    a.head_split, a.qi_rm = split, qi
    if dec.mlp_kind == "moe":
        dec.mlp.route_rm = route


def _variant(mesh, dec, batch, rows, pos, n):
    r = DecodeRunner(dec, mesh, batch)
    eager = r.eager_step(rows, pos)
    r.capture(rows, pos)
    traced = r.traced_step(rows, pos)  # warm replay
    ttnn.synchronize_device(mesh)
    t = time.perf_counter()
    for _ in range(n):
        r.traced_step(rows, pos, sync=False)
    ttnn.synchronize_device(mesh)
    us = (time.perf_counter() - t) / n * 1e6
    traced = r.traced_step(rows, pos)
    r.release()
    return eager, traced, us


def _run(mesh, dec, batch, rows, pos, variants, n, act):
    ref = None
    for name, kw in variants:
        _set(dec, **kw)
        try:
            e, tr, us = _variant(mesh, dec, batch, rows, pos, n)
        except Exception as ex:  # noqa: BLE001
            msg = " | ".join(l.strip() for l in str(ex).splitlines() if l.strip() and "backtrace" not in l.lower())
            print(f"{name:34}: ERROR {msg[:400]}", flush=True)
            continue
        e, tr = e[act], tr[act]
        if ref is None:
            ref = (e, tr)
        ok_e, ok_t = torch.equal(e, ref[0]), torch.equal(tr, ref[1])
        print(
            f"{name:34}: traced {us:8.1f} us/step (host wall, {n} replays)  eager==control {ok_e}  traced==control "
            f"{ok_t}  PCC traced {pcc(tr, ref[1]):.6f}  eager==traced {torch.equal(e, tr)}",
            flush=True,
        )


def one(mesh, rd, n):
    from models.autoports.zai_org_glm_5_3_flash.tests.test_functional_decoder import reference

    L, P = 3, 4000
    x, _ = reference(L, "real", 1000 + P, P, 3)  # the cached request of test_lengths[dsa_moe-4000]
    dec = build_layer(mesh, rd, L, "real", batch=1, decoder_cls=_Dec)
    prefill_user(dec, mesh, x, P, 0)
    v = [("control (r3 default)", {})]
    v += [(f"dsa_q_mm in0 {ib}", {"q_mm": ib}) for ib in (8, 16, 32, 64)]
    v += [
        ("head split", {"split": True}),
        ("qi_rm", {"qi": True}),
        ("dsa_q_mm 64 + head split", {"q_mm": 64, "split": True}),
        ("dsa_q_mm 64 + head split + qi_rm", {"q_mm": 64, "split": True, "qi": True}),
        ("control again", {}),
    ]
    print(f"===== one: L{L} 1 slot, prefill {P}, decode pos {P}", flush=True)
    _run(mesh, dec, 1, {0: x[P]}, [P], v, n, [0])
    del dec


def b32(mesh, rd, n):
    L, B, P = 3, 32, 128
    dec = build_layer(mesh, rd, L, "real", batch=B, decoder_cls=_Dec)
    g = torch.Generator().manual_seed(0)
    rows = {s: torch.randn(H4, generator=g) * 0.02 for s in range(B)}
    v = [
        ("control (r3 default)", {}),
        ("qi_rm", {"qi": True}),
        ("route_rm", {"route": True}),
        ("qi_rm + route_rm", {"qi": True, "route": True}),
        ("control again", {}),
    ]
    print(f"===== b32: L{L} {B} slots, decode-only pos {P}", flush=True)
    _run(mesh, dec, B, rows, [P] * B, v, n, list(range(B)))
    del dec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="one,b32")
    ap.add_argument("--n", type=int, default=50)
    a = ap.parse_args()
    rd = CheckpointReader()
    mesh = open_mesh()
    try:
        for sec in a.only.split(","):
            {"one": one, "b32": b32}[sec](mesh, rd, a.n)
    finally:
        close_mesh(mesh)


if __name__ == "__main__":
    main()

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Bit-exactness check of fused-decoder rewrite flags on real-weight layers (one die).

For the flags off, then on (a fresh layer each time), at 32 slots and at 1 slot: prefill one user (2052 tokens: a
2048-row chunk, then a 4-row chunk on the 128-row bucket; for DSA layers the second chunk runs the sparse indexer path)
into a slot, then 5 decode steps of every slot: one eager step, then a captured trace replayed 4 times with new positions
and rows (the in-place conv-history shift feeds the next step). Every prefill and decode output with the flags off and on
must be bit-identical.

Usage: python -m models.autoports.zai_org_glm_5_3_flash.tools.fused_flag_check FLAG[,FLAG...] LAYER [LAYER ...]
  e.g. dsa_heads_op,mlp_act_fold,kda_decode_fold 0 3 4
"""

import gc
import sys

import torch

from models.autoports.zai_org_glm_5_3_flash.tests.harness import (
    H4,
    DecodeRunner,
    build_layer,
    close_mesh,
    open_mesh,
    prefill_user,
)
from models.autoports.zai_org_glm_5_3_flash.tt.ckpt import CheckpointReader
from models.autoports.zai_org_glm_5_3_flash.tt.fused_decoder import FusedDecoder, FusedFlags

LENGTH, STEPS = 2052, 5


def _decoder_cls(flag: str, on: bool):
    flags = FusedFlags(**{f: on for f in flag.split(",")})

    class _Fused(FusedDecoder):
        @classmethod
        def from_state_dict(cls, *a, **kw):
            return FusedDecoder.from_state_dict(*a, flags=flags, **kw)

    return _Fused


def run(mesh, rd, layer: int, flag: str, on: bool, batch: int):
    dec = build_layer(
        mesh, rd, layer, batch=batch, prefill_lengths=(128, 2048), max_context=4096, decoder_cls=_decoder_cls(flag, on)
    )
    slot = min(5, batch - 1)
    g = torch.Generator().manual_seed(layer)
    x = torch.randn(LENGTH, H4, generator=g) * 0.02
    steps = [{s: torch.randn(H4, generator=g) * 0.02 for s in range(batch)} for _ in range(STEPS)]
    outs = [prefill_user(dec, mesh, x, LENGTH, slot)]

    def pos(i):
        p = [i] * batch
        p[slot] = LENGTH + i
        return p

    runner = DecodeRunner(dec, mesh, batch)
    outs.append(runner.eager_step(steps[0], pos(0)))
    runner.capture(steps[1], pos(1))
    for i in range(1, STEPS):
        outs.append(runner.traced_step(steps[i], pos(i)))
    runner.release()
    del runner, dec
    gc.collect()
    return outs


def main():
    flag, layers = sys.argv[1], [int(a) for a in sys.argv[2:]]
    rd = CheckpointReader()
    mesh = open_mesh()
    try:
        for batch in (32, 1):
            for layer in layers:
                off, on = run(mesh, rd, layer, flag, False, batch), run(mesh, rd, layer, flag, True, batch)
                names = ["prefill"] + ["decode eager"] + [f"decode traced {i}" for i in range(1, STEPS)]
                for name, a, b in zip(names, off, on):
                    print(
                        f"B{batch} L{layer} {name}: bit-exact {torch.equal(a, b)}  max|d| {(a - b).abs().max().item():.3e}"
                        f"  finite {bool(torch.isfinite(b).all())}",
                        flush=True,
                    )
    finally:
        close_mesh(mesh)


if __name__ == "__main__":
    main()

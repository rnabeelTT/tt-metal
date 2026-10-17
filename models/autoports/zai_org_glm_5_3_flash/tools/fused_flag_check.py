# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Bit-exactness check of fused-decoder rewrite flags on real-weight layers (one die, 32 slots).

For the flags off, then on (a fresh layer each time): prefill one user (2052 tokens: a 2048-row chunk, then a 4-row chunk
on the 128-row bucket; for DSA layers the second chunk runs the sparse indexer path) into slot 5, then one eager decode
step of all 32 slots (every slot active, the other slots at position 0 with their own rows). The prefill and decode
outputs with the flags off and on must be bit-identical.

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

LENGTH, SLOT, BATCH = 2052, 5, 32


def _decoder_cls(flag: str, on: bool):
    flags = FusedFlags(**{f: on for f in flag.split(",")})

    class _Fused(FusedDecoder):
        @classmethod
        def from_state_dict(cls, *a, **kw):
            return FusedDecoder.from_state_dict(*a, flags=flags, **kw)

    return _Fused


def run(mesh, rd, layer: int, flag: str, on: bool):
    dec = build_layer(
        mesh, rd, layer, batch=BATCH, prefill_lengths=(128, 2048), max_context=4096, decoder_cls=_decoder_cls(flag, on)
    )
    g = torch.Generator().manual_seed(layer)
    x = torch.randn(LENGTH, H4, generator=g) * 0.02
    rows = {s: torch.randn(H4, generator=g) * 0.02 for s in range(BATCH)}
    yp = prefill_user(dec, mesh, x, LENGTH, SLOT)
    pos = [0] * BATCH
    pos[SLOT] = LENGTH
    runner = DecodeRunner(dec, mesh, BATCH)
    yd = runner.eager_step(rows, pos)
    del runner, dec
    gc.collect()
    return yp, yd


def main():
    flag, layers = sys.argv[1], [int(a) for a in sys.argv[2:]]
    rd = CheckpointReader()
    mesh = open_mesh()
    try:
        for layer in layers:
            off, on = run(mesh, rd, layer, flag, False), run(mesh, rd, layer, flag, True)
            for i, name in enumerate(("prefill", "decode")):
                a, b = off[i], on[i]
                print(
                    f"{flag} L{layer} {name}: bit-exact {torch.equal(a, b)}  max|d| {(a - b).abs().max().item():.3e}"
                    f"  finite {bool(torch.isfinite(b).all())}",
                    flush=True,
                )
    finally:
        close_mesh(mesh)


if __name__ == "__main__":
    main()

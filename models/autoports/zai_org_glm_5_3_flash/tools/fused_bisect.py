# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic: run the real-weight case of tests/test_functional_decoder.py (prefill 100 at slot 5, 3 decode steps) on the
fused decoder with selected rewrites off (--off mhc_mix_op,mhc_pre_op ...), to bisect a fused-path regression."""

import argparse

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tests import test_functional_decoder as tf
from models.autoports.zai_org_glm_5_3_flash.tests.harness import build_layer, open_mesh
from models.autoports.zai_org_glm_5_3_flash.tt.ckpt import CheckpointReader
from models.autoports.zai_org_glm_5_3_flash.tt.fused_decoder import FusedDecoder, FusedFlags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--off", default="")
    a = ap.parse_args()
    off = [f for f in a.off.split(",") if f]
    flags = FusedFlags(**{f: False for f in off})

    class _F(FusedDecoder):
        @classmethod
        def from_state_dict(cls, *x, **kw):
            return FusedDecoder.from_state_dict(*x, flags=flags, **kw)

    mesh = open_mesh()
    try:
        dec = build_layer(mesh, CheckpointReader(), a.layer, "real", batch=tf.BATCH, decoder_cls=_F)
        print(f"=== off={off}")
        try:
            tf.run_user(mesh, dec, a.layer, "real", seed=1, prefill_len=100, decode_steps=3, slot=5, traced=False)
            print("RESULT pass")
        except AssertionError as e:
            print(f"RESULT fail: {e}")
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic: the TT MoE's top-8 expert set for one user at given decode steps (real weights, real tokens as in
tests/test_functional_decoder.py), to compare with HF's (scripts/route_margin.py): a failing near-tie token is a router
flip only if the sets differ. Run with GLM53_MOE_DEBUG=1.
usage: python tools/route_flip_check.py --layer 3 --seed 3051 --prefill 2051 --steps 3 --slot 3 [--fused]"""

import argparse
import os

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tests import test_functional_decoder as tf
from models.autoports.zai_org_glm_5_3_flash.tests.harness import DecodeRunner, build_layer, open_mesh, pcc, prefill_user
from models.autoports.zai_org_glm_5_3_flash.tt.ckpt import CheckpointReader


def main():
    ap = argparse.ArgumentParser()
    for a in ("layer", "seed", "prefill", "steps", "slot"):
        ap.add_argument(f"--{a}", type=int, required=True)
    ap.add_argument("--fused", action="store_true", help="tt/fused_decoder.FusedDecoder (default: functional)")
    a = ap.parse_args()
    assert os.environ.get("GLM53_MOE_DEBUG") == "1"
    x, y = tf.reference(a.layer, "real", a.seed, a.prefill, a.steps)
    mesh = open_mesh()
    try:
        cls = None
        if a.fused:
            from models.autoports.zai_org_glm_5_3_flash.tt.fused_decoder import FusedDecoder as cls
        dec = build_layer(mesh, CheckpointReader(), a.layer, "real", batch=tf.BATCH, decoder_cls=cls)
        prefill_user(dec, mesh, x, a.prefill, a.slot)
        runner = DecodeRunner(dec, mesh, tf.BATCH)
        for i in range(a.steps):
            pos = [-1] * tf.BATCH
            pos[a.slot] = a.prefill + i
            o = runner.eager_step({a.slot: x[a.prefill + i]}, pos)[a.slot]
            ids = ttnn.to_torch(dec.mlp.last_ids).reshape(tf.BATCH, -1)[a.slot, :8].long()
            d = pcc(o - x[a.prefill + i], y[a.prefill + i] - x[a.prefill + i])
            print(f"pos {a.prefill + i}: delta PCC {d:.6f} TT top-8 {sorted(ids.tolist())}")
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 1 long-context test: one decoder layer per kind at the advertised context (max_position_embeddings = 1,048,576).

A 1,048,000-token prompt (not a multiple of the 2048 chunk, the 128 page or the 4-token pool) is prefilled in 2048-row
chunks into decode slot 0 of a 1-slot layer, then 3 decode steps run at positions 1,048,000..1,048,002. The input
stream is generated chunk by chunk (synthetic.synthetic_stream_segment: per-channel statistics of the real stream of
the layer kind) on both sides; the HF reference (tests/hf_ref/hf_layer_ref.py "long" mode, checked against the
full forward at 4K) fills the layer cache segment by segment and runs the HF layer on the last 64 prompt rows and the
decode steps. Compared: the last 64 prompt rows and the 3 decode rows.
"""

from __future__ import annotations

import json
import os
import time

import pytest
import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tests.harness import H4, STATS, DecodeRunner, build_layer, open_mesh, pcc
from models.autoports.zai_org_glm_5_3_flash.tests.hf_ref.client import hf_layer_reference
from models.autoports.zai_org_glm_5_3_flash.tt.ckpt import CheckpointReader
from models.autoports.zai_org_glm_5_3_flash.tt.synthetic import synthetic_stream_segment

ADVERTISED = 1_048_576
N_PROMPT = int(os.environ.get("GLM53_LONG_N", "1048000"))
TAIL, DECODE, CHUNK, SEED = 64, 3, 2048, 21
KINDS = {"kda_dense": 0, "dsa_moe": 3, "kda_moe": 4}
DELTA_BAR = 0.995

pytestmark = pytest.mark.skipif(not os.environ.get("GLM53_CKPT"), reason="needs the E192 checkpoint (GLM53_CKPT)")


@pytest.mark.parametrize("kind", list(KINDS))
def test_long_context(kind):
    layer = KINDS[kind]
    stats = json.loads(STATS.read_text())
    assert N_PROMPT + DECODE <= ADVERTISED
    ref = hf_layer_reference(
        {
            "layer": layer,
            "weights": "real",
            "long": {
                "n_prompt": N_PROMPT,
                "tail": TAIL,
                "decode": DECODE,
                "seed": SEED,
                "stats": str(STATS),
                "seg": CHUNK,
            },
        },
        timeout=6 * 3600,
    )
    xr = ref["x"][0].float().reshape(-1, H4)  # [TAIL + DECODE, H4]
    yr = ref["y"][0].float().reshape(-1, H4)
    rd = CheckpointReader()
    mesh = open_mesh()
    try:
        max_ctx = ((N_PROMPT + DECODE + CHUNK - 1) // CHUNK) * CHUNK  # whole prefill chunks (1,048,576 here)
        dec = build_layer(mesh, rd, layer, "real", batch=1, prefill_lengths=(CHUNK,), max_context=max_ctx)
        t0 = time.time()
        st, last = None, None
        for c0 in range(0, N_PROMPT, CHUNK):
            ln = min(CHUNK, N_PROMPT - c0)
            xp = torch.zeros(1, 1, CHUNK, H4)
            xp[0, 0, :ln] = synthetic_stream_segment(stats, layer, SEED, c0, ln).float().reshape(ln, H4)
            X = ttnn.from_torch(xp, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh)
            Y, st = dec.prefill_forward(X, inputs=dec.prefill_inputs(CHUNK, start=c0, length=ln, slot=0), attn_state=st)
            if c0 + ln == N_PROMPT:
                last = ttnn.to_torch(Y)[0, 0, ln - TAIL : ln].float()
            ttnn.deallocate(X)
            ttnn.deallocate(Y)
        prefill_s = time.time() - t0
        dec.commit_prefill(0, st)
        d = pcc(last - xr[:TAIL], yr[:TAIL] - xr[:TAIL])
        print(f"L{layer} prefill {N_PROMPT} tokens in {prefill_s:.0f} s; last {TAIL} rows delta PCC {d:.6f}")
        assert torch.isfinite(last).all()
        assert d >= DELTA_BAR, d
        runner = DecodeRunner(dec, mesh, 1)
        for i in range(DECODE):
            pos = [N_PROMPT + i]
            rows = {0: xr[TAIL + i]}
            if i == 0:
                o = runner.eager_step(rows, pos)[0]
                runner.capture({0: xr[TAIL + 1]}, [N_PROMPT + 1])
            else:
                o = runner.traced_step(rows, pos)[0]
            d = pcc(o - xr[TAIL + i], yr[TAIL + i] - xr[TAIL + i])
            print(f"L{layer} decode position {N_PROMPT + i}: delta PCC {d:.6f}")
            assert torch.isfinite(o).all()
            assert d >= DELTA_BAR, d
        runner.release()
    finally:
        ttnn.close_mesh_device(mesh)

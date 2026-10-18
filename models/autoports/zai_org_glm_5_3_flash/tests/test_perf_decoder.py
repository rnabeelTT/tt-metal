# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 1 perf windows (run under Tracy: python -m tracy -r -p -v -m pytest <this file>::<case>).

Warmed prefill of one chunk (128 or 2048 rows, start 0) and a traced decode step (3 replays) of one layer per kind,
on one die, real weights. Signposts PERF_PREFILL / PERF_PREFILL_END and PERF_DECODE / PERF_DECODE_END delimit the
measured windows for tt-perf-report.
"""

from __future__ import annotations

__import__("os").environ.setdefault("GLM53_PRECISION", "none")  # stage 8: stage-7 precision policy unless set
import os
import time

import pytest
import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tests.harness import (
    H4,
    DecodeRunner,
    build_layer,
    close_mesh,
    open_mesh,
    stream_in,
)
from models.autoports.zai_org_glm_5_3_flash.tt.ckpt import CheckpointReader


def _decoder_cls():
    """GLM53_DECODER=fused: tt/fused_decoder.FusedDecoder (FusedFlags from GLM53_FUSED_OFF=comma list of flags to turn
    off, for before / after measurements); default the functional decoder."""
    kind = os.environ.get("GLM53_DECODER", "functional")
    if kind in ("optimized", "multichip"):  # multichip: run with GLM53_TP=4 (a (1, 4) mesh)
        from models.autoports.zai_org_glm_5_3_flash.tt.optimized_decoder import CANDIDATES, SELECTED, OptimizedDecoder

        if kind == "multichip":
            from models.autoports.zai_org_glm_5_3_flash.tt.multichip_decoder import MultichipDecoder as OptimizedDecoder

        opt = CANDIDATES[os.environ["GLM53_OPT"]] if os.environ.get("GLM53_OPT") else SELECTED
        print(f"optimized decoder policy: {opt.describe()}")

        class _Opt(OptimizedDecoder):
            @classmethod
            def from_state_dict(cls, *a, **kw):
                return OptimizedDecoder.from_state_dict(*a, opt=opt, **kw)

        return _Opt
    if kind != "fused":
        return None
    from models.autoports.zai_org_glm_5_3_flash.tt.fused_decoder import FusedDecoder, FusedFlags

    off = [f for f in os.environ.get("GLM53_FUSED_OFF", "").split(",") if f]
    on = [f for f in os.environ.get("GLM53_FUSED_ON", "").split(",") if f]
    flags = FusedFlags(**{f: False for f in off}, **{f: True for f in on})

    class _Fused(FusedDecoder):
        @classmethod
        def from_state_dict(cls, *a, **kw):
            return FusedDecoder.from_state_dict(*a, flags=flags, **kw)

    return _Fused


try:
    from tracy import signpost
except ImportError:  # perf windows are optional outside Tracy

    def signpost(name):
        pass


KINDS = {"kda_dense": 0, "dsa_moe": 3, "kda_moe": 4}
pytestmark = pytest.mark.skipif(not os.environ.get("GLM53_CKPT"), reason="needs the E192 checkpoint (GLM53_CKPT)")


def _flush(mesh):
    if os.environ.get("TT_METAL_DEVICE_PROFILER") == "1" and hasattr(ttnn, "ReadDeviceProfiler"):
        ttnn.synchronize_device(mesh)
        ttnn.ReadDeviceProfiler(mesh)


def _x(n, seed):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(1, 1, n, H4, generator=g) * 0.02).float()


@pytest.mark.parametrize("rows", [128, 2048])
@pytest.mark.parametrize("kind", list(KINDS))
def test_perf_prefill(kind, rows):
    torch.manual_seed(int(os.environ.get("GLM53_PERF_SEED", "0")))  # reproducible inputs (32-user A/Bs)
    rd = CheckpointReader()
    mesh = open_mesh()
    try:
        dec = build_layer(
            mesh, rd, KINDS[kind], batch=8, prefill_lengths=(rows,), max_context=4096, decoder_cls=_decoder_cls()
        )
        X = stream_in(mesh, _x(rows, 1))
        inputs = dec.prefill_inputs(rows, start=0, length=rows, slot=0)
        Y, _ = dec.prefill_forward(X, inputs=inputs)  # compile
        ttnn.deallocate(Y)
        ttnn.synchronize_device(mesh)
        _flush(mesh)
        signpost("PERF_PREFILL")
        t0 = time.time()
        Y, _ = dec.prefill_forward(X, inputs=inputs)
        ttnn.synchronize_device(mesh)
        dt = time.time() - t0
        signpost("PERF_PREFILL_END")
        print(f"PERF {kind} prefill {rows} rows: {dt * 1e3:.2f} ms host wall (warmed)")
    finally:
        close_mesh(mesh)


@pytest.mark.parametrize("batch", [1, 32])
@pytest.mark.parametrize("kind", list(KINDS))
def test_perf_traced_decode(kind, batch):
    # GLM53_PERF_SEED: decode input rows (the 32-user active-expert set is data dependent); GLM53_PERF_POS: decode
    # position (default 128, the dense-attention regime; >= 2051 puts the DSA layers in the sparse-attention regime,
    # max_context grows to hold it)
    torch.manual_seed(int(os.environ.get("GLM53_PERF_SEED", "0")))
    p0 = int(os.environ.get("GLM53_PERF_POS", "128"))
    max_ctx = max(4096, -(-(p0 + 64) // 2048) * 2048)
    rd = CheckpointReader()
    mesh = open_mesh()
    try:
        dec = build_layer(
            mesh, rd, KINDS[kind], batch=batch, prefill_lengths=(128,), max_context=max_ctx, decoder_cls=_decoder_cls()
        )
        runner = DecodeRunner(dec, mesh, batch)
        rows = {s: torch.randn(H4) * 0.02 for s in range(batch)}
        pos = [p0] * batch
        runner.eager_step(rows, pos)  # compile
        runner.capture(rows, [p0 + 1] * batch)
        runner.traced_step(rows, [p0 + 1] * batch)  # warm replay
        ttnn.synchronize_device(mesh)
        _flush(mesh)
        signpost("PERF_DECODE")
        t0 = time.time()
        iters = 3
        for i in range(iters):
            runner.traced_step(rows, [p0 + 2 + i] * batch, sync=False)
        ttnn.synchronize_device(mesh)
        dt = (time.time() - t0) / iters
        signpost("PERF_DECODE_END")
        print(
            f"PERF {kind} traced decode batch {batch}: {dt * 1e3:.3f} ms/step host wall ({iters} replays, incl. input copies)"
        )
        runner.release()
    finally:
        close_mesh(mesh)

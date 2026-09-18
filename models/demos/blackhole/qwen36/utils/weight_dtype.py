# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Weight dtype for the qwen36 hybrid stack's DENSE (non-expert) 2D projections.

The Gated DeltaNet and full-attention modules both loaded every 2D projection weight
at a hard-coded ``ttnn.bfloat8_b`` -- the GDN fused QKV, a+b, mega-fused and out
projections, and attention's q/k/v/o. That is a defensible default, but hard-coding it
made the backbone's precision the one part of this model's error budget that no sweep
could reach: ``ORNITH_ROUTED_DTYPE``, ``ORNITH_KV_DTYPE``, ``ORNITH_LM_HEAD_DTYPE``,
``ORNITH_EXPERT_ACT`` and the two fidelity knobs all exist, and every one of them
varies an expert, KV-cache or LM-head dtype. The 40 layers of dense projections that
the residual stream actually accumulates through could not be varied at all.

That matters here because this stack's error is accumulation, not a broken layer:
teacher-forced, every one of the 40 layers lands between 0.9935 and 0.9995 logit PCC,
while free-running the same model reaches only 0.968 / 0.942 / 0.934. A Gated DeltaNet
layer injects ~2.5x the error of a full-attention one and there are three times as
many, so GDN carries ~80% of the total -- and its ~16% per-layer relative error was
recorded as unexplained on the grounds that "bf16 arithmetic accounts for ~0.4%".
The weights were never bf16. They are bfloat8_b, so bf16 was the wrong yardstick.

``QWEN36_DENSE_WEIGHT_DTYPE`` selects it instead, defaulting to ``bfloat8_b`` so the
dense Qwen3.5/3.6 models that share these modules are unaffected unless a caller asks.
Exposed as one helper rather than per-module reads so the attention and GDN paths
cannot drift apart -- they are the same error budget, the same argument as
``math_fidelity()`` next door.

Safe to flip: ``ttnn.as_tensor`` appends ``_dtype_<NAME>_layout_<NAME>.tensorbin`` to
its ``cache_file_name``, so a changed dtype gets its own cache entry rather than
silently loading the previously cached one. The uncached ``from_torch`` fused weights
are rebuilt every run regardless.
"""
import os

import ttnn

_DTYPES = {
    "bfloat8_b": ttnn.bfloat8_b,
    "bfloat16": ttnn.bfloat16,
    "bfloat4_b": ttnn.bfloat4_b,
}


def dense_weight_dtype(default: str = "bfloat8_b") -> "ttnn.DataType":
    """Resolve ``QWEN36_DENSE_WEIGHT_DTYPE``, falling back to ``default``."""
    name = os.environ.get("QWEN36_DENSE_WEIGHT_DTYPE", default).strip()
    if name not in _DTYPES:
        raise ValueError(f"QWEN36_DENSE_WEIGHT_DTYPE={name!r} is not one of {sorted(_DTYPES)}")
    return _DTYPES[name]

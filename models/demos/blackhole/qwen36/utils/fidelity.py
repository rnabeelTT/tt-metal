# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Math fidelity for the qwen36 hybrid stack's non-MoE matmuls.

Both the full-attention and the Gated DeltaNet modules built their compute kernel
configs inline at ``MathFidelity.LoFi``. That is the right default for a dense model
whose weights dominate the error budget, but it is the wrong one to hard-code: LoFi
truncates the bf16 mantissa in the math unit, and on a 40-layer hybrid stack that
truncation is applied to every in-projection of every layer and accumulates through
the residual stream.

``QWEN36_MATH_FIDELITY`` selects it instead, defaulting to ``lofi`` so nothing
changes unless a caller asks. Exposed as one helper rather than two env reads so the
attention and GDN paths cannot drift apart -- they are the same error budget.
"""
import os

import ttnn

_FIDELITIES = {
    "lofi": ttnn.MathFidelity.LoFi,
    "hifi2": ttnn.MathFidelity.HiFi2,
    "hifi3": ttnn.MathFidelity.HiFi3,
    "hifi4": ttnn.MathFidelity.HiFi4,
}


def math_fidelity(default: str = "lofi") -> "ttnn.MathFidelity":
    """Resolve ``QWEN36_MATH_FIDELITY``, falling back to ``default``."""
    name = os.environ.get("QWEN36_MATH_FIDELITY", default).strip().lower()
    if name not in _FIDELITIES:
        raise ValueError(f"QWEN36_MATH_FIDELITY={name!r} is not one of {sorted(_FIDELITIES)}")
    return _FIDELITIES[name]

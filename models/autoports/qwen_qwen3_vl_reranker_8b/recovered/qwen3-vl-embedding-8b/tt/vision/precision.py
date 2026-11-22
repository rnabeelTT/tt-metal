# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Precision switches for the vendored qwen36 vision tower (Qwen3-VL reranker use: single die, no CCL).

``sdpa_bf16``: keep Q/K/V bf16 into SDPA (the qwen36 code typecasts them to bfp8).
``wo_dtype``: attention output projection dtype (qwen36: the decoder activation dtype or bfp8).
``mlp_ckc``: callable(args) -> compute kernel config for the two MLP matmuls, or None for the qwen36 default.
``mlp_weight_dtype``: fc1 / fc2 weight dtype, or None for the qwen36 default (bfp8).
"""
import os

import ttnn


def _hifi4_fp32(args):
    return ttnn.init_device_compute_kernel_config(
        args.mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


def _hifi2_fp32(args):
    return ttnn.init_device_compute_kernel_config(
        args.mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


MLP_CKC = {"default": lambda args: None, "hifi4_fp32": _hifi4_fp32, "hifi2_fp32": _hifi2_fp32}

VISION_PRECISION = {
    "sdpa_bf16": os.environ.get("QWEN3VL_VISION_SDPA_BF16", "1") == "1",
    "wo_dtype": {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b}[os.environ.get("QWEN3VL_VISION_WO_DTYPE", "bf16")],
    "mlp_ckc": MLP_CKC[os.environ.get("QWEN3VL_VISION_MLP_CKC", "hifi4_fp32")],
    "ln_fp32": os.environ.get("QWEN3VL_VISION_LN_FP32", "1") == "1",
    "mlp_weight_dtype": {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, "default": None}[
        os.environ.get("QWEN3VL_VISION_MLP_WDTYPE", "default")
    ],
}


def vision_mm(x, w, bias=None, activation=None, ckc=None, dtype=None):
    """Vision projections through minimal_matmul with the tt_dit Blackhole blocking (the qwen36 ``ttnn.linear``
    program configs ran the 1536 -> 1152 output projection on 24 cores). Leading dims fold into M."""
    from models.tt_dit.utils.matmul import get_matmul_config

    shape = list(x.shape)
    M = 1
    for d in shape[:-1]:
        M *= int(d)
    K, N = int(w.shape[-2]), int(w.shape[-1])
    key = (M, K, N)
    cfg = _VMM_CFG.get(key)
    if cfg is None:
        cfg = _VMM_CFG[key] = get_matmul_config(M, K, N, x.device().compute_with_storage_grid_size())
    kw = {}
    if activation is not None:
        kw["fused_activation"] = activation
    return ttnn.experimental.minimal_matmul(
        x, w, bias_tensor=bias, config=cfg, compute_kernel_config=ckc, dtype=dtype, **kw
    )


_VMM_CFG = {}
VISION_PRECISION["use_minimal_matmul"] = os.environ.get("QWEN3VL_VISION_MINIMAL_MM", "1") == "1"
VISION_PRECISION["sdpa_chunk"] = int(os.environ.get("QWEN3VL_VISION_SDPA_CHUNK", "256"))  # 0 = qwen36 default
VISION_PRECISION["sdpa_fidelity"] = os.environ.get(
    "QWEN3VL_VISION_SDPA_FIDELITY", "hifi2"
)  # "" = qwen36 default (HiFi4)

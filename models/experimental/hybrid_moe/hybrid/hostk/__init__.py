# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""ctypes binding of hostk.c: one layer's missed experts in one call, on the library's torch CPU int4 pack."""

import ctypes
import os
import subprocess

_DIR = os.path.dirname(os.path.abspath(__file__))
_SO = os.path.join(_DIR, "libhostk.so")
_lib = None
MAXR = 16  # hostk.c MAXR


def build():
    """Compile libhostk.so (on the host: the serving image has no compiler; serve_397b.sh calls this)."""
    src = os.path.join(_DIR, "hostk.c")
    cmd = ["gcc", "-O3", "-march=x86-64-v4", "-mtune=znver4", "-fPIC", "-shared", "-o", _SO, src, "-lpthread", "-lm"]
    subprocess.run(cmd, check=True)


def load(nthreads):
    global _lib
    if _lib is None:
        if not os.path.exists(_SO):
            raise FileNotFoundError(f"{_SO} not built: run `python -m hybrid.hostk` on the host (needs gcc)")
        lib = ctypes.CDLL(_SO)
        lib.hostk_init.argtypes = [ctypes.c_int]
        lib.hostk_moe.argtypes = [ctypes.c_int] * 4 + [ctypes.c_void_p, ctypes.c_int] + [ctypes.c_void_p] * 7
        lib.hostk_init(int(nthreads))
        _lib = lib
    return _lib


def moe(lib_t, X, rows, W, out):
    """out [R, H] fp32 += the weighted experts: X [R, H] fp32, rows [E] int64 (library rows), W [E, R] fp32
    (0 = row does not use the expert). lib_t: the layer's gu_p / gu_sz / dn_p / dn_sz / gs tensors (contiguous)."""
    R, H = X.shape
    I = lib_t["dn_p"].shape[2] * 2
    for t in (X, rows, W, out):
        if not t.is_contiguous():
            raise ValueError("hostk.moe: non-contiguous input")
    rc = _lib.hostk_moe(
        R,
        H,
        I,
        int(lib_t["gs"]),
        X.data_ptr(),
        rows.numel(),
        rows.data_ptr(),
        W.data_ptr(),
        lib_t["gu_p"].data_ptr(),
        lib_t["gu_sz"].data_ptr(),
        lib_t["dn_p"].data_ptr(),
        lib_t["dn_sz"].data_ptr(),
        out.data_ptr(),
    )
    if rc != 0:
        raise RuntimeError(f"hostk_moe returned {rc} (R={R} H={H} I={I} E={rows.numel()})")

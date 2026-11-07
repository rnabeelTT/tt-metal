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


def build(so=_SO):
    """Compile libhostk.so (on the host: the serving image has no compiler; serve_397b.sh calls this). The new file
    replaces the old one by rename, so a server that has the old one mapped keeps running."""
    src = os.path.join(_DIR, "hostk.c")
    tmp = so + ".tmp"
    cmd = ["gcc", "-O3", "-march=x86-64-v4", "-mtune=znver4", "-fPIC", "-shared", "-o", tmp, src, "-lpthread", "-lm"]
    subprocess.run(cmd, check=True)
    os.replace(tmp, so)


def load(nthreads, so=_SO):
    global _lib
    if _lib is None:
        if not os.path.exists(so):
            raise FileNotFoundError(f"{so} not built: run `python -m hybrid.hostk` on the host (needs gcc)")
        lib = ctypes.CDLL(so)
        lib.hostk_init.argtypes = [ctypes.c_int]
        lib.hostk_moe.argtypes = (
            [ctypes.c_int] * 4
            + [ctypes.c_void_p, ctypes.c_int]
            + [ctypes.c_void_p] * 3
            + [ctypes.c_int]
            + [ctypes.c_void_p] * 5
        )
        lib.hostk_planarize.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int, ctypes.c_int]
        lib.hostk_init(int(nthreads))
        _lib = lib
    return _lib


def planarize(p, inverse=False):
    """In place: a uint8 int4-pack tensor [..., N, K/2] to the bit-plane layout (inverse: back to torch's pack)."""
    import torch

    if p.dtype != torch.uint8 or not p.is_contiguous() or p.shape[-2] % 64:
        raise ValueError(f"hostk.planarize: need a contiguous uint8 [..., N % 64 == 0, K/2], got {p.dtype} {p.shape}")
    rc = _lib.hostk_planarize(p.data_ptr(), p.numel() // (p.shape[-1] * 64), p.shape[-1] * 2, int(inverse))
    if rc != 0:
        raise RuntimeError(f"hostk_planarize returned {rc} (shape {tuple(p.shape)})")


def moe_prepared(lib_t, X, rows, W, out, lo=None):
    """moe() split for threads: the checks and the argument list here (the caller's thread, holding the GIL), the
    returned zero-argument function only makes the C call (ctypes releases the GIL for it). The tensors must stay
    alive until it has run."""
    import functools

    args = _moe_args(lib_t, X, rows, W, out, lo)
    return functools.partial(_call, args, X.shape, rows.numel())


def _call(args, shape, n):
    rc = _lib.hostk_moe(*args)
    if rc != 0:
        raise RuntimeError(f"hostk_moe returned {rc} (R, H = {tuple(shape)}, E={n})")


def moe(lib_t, X, rows, W, out, lo=None):
    _call(_moe_args(lib_t, X, rows, W, out, lo), X.shape, rows.numel())


def _moe_args(lib_t, X, rows, W, out, lo=None):
    """out [R, H] fp32 += the weighted experts: X [R, H] fp32, rows [E] int64 (library rows), W [E, R] fp32
    (0 = row does not use the expert). lib_t: the layer's gu_p / gu_sz / dn_p / dn_sz / gs tensors (contiguous),
    with "planar" set once planarize() ran on gu_p / dn_p. lo [E] int8 (bit-plane layout only): 1 = that expert
    reads only its hi plane (2-bit approximation)."""
    R, H = X.shape
    I = lib_t["dn_p"].shape[2] * 2
    import torch

    planar = bool(lib_t.get("planar", False))
    if lo is not None and (not planar or lo.dtype != torch.int8 or lo.numel() != rows.numel()):
        raise ValueError(
            f"hostk.moe: lo needs the bit-plane layout and int8 [E] (planar={planar}, lo={lo.dtype} "
            f"{tuple(lo.shape)}, E={rows.numel()})"
        )
    for t in (X, rows, W, out) + ((lo,) if lo is not None else ()):
        if not t.is_contiguous():
            raise ValueError("hostk.moe: non-contiguous input")
    return (
        R,
        H,
        I,
        int(lib_t["gs"]),
        X.data_ptr(),
        rows.numel(),
        rows.data_ptr(),
        W.data_ptr(),
        lo.data_ptr() if lo is not None else None,
        int(planar),
        lib_t["gu_p"].data_ptr(),
        lib_t["gu_sz"].data_ptr(),
        lib_t["dn_p"].data_ptr(),
        lib_t["dn_sz"].data_ptr(),
        out.data_ptr(),
    )

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The C++ object behind a nanobind-bound Python object (ttnn's MeshDevice, Tensor), for the ctypes extensions here.
nanobind's instance header: PyObject_HEAD, int32 offset, uint32 flags (bit 2: direct = the C++ object is stored at
self + offset; otherwise self + offset holds a pointer to it)."""

import ctypes


def cpp_ptr(obj):
    base = id(obj)
    hdr = ctypes.sizeof(ctypes.c_ssize_t) + ctypes.sizeof(ctypes.c_void_p)  # ob_refcnt, ob_type
    offset = ctypes.c_int32.from_address(base + hdr).value
    flags = ctypes.c_uint32.from_address(base + hdr + 4).value
    direct = (flags >> 2) & 1
    addr = base + offset
    return addr if direct else ctypes.c_void_p.from_address(addr).value

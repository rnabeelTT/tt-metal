# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Probe tt-metal's pinned-memory limits on the (1, 4) mesh. usage (in the image): python dbg_pin.py <out dir>"""
import ctypes
import sys

sys.path.insert(0, "/poc")
import ttnn  # noqa: E402
from hybrid.pin.nbptr import cpp_ptr  # noqa: E402

ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
try:
    lib = ctypes.CDLL("/poc/hybrid/pin/pinprobe.so")
    out = (ctypes.c_uint64 * 4)()
    p = cpp_ptr(mesh)
    print(f"mesh object {hex(id(mesh))} -> C++ {hex(p)}", flush=True)
    lib.pin_params(ctypes.c_void_p(p), out)
    print(
        f"RESULT max_pins {out[0]} max_total_pin_size {out[1] / 2**30:.2f} GiB can_map_to_noc {out[2]} "
        f"supports_read_only {out[3]}",
        flush=True,
    )
finally:
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

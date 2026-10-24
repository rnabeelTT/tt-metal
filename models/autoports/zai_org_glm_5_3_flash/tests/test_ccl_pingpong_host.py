# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only test of the stage-12 persistent all-gather buffers (MeshCCL, ag_persistent=True in DEFAULT_MC): one
output buffer per (call site, shape, dtype, layer parity), shared by the layers of one parity on a TT_CCL. No device:
ttnn.zeros and the all-gather op are stand-ins that record the buffer each call writes."""

import dataclasses
import types

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tt import multichip_decoder as MC


class _X:
    def __init__(self, shape, dtype):
        self.shape, self.dtype = shape, dtype

    def is_sharded(self):
        return False


class _Buf:
    def memory_config(self):
        return None


def test_persistent_all_gather_ping_pong(monkeypatch):
    assert MC.DEFAULT_MC.ag_persistent  # stage 12 default
    written = []
    monkeypatch.setattr(ttnn, "zeros", lambda *a, **k: _Buf())
    monkeypatch.setattr(ttnn, "deallocate", lambda *a, **k: None)
    monkeypatch.setattr(
        ttnn.experimental,
        "all_gather_async",
        lambda xi, persistent_output_buffer, **k: written.append(persistent_output_buffer),
    )
    tt_ccl = types.SimpleNamespace(
        get_and_cycle_ag_semaphore_handles=lambda: None,
        get_and_cycle_barrier_semaphore_handle=lambda: None,
        get_num_links=lambda: 1,
    )
    mesh = types.SimpleNamespace(shape=(1, 4))
    mc = dataclasses.replace(MC.DEFAULT_MC, num_links=1)
    ccls = []
    for layer in range(4):  # the parity MultichipDecoder.from_state_dict sets
        c = MC.MeshCCL(mesh, tt_ccl, mc)
        c.parity = layer % 2
        c.topology = lambda: ttnn.Topology.Linear  # "auto" probes the device
        ccls.append(c)
    x = _X([1, 1, 32, 1024], ttnn.bfloat16)
    out = [c.all_gather(x, dim=3, dtype=ttnn.bfloat16, decode=True, site="h_attn") for c in ccls]
    assert written == out
    assert out[0] is out[2] and out[1] is out[3]  # layer L's buffer is next written at layer L + 2
    assert out[0] is not out[1]  # adjacent layers never share a buffer
    assert all(c.is_persistent(o) for c, o in zip(ccls, out))
    keys = set(tt_ccl._glm53_ag_bufs)
    assert keys == {("h_attn", (1, 1, 32, 4096), ttnn.bfloat16, 0), ("h_attn", (1, 1, 32, 4096), ttnn.bfloat16, 1)}
    # another call site at the same parity takes its own buffer
    other = ccls[0].all_gather(x, dim=3, dtype=ttnn.bfloat16, decode=True, site="h_ffn")
    assert other is not out[0] and len(tt_ccl._glm53_ag_bufs) == 3

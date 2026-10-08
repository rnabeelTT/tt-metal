# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device-resident real-length inputs of one masked fixed-bucket prefill (stage 7, traced prefill).

The masked-bucket prefill pads a segment of ``actual_len`` real tokens to a fixed ``bucket`` and masks the GDN
recurrence / conv state to the real length. In the stock path every GDN layer builds its masks from the host int with
``ttnn.from_torch`` (a blocking host write per layer, and a "write during trace capture" fatal under a trace). Here the
masks are PERSISTENT device tensors, staged once per segment with ``copy_host_to_device_tensor``; the kernels only see
their addresses, so a bucket forward is a fixed program that can be captured as a trace and replayed for any length.

Tensors (all replicated across the mesh, TILE layout):
  mask_f32   [1, bucket, 1] fp32  1.0 for positions < actual_len (beta / g mask of the fused chunk kernel)
  mask_q     [1, bucket, 1] bf16  same mask for the flat q/k/v rows
  conv_sel   [1, K-1, bucket+K-1] bf16 one-hot rows selecting the last K-1 REAL conv inputs (decode conv window)
  logit_sel  [1, 1, 32, bucket] bf16: row 0 one-hot selecting hidden row actual_len-1 (next-token logits), rows 1..31 zero
             (a 32-row selection so the bucket program's logits have the decode sampler's [1,1,32,vocab/TP] spec)
"""
import torch

import ttnn


class PrefillLen:
    """Per-bucket persistent length inputs; ``stage(actual_len)`` writes them (host -> device copy, no allocation)."""

    def __init__(self, mesh_device, bucket: int, conv_k: int, q_dtype=ttnn.bfloat16):
        self.mesh_device = mesh_device
        self.bucket = int(bucket)
        self.K = int(conv_k)
        self.actual_len = None
        self._rep = ttnn.ReplicateTensorToMesh(mesh_device)
        z = torch.zeros
        self.mask_f32 = self._dev(z(1, self.bucket, 1), ttnn.float32)
        self.mask_q = self._dev(z(1, self.bucket, 1), q_dtype)
        self.conv_sel = self._dev(z(1, self.K - 1, self.bucket + self.K - 1), ttnn.bfloat16)
        self.logit_sel = self._dev(z(1, 1, 32, self.bucket), ttnn.bfloat16)

    def _dev(self, t, dtype):
        return ttnn.from_torch(
            t,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self._rep,
        )

    def _host(self, t, dtype):
        return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=None, mesh_mapper=self._rep)

    def stage(self, actual_len: int):
        """Write the masks for ``actual_len`` real tokens (1 <= actual_len <= bucket). Idempotent per length."""
        actual_len = int(actual_len)
        assert 1 <= actual_len <= self.bucket, (actual_len, self.bucket)
        if self.actual_len == actual_len:
            return
        m = torch.zeros(1, self.bucket, 1, dtype=torch.float32)
        m[:, :actual_len, :] = 1.0
        ttnn.copy_host_to_device_tensor(self._host(m, ttnn.float32), self.mask_f32)
        ttnn.copy_host_to_device_tensor(self._host(m, self.mask_q.dtype), self.mask_q)
        sel = torch.zeros(1, self.K - 1, self.bucket + self.K - 1, dtype=torch.float32)
        for j in range(self.K - 1):
            sel[0, j, actual_len + j] = 1.0  # x[i] sits at x_padded index (K-1)+i -> rows actual_len-(K-1)+j
        ttnn.copy_host_to_device_tensor(self._host(sel, ttnn.bfloat16), self.conv_sel)
        ls = torch.zeros(1, 1, 32, self.bucket, dtype=torch.float32)
        ls[0, 0, 0, actual_len - 1] = 1.0
        ttnn.copy_host_to_device_tensor(self._host(ls, ttnn.bfloat16), self.logit_sel)
        self.actual_len = actual_len

    # duck-typed protocol read by the kernels (``hasattr(valid_len, "mask_f32")``)
    def __int__(self):
        return int(self.actual_len)

    def __repr__(self):
        return f"PrefillLen(bucket={self.bucket}, actual_len={self.actual_len})"

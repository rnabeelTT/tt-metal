# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The Qwen3.5-9B Gated DeltaNet layer — composes config/weights/state/prefill/decode.

Wraps the experimental ``gated_deltanet_forward_ttnn()`` and the on-device GDN prefill
kernel into a module that manages weight tensors, recurrent state, and conv state.
"""
import ttnn
from models.demos.blackhole.qwen36.tt.gdn.config import GDNConfig
from models.demos.blackhole.qwen36.tt.gdn.decode import recurrent_forward
from models.demos.blackhole.qwen36.tt.gdn.state import init_recurrent_state, restore_split_conv_from_fused
from models.demos.blackhole.qwen36.tt.gdn.weights import load_gdn_weights


class Qwen36GatedDeltaNet:
    """Gated DeltaNet (linear attention) layer for Qwen3.5-9B.

    Maintains fixed-size recurrent state [B, H, K, V] that replaces the KV cache.
    Also maintains conv states [B, kernel_size-1, D] for causal conv1d history.
    Supports two modes:
      - "recurrent": single-token decode (T=1), O(1) memory
      - "chunk": multi-token prefill (T>1), chunked parallel processing
    """

    def __init__(self, mesh_device, config: GDNConfig, state_dict, tensor_cache_path=None):
        self.device = mesh_device
        self.cfg = config

        # Mirror config-derived scalar dims so the forward bodies read them directly.
        self.num_heads = config.num_heads
        self.num_v_heads = config.num_v_heads
        self.head_k_dim = config.head_k_dim
        self.head_v_dim = config.head_v_dim
        self.conv_kernel_size = config.conv_kernel_size
        self.norm_eps = config.norm_eps
        self.long_prefill_chunk_size = config.long_prefill_chunk_size

        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        self.compute_kernel_config_decode = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        self.weights = load_gdn_weights(mesh_device, config, state_dict, tensor_cache_path)

        # ---- Runtime state (plain instance attributes, exact same names as before;
        # poked directly by the trace machinery in model.py / qwen36_vllm.py) ----
        self.recurrent_state = None
        # Conv states: ttnn tensors on device [B, kernel_size-1, D]
        self.conv_state_q = None
        self.conv_state_k = None
        self.conv_state_v = None
        # Fused conv state [B, kernel_size-1, D_total] where D_total = q_dim + k_dim + v_dim
        self.fused_conv_state = None
        self.split_conv_state = None
        # Trace capture support
        self.use_inplace_state = False
        # When True (set during chunk-outer traced-prefill capture), the chunk (prefill)
        # path writes recurrent + conv state into the persistent external buffers IN PLACE
        # (ttnn.copy) instead of reassigning a fresh tensor, so the state carries across
        # execute_trace() replays (each replay re-runs the same baked buffer addresses).
        # Eager prefill keeps the reassign path. See Qwen36Model.capture_prefill_trace_chunked.
        self._chunk_inplace_state = False

    def forward(self, x, mode="recurrent", chunk_size=None, valid_len=None):
        return recurrent_forward(self, x, mode=mode, chunk_size=chunk_size, valid_len=valid_len)

    def set_external_state(self, recurrent_state, conv_state):
        """Point layer at externally-allocated state buffers.
        Sets use_inplace_state=True so all forward passes write state inplace (preserving buffer addresses).
        Does NOT create split_conv_state — that happens after prefill when there is real data to split.
        """
        # The batch dim is whatever the caller allocated. Everything downstream reads it
        # off the tensor (`B = q.shape[0]` in the delta rule, likewise the conv), so the
        # only thing that ever required batch 1 here was this assert -- it hardcoded a
        # leading 1 and rejected a perfectly good [B, ...] state with
        # "recurrent_state shape mismatch: [4, 32, 128, 128] != [1, 32, 128, 128]".
        # The trailing three dims are the real contract and are still checked.
        got = list(recurrent_state.shape)
        expected_tail = [self.num_v_heads, self.head_k_dim, self.head_v_dim]
        assert (
            len(got) == 4 and got[1:] == expected_tail and got[0] >= 1
        ), f"recurrent_state shape mismatch: {got} != [B, {', '.join(map(str, expected_tail))}]"
        assert (
            conv_state.shape[1] == self.conv_kernel_size - 1
        ), f"conv_state dim 1 mismatch: {conv_state.shape[1]} != {self.conv_kernel_size - 1}"
        # The caller is installing FRESH state buffers, so every view derived from the
        # PREVIOUS ones is stale and must go with them. The decode conv reads
        # `split_conv_state` and gdn/decode.py only rebuilds it "when it is None", so a
        # leftover split state silently carries the last sequence's conv window into the
        # next one: after the documented free_kv_caches() + allocate_kv_caches() reset the
        # recurrent and fused conv states read back as exactly zero while all 30 GDN
        # layers still held a split state with absmax ~30. free_kv_caches never released
        # those buffers either, so they also leaked one conv state per layer per
        # generation. Dropping them here is what _reset_dn_state_inplace already does for
        # the in-place trace path ("rebuilt lazily on first decode").
        self.drop_derived_conv_state()
        self.recurrent_state = recurrent_state
        self.fused_conv_state = conv_state
        self.use_inplace_state = True

    def drop_derived_conv_state(self):
        """Release the conv state derived from a previous fused/per-stream buffer.

        `split_conv_state` owns its buffers (gdn/state.py clones them), so they are
        deallocated; the per-stream conv states are views handed back by the forward and
        are only dropped. Both are rebuilt lazily on the next decode.
        """
        if self.split_conv_state is not None:
            for buf in self.split_conv_state:
                ttnn.deallocate(buf)
            self.split_conv_state = None
        self.conv_state_q = None
        self.conv_state_k = None
        self.conv_state_v = None

    def _restore_split_conv_from_fused(self):
        """Copy fused_conv_state slices into existing split_conv_state buffers.
        Preserves device addresses (critical for trace replay).
        Kept as a method because model.py calls it on the instance.
        """
        restore_split_conv_from_fused(self)

    def reset_state(self, batch_size=None):
        if batch_size is not None:
            init_recurrent_state(self, batch_size)
        else:
            self.recurrent_state = None
        self.drop_derived_conv_state()
        self.fused_conv_state = None

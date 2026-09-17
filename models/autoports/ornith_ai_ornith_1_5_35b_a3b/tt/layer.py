# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Ornith decoder layer: the Qwen3.5 hybrid layer with an MoE feed-forward.

Everything else in the layer is unchanged from the qwen36 port -- Gated DeltaNet,
gated full attention, partial/m-RoPE, zero-centred RMSNorm, the residual pattern, the
tensor-parallel norm fusions. Only the feed-forward differs, so this subclass overrides
exactly that hook and inherits the rest.
"""
import ttnn
from models.autoports.ornith_ai_ornith_1_5_35b_a3b.tt.moe import OrnithMoE
from models.demos.blackhole.qwen36.tt.layer import Qwen36DecoderLayer
from models.demos.blackhole.qwen36.utils.substate import substate


class _FracturedFF:
    """Adapt a REPLICATED feed-forward to the tensor-parallel layer contract.

    Under TP the residual stream carried between layers is FRACTURED along the hidden
    dim (embedding is ShardTensor2dMesh(dims=(None, 3)); GDN/attention out-projections
    reduce-scatter), and the layer adds the FF output straight onto it:
    ``output = ttnn.add(h, ff_output)`` (qwen36/tt/layer.py:285). Qwen36MLP satisfies
    that because its down-projection is row-parallel + reduce-scatter.

    OrnithMoE does not: its experts are REPLICATED (expert parallelism is opt-in via
    ORNITH_MOE_EP), so every die computes the same full-width [.,.,T,dim] result. Adding
    that to a dim/tp residual raises "Invalid subtile broadcast type" from ttnn.add.

    Rather than re-plumb the MoE's 256-expert bank for column/row parallelism, slice the
    replicated result to this die's shard with ``ttnn.mesh_partition`` -- a local, no-
    communication op (every die already holds an identical tensor, so taking the i-th
    slice on die i reconstitutes exactly the fractured residual). The arithmetic is
    unchanged; only the placement is.
    """

    def __init__(self, inner, cluster_axis=1):
        self.inner = inner
        self.cluster_axis = cluster_axis

    def __getattr__(self, name):
        # Delegate everything else (e.g. `.bank`, used by tools/repro_nondeterminism.py).
        return getattr(self.inner, name)

    def forward(self, x, mode: str = "decode"):
        y = self.inner.forward(x, mode=mode)
        frac = ttnn.mesh_partition(y, dim=-1, cluster_axis=self.cluster_axis)
        ttnn.deallocate(y)
        return frac


class OrnithDecoderLayer(Qwen36DecoderLayer):
    """Qwen3.5 hybrid layer whose MLP is a 256-expert top-8 MoE block."""

    def _build_feed_forward(self, mesh_device, args, state_dict, layer_num, tensor_cache_path, tt_ccl):
        moe_state = substate(state_dict, f"layers.{layer_num}.mlp")
        cache = (tensor_cache_path / f"layers.{layer_num}") if tensor_cache_path else None
        # tt_ccl carries the collective context the tensor-parallel reduce needs.
        moe = OrnithMoE(mesh_device, args, moe_state, layer_id=layer_num, tensor_cache_path=cache, tt_ccl=tt_ccl)
        if getattr(args, "num_devices", 1) > 1:
            return _FracturedFF(moe, cluster_axis=1)
        return moe

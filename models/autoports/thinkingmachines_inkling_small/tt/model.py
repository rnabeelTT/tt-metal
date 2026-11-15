# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Inkling-Small model on the qwen35 autoport framework (exports ``Qwen35MoeModel`` / ``create_model``, the names the
hybrid overlay and the serving glue import).

The M2.7 port's model class with the Inkling decoder layer. Model-level differences are folded into the weights by
tt/weights.StreamingStateDict (normed embedding table, final norm / 16, dense global_scale); there is no RoPE (the
framework's RoPE tables are built at a dummy width and never read by the layers). The 4 short-conv states per layer
are per-sequence: reset_sequence_state() before a new sequence (the serving glue calls it on prefill).
"""

from __future__ import annotations

import json
import os

from loguru import logger

from models.autoports.minimaxai_minimax_m2_7.tt import precision
from models.autoports.minimaxai_minimax_m2_7.tt.model import Qwen35MoeModel as _M27Model
from models.autoports.minimaxai_minimax_m2_7.tt.model import _CacheRoot
from models.autoports.minimaxai_minimax_m2_7.tt.multichip_decoder import MULTICHIP_SELECTED, MultichipDecoder
from models.autoports.thinkingmachines_inkling_small.tt.layer import InklingDecoderLayer
from models.autoports.thinkingmachines_inkling_small.tt.model_config import Qwen35MoeModelArgs, check_mesh_shape
from models.tt_transformers.tt.common import Mode

ROPE_DUMMY_DIM = 64  # the framework builds RoPE tables at args.rope_head_dim; Inkling's layers never read them


class Qwen35MoeModel(_M27Model):
    DECODER_LAYER_CLS = InklingDecoderLayer

    def __init__(self, mesh_device, args, state_dict, tensor_cache_path=None):
        if not args.rope_head_dim:
            args.rope_head_dim = ROPE_DUMMY_DIM
            args.partial_rotary_factor = ROPE_DUMMY_DIM / args.head_dim
        super().__init__(mesh_device, args, state_dict, tensor_cache_path)

    def prefill_tokens(self, ids, page_table_host, start=0, chunk=None, logits_rows="last", buckets=None,
                       on_chunk=None, group=None):
        """Eager chunked prefill of token ids [T] (torch) from position `start` (a multiple of the chunk; the conv
        states must hold the sequence up to start). Chunks of `chunk` rows (default the layers' prefill_chunk). A
        shorter (last) chunk is padded to the smallest of `buckets` (row counts, multiples of 64) that holds it, else to
        the next multiple of 64: only its real rows advance the conv states, and the hybrid MoE computes no host experts
        for the padding rows (HYB.pf_valid). Bucketing bounds the set of prefill programs, so a server can compile them
        all before its trace captures.

        Layer-major in groups of `group` chunks (default INKLING_PREFILL_GROUP, 8): every layer runs the group's chunks
        in order before the next layer (each layer's KV / conv states still advance chunk by chunk), so the experts
        the hybrid overlay streams into its staging for one layer serve the whole group instead of one chunk.

        Returns the logits of the last real row ([V] torch fp32) or, logits_rows="all", of every real row ([T, V]).
        on_chunk(c0, n, Tp, x): called after each chunk's layers with the chunk's final hidden x [1, 1, Tp, H/N]
        (fractured, before the final norm; e.g. the MTP drafter's prompt pass), n real rows of Tp."""
        import os
        import sys

        import torch

        import ttnn

        chunk = chunk or self.layers[0].attention.prefill_chunk
        group = int(group or os.environ.get("INKLING_PREFILL_GROUP", "8"))
        hyb = getattr(sys.modules.get("hybrid.patch"), "HYB", None)
        rep = ttnn.ReplicateTensorToMesh(self.mesh_device)
        V = int(getattr(self.args, "unpadded_vocab_size", self.args.vocab_size))
        T = int(ids.shape[0])
        chunks = []
        for c0 in range(0, T, chunk):
            n = min(chunk, T - c0)
            Tp = -(-n // 64) * 64
            if buckets:
                Tp = min((int(b) for b in buckets if int(b) >= n), default=Tp)
            chunks.append((c0, n, Tp, n if n < Tp else None))
        out = []
        for g0 in range(0, len(chunks), group):
            grp = chunks[g0 : g0 + group]
            xs = []
            for c0, n, Tp, _ in grp:
                tok = torch.zeros(1, 1, 1, Tp, dtype=torch.int32)
                tok[0, 0, 0, :n] = ids[c0 : c0 + n].to(torch.int32)
                tok_tt = ttnn.from_torch(tok.reshape(1, Tp), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT,
                                         device=self.mesh_device, mesh_mapper=rep)
                x = self.embd(tok_tt)
                xs.append(ttnn.reshape(x, (1, 1, Tp, x.shape[-1])))
                ttnn.deallocate(tok_tt)
            memlog = os.environ.get("INKLING_PREFILL_MEMLOG") == "1" and g0 == 0
            for li, layer in enumerate(self.layers):
                if memlog:
                    mv = ttnn.get_memory_view(self.mesh_device, ttnn.BufferType.DRAM)
                    print(f"prefill memlog layer {li}: DRAM free {mv.total_bytes_free_per_bank * mv.num_banks / 2**20:.0f}"
                          f" MiB, largest block {mv.largest_contiguous_bytes_free_per_bank / 2**20:.0f} MiB/bank",
                          flush=True)
                for i, (c0, n, Tp, valid) in enumerate(grp):
                    if hyb is not None:
                        hyb.pf_valid = valid
                    try:
                        xs[i] = layer.forward_prefill(xs[i], start + c0, page_table_host, valid=valid)
                    finally:
                        if hyb is not None:
                            hyb.pf_valid = None
            for (c0, n, Tp, _), x in zip(grp, xs):
                if on_chunk is not None:
                    on_chunk(c0, n, Tp, x)
                if logits_rows == "all" or c0 + n == T:
                    # rows [r0, r1) of the chunk: everything, or the 32-row tile holding the last real row
                    r0, r1 = (0, Tp) if logits_rows == "all" else (((n - 1) // 32) * 32, ((n - 1) // 32) * 32 + 32)
                    # the LM head in 128-row slices: all rows' logits at once (rows x vocab/N) can exceed the DRAM
                    # left beside the experts
                    ts = []
                    for s0 in range(r0, r1, 128):
                        xsl = ttnn.slice(x, (0, 0, s0, 0), (1, 1, min(s0 + 128, r1), x.shape[-1]))
                        xn = self.norm(xsl, mode=Mode.PREFILL)
                        lg = self._lm_head(xn)
                        ts.append(ttnn.to_torch(ttnn.get_device_tensors(lg)[0]).float().reshape(-1, lg.shape[-1])[:, :V])
                        for z in (xsl, xn, lg):
                            ttnn.deallocate(z)
                    t = torch.cat(ts)
                    out.append(t[:n] if logits_rows == "all" else t[n - 1 - r0 : n - r0])
                ttnn.deallocate(x)
        res = torch.cat(out)
        return res[-1] if logits_rows == "last" else res

    def reset_sequence_state(self):
        """Zero every layer's short-conv states (a new sequence)."""
        for layer in self.layers:
            layer.reset()


def create_model(mesh_device, *, max_batch_size=1, max_seq_len=None, n_layers=None, layer_indices=None, mc=None):
    """(args, model) for the mesh; ``layer_indices`` selects a reduced model (probes, layer tests)."""
    mc = mc or MULTICHIP_SELECTED
    policy, policy_path = precision.load_policy(num_devices=mesh_device.get_num_devices())
    mc = precision.apply_policy(mc, policy)
    logger.info(f"precision policy {policy.get('config_id')} ({policy_path or 'code defaults'}): "
                f"{json.dumps(precision.describe_config(mc), sort_keys=True)}")
    check_mesh_shape(mesh_device)
    args = Qwen35MoeModelArgs(mesh_device, max_batch_size=max_batch_size, max_seq_len=max_seq_len or 32768)
    MultichipDecoder._args_hook(args, mc, mc.to_optimization_config(), mesh_device)
    if layer_indices is not None:
        args.layer_indices = [int(i) for i in layer_indices]
        args.n_layers = len(args.layer_indices)
    elif n_layers is not None:
        args.n_layers = int(n_layers)
        args.attention_type_list = args.attention_type_list[: int(n_layers)]
    use_cache = os.environ.get("QWEN35_TENSOR_CACHE") == "1"
    cache_path = _CacheRoot(args.weight_cache_path(), disabled=not use_cache)
    state_dict = args.load_state_dict()
    model = Qwen35MoeModel(mesh_device, args, state_dict, tensor_cache_path=cache_path)
    model.mc = mc
    model.precision_policy = policy
    return args, model

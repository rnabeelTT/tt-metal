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
import time

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
        self._pad_lm_head(state_dict)
        from models.autoports.thinkingmachines_inkling_small.tt import layer as _layer

        if _layer.REPLICATED and self.num_devices > 1:  # replicated-residual verify rows (tt/layer.py)
            for layer in self.layers:
                layer.enable_replicated()

    def _pad_lm_head(self, state_dict):
        """The vocab-sharded LM head with each die's shard a whole number of tiles: 201024 / 4 = 50256 columns per die
        is half a tile short, so every _lm_head all-gather took all_gather_async's composite path (untilize, 5 permutes,
        a concat, tilize: ~1.3 ms per call, measured with tools/draft_prof.py) on each verify pass and draft depth. Zero
        columns up to a multiple of 32 x N (201088) make it the native gather; their logits (0) sit past the real vocab
        (unpadded_vocab_size 200058), which every consumer slices away. INKLING_PAD_LM_HEAD=0 keeps the stock head."""
        import os

        import torch
        import ttnn

        N = int(self.num_devices)
        if (
            N < 2
            or not getattr(self, "_lmhead_vocab_sharded", False)
            or os.environ.get("INKLING_PAD_LM_HEAD", "1") != "1"
        ):
            return
        w = state_dict["output.weight"]  # [vocab, dim]
        V = int(w.shape[0])
        Vp = -(-V // (32 * N)) * (32 * N)
        if Vp == V:
            return
        wt = torch.zeros(int(w.shape[1]), Vp, dtype=torch.bfloat16)
        wt[:, :V] = w.T.to(torch.bfloat16)
        dtype = self.lm_head_weight.dtype
        ttnn.deallocate(self.lm_head_weight)
        self.lm_head_weight = ttnn.as_tensor(
            wt,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=-1),
        )

    def prefill_tokens(
        self,
        ids,
        page_table_host,
        start=0,
        chunk=None,
        logits_rows="last",
        buckets=None,
        on_chunk=None,
        group=None,
        on_layer=None,
        mm=None,
    ):
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
        (fractured, before the final norm; e.g. the MTP drafter's prompt pass), n real rows of Tp.
        on_layer(li, end, part): called when layer li's conv states hold the sequence up to end (the absolute position
        after a chunk's real rows): part "attn" for the attention-side convs (k, v, attn), "mlp" for the MLP conv
        (pipelined layers advance the attention side ahead of the MLP side; e.g. the engine's prefix snapshots)."""

        from models.autoports.thinkingmachines_inkling_small.tt.attention import prefill_uploads_clear

        prefill_uploads_clear()  # (left over by a prefill that raised, or by direct forward_prefill calls)
        try:
            out = self._prefill_tokens(
                ids, page_table_host, start, chunk, logits_rows, buckets, on_chunk, group, on_layer, mm
            )
        finally:
            prefill_uploads_clear()
        for layer in self.layers:  # the replicated rows' full-width conv states from the prefill's (no-op otherwise)
            layer.sync_replicated_states()
        return out

    def _final_norm_decode(self, x):
        """The final norm of the verify rows: on the replicated residual (INKLING_REPLICATED) the sharded norm program
        without its all-gather (the TP path's lm_head norm config, output in DRAM for the LM head)."""
        import ttnn

        if self.num_devices > 1 and int(x.shape[-1]) == self.args.dim:
            nc = dict(self.args.get_norm_config("lm_head", Mode.DECODE))
            nc["output_mem_config"] = ttnn.DRAM_MEMORY_CONFIG
            x_in = ttnn.to_memory_config(x, nc["sharded_output_config"])
            out = self.norm.norm(x_in, mode=Mode.DECODE, in_sharded=True, out_sharded=True, norm_config=nc)
            ttnn.deallocate(x_in)
            return out
        return super()._final_norm_decode(x)

    def _prefill_tokens(
        self, ids, page_table_host, start, chunk, logits_rows, buckets, on_chunk, group, on_layer, mm=None
    ):
        import os
        import sys

        import torch
        import ttnn

        chunk = chunk or self.layers[0].attention.prefill_chunk
        group = int(group or os.environ.get("INKLING_PREFILL_GROUP", "8"))
        # INKLING_PREFILL_PIPE (default 1): a hybrid layer's chunks overlap -- the device runs the next chunks'
        # attention (INKLING_PREFILL_AHEAD of them, default 2) and the current chunk's experts while the host plans
        pipe = os.environ.get("INKLING_PREFILL_PIPE", "1") == "1"
        ahead = max(1, int(os.environ.get("INKLING_PREFILL_AHEAD", "2")))
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
                tok_tt = ttnn.from_torch(
                    tok.reshape(1, Tp),
                    dtype=ttnn.uint32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=self.mesh_device,
                    mesh_mapper=rep,
                )
                x = self.embd(tok_tt)
                x = ttnn.reshape(x, (1, 1, Tp, x.shape[-1]))
                if mm is not None:  # image / audio rows in place of their placeholder tokens' embeddings (tt/mm.py)
                    from models.autoports.thinkingmachines_inkling_small.tt import mm as _mm

                    x = _mm.merge(x, mm, start + c0, n, Tp, self.mesh_device, int(self.args.dim))
                xs.append(x)
                ttnn.deallocate(tok_tt)
            memlog = os.environ.get("INKLING_PREFILL_MEMLOG") == "1" and g0 == 0
            for li, layer in enumerate(self.layers):
                if memlog:
                    mv = ttnn.get_memory_view(self.mesh_device, ttnn.BufferType.DRAM)
                    print(
                        f"prefill memlog layer {li}: DRAM free {mv.total_bytes_free_per_bank * mv.num_banks / 2**20:.0f}"
                        f" MiB, largest block {mv.largest_contiguous_bytes_free_per_bank / 2**20:.0f} MiB/bank",
                        flush=True,
                    )
                if pipe and hyb is not None and layer.prefill_pipelined():
                    self._prefill_layer_pipelined(layer, li, grp, xs, start, page_table_host, hyb, on_layer, ahead)
                    continue
                for i, (c0, n, Tp, valid) in enumerate(grp):
                    if hyb is not None:
                        hyb.pf_valid = valid
                    try:
                        xs[i] = layer.forward_prefill(xs[i], start + c0, page_table_host, valid=valid)
                    finally:
                        if hyb is not None:
                            hyb.pf_valid = None
                    if on_layer is not None:
                        on_layer(li, start + c0 + n, "attn")
                        on_layer(li, start + c0 + n, "mlp")
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
                        ts.append(
                            ttnn.to_torch(ttnn.get_device_tensors(lg)[0]).float().reshape(-1, lg.shape[-1])[:, :V]
                        )
                        for z in (xsl, xn, lg):
                            ttnn.deallocate(z)
                    t = torch.cat(ts)
                    out.append(t[:n] if logits_rows == "all" else t[n - 1 - r0 : n - r0])
                ttnn.deallocate(x)
        res = torch.cat(out)
        return res[-1] if logits_rows == "last" else res

    @staticmethod
    def _prefill_layer_pipelined(layer, li, grp, xs, start, page_table_host, hyb, on_layer, ahead):
        """One hybrid layer over a group's chunks, overlapped: part 1 (device only) of chunk i + ahead is queued before
        the host plans chunk i, and chunk i - 1 completes (its CPU experts waited for) after chunk i's experts are
        queued. Every conv state still advances chunk by chunk in order."""
        pre, pending = {}, None

        def issue(j):
            c0, n, Tp, valid = grp[j]
            t0 = time.perf_counter()
            pre[j] = layer.prefill_pre(xs[j], start + c0, page_table_host, valid=valid)
            # host time of part 1 (the attention segment's dispatch; its _pf_pre share is t_pf_pre)
            hyb.st["t_pf_part1"] = hyb.st.get("t_pf_part1", 0.0) + time.perf_counter() - t0
            if on_layer is not None:
                on_layer(li, start + c0 + n, "attn")

        def complete(item):
            j, done = item
            hyb.pf_valid = grp[j][3]
            try:
                xs[j] = done()
            finally:
                hyb.pf_valid = None
            if on_layer is not None:
                on_layer(li, start + grp[j][0] + grp[j][1], "mlp")

        for j in range(min(ahead, len(grp))):
            issue(j)
        for i in range(len(grp)):
            hyb.pf_valid = grp[i][3]
            try:
                go = layer.prefill_moe(pre.pop(i))  # host plan (background expert copies start)
            finally:
                hyb.pf_valid = None
            # CQ0 work while the copies run: the next chunk's part 1, the previous chunk's end
            if i + ahead < len(grp):
                issue(i + ahead)
            if pending is not None:
                complete(pending)
            hyb.pf_valid = grp[i][3]
            try:
                done = go()  # this chunk's device experts
            finally:
                hyb.pf_valid = None
            pending = (i, done)
        complete(pending)

    def warm_last_row_slices(self, buckets):
        """Compile prefill_tokens' last-row logits slice at every 32-row offset of every bucket (its offset follows the
        prompt length, and a slice program first compiled after the decode trace captures can have its kernels
        overwritten by the replays)."""
        import torch
        import ttnn

        H = int(self.args.dim) // int(self.mesh_device.get_num_devices())  # the fractured hidden of a chunk's output
        for Tp in buckets:
            x = ttnn.from_torch(
                torch.zeros(1, 1, int(Tp), H, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
            for s0 in range(0, int(Tp), 32):
                ttnn.deallocate(ttnn.slice(x, (0, 0, s0, 0), (1, 1, s0 + 32, H)))
            ttnn.deallocate(x)

    def reset_sequence_state(self):
        """Zero every layer's short-conv states (a new sequence)."""
        for layer in self.layers:
            layer.reset()


def create_model(mesh_device, *, max_batch_size=1, max_seq_len=None, n_layers=None, layer_indices=None, mc=None):
    """(args, model) for the mesh; ``layer_indices`` selects a reduced model (probes, layer tests)."""
    mc = mc or MULTICHIP_SELECTED
    policy, policy_path = precision.load_policy(num_devices=mesh_device.get_num_devices())
    mc = precision.apply_policy(mc, policy)
    logger.info(
        f"precision policy {policy.get('config_id')} ({policy_path or 'code defaults'}): "
        f"{json.dumps(precision.describe_config(mc), sort_keys=True)}"
    )
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

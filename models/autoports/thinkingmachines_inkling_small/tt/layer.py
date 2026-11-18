# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Inkling decoder layer on the qwen35 autoport framework (TP over a (1, N) mesh, fractured residual [1, 1, T, H/N]):

    a = attention(attention_norm(x))          tt/attention.InklingAttention -> reduce-scatter -> fractured
    h = x + attn_sconv(a)                      fp32 short conv on the die's hidden slice (depthwise: no collective)
    f = feed_forward(ffn_norm(h))              dense MLP (layers 0-1) or the Inkling MoE (tt/optimized_decoder.py)
    out = h + mlp_sconv(f)

Same contract as the qwen36 layer (``forward(x, cos, sin, mode, position_tensor, page_table, ...)``; cos / sin are
ignored: no RoPE), the norms and the feed-forward built by the shared code (so the hybrid overlay's MoE patches apply).
Implemented: one-row decode. Prefill / spec-verify rows come next.
"""

from __future__ import annotations

import os

import torch
import ttnn

from models.autoports.minimaxai_minimax_m2_7.tt.layer import Qwen35MoeDecoderLayer as _M27Layer
from models.autoports.thinkingmachines_inkling_small.tt.rows_mm import rows_linear
from models.autoports.thinkingmachines_inkling_small.tt.attention import InklingAttention
from models.autoports.thinkingmachines_inkling_small.tt.sconv import ShortConv
from models.demos.blackhole.qwen36.utils.substate import substate
from models.tt_transformers.tt.common import Mode


class _HFConfigView:
    """The fields InklingAttention reads, from the framework args (args.inkling = the raw text config)."""

    def __init__(self, args):
        r = args.inkling
        L = args.n_layers_total if hasattr(args, "n_layers_total") else len(args.attention_type_list)
        local = set(r["local_layer_ids"])
        self.layer_types = ["hybrid_sliding" if i in local else "hybrid" for i in range(max(L, 64))]
        for k in (
            "hidden_size",
            "head_dim",
            "num_attention_heads",
            "num_key_value_heads",
            "swa_head_dim",
            "swa_num_attention_heads",
            "swa_num_key_value_heads",
            "d_rel",
            "rel_extent",
            "sliding_window_size",
            "log_scaling_n_floor",
            "log_scaling_alpha",
            "rms_norm_eps",
        ):
            setattr(self, k, r[k])


class InklingDenseMLP:
    """Layers 0-1: SwiGLU (I 16384, global_scale folded into down_proj by the state dict), TP: gate/up column-parallel,
    down row-parallel + reduce-scatter (fractured output, like the MoE). bfp8 weights, HiFi4 / fp32 accumulation: two
    layers only, so precision over the shared Qwen36MLP's bfp4 gate/up."""

    def __init__(self, mesh, args, sd, tt_ccl, dtype=ttnn.bfloat8_b):
        self.mesh, self.args, self.tt_ccl = mesh, args, tt_ccl
        self.N = mesh.get_num_devices()
        col, row = ttnn.ShardTensorToMesh(mesh, dim=-1), ttnn.ShardTensorToMesh(mesh, dim=-2)
        up = lambda t, m: ttnn.as_tensor(
            t.to(torch.bfloat16).T.contiguous(),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,  # noqa
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=m,
        )
        self.w1 = up(sd["gate_proj.weight"], col)
        self.w3 = up(sd["up_proj.weight"], col)
        self.w2 = up(sd["down_proj.weight"], row)
        self.ckc = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def forward(self, x, mode="decode", partial=False):
        if x.is_sharded():
            x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
        mc = ttnn.L1_MEMORY_CONFIG if int(x.shape[-2]) <= 32 else ttnn.DRAM_MEMORY_CONFIG
        g = rows_linear(x, self.w1, self.mesh, self.ckc, mc, tag="dense")
        u = rows_linear(x, self.w3, self.mesh, self.ckc, mc, tag="dense")
        h = ttnn.multiply(g, u, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], memory_config=mc)
        ttnn.deallocate(g)
        ttnn.deallocate(u)
        part = rows_linear(h, self.w2, self.mesh, self.ckc, ttnn.DRAM_MEMORY_CONFIG, tag="dense")
        ttnn.deallocate(h)
        if self.N == 1 or partial == "raw":  # raw: the per-device partial for the replicated boundary's all-reduce
            return part
        from models.tt_transformers.tt.ccl import tt_all_reduce

        return tt_all_reduce(
            part,
            self.mesh,
            self.tt_ccl,
            cluster_axis=0,
            dim=3,
            topology=self.args.ccl_topology(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )


# INKLING_REPLICATED (default 0): the verify / decode rows keep the residual replicated on every die (qwen36
# decode_boundary.py's residual layout: [1, 1, R, H] width-sharded over 16 cores); a layer's two boundaries are one bf16
# all-reduce each (+ the full-width short conv + the add) instead of reduce-scatter + all-gather, and its norms run
# locally. Prefill keeps the fractured contract; the full-width conv states are gathered from the fractured ones after
# every prefill (sync_replicated_states).
REPLICATED = os.environ.get("INKLING_REPLICATED", "0") == "1"


class RepBoundary:
    """The model-wide bf16 all-reduce boundary (one scratch buffer + global semaphore per site, "mixer" / "mlp": the
    two sites strictly alternate, as qwen36 DecodeBoundary). bf16 partials in: the BFP8 partial of DecodeBoundary is
    ~2x faster (18.6 vs 29.8 us at 3 rows) but its sum is ~4x further from the exact one (tools/ccl_bench.py)."""

    def __init__(self, mesh, hidden, topology):
        from models.demos.blackhole.qwen36.tt.decode_boundary import norm_config, residual_memory_config

        # INKLING_REP_CORES (default 8,8): the residual / norm / all-reduce shard grid. The scratch buffers are
        # persistent L1 and an L1 address is reserved on every core: on qwen36's 16 cores the two bf16 sites take
        # 2 x 64 KB per core, which the 1536-row prefill's circular buffers need (they clashed by ~63 KB); 64 cores: 2 x 16
        CORES = tuple(int(v) for v in os.environ.get("INKLING_REP_CORES", "8,8").split(","))
        self.mesh, self.hidden, self.topology = mesh, hidden, topology
        self.mc = residual_memory_config(hidden, CORES)
        self.norm_config = norm_config(hidden, CORES)
        nd = mesh.get_num_devices()
        n = CORES[0] * CORES[1]
        scratch_mc = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, hidden * nd // n),
            core_grid=ttnn.CoreGrid(x=CORES[0], y=CORES[1]),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        grid = mesh.compute_with_storage_grid_size()
        sem_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
        self.sites = {
            site: (
                ttnn.empty([1, 1, ttnn.TILE_SIZE, hidden * nd], ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh, scratch_mc),
                ttnn.create_global_semaphore(mesh, sem_cores, 0),
            )
            for site in ("mixer", "mlp")
        }

    def all_reduce(self, partial, site):
        """Sum over the dies of partial [1, 1, R, H] (bf16) -> replicated, in the residual layout; frees partial."""
        p = ttnn.to_memory_config(partial, self.mc, dtype=ttnn.bfloat16)
        if p.buffer_address() != partial.buffer_address():
            ttnn.deallocate(partial)
        scratch, sem = self.sites[site]
        red = ttnn.experimental.all_reduce_async(
            p,
            scratch,
            cluster_axis=1,
            mesh_device=self.mesh,
            multi_device_global_semaphore=sem,
            memory_config=self.mc,
            dtype=ttnn.bfloat16,
            topology=self.topology,
            num_links=1,
        )
        ttnn.deallocate(p)
        return red

    def gather(self, x_frac, tt_ccl):
        """The fractured [1, 1, R, H / N] rows (the embedding) -> the replicated residual."""
        return ttnn.experimental.all_gather_async(
            x_frac,
            persistent_output_buffer=None,
            dim=3,
            multi_device_global_semaphore=tt_ccl.get_and_cycle_ag_semaphore_handles(),
            num_links=1,
            topology=self.topology,
            memory_config=self.mc,
            barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(),
            chunks_per_sync=10,
            num_workers_per_link=2,
            num_buffers_per_channel=2,
        )


def rep_boundary(args, mesh):
    b = args.__dict__.get("_inkling_rep_boundary")
    if b is None:
        b = RepBoundary(mesh, args.dim, args.ccl_topology())
        args._inkling_rep_boundary = b
    return b


class InklingDecoderLayer(_M27Layer):
    def __init__(self, mesh_device, args, state_dict, layer_num, tensor_cache_path=None, tt_ccl=None):
        # not Qwen36DecoderLayer.__init__: it would build the qwen attention; the parts kept are rebuilt here
        self.layer_num = layer_num
        self.device = mesh_device
        self.args = args
        self.tt_ccl = tt_ccl
        self.num_devices = getattr(args, "num_devices", 1)
        self._tp_modules = True
        self.is_full_attention = True  # every layer is attention (the model loop passes position / page table)
        self._decode_replicated = False
        self._fuse_norm_agmm = False
        self._fuse_ff_agmm = False
        self._addnorm = None
        self.attention_norm = self._make_norm(
            mesh_device, args, state_dict, layer_num, "input_layernorm", tensor_cache_path, tt_ccl, "attention_norm"
        )
        self.ffn_norm = self._make_norm(
            mesh_device, args, state_dict, layer_num, "post_attention_layernorm", tensor_cache_path, tt_ccl, "ff_norm"
        )
        attn_sd = substate(state_dict, f"layers.{layer_num}.self_attn")
        self.attention = InklingAttention(
            mesh_device,
            _HFConfigView(args),
            {k: attn_sd[k] for k in attn_sd},
            layer_num,
            tt_ccl=tt_ccl,
            block_size=int(getattr(args, "kv_block_size", 64)),
            max_seq_len=int(args.max_seq_len),
            weight_dtype=getattr(args, "attn_weight_dtype", None) or ttnn.bfloat8_b,
            kv_dtype=getattr(args, "inkling_kv_dtype", None) or getattr(args, "kv_dtype", None) or ttnn.bfloat8_b,
            own_global_cache=bool(getattr(args, "inkling_own_kv", True)),
            # INKLING_PREFILL_CHUNK: rows per prefill launch (default 2048); 1024 halves the prefill transients (the
            # MoE's dispatch buffers) for profiles whose experts leave little DRAM
            prefill_chunk=int(os.environ.get("INKLING_PREFILL_CHUNK", "2048")),
        )
        H, N = args.dim, self.num_devices
        shard = ttnn.ShardTensorToMesh(mesh_device, dim=-1) if N > 1 else None
        self.attn_sconv = ShortConv(
            mesh_device, state_dict[f"layers.{layer_num}.attn_sconv.weight"].reshape(H, -1), shard, H // N
        )
        self.mlp_sconv = ShortConv(
            mesh_device, state_dict[f"layers.{layer_num}.mlp_sconv.weight"].reshape(H, -1), shard, H // N
        )
        # the full-width conv weights, for enable_replicated() (the target model's layers; the MTP block stays fractured)
        self._sconv_w = {
            k: state_dict[f"layers.{layer_num}.{k}.weight"].reshape(H, -1).clone() for k in ("attn_sconv", "mlp_sconv")
        }
        self.rep = None
        self.feed_forward = self._build_feed_forward(
            mesh_device, args, state_dict, layer_num, tensor_cache_path, tt_ccl
        )

    def _build_feed_forward(self, mesh_device, args, state_dict, layer_num, tensor_cache_path, tt_ccl):
        if not args.is_moe_layer(layer_num):  # dense SwiGLU (global_scale folded into down_proj by the state dict)
            return InklingDenseMLP(mesh_device, args, substate(state_dict, f"layers.{layer_num}.mlp"), tt_ccl)
        from models.autoports.thinkingmachines_inkling_small.tt.optimized_decoder import Qwen35MoEOptimized

        mlp_state = substate(state_dict, f"layers.{layer_num}.mlp")
        mlp_cache = (tensor_cache_path / f"layers.{layer_num}") if tensor_cache_path else None
        return Qwen35MoEOptimized(mesh_device, args, mlp_state, mlp_cache, tt_ccl=tt_ccl, layer_num=layer_num)

    def enable_replicated(self):
        """Replicated-residual rows (INKLING_REPLICATED): the boundary and the two full-width convs (built before any
        trace capture)."""
        if self.num_devices == 1 or self.rep is not None:
            return
        H = self.args.dim
        rep = ttnn.ReplicateTensorToMesh(self.device)
        self.rep = rep_boundary(self.args, self.device)
        self.attn_sconv_r = ShortConv(self.device, self._sconv_w["attn_sconv"], rep, H)
        self.mlp_sconv_r = ShortConv(self.device, self._sconv_w["mlp_sconv"], rep, H)

    def sync_replicated_states(self):
        """The full-width conv states := the fractured ones (gathered over the dies), after a prefill wrote them."""
        if self.rep is None:
            return
        for frac, full in ((self.attn_sconv, self.attn_sconv_r), (self.mlp_sconv, self.mlp_sconv_r)):
            t = ttnn.to_layout(frac.st, ttnn.TILE_LAYOUT)
            g = ttnn.experimental.all_gather_async(
                t,
                persistent_output_buffer=None,
                dim=3,
                multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(),
                num_links=1,
                topology=self.args.ccl_topology(),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
                chunks_per_sync=10,
                num_workers_per_link=2,
                num_buffers_per_channel=2,
            )
            ttnn.deallocate(t)
            r = ttnn.to_layout(g, ttnn.ROW_MAJOR_LAYOUT)
            ttnn.deallocate(g)
            ttnn.copy(r, full.st)
            ttnn.deallocate(r)

    def reset(self):
        self.attention.reset()
        self.attn_sconv.reset()
        self.mlp_sconv.reset()
        if self.rep is not None:
            self.attn_sconv_r.reset()
            self.mlp_sconv_r.reset()

    def forward(
        self,
        x,
        cos=None,
        sin=None,
        mode="decode",
        chunk_size=128,
        position_tensor=None,
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        chunk_start_idx_tensor=None,
        valid_len=None,
        gdn_collect=False,
        tau=None,
    ):
        if mode == "rows":
            return self.forward_rows(x, position_tensor, page_table, tau=tau)
        if mode != "decode":
            raise NotImplementedError("Inkling prefill: next step (tt/attn prefill kernel); use mode='rows' groups")
        norm_cfg = self.args.get_norm_config("attn", Mode.DECODE) if self.num_devices > 1 else None
        attn_in = self.attention_norm(x, mode=Mode.DECODE, norm_config=norm_cfg)
        if attn_in.is_sharded():
            attn_in = ttnn.sharded_to_interleaved(attn_in, ttnn.L1_MEMORY_CONFIG)
        a = self.attention.forward_decode(attn_in, position_tensor, page_table, tau=tau)
        ttnn.deallocate(attn_in)
        a2 = self.attn_sconv.decode(a)
        ttnn.deallocate(a)
        h = ttnn.add(x, a2)
        ttnn.deallocate(a2)
        ff_in = self.ffn_norm(h, mode=Mode.DECODE, norm_config=norm_cfg)
        f = self.feed_forward.forward(ff_in, mode="decode")
        ttnn.deallocate(ff_in)
        f2 = self.mlp_sconv.decode(f)
        ttnn.deallocate(f)
        out = ttnn.add(h, f2)
        ttnn.deallocate(h)
        ttnn.deallocate(f2)
        return out

    def forward_rows(self, x, positions, page_table, tau=None, commit_all=True, kv_idx=None):
        """R <= 8 rows at consecutive positions (spec verify / prefill row groups): x [1, 1, R, H/N] fractured,
        positions [R] int32. commit_all=False leaves the 4 conv states for commit_rows(n) (verify)."""
        if self.rep is not None:
            return self._forward_rows_rep(x, positions, page_table, tau, commit_all, kv_idx)
        R = int(x.shape[2])
        norm_cfg = self.args.get_norm_config("attn", Mode.DECODE) if self.num_devices > 1 else None
        attn_in = self.attention_norm(x, mode=Mode.DECODE, norm_config=norm_cfg)
        if attn_in.is_sharded():
            attn_in = ttnn.sharded_to_interleaved(attn_in, ttnn.L1_MEMORY_CONFIG)
        a = self.attention.forward_rows(attn_in, positions, page_table, tau=tau, commit_all=commit_all, kv_idx=kv_idx)
        ttnn.deallocate(attn_in)
        a2 = self.attn_sconv.rows(a, commit_all=commit_all)
        ttnn.deallocate(a)
        h = ttnn.add(x, a2)
        ttnn.deallocate(a2)
        ff_in = self.ffn_norm(h, mode=Mode.DECODE, norm_config=norm_cfg)
        f = self.feed_forward.forward(ff_in, mode="decode")
        ttnn.deallocate(ff_in)
        f2 = self.mlp_sconv.rows(f, commit_all=commit_all)
        ttnn.deallocate(f)
        out = ttnn.add(h, f2)
        ttnn.deallocate(h)
        ttnn.deallocate(f2)
        return out

    def _forward_rows_rep(self, x, positions, page_table, tau, commit_all, kv_idx):
        """forward_rows on the replicated residual: x [1, 1, R, H] in the boundary's residual layout (or the fractured
        [1, 1, R, H / N] embedding rows: gathered first) -> the replicated residual."""
        bnd = self.rep
        if int(x.shape[-1]) != self.args.dim:
            x = bnd.gather(x, self.tt_ccl)
        nc = bnd.norm_config
        attn_in = self.attention_norm.norm(x, mode=Mode.DECODE, in_sharded=True, out_sharded=True, norm_config=nc)
        a_in = ttnn.sharded_to_interleaved(attn_in, ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(attn_in)
        part = self.attention.forward_rows(
            a_in, positions, page_table, tau=tau, commit_all=commit_all, reduce=False, kv_idx=kv_idx
        )
        ttnn.deallocate(a_in)
        a = bnd.all_reduce(part, "mixer")
        a2 = self.attn_sconv_r.rows(a, commit_all=commit_all)
        ttnn.deallocate(a)
        h = ttnn.add(x, a2, memory_config=bnd.mc)
        ttnn.deallocate(a2)
        ff_in = self.ffn_norm.norm(h, mode=Mode.DECODE, in_sharded=True, out_sharded=True, norm_config=nc)
        f_in = ttnn.sharded_to_interleaved(ff_in, ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(ff_in)
        fpart = self.feed_forward.forward(f_in, mode="decode", partial="raw")
        ttnn.deallocate(f_in)
        f = bnd.all_reduce(fpart, "mlp")
        f2 = self.mlp_sconv_r.rows(f, commit_all=commit_all)
        ttnn.deallocate(f)
        out = ttnn.add(h, f2, memory_config=bnd.mc)
        ttnn.deallocate(h)
        ttnn.deallocate(f2)
        return out

    def commit_rows(self, n: int):
        """Spec verify: every conv state advances past the first n rows of the last forward_rows."""
        self.attention.commit_rows(n)
        if self.rep is not None:
            self.attn_sconv_r.commit_rows(n)
            self.mlp_sconv_r.commit_rows(n)
            return
        self.attn_sconv.commit_rows(n)
        self.mlp_sconv.commit_rows(n)

    def prefill_pipelined(self):
        """Whether this layer's prefill can run split (prefill_pre / prefill_moe): a hybrid MoE layer on the host-routed
        path (hybrid.patch.pf_pipelined)."""
        import sys

        pm = sys.modules.get("hybrid.patch")
        return pm is not None and hasattr(self.feed_forward, "_router_logits") and pm.pf_pipelined(self.feed_forward)

    def prefill_pre(self, x, p0: int, page_table_host=None, valid=None):
        """forward_prefill, part 1 (queued on the device, no host wait): attention, its convs (committed), the MoE input
        and its router logits with their non-blocking reads. -> the context for prefill_moe."""
        import sys

        mc = Mode.PREFILL
        norm_cfg = self.args.get_norm_config("attn", mc) if self.num_devices > 1 else None
        attn_in = self.attention_norm(x, mode=mc, norm_config=norm_cfg)
        if attn_in.is_sharded():
            attn_in = ttnn.sharded_to_interleaved(attn_in, ttnn.DRAM_MEMORY_CONFIG)
        a = self.attention.forward_prefill(attn_in, p0, page_table_host, valid=valid)
        ttnn.deallocate(attn_in)
        a2 = self.attn_sconv.prefill_commit(a, valid)
        ttnn.deallocate(a)
        h = ttnn.add(x, a2)
        ttnn.deallocate(a2)
        ff_cfg = self.args.get_norm_config("ff", mc) if self.num_devices > 1 else None
        ff_in = self.ffn_norm(h, mode=mc, norm_config=ff_cfg)
        if ff_in.is_sharded():
            ff_in = ttnn.sharded_to_interleaved(ff_in, ttnn.DRAM_MEMORY_CONFIG)
        ctx = sys.modules["hybrid.patch"]._pf_pre(self.feed_forward, ff_in)
        return dict(h=h, ff_in=ff_in, ctx=ctx, valid=valid)

    def prefill_moe(self, pre):
        """forward_prefill, part 2: the host plans the MoE step; returns go() -> queues the device experts -> done() ->
        the layer output (waits for the CPU experts, then the MLP conv (committed) and the residual add)."""
        import sys

        go_moe = sys.modules["hybrid.patch"]._pf_plan(self.feed_forward, pre["ctx"])

        def go():
            fin = go_moe()

            def done():
                return self._prefill_moe_done(pre, fin)

            return done

        return go

    def _prefill_moe_done(self, pre, fin):
        """prefill_moe's end: the MoE output (fin), the MLP conv (committed), the residual add."""
        valid = pre["valid"]
        f = fin()
        ttnn.deallocate(pre["ff_in"])
        f2 = self.mlp_sconv.prefill_commit(f, valid)
        ttnn.deallocate(f)
        out = ttnn.add(pre["h"], f2)
        ttnn.deallocate(pre["h"])
        ttnn.deallocate(f2)
        return out

    def forward_prefill(self, x, p0: int, page_table_host=None, valid=None):
        """A prefill chunk (T rows, T % 64 == 0, p0 % block == 0): x [1, 1, T, H/N] fractured -> same. One attention
        launch for the chunk (tt/attention.forward_prefill), the MoE's prefill path, every conv state past all T rows
        (or past the first `valid`: a padded chunk's real rows)."""
        mc = Mode.PREFILL
        norm_cfg = self.args.get_norm_config("attn", mc) if self.num_devices > 1 else None
        attn_in = self.attention_norm(x, mode=mc, norm_config=norm_cfg)
        if attn_in.is_sharded():
            attn_in = ttnn.sharded_to_interleaved(attn_in, ttnn.DRAM_MEMORY_CONFIG)
        a = self.attention.forward_prefill(attn_in, p0, page_table_host, valid=valid)
        ttnn.deallocate(attn_in)
        a2 = self.attn_sconv.prefill_commit(a, valid)
        ttnn.deallocate(a)
        h = ttnn.add(x, a2)
        ttnn.deallocate(a2)
        ff_cfg = self.args.get_norm_config("ff", mc) if self.num_devices > 1 else None
        ff_in = self.ffn_norm(h, mode=mc, norm_config=ff_cfg)
        if ff_in.is_sharded():
            ff_in = ttnn.sharded_to_interleaved(ff_in, ttnn.DRAM_MEMORY_CONFIG)
        f = self.feed_forward.forward(ff_in, mode="prefill")
        ttnn.deallocate(ff_in)
        f2 = self.mlp_sconv.prefill_commit(f, valid)
        ttnn.deallocate(f)
        out = ttnn.add(h, f2)
        ttnn.deallocate(h)
        ttnn.deallocate(f2)
        return out

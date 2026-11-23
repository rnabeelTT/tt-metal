# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Hy4 latent MoE (HF HYV4FlashMoE) and the dense MLP of layer 0, TP over the (1, N) mesh.

    logits = x @ W_r^T (fp32); s = sigmoid(logits); pick = top16(s + bias); w = s[pick] / sum * 3.998
    xl     = x @ W_down^T                                     (3072 -> 1536 latent, replicated)
    routed = sum_k w_k * D_k(clamp_silu(G_k xl) * clamp(U_k xl))   (latent 1536; expert intermediate split over TP)
    out    = combine_output_norm(W_up(routed) + shared(x))       (W_up 1536 -> 3072 applied to the per-die partial;
                                                                  one all-reduce of the sum)

Experts are packed per TP shard as [up_d | gate_d] (so a die's gate/up matmul is one sparse matmul), routed with
ttnn.sparse_matmul in "gather" mode for a few rows (decode / verify) or chunk-masked mode for prefill. SwiGLU with the
reference's clamps: gate <= 10, |up| <= 10.
"""

from __future__ import annotations

import os

import torch
import ttnn

from models.autoports.tencent_hy4_flash.tt.common import (AllReduce, bspec_policy, Boundary, boundary_rows_out, RMSNorm, StreamLinear, TILE, ckc, dl_split, mm, pair_enabled, stream_pair, num_devices, wfid,
                                                           stream_enabled, upload)

class _Args:
    """The model-wide cache holder FusedMoEDecode keys its kernels / buffers on."""


_FUSED_ARGS = _Args()
GATHER_MAX = int(os.environ.get("HY4_MOE_GATHER_MAX", "1"))  # more rows: the chunk-masked path (fixed op count)
PREFILL_CHUNK = int(os.environ.get("HY4_MOE_CHUNK", "32"))


def _lshard_env() -> str:
    """HY4_MOE_LSHARD (latent-down column shard + all_gather; "1": the shard copy replaces the full-width one, -~6 MB/die
    per layer of DRAM; "build": both, for A/B): default on (with the DFly2 drafter too)."""
    return os.environ.get("HY4_MOE_LSHARD", "1")


def _lshard_topology():
    return ttnn.Topology.Ring if os.environ.get("HY4_LSHARD_TOPO", "linear") == "ring" else ttnn.Topology.Linear


def _stream_defaults():
    """Hy4's routed-expert streaming knobs of the qwen36 fused multi-row MoE (moe_decode/union.py), unless set; all
    exact (data movement, core placement and counts only, bit-identical logits). HY4_MOE_STREAM=0: the qwen36 defaults.
    * 17+ rows (batched-spec verify passes of 6-8 users): the activation buffer in one page per (slot, row group)
      (QWEN35_MOE_ACT_BLK), down-weight chunks of 2 union slots (QWEN35_MOE_DN_CHUNK), the far bank block on the
      writer's NOC1 (QWEN35_MOE_GU_FARW) with 8 cores per column group (QWEN35_MOE_GUR_READERS), gate|up and down in ONE
      program (QWEN35_MOE_GUDN: the down chunks start while the gate|up cores stream later slots).
    * Every row count (QWEN35_MOE_PER_DEVICE): each chip's MoE cores by ITS DRAM banks (the mesh query answers for the
      front chip; QB2's chips differ in harvested columns: the centre DRAM column's workers at logical x 5 / 6 / 7).
    * 2..16 rows (HY4_MOE_GU1B=1): one-bank gate|up (QWEN35_MOE_GU1B, ExpertGateUpRows.onebank): every gate|up core
      streams one DRAM bank from its row (two-bank cores were capped at ~330 GB/s per die by the far bank's path),
      partner cores swap their gate results as FP32 tiles; READERS / ONROW cores per bank.
    In-model 8 lanes x r rows pass (traced, device + logits read): 8 rows 33.4 -> 30.7 ms, 16: 44.1 -> 39.0, 24: 67.4 ->
    55.7, 32: 76.0 -> 61.6 (vs the 17+ row knobs alone); plain batched decode 2 / 4 / 8 users 25.2 / 28.8 / 35.1 ->
    24.3 / 27.1 / 32.0 ms per step."""
    if os.environ.get("HY4_MOE_STREAM", "1") != "1":
        return
    for k, v in (("QWEN35_MOE_ACT_BLK", "17"), ("QWEN35_MOE_DN_CHUNK", "17:2"), ("QWEN35_MOE_GU_FARW", "17"),
                 ("QWEN35_MOE_GUR_READERS", "17:8"), ("QWEN35_MOE_GUDN", "17"), ("QWEN35_MOE_PER_DEVICE", "1")):
        os.environ.setdefault(k, v)
    if os.environ.get("HY4_MOE_GU1B", "1") == "1":
        # gate|up us per MoE layer and die (bench): 2 rows 66 -> 48, 4: 84 -> 60, 8: 159 -> 118, 16: 308 -> 240
        for k, v in (("QWEN35_MOE_GU1B", "2-16"), ("QWEN35_MOE_GU1B_READERS", "2:2,5:3,9:4"),
                     ("QWEN35_MOE_GU1B_ONROW", "2:2,5:3,9:4")):
            os.environ.setdefault(k, v)


def _dt(name, default):
    return {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}[os.environ.get(name, default)]


def _smm_cfg(M, N, K, target_cores=64, in0_block_w_cap=32, grid=(13, 10)):
    """A MatmulMultiCoreReuseMultiCast1DProgramConfig for sparse_matmul (one group of M rows x N cols)."""
    from models.autoports.minimaxai_minimax_m2_7.tt.moe import sparse_matmul_program_config

    return sparse_matmul_program_config(M, N, K, target_cores=target_cores, in0_block_w_cap=in0_block_w_cap)


class Hy4DenseMLP:
    """Layer 0: SwiGLU 3072 -> 13824 -> 3072 (gate/up column parallel, down row parallel), then post_fc2_norm."""

    def __init__(self, mesh, cfg, sd):
        self.mesh = mesh
        dt = _dt("HY4_DENSE_DTYPE", "bfp8")
        bf = lambda t: t.to(torch.bfloat16)  # noqa: E731
        I = cfg.dense_inter
        self.w1 = upload(mesh, bf(sd["gate_proj.weight"]).T.reshape(1, 1, cfg.hidden, I), dt, mapper=("shard", -1))
        self.w3 = upload(mesh, bf(sd["up_proj.weight"]).T.reshape(1, 1, cfg.hidden, I), dt, mapper=("shard", -1))
        self.w2 = upload(mesh, bf(sd["down_proj.weight"]).T.reshape(1, 1, I, cfg.hidden), dt, mapper=("shard", -2))
        self.norm = RMSNorm(mesh, sd["post_fc2_norm.weight"], cfg.rms_eps)
        self.ar = AllReduce(mesh)
        self.ckc = ckc(mesh)
        self.dec = None
        if stream_enabled(mesh):  # decode: [gate_d | up_d] in one streamed linear, down_d
            N = num_devices(mesh)
            Il = I // N
            g, u, d = bf(sd["gate_proj.weight"]).T, bf(sd["up_proj.weight"]).T, bf(sd["down_proj.weight"]).T
            cdir = os.environ.get("HY4_TENSOR_CACHE")
            cache = (lambda n: os.path.join(cdir, f"layers.0.mlp.{n}")) if cdir else (lambda n: None)
            self.Il = Il
            self.dec = {
                "gu": StreamLinear(mesh, [torch.cat([g[:, i * Il : (i + 1) * Il], u[:, i * Il : (i + 1) * Il]], 1)
                                          for i in range(N)], dt, cache=cache(f"gu.{dt}")),
                "d": StreamLinear(mesh, [d[i * Il : (i + 1) * Il, :] for i in range(N)], dt, cache=cache(f"d.{dt}")),
            }

    def forward_part(self, x, T=None, mem=None):
        """forward() without the all-reduce and norm: the per-die down-projection partial."""
        mem = mem or ttnn.DRAM_MEMORY_CONFIG
        g = mm(x, self.w1, wfid(self.ckc, self.w1), mem)
        u = mm(x, self.w3, wfid(self.ckc, self.w3), mem)
        h = ttnn.multiply(u, g, input_tensor_b_activations=[ttnn.UnaryOpType.SILU], memory_config=mem)
        ttnn.deallocate(g)
        ttnn.deallocate(u)
        part = mm(h, self.w2, wfid(self.ckc, self.w2), mem)
        ttnn.deallocate(h)
        return part

    def forward(self, x, mem=None, R=None):
        mem = mem or ttnn.DRAM_MEMORY_CONFIG
        dec = self.dec is not None and int(x.shape[-2]) == 32
        if dec:
            if dl_split():
                g, u = self.dec["gu"](x, mem, splits=[(0, self.Il), (self.Il, 2 * self.Il)])
            else:
                gu = self.dec["gu"](x, mem)
                g = ttnn.slice(gu, (0, 0, 0, 0), (1, 1, 32, self.Il), memory_config=mem)
                u = ttnn.slice(gu, (0, 0, 0, self.Il), (1, 1, 32, 2 * self.Il), memory_config=mem)
                ttnn.deallocate(gu)
        else:
            g = mm(x, self.w1, wfid(self.ckc, self.w1), mem)
            u = mm(x, self.w3, wfid(self.ckc, self.w3), mem)
        h = ttnn.multiply(u, g, input_tensor_b_activations=[ttnn.UnaryOpType.SILU], memory_config=mem)
        ttnn.deallocate(g)
        ttnn.deallocate(u)
        if dec:
            part = self.dec["d"](h, mem, rows_out=boundary_rows_out(self.mesh, R))
        else:
            part = mm(h, self.w2, wfid(self.ckc, self.w2), mem)
        ttnn.deallocate(h)
        if dec and R is not None and Boundary.enabled(self.mesh, R):
            if getattr(self, "bnd", None) is None:
                self.bnd = Boundary(self.mesh, int(part.shape[-1]), self.norm.eps)
            out = self.bnd(part, R, self.norm, "mlp")
            ttnn.deallocate(part)
            return out
        out = self.ar(part, mem)
        ttnn.deallocate(part)
        return self.norm(out, mem)


class Hy4MoE:
    def __init__(self, mesh, cfg, sd, layer_idx: int, cache_dir=None):
        self.mesh, self.cfg, self.li = mesh, cfg, layer_idx
        self.N = tp = num_devices(mesh)
        self.E, self.k, self.L, self.I = cfg.n_experts, cfg.top_k, cfg.latent, cfg.moe_inter
        assert self.I % (tp * TILE) == 0
        self.Il = Il = self.I // tp
        self.limit = float(cfg.swiglu_limit)
        self.scale = float(cfg.routed_scale)
        self.ar = AllReduce(mesh)
        bf = lambda t: t.to(torch.bfloat16)  # noqa: E731
        cache = (lambda n: os.path.join(cache_dir, f"layers.{layer_idx}.mlp.{n}")) if cache_dir else (lambda n: None)
        gu_dt = _dt("HY4_EXPERT_GU_DTYPE", "bfp4")
        dn_dt = _dt("HY4_EXPERT_DN_DTYPE", "bfp4")
        sm_dt = _dt("HY4_MOE_SMALL_DTYPE", "bf16")
        # router (fp32 logits), selection bias
        self.wr = upload(mesh, sd["gate.weight"].float().T.reshape(1, 1, cfg.hidden, self.E), ttnn.float32,
                         cache=cache("router.fp32"))
        self.bias = upload(mesh, sd["gate.e_score_correction_bias"].float().reshape(1, 1, 1, self.E), ttnn.float32)
        self._bias_host = sd["gate.e_score_correction_bias"].float().reshape(-1).clone()
        self.route_kernel = os.environ.get("HY4_ROUTE_KERNEL", "1") == "1" and tp > 1
        self.bias_q23 = None
        if self.route_kernel:
            from models.demos.blackhole.qwen36.tt.moe_decode.union import upload_selection_bias

            self.bias_q23 = upload_selection_bias(mesh, self._bias_host)
        # latent projections
        self.wd = upload(mesh, bf(sd["routed_expert_down_proj.weight"]).T.reshape(1, 1, cfg.hidden, self.L), sm_dt,
                         cache=cache(f"latent_down.{sm_dt}"))
        self.wu = upload(mesh, bf(sd["routed_expert_up_proj.weight"]).T.reshape(1, 1, self.L, cfg.hidden), sm_dt,
                         cache=cache(f"latent_up.{sm_dt}"))
        # routed experts: gate_up [E, 2I, L] (gate rows first), down [E, L, I]
        E, L, I = self.E, self.L, self.I
        # (the names ttnn.as_tensor gives the files: "<name>_dtype_<DTYPE>_layout_TILE.tensorbin"; the old check of
        # "<name>.tensorbin" never matched, so every start re-read the BF16 experts and repacked them (~90 s))
        have_cache = bool(cache_dir) and all(
            os.path.exists(f"{cache(f'experts.{n}.{dt}')}_dtype_{dt.name}_layout_TILE.tensorbin")
            for n, dt in (("up_gate", gu_dt), ("down", dn_dt)))
        if have_cache:
            gu_packed = torch.zeros(1)  # unused: as_tensor reads the cache file
            dn_t = torch.zeros(1)
        else:
            gu = sd["experts.gate_up_proj"]
            gate_t = gu[:, :I, :].transpose(1, 2)  # [E, L, I]
            up_t = gu[:, I:, :].transpose(1, 2)
            gu_packed = torch.cat(
                [torch.cat([up_t[:, :, d * Il : (d + 1) * Il], gate_t[:, :, d * Il : (d + 1) * Il]], dim=-1)
                 for d in range(tp)], dim=-1
            ).unsqueeze(0).contiguous()  # [1, E, L, tp * 2 Il]
            del gu, gate_t, up_t
            dn_t = sd["experts.down_proj"].transpose(1, 2).unsqueeze(0).contiguous()  # [1, E, I, L]
        self.up_gate = self._up(gu_packed, gu_dt, -1, cache(f"experts.up_gate.{gu_dt}"), have_cache, (1, E, L, tp * 2 * Il))
        self.down = self._up(dn_t, dn_dt, -2, cache(f"experts.down.{dn_dt}"), have_cache, (1, E, I, L))
        del gu_packed, dn_t
        # shared expert (TP)
        # (tensor-cached: a plain upload tilized them on the host at every start)
        self.sw1 = upload(mesh, bf(sd["shared_experts.gate_proj.weight"]).T.reshape(1, 1, cfg.hidden, -1), sm_dt,
                          mapper=("shard", -1), cache=cache(f"shared_gate.{sm_dt}.tp{tp}"))
        self.sw3 = upload(mesh, bf(sd["shared_experts.up_proj.weight"]).T.reshape(1, 1, cfg.hidden, -1), sm_dt,
                          mapper=("shard", -1), cache=cache(f"shared_up.{sm_dt}.tp{tp}"))
        self.sw2 = upload(mesh, bf(sd["shared_experts.down_proj.weight"]).T.reshape(1, 1, -1, cfg.hidden), sm_dt,
                          mapper=("shard", -2), cache=cache(f"shared_down.{sm_dt}.tp{tp}"))
        self.norm = RMSNorm(mesh, sd["combine_output_norm.weight"], cfg.rms_eps)
        self.ones_mask = upload(mesh, torch.ones(1, 1, 1, E, dtype=torch.bfloat16), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
        self.iota = upload(mesh, torch.arange(E, dtype=torch.float32).reshape(1, 1, 1, E), ttnn.float32)
        self.tile = ttnn.Tile([TILE, TILE])
        self.ckc_r = ckc(mesh, ttnn.MathFidelity.HiFi4, fp32=True)
        # tt-metal #49068: no fp32 dest on BH sparse mm. bfp4 experts: LoFi covers their 3 mantissa bits (as the fused
        # decode kernels); HY4_PF_EXPERT_FID=HiFi2 restores the old prefill fidelity
        efid = getattr(ttnn.MathFidelity, os.environ.get("HY4_PF_EXPERT_FID", "LoFi"))
        self.ckc_e = ckc(mesh, efid, fp32=False)
        self.ckc_s = ckc(mesh, ttnn.MathFidelity.HiFi4, fp32=True)
        self.cfg_gu_g = _smm_cfg(1, 2 * Il, L, target_cores=64, in0_block_w_cap=96)
        self.cfg_dn_g = _smm_cfg(1, L, Il, target_cores=64, in0_block_w_cap=32)
        self.cfg_gu_m = _smm_cfg(PREFILL_CHUNK, 2 * Il, L, target_cores=32, in0_block_w_cap=64)
        self.cfg_dn_m = _smm_cfg(PREFILL_CHUNK, L, Il, target_cores=32, in0_block_w_cap=32)
        # decode: streamed copies (tt/common.StreamLinear): x -> [latent down | shared gate_d | shared up_d];
        # [routed partial | shared hidden_d] -> [latent up ; shared down_d] (the add of the two folded into K)
        self.dec = None
        if stream_enabled(mesh):
            Sl = int(sd["shared_experts.gate_proj.weight"].shape[0]) // tp
            sg, su = bf(sd["shared_experts.gate_proj.weight"]).T, bf(sd["shared_experts.up_proj.weight"]).T
            sdn = bf(sd["shared_experts.down_proj.weight"]).T
            wd_t, wu_t = bf(sd["routed_expert_down_proj.weight"]).T, bf(sd["routed_expert_up_proj.weight"]).T
            self.Sl = Sl
            # router: x (bf16, exact) against [W_hi | W_lo] (W_hi = bf16(W), W_lo = bf16(W - W_hi)), fp32 out, the two
            # halves added: ~16 mantissa bits of the fp32 router weight (the interleaved fp32 matmul truncates its
            # operands to tf32) at stream speed (43 -> ~20 us)
            wr32 = sd["gate.weight"].float().T.contiguous()  # [H, E]
            w_hi = wr32.to(torch.bfloat16)
            w_lo = (wr32 - w_hi.float()).to(torch.bfloat16)
            self.dec_router = StreamLinear(mesh, [torch.cat([w_hi, w_lo], 1)] * tp, ttnn.bfloat16,
                                           cache=cache("router_hilo.bf16"), out_dtype=ttnn.float32)
            # the selection bias replicated over a tile's rows (tt/router_post, HY4_ROUTER_POST)
            self.bias_rows = upload(mesh, self._bias_host.reshape(1, 1, 1, self.E).expand(1, 1, TILE, self.E)
                                    .contiguous(), ttnn.float32)
            # HY4_MOE_DEC_DTYPE: the decode copies' dtype (default: the small weights' dtype; bfp8 is a quality tradeoff)
            dd = _dt("HY4_MOE_DEC_DTYPE", os.environ.get("HY4_MOE_SMALL_DTYPE", "bf16"))
            self.dec = {
                "out": StreamLinear(mesh, [torch.cat([wu_t, sdn[i * Sl : (i + 1) * Sl, :]], 0) for i in range(tp)],
                                    dd, cache=cache(f"dec_out.{dd}")),
            }
            # HY4_MOE_LSHARD=1: no full-width copy (its ~14 MB/die/layer of DRAM: 8 x 12K served prefills OOMed with
            # both copies); rows the fused kernels do not take (T > rows_max) go through the interleaved-weight path
            if not (_lshard_env() == "1" and L % (tp * TILE) == 0):
                self.dec["in"] = StreamLinear(
                    mesh, [torch.cat([wd_t, sg[:, i * Sl : (i + 1) * Sl], su[:, i * Sl : (i + 1) * Sl]], 1)
                           for i in range(tp)], dd, cache=cache(f"dec_in.{dd}"))
            # HY4_MOE_LSHARD (1 / build: the copy only): the latent down projection column-sharded over the dies
            # ([xl_d | shared gate_d | shared up_d], xl_d = latent columns [d L / tp, (d + 1) L / tp)), its rows
            # all-gathered for the fused experts: each die streams a quarter of W_down (exact: whole columns)
            if _lshard_env() in ("1", "build") and L % (tp * TILE) == 0:
                Lq = L // tp
                self.dec["in_s"] = StreamLinear(
                    mesh, [torch.cat([wd_t[:, i * Lq : (i + 1) * Lq], sg[:, i * Sl : (i + 1) * Sl],
                                      su[:, i * Sl : (i + 1) * Sl]], 1) for i in range(tp)],
                    dd, cache=cache(f"dec_in_s.{dd}"))
        # decode: the M2.7 / Qwen fused MoE kernels (route + gate|up + down in 3 programs), with Hy4's SwiGLU clamps
        # (MOE_SWIGLU_LIMIT, moe_decode/kernels SWIGLU_LIMIT), the router run by this port on the full-width hidden
        self.fused = None
        # decode, one row: the M2.7 / Qwen fused MoE kernels (gate|up and down, 2 programs; top-16 through the 128-byte
        # route record ROUTE_WIDE; the "keys" router: sigmoid + bias on the Tensix, top-k on the data-movement cores)
        if os.environ.get("HY4_MOE_FUSED", "1") == "1" and tp > 1 and self.dec is not None:
            os.environ["MOE_SWIGLU_LIMIT"] = str(self.limit)
            _stream_defaults()
            from models.demos.blackhole.qwen36.tt.moe_decode import fused as moe_fused

            if moe_fused.supported(E, L, Il, gu_dt, banks=len(moe_fused.bank_workers(mesh))):
                self.fused = moe_fused.FusedMoEDecode(
                    mesh, _FUSED_ARGS, hidden=L, inter_local=Il, num_experts=E, k=self.k, tp=tp, dtype=gu_dt,
                    down_dtype=dn_dt, router_weight=None, shared_gate_weight=None, shared_gate_proj=None,
                    shared_up_proj=None, shared_down_proj=None, cache_name=lambda n: None, router_ckc=self.ckc_r,
                    router="keys", selection_bias=sd["gate.e_score_correction_bias"],
                    router_logits=lambda x1: self.router_row(self._x_full),
                )
                # verify rows (DFly2): the multi-row fused kernels (the union of the rows' experts), up to
                # HY4_MOE_FUSED_ROWS_MAX rows; their buffers are uploaded here, before any trace capture
                # (+ the batched spec verify passes' row counts u (K + 1), tt/common.py bspec_policy)
                self.rows_max = int(os.environ.get("HY4_MOE_FUSED_ROWS_MAX", "8"))
                self.rows_set = set(range(2, self.rows_max + 1)) | {u * (K + 1) for u, K in bspec_policy().items()}
                for m in sorted(self.rows_set):
                    self.fused.rows_ops(m)

    def _up(self, t, dtype, dim, cache_name, have_cache, shape):
        kw = {"mesh_mapper": ttnn.ShardTensorToMesh(self.mesh, dim=dim)} if self.N > 1 else {}
        if have_cache:
            # shape only; the cache file supplies the data. torch.empty: never touched (torch.zeros of the 4.8 + 2.4 GB
            # placeholders page-faulted ~3.5 s per layer, ~150 s of the build); a failed load raises instead of
            # "repairing" the cache file from the placeholder
            t = torch.empty(shape, dtype=torch.bfloat16)
            repair = getattr(ttnn, "as_tensor_cache_repair", None)
            if repair is not None:
                with repair(False):
                    return ttnn.as_tensor(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=self.mesh,
                                          memory_config=ttnn.DRAM_MEMORY_CONFIG, cache_file_name=cache_name, **kw)
        return ttnn.as_tensor(t if t.dtype == torch.bfloat16 else t.to(torch.bfloat16), dtype=dtype,
                              layout=ttnn.TILE_LAYOUT, device=self.mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                              cache_file_name=cache_name, **kw)

    # ------------------------------------------------------------------ routing
    _RS = {}

    def route_decode(self, x, T, mem):
        """Decode rows: the fp32 router matmul, then the sigmoid + bias top-16 and normalization on one data-movement
        core per real row (qwen36 moe_decode RouteScores, Q23 fixed point) instead of ttnn.topk over [32, 512] fp32 on
        one core (~80 us). -> (scores bf16 [1,1,T,16] x 3.998 (new tensor), ids uint16 [1,1,T,16] (persistent: do not
        deallocate))."""
        from models.demos.blackhole.qwen36.tt.moe_decode.union import RouteScores, upload_selection_bias

        rs = Hy4MoE._RS.get(id(self.mesh))
        if rs is None:
            rs = Hy4MoE._RS[id(self.mesh)] = RouteScores(self.mesh, self.E, self.k, router="keys")
        row = self.router_row(x)
        sc, ids = rs(row, T, bias=self.bias_q23)
        ttnn.deallocate(row)
        return ttnn.multiply(sc, self.scale, memory_config=mem), ids

    def router_logits_dec(self, x):
        """Decode rows: fp32 router logits [1, 1, 32, E] from the streamed hi / lo bf16 router."""
        L1 = ttnn.L1_MEMORY_CONFIG
        if dl_split():
            a, b = self.dec_router(x, L1, splits=[(0, self.E), (self.E, 2 * self.E)])
        else:
            y = self.dec_router(x, L1)
            a = ttnn.slice(y, (0, 0, 0, 0), (1, 1, 32, self.E), memory_config=L1)
            b = ttnn.slice(y, (0, 0, 0, self.E), (1, 1, 32, 2 * self.E), memory_config=L1)
            ttnn.deallocate(y)
        lg = ttnn.add(a, b, memory_config=L1)
        ttnn.deallocate(a)
        ttnn.deallocate(b)
        return lg

    def router_row(self, x, T=None, ab=None):
        """[keys | s] FP32 [1, 1, rows, 2E]: s = sigmoid(x W_r) (fp32 logits, as the reference), keys = s + bias (the
        selection score); the routing kernels (router "keys") take the top-k of the keys and normalize s. T: return
        rows 0..T-1 only (HY4_ROUTER_POST=1, decode rows: one fused program after the router linear)."""
        L1 = ttnn.L1_MEMORY_CONFIG
        if (T is not None and os.environ.get("HY4_ROUTER_POST", "1") == "1"
                and getattr(self, "dec_router", None) is not None and int(x.shape[-2]) == 32):
            from models.autoports.tencent_hy4_flash.tt.router_post import RouterPostOp

            if ab is not None:  # the router linear already ran (paired with dec_in_s: HY4_PAIR)
                a, b = ab
            elif dl_split():
                a, b = self.dec_router(x, L1, splits=[(0, self.E), (self.E, 2 * self.E)])
            else:
                y = self.dec_router(x, L1)
                a = ttnn.slice(y, (0, 0, 0, 0), (1, 1, 32, self.E), memory_config=L1)
                b = ttnn.slice(y, (0, 0, 0, self.E), (1, 1, 32, 2 * self.E), memory_config=L1)
                ttnn.deallocate(y)
            row = RouterPostOp.get(self.mesh)(a, b, self.bias_rows, T, L1)
            ttnn.deallocate(a)
            ttnn.deallocate(b)
            return row
        if getattr(self, "dec_router", None) is not None and int(x.shape[-2]) == 32:
            logits = self.router_logits_dec(x)
        else:
            logits = self._logits_fp32(x)
        sg = ttnn.sigmoid(logits, memory_config=L1)
        ttnn.deallocate(logits)
        key = ttnn.add(sg, self.bias, memory_config=L1)
        row = ttnn.concat([key, sg], dim=-1, memory_config=L1)
        ttnn.deallocate(key)
        ttnn.deallocate(sg)
        return row

    def route(self, x, mem):
        """-> (weights bf16 [1,1,T,16] incl. x 3.998, ids uint16 [1,1,T,16])."""
        T = int(x.shape[-2])
        xf = ttnn.typecast(x, ttnn.float32, memory_config=mem)
        logits = ttnn.matmul(xf, self.wr, dtype=ttnn.float32, compute_kernel_config=self.ckc_r, memory_config=mem)
        ttnn.deallocate(xf)
        s = ttnn.sigmoid(logits, memory_config=mem)
        ttnn.deallocate(logits)
        choice = ttnn.add(s, self.bias, memory_config=mem)
        _, ids = ttnn.topk(choice, k=self.k, dim=-1, largest=True, sorted=True, memory_config=mem)
        ttnn.deallocate(choice)
        # w[t, j] = s[t, ids[t, j]] (exact)
        if os.environ.get("HY4_ROUTE_GATHER", "1") == "1":
            # a gather along the expert dim (the one-hot [T, k, E] fp32 select below was ~3.5 ms per layer at 2K rows)
            gi = ids if ids.dtype in (ttnn.uint16, ttnn.uint32) else ttnn.typecast(ids, ttnn.uint32, memory_config=mem)
            w = ttnn.gather(s, -1, gi, memory_config=mem)  # [1, 1, T, k] fp32
            if gi is not ids:
                ttnn.deallocate(gi)
            ttnn.deallocate(s)
        else:
            idf = ttnn.typecast(ids, ttnn.float32, memory_config=mem)
            oh = ttnn.eq(ttnn.reshape(idf, (1, T, self.k, 1)), self.iota, memory_config=mem)  # [1,T,k,E]
            sel = ttnn.multiply(oh, ttnn.reshape(s, (1, T, 1, self.E)), memory_config=mem)
            ttnn.deallocate(oh)
            ttnn.deallocate(s)
            w = ttnn.reshape(ttnn.sum(sel, dim=-1, keepdim=True, memory_config=mem), (1, 1, T, self.k))
            ttnn.deallocate(sel)
        den = ttnn.add(ttnn.sum(w, dim=-1, keepdim=True, memory_config=mem), 1e-20, memory_config=mem)
        wn = ttnn.multiply(ttnn.div(w, den, memory_config=mem), self.scale, memory_config=mem)
        ttnn.deallocate(w)
        scores = ttnn.typecast(wn, ttnn.bfloat16, memory_config=mem)
        ids16 = ttnn.typecast(ids, ttnn.uint16, memory_config=mem) if ids.dtype != ttnn.uint16 else ids
        return scores, ids16

    def _swiglu(self, up, gate, mem):
        g = ttnn.clamp(gate, max=self.limit, memory_config=mem)
        u = ttnn.clamp(up, min=-self.limit, max=self.limit, memory_config=mem)
        return ttnn.multiply(u, g, input_tensor_b_activations=[ttnn.UnaryOpType.SILU], memory_config=mem)

    # ------------------------------------------------------------------ experts
    def _gather(self, xl, scores, ids, T, mem):
        """Few rows: per row, the k picked experts' gate/up and down as two sparse matmuls."""
        k, Il, L = self.k, self.Il, self.L
        ids_rm = ttnn.to_layout(ids, ttnn.ROW_MAJOR_LAYOUT)
        outs = []
        for u in range(T):
            ids_u = ttnn.slice(ids_rm, [0, 0, u, 0], [1, 1, u + 1, k])
            x_u = ttnn.slice(xl, [0, 0, u, 0], [1, 1, u + 1, L])
            fused = ttnn.sparse_matmul(x_u, self.up_gate, sparsity=self.ones_mask, indices=ids_u, is_input_a_sparse=False,
                                       is_input_b_sparse=True, memory_config=mem, output_tile=self.tile,
                                       program_config=self.cfg_gu_g, compute_kernel_config=self.ckc_e,
                                       dtype=ttnn.bfloat16)
            fused = ttnn.reshape(fused, (1, k, 1, 2 * Il))
            up = ttnn.slice(fused, [0, 0, 0, 0], [1, k, 1, Il])
            gate = ttnn.slice(fused, [0, 0, 0, Il], [1, k, 1, 2 * Il])
            ttnn.deallocate(fused)
            din = self._swiglu(up, gate, mem)
            ttnn.deallocate(up)
            ttnn.deallocate(gate)
            down = ttnn.sparse_matmul(din, self.down, sparsity=self.ones_mask, indices=ids_u, is_input_a_sparse=True,
                                      is_input_b_sparse=True, memory_config=mem, output_tile=self.tile,
                                      program_config=self.cfg_dn_g, compute_kernel_config=self.ckc_e,
                                      dtype=ttnn.bfloat16)
            ttnn.deallocate(din)
            down = ttnn.reshape(down, (1, k, 1, L))
            sc_u = ttnn.reshape(ttnn.slice(scores, [0, 0, u, 0], [1, 1, u + 1, k]), (1, k, 1, 1))
            weighted = ttnn.multiply(down, sc_u, memory_config=mem)
            ttnn.deallocate(down)
            outs.append(ttnn.sum(weighted, dim=1, keepdim=True, memory_config=mem))  # [1,1,1,L]
            ttnn.deallocate(weighted)
            ttnn.deallocate(ids_u)
            ttnn.deallocate(x_u)
        ttnn.deallocate(ids_rm)
        acc = ttnn.concat(outs, dim=2, memory_config=mem) if T > 1 else outs[0]
        if T > 1:
            for o in outs:
                ttnn.deallocate(o)
        return ttnn.reshape(acc, (1, 1, T, L))

    def _masked(self, xl, scores, ids, T, mem):
        """Many rows: chunks of PREFILL_CHUNK rows; each chunk runs every expert any of its rows picked."""
        E, Il, L, C = self.E, self.Il, self.L, PREFILL_CHUNK
        Tp = -(-T // C) * C
        # dense routing weights [1,1,T,E] (zero for unpicked)
        z = ttnn.zeros((1, 1, int(scores.shape[-2]), E), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh,
                       memory_config=mem)
        idx = ttnn.typecast(ids, ttnn.uint32, memory_config=mem)
        routing = ttnn.scatter(z, dim=-1, index=idx, src=scores)
        ttnn.deallocate(z)
        ttnn.deallocate(idx)
        if Tp != int(routing.shape[-2]):
            routing = ttnn.pad(routing, [(0, 0), (0, 0), (0, Tp - int(routing.shape[-2])), (0, 0)], value=0.0)
        if Tp != int(xl.shape[-2]):
            xl = ttnn.pad(xl, [(0, 0), (0, 0), (0, Tp - int(xl.shape[-2])), (0, 0)], value=0.0)
        outs = []
        for s0 in range(0, Tp, C):
            x_c = ttnn.slice(xl, [0, 0, s0, 0], [1, 1, s0 + C, L], memory_config=mem)
            r_c = ttnn.slice(routing, [0, 0, s0, 0], [1, 1, s0 + C, E], memory_config=mem)
            mask = ttnn.to_layout(ttnn.sum(r_c, dim=2, keepdim=True, memory_config=mem), ttnn.ROW_MAJOR_LAYOUT,
                                  memory_config=mem)
            fused = ttnn.sparse_matmul(x_c, self.up_gate, sparsity=mask, nnz=None, memory_config=mem,
                                       output_tile=self.tile, program_config=self.cfg_gu_m,
                                       compute_kernel_config=self.ckc_e, dtype=ttnn.bfloat16)
            ttnn.deallocate(x_c)
            fused = ttnn.reshape(fused, (1, E, C, 2 * Il))
            up = ttnn.slice(fused, [0, 0, 0, 0], [1, E, C, Il], memory_config=mem)
            gate = ttnn.slice(fused, [0, 0, 0, Il], [1, E, C, 2 * Il], memory_config=mem)
            ttnn.deallocate(fused)
            din = self._swiglu(up, gate, mem)
            ttnn.deallocate(up)
            ttnn.deallocate(gate)
            down = ttnn.sparse_matmul(din, self.down, sparsity=mask, nnz=None, is_input_a_sparse=True,
                                      is_input_b_sparse=True, memory_config=mem, output_tile=self.tile,
                                      program_config=self.cfg_dn_m, compute_kernel_config=self.ckc_e,
                                      dtype=ttnn.bfloat16)
            ttnn.deallocate(din)
            w_c = ttnn.permute(r_c, (0, 3, 2, 1), memory_config=mem)  # [1,E,C,1]
            weighted = ttnn.multiply(ttnn.reshape(down, (1, E, C, L)), w_c, memory_config=mem)
            for t_ in (down, w_c, r_c, mask):
                ttnn.deallocate(t_)
            outs.append(ttnn.experimental.fast_reduce_nc(weighted, dims=[1]))  # [1,1,C,L]
            ttnn.deallocate(weighted)
        ttnn.deallocate(routing)
        acc = outs[0] if len(outs) == 1 else ttnn.concat(outs, dim=2, memory_config=mem)
        if len(outs) > 1:
            for o in outs:
                ttnn.deallocate(o)
        acc = ttnn.reshape(acc, (1, 1, Tp, L))
        if Tp != T:
            acc = ttnn.slice(acc, [0, 0, 0, 0], [1, 1, T, L], memory_config=mem)
        return acc

    def _shared(self, x, mem):
        g = mm(x, self.sw1, wfid(self.ckc_s, self.sw1), mem)
        u = mm(x, self.sw3, wfid(self.ckc_s, self.sw3), mem)
        h = ttnn.multiply(u, g, input_tensor_b_activations=[ttnn.UnaryOpType.SILU], memory_config=mem)
        ttnn.deallocate(g)
        ttnn.deallocate(u)
        out = mm(h, self.sw2, wfid(self.ckc_s, self.sw2), mem)
        ttnn.deallocate(h)
        return out

    def _logits_fp32(self, x):
        xf = ttnn.typecast(x, ttnn.float32, memory_config=ttnn.L1_MEMORY_CONFIG)
        lg = ttnn.matmul(xf, self.wr, dtype=ttnn.float32, compute_kernel_config=self.ckc_r,
                         memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(xf)
        return lg

    def _dec_tail(self, part, T, mem):
        """The decode MoE's per-die partial -> all-reduce + norm (the fabric boundary for its row counts)."""
        if Boundary.enabled(self.mesh, T):
            if getattr(self, "bnd", None) is None:
                self.bnd = Boundary(self.mesh, self.cfg.hidden, self.cfg.rms_eps)
            out = self.bnd(part, T, self.norm, "mlp")
            ttnn.deallocate(part)
            return out
        full = self.ar(part, mem)
        ttnn.deallocate(part)
        return self.norm(full, mem)

    def forward_decode(self, x, T: int, mem=None):
        """Decode / verify rows (T <= 32 real rows of x [1, 1, 32, 3072]): the fused kernels when available."""
        mem = mem or ttnn.DRAM_MEMORY_CONFIG
        fused_ok = self.fused is not None and (T == 1 or T in getattr(self, "rows_set", ()))
        if (self.dec is not None and int(x.shape[-2]) == 32
                and ("in" in self.dec or (fused_ok
                                          and _lshard_env() == "1"))):
            # streamed linears: [latent down | shared gate | shared up] and [latent up ; shared down]; the experts by
            # the fused kernels (one row) or the per-row gather (verify rows)
            L, Sl = self.L, self.Sl
            fusedT = fused_ok
            ab = None
            if fusedT and "in_s" in self.dec and _lshard_env() == "1":
                Lq = L // self.N
                in_kw = dict(x=x, mem=mem, splits=[(0, Lq, T), (Lq, Lq + Sl), (Lq + Sl, Lq + 2 * Sl)])
                if (pair_enabled() and getattr(self, "dec_router", None) is not None and dl_split()
                        and os.environ.get("HY4_ROUTER_POST", "1") == "1"):
                    # HY4_PAIR: dec_in_s + the router linear as one program (the router's [keys | s] halves kept for
                    # router_row; same values)
                    (xq, g, u), ab = stream_pair(self.dec["in_s"], in_kw, self.dec_router,
                                                 dict(x=x, mem=ttnn.L1_MEMORY_CONFIG,
                                                      splits=[(0, self.E), (self.E, 2 * self.E)]))
                else:
                    xq, g, u = self.dec["in_s"](**in_kw)
                xl = ttnn.all_gather(xq, dim=3, cluster_axis=1, topology=_lshard_topology(), memory_config=mem)
                ttnn.deallocate(xq)
            elif dl_split():
                # the parts written straight by the linear (the fused experts' rows: only the real rows)
                xl, g, u = self.dec["in"](x, mem, splits=[(0, L, T if fusedT else None), (L, L + Sl),
                                                          (L + Sl, L + 2 * Sl)])
            else:
                y = self.dec["in"](x, mem)
                xl = ttnn.slice(y, (0, 0, 0, 0), (1, 1, 32, L), memory_config=mem)
                g = ttnn.slice(y, (0, 0, 0, L), (1, 1, 32, L + Sl), memory_config=mem)
                u = ttnn.slice(y, (0, 0, 0, L + Sl), (1, 1, 32, L + 2 * Sl), memory_config=mem)
                ttnn.deallocate(y)
            h = ttnn.multiply(u, g, input_tensor_b_activations=[ttnn.UnaryOpType.SILU], memory_config=mem)
            ttnn.deallocate(g)
            ttnn.deallocate(u)
            if fusedT:
                row = self.router_row(x, T, ab=ab)  # [keys | s] (streamed router): rows 0..T-1 are the tokens'
                if int(row.shape[-2]) == T:
                    rowT = row
                else:
                    rowT = ttnn.slice(row, (0, 0, 0, 0), (1, 1, T, 2 * self.E), memory_config=ttnn.L1_MEMORY_CONFIG)
                    ttnn.deallocate(row)
                if int(xl.shape[-2]) == T:
                    xT = xl
                else:
                    xT = ttnn.slice(xl, (0, 0, 0, 0), (1, 1, T, L), memory_config=mem)
                    ttnn.deallocate(xl)
                if T == 1:
                    partT = self.fused(xT, self.up_gate, self.down, logits=rowT)  # [1, 1, 1, L] (model-wide: keep)
                else:
                    partT = self.fused.rows(xT, self.up_gate, self.down, logits=rowT)  # [1, 1, T, L] (model-wide)
                ttnn.deallocate(xT)
                rT = ttnn.multiply(partT, self.scale, memory_config=mem)  # the fused router normalizes to 1
                if os.environ.get("HY4_MOE_XSPLIT", "1") == "1" and T <= 16:
                    # the out linear reads [routed | h] as two K parts (its rows 0..T-1 only): no pad / concat
                    part = self.dec["out"]((rT, h), mem, rows_out=boundary_rows_out(self.mesh, T))
                    ttnn.deallocate(rT)
                    ttnn.deallocate(h)
                    return self._dec_tail(part, T, mem)
                if T == 32:  # (a zero pad returns its input: deallocating it freed `routed`, a segfault in the concat)
                    routed = rT
                else:
                    routed = ttnn.pad(rT, [(0, 0), (0, 0), (0, 32 - T), (0, 0)], value=0.0)
                    ttnn.deallocate(rT)
            else:
                if self.route_kernel:
                    scores, ids = self.route_decode(x, T, mem)
                else:
                    scores, ids = self.route(x, mem)
                routed = self._gather(xl, scores, ids, T, mem)
                if T != 32:
                    routed = ttnn.pad(routed, [(0, 0), (0, 0), (0, 32 - T), (0, 0)], value=0.0)
                for t_ in (xl, scores) + (() if self.route_kernel else (ids,)):
                    ttnn.deallocate(t_)
            cat = ttnn.concat([routed, h], dim=-1, memory_config=mem)
            ttnn.deallocate(routed)
            ttnn.deallocate(h)
            part = self.dec["out"](cat, mem, rows_out=boundary_rows_out(self.mesh, T))
            ttnn.deallocate(cat)
            return self._dec_tail(part, T, mem)
        if not fused_ok:
            # trace-safe rows path: per-row gather (the chunk-masked prefill path writes from the host: ttnn.zeros)
            scores, ids = self.route(x, mem)
            xl = mm(x, self.wd, wfid(self.ckc_s, self.wd), mem)
            routed = self._gather(xl, scores, ids, T, mem)
            Tp = int(x.shape[-2])
            if Tp != T:
                routed = ttnn.pad(routed, [(0, 0), (0, 0), (0, Tp - T), (0, 0)], value=0.0)
            for t_ in (xl, scores, ids):
                ttnn.deallocate(t_)
            up = mm(routed, self.wu, wfid(self.ckc_s, self.wu), mem)  # [T, 3072]
            ttnn.deallocate(routed)
            sh = self._shared(x, mem)
            part = ttnn.add(sh, up, memory_config=mem)  # rows >= T are padding
            ttnn.deallocate(up)
            ttnn.deallocate(sh)
            full = self.ar(part, mem)
            ttnn.deallocate(part)
            return self.norm(full, mem)
        Tp = int(x.shape[-2])
        xr = x if T == Tp else ttnn.slice(x, (0, 0, 0, 0), (1, 1, T, int(x.shape[-1])), memory_config=mem)
        self._x_full = xr
        xl = mm(xr, self.wd, wfid(self.ckc_s, self.wd), mem)  # [1,1,T,1536]
        if T == 1:
            routed = self.fused(xl, self.up_gate, self.down)  # model-wide [1,1,1,1536] partial sum
        else:
            routed = self.fused.rows(xl, self.up_gate, self.down)
        ttnn.deallocate(xl)
        up = mm(routed, self.wu, wfid(self.ckc_s, self.wu), mem)  # partial [T,3072]
        up = ttnn.multiply(up, self.scale, memory_config=mem)  # the routed scaling (the fused router normalizes to 1)
        sh = self._shared(x, mem)
        part = ttnn.add(sh, up, memory_config=mem)  # [Tp, H] + [T, H]: rows >= T are padding
        ttnn.deallocate(up)
        ttnn.deallocate(sh)
        if xr is not x:
            ttnn.deallocate(xr)
        self._x_full = None
        full = self.ar(part, mem)
        ttnn.deallocate(part)
        return self.norm(full, mem)

    def forward_part(self, x, T: int, mem=None, routed=None):
        """Prefill: forward() without the all-reduce and norm (the per-die partial [1, 1, Tp, 3072]). routed: the
        rows' (scores, ids) when the caller routed them (row-local router, tt/model.py _mlp_part_split)."""
        mem = mem or ttnn.DRAM_MEMORY_CONFIG
        Tp = int(x.shape[-2])
        scores, ids = routed if routed is not None else self.route(x, mem)
        xl = mm(x, self.wd, wfid(self.ckc_s, self.wd), mem)
        if getattr(self, "_sorted", None) is None:
            from models.autoports.tencent_hy4_flash.tt.prefill_moe import SortedExperts

            self._sorted = SortedExperts(self)
        sub = int(os.environ.get("HY4_MOE_SUB", "2048"))  # the sorted dispatch admits ~3K rows at top-16 of 512
        if Tp <= sub:
            routed = self._sorted(xl, scores, ids, Tp, mem)
        else:
            parts = []
            for r0 in range(0, Tp, sub):
                r1 = min(Tp, r0 + sub)
                sl = lambda t, w: ttnn.slice(t, (0, 0, r0, 0), (1, 1, r1, w), memory_config=mem)  # noqa: E731
                xs, ss, is_ = sl(xl, self.L), sl(scores, self.k), sl(ids, self.k)
                parts.append(self._sorted(xs, ss, is_, r1 - r0, mem))
                for t_ in (xs, ss, is_):
                    ttnn.deallocate(t_)
            routed = ttnn.concat(parts, dim=2, memory_config=mem)
            for t_ in parts:
                ttnn.deallocate(t_)
        for t_ in (xl, scores, ids):
            ttnn.deallocate(t_)
        up = mm(routed, self.wu, wfid(self.ckc_s, self.wu), mem)
        ttnn.deallocate(routed)
        sh = self._shared(x, mem)
        part = ttnn.add(up, sh, memory_config=mem)
        ttnn.deallocate(up)
        ttnn.deallocate(sh)
        return part

    def forward(self, x, T: int, mem=None):
        """x [1, 1, Tp, 3072] bf16 replicated (normed); rows >= T are padding. -> [1, 1, Tp, 3072] replicated."""
        mem = mem or ttnn.DRAM_MEMORY_CONFIG
        Tp = int(x.shape[-2])
        scores, ids = self.route(x, mem)
        xl = mm(x, self.wd, wfid(self.ckc_s, self.wd), mem)  # [1,1,Tp,1536]
        if T <= GATHER_MAX:
            routed = self._gather(xl, scores, ids, T, mem)
            if Tp != T:
                routed = ttnn.pad(routed, [(0, 0), (0, 0), (0, Tp - T), (0, 0)], value=0.0)
        elif os.environ.get("HY4_MOE_SORTED", "1") == "1":
            # prefill: expert-sorted dispatch + one grouped sparse matmul per projection (tt/prefill_moe.py)
            if getattr(self, "_sorted", None) is None:
                from models.autoports.tencent_hy4_flash.tt.prefill_moe import SortedExperts

                self._sorted = SortedExperts(self)
            routed = self._sorted(xl, scores, ids, Tp, mem)
        else:
            routed = self._masked(xl, scores, ids, Tp, mem)
        ttnn.deallocate(xl)
        ttnn.deallocate(scores)
        ttnn.deallocate(ids)
        up = mm(routed, self.wu, wfid(self.ckc_s, self.wu), mem)  # partial [Tp,3072]
        ttnn.deallocate(routed)
        sh = self._shared(x, mem)
        part = ttnn.add(up, sh, memory_config=mem)
        ttnn.deallocate(up)
        ttnn.deallocate(sh)
        full = self.ar(part, mem)
        ttnn.deallocate(part)
        return self.norm(full, mem)

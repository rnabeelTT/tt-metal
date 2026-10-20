# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Full GLM-5.3-Flash (E192) text model on a (1, tp) mesh (stage 6).

  tokens -> embedding (hidden-sharded table: die d looks up columns [d*C/tp, (d+1)*C/tp)) -> 4 identical mHC streams
  in the stream-interleaved layout of tt/multichip_decoder.py -> 45 MultichipDecoder layers -> hc_head (mean of the
  streams, local columns) -> all-gather -> final RMSNorm -> LM head (vocab-sharded; stage-8 default bfp4 LoFi,
  GLM53_PRECISION=none: bfp8) -> logits [.., V/tp] per die
  -> on-device sampling (models.common.sampling.SamplingGenerator) or host logits.

Prefill runs one user's prompt in chunks of the constructed prefill lengths (the last chunk on the smallest length
that holds it); KDA layers carry their recurrent state chunk to chunk, DSA layers write the paged latent cache. Decode
runs all S slots per step; the per-step DSA / KDA inputs are shared by every layer and derived on device
(tt/decode_inputs.py), so a traced decode step needs no host input other than the token on the first step.
"""

from __future__ import annotations

import dataclasses
import os
import time
from types import SimpleNamespace

import torch
from loguru import logger

import ttnn
from models.tt_transformers.tt.ccl import TT_CCL, tt_all_gather

from .ckpt import CheckpointReader
from .decode_inputs import DeviceDecodeInputs
from .multichip_decoder import DEFAULT_MC, TOPOLOGY, MultichipDecoder, _ds_grid, _width_shard, sharded_rms_norm
from .optimized_decoder import SELECTED, OptConfig
from .precision import PrecisionOverrides
from .weight_cache import cached, weight_cache_from_env

DRAM = ttnn.DRAM_MEMORY_CONFIG
HC = 4

# Stage-7 terminal path (GLM53_HEAD, comma list; "legacy" = the stage-6 head):
#   norm_sharded  decode final RMSNorm width-sharded on 32 cores (the interleaved norm runs one tile of rows on 1 core)
#   lm_ds         LM head on a DRAM width-sharded weight: each die's [C, V/tp] slice zero-padded to a multiple of
#                 32 x 8 DRAM banks, decode rows on the DRAM-sharded matmul, prefill rows on a 2D config; the padded
#                 columns are sliced off, so the logits keep the [.., V/tp] per-die contract of the sampler / host.
HEAD_DEFAULT = "norm_sharded"  # lm_ds measured slower (doc/optimized_full_model/README.md); option kept
# Prefill chunk plan (GLM53_PREFILL_PLAN): "cost" = fewest estimated ms over the constructed lengths (largest chunks
# first, the last chunk holds the remainder), "legacy" = stage 6 (the largest length, then the smallest that fits).
# GLM53_PREFILL_COST "n:ms,..." overrides the per-chunk estimates (full 45-layer model, TP4, measured stage 7).
# measured TTFT of one chunk (full model, TP4, 1 slot, s7_cand_s1); 256 / 1024 interpolated
PREFILL_COST_MS = {128: 232.0, 256: 290.0, 512: 378.0, 1024: 600.0, 2048: 1033.0}
PREFILL_LENGTHS_DEFAULT = (128, 512, 2048)
# Traced prefill (GLM53_PREFILL_TRACE, comma list of constructed lengths; "none" = off): the FIRST chunk of a prompt
# (start 0, DSA flash path) replays a trace captured on persistent inputs (tokens, KDA bounds, DSA row / mask / page
# rows) of that length; later chunks run eagerly. Only short chunks are host-dispatch bound (stage-7 measurement).
PREFILL_TRACE_DEFAULT = "none"  # measured: 128-row trace 222 vs 232 ms eager; not worth ~30 MB + trace region


def _env_set(name, default):
    v = os.environ.get(name, default)
    return {x.strip() for x in v.split(",") if x.strip() and x.strip() not in ("legacy", "none")}


def prefill_lengths_from_env(default=PREFILL_LENGTHS_DEFAULT):
    env = os.environ.get("GLM53_PREFILL_LENGTHS")
    return tuple(sorted(int(x) for x in env.split(","))) if env else tuple(default)


def prefill_cost_from_env(lengths):
    cost = dict(PREFILL_COST_MS)
    env = os.environ.get("GLM53_PREFILL_COST")
    if env:
        for kv in env.split(","):
            k, v = kv.split(":")
            cost[int(k)] = float(v)
    # unknown lengths: linear in rows between the known points (an estimate only steers the plan)
    for n in lengths:
        if n not in cost:
            cost[n] = PREFILL_COST_MS[2048] * n / 2048 + PREFILL_COST_MS[128] * 0.5
    return {n: cost[n] for n in lengths}


class Glm53Model:
    def __init__(
        self,
        mesh,
        rd: CheckpointReader | None = None,
        *,
        max_batch: int = 32,
        max_context: int = 4096,
        prefill_lengths=None,
        num_blocks: int | None = None,
        layer_indices=None,
        opt: OptConfig | None = None,
        weight_cache="env",
        allocate_dsa: bool = True,
        lockstep_sampling: bool = False,
        dsa_shard: bool | None = None,
    ):
        """weight_cache: "env" = tt/weight_cache.weight_cache_from_env (GLM53_CACHE_DIR), None = no cache, or a
        WeightCache. A cached build loads the final device weights without reading the checkpoint tensors.
        allocate_dsa=False (vLLM serving): the DSA layers' paged pools and the shared decode inputs are allocated
        later by allocate_dsa_caches(num_blocks), when vLLM's block count is known. lockstep_sampling: every die
        feeds back die 0's sampled token (tt/serving_state.install_lockstep_sampling; required for sampled serving
        on a multi-die mesh). dsa_shard: the DSA latent pool layout (True = block-sharded over the dies, tt/dsa_shard.py;
        False = replicated; None = MCOpts.dsa_shard / GLM53_DSA_SHARD). tt/generator.plan_context chooses it from the
        requested context."""
        t_load = time.time()
        self.mesh = mesh
        self.rd = rd or CheckpointReader()
        tc = self.rd.text_config
        self.tc = tc
        self.tp = tuple(mesh.shape)[1]
        self.S = max_batch
        # sampler rows: models.common.sampling takes multiples of 32; with fewer slots the logits are zero-padded
        self.SR = -(-max_batch // 32) * 32
        self.max_context = max_context
        self.prefill_lengths = tuple(sorted(prefill_lengths)) if prefill_lengths else prefill_lengths_from_env()
        self.prefill_plan = os.environ.get("GLM53_PREFILL_PLAN", "cost")
        self.prefill_cost = prefill_cost_from_env(self.prefill_lengths)
        self.head_opts = _env_set("GLM53_HEAD", HEAD_DEFAULT) if mesh.get_num_devices() > 1 else set()
        self.prefill_trace_lengths = tuple(
            sorted(
                int(x) for x in _env_set("GLM53_PREFILL_TRACE", PREFILL_TRACE_DEFAULT) if int(x) in self.prefill_lengths
            )
        )
        self._ptr_inputs = {}  # n -> persistent inputs of the traced prefill (prepare_prefill_traces)
        self._ptraces = {}  # n -> captured trace (capture_prefill_traces)
        self.hidden = tc["hidden_size"]
        self.vocab = tc["vocab_size"]
        assert self.vocab % (self.tp * 32) == 0 and self.hidden % (self.tp * 32) == 0
        self.layer_indices = list(range(tc["num_hidden_layers"])) if layer_indices is None else list(layer_indices)
        self.tt_ccl = TT_CCL(mesh) if self.tp > 1 else None
        rep = ttnn.ReplicateTensorToMesh(mesh)
        shard = (lambda d: ttnn.ShardTensorToMesh(mesh, dim=d)) if self.tp > 1 else (lambda d: rep)
        opt = opt or SELECTED
        self.weight_cache = weight_cache_from_env(mesh, self.rd, opt) if weight_cache == "env" else weight_cache
        top_cache = self.weight_cache.component("top") if self.weight_cache is not None else None

        # [V, C] hidden-sharded. Not cached: ttnn.load_tensor of this 1.27 GB ROW_MAJOR table stalled for 20+ min at
        # 100 % CPU (2026-10-16); the bf16 checkpoint read + upload takes seconds
        self.embed_w = ttnn.from_torch(
            self.rd.top("embed_tokens.weight").to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            mesh_mapper=shard(-1),
        )
        self.norm_w = ttnn.from_torch(
            self.rd.top("norm.weight").to(torch.bfloat16).reshape(1, 1, -1, 32),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            mesh_mapper=rep,
        )
        if "lm_ds" in self.head_opts:
            self._build_lm_head_ds(top_cache, shard)
        else:
            self.lm_head = cached(
                top_cache,
                "lm_head",
                lambda: ttnn.from_torch(
                    self.rd.top("lm_head.weight").T.contiguous(),  # [C, V], vocab-sharded
                    dtype=ttnn.bfloat8_b,
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh,
                    memory_config=DRAM,
                    mesh_mapper=shard(-1),
                ),
            )
        self.ckc = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self.norm_ckc = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

        # stage-8 precision (GLM53_PRECISION, tt/precision.py; unset = selected default, none = stage 7): LM head
        PrecisionOverrides.from_env().apply_model(self, 0, tc["num_hidden_layers"] - 1)
        mc = None
        if dsa_shard is not None:  # explicit layout (wins over GLM53_DSA_SHARD; GLM53_MC still applies on top)
            mc = dataclasses.replace(DEFAULT_MC, dsa_shard=bool(dsa_shard))
            saved_env = os.environ.pop("GLM53_DSA_SHARD", None)
        self.layers = []
        for i in self.layer_indices:
            t0 = time.time()
            dec = MultichipDecoder.from_state_dict(
                self.rd.layer_state(i, lazy=True),  # read only on weight-cache misses
                hf_config=tc,
                layer_idx=i,
                mesh_device=mesh,
                max_batch=max_batch,
                prefill_lengths=self.prefill_lengths,
                dsa_kwargs={"max_context": max_context, "num_blocks": num_blocks},
                tt_ccl=self.tt_ccl,
                mc=mc,
                opt=opt,
                weight_cache=self.weight_cache,
            )
            if (allocate_dsa and allocate_dsa != "early") or dec.attn_kind != "dsa":
                dec.allocate_decode_state(max_batch)
            self.layers.append(dec)
            logger.info(f"layer {i} ({dec.attn_kind}/{dec.mlp_kind}) loaded in {time.time() - t0:.1f}s")
        if dsa_shard is not None and saved_env is not None:
            os.environ["GLM53_DSA_SHARD"] = saved_env
        dsa = next((d.attn for d in self.layers if d.attn_kind == "dsa"), None)
        kda = next((d.attn for d in self.layers if d.attn_kind == "kda"), None)
        self.dsa_shard = bool(getattr(dsa, "dsa_shard", False))
        early = allocate_dsa == "early"  # diagnostic (GLM53_DEFER_POOL=2): the deferred API, before the sampler
        self.inputs = (
            DeviceDecodeInputs(mesh, dsa=dsa, kda=kda, batch=max_batch) if allocate_dsa and not early else None
        )
        self._install_inactive_skip()
        if early:
            self.allocate_dsa_caches(num_blocks or max_batch * (max_context // 128) + 1)
        # the sampler is built after the DSA pools and the shared decode inputs: built before a deferred pool
        # (allocate_dsa_caches after construction), traced decode replays diverged from eager (stage-6/7 rework
        # bisect, 2026-10-18: reduced 8-slot probe, deferred pool after the sampler fails, deferred pool before the
        # sampler passes). Deferred builds create it in allocate_dsa_caches.
        self._lockstep = lockstep_sampling
        self.sampling = None
        if self.inputs is not None:
            self._init_sampler()
        wc = self.weight_cache
        self.load_seconds = time.time() - t_load
        logger.info(
            f"model weights loaded in {self.load_seconds:.1f}s ("
            + (f"weight cache {wc.root}: {wc.hits} hits, {wc.misses} stored)" if wc else "no weight cache)")
        )

    def _init_sampler(self):
        self.sampling = self._make_sampler()
        if self.sampling is not None and os.environ.get("GLM53_PENALTY_CLONE", "1") == "1":
            # TTPenalties.update_output_tokens reshapes the token tensor it is given and deallocates that view
            # (token_bin_counts_and_mask: new_tokens.deallocate()); handed the decode token buffer (tt_out_tok) it can
            # free it, after which the decode trace reads its token input from recycled memory. Count a copy instead.
            tp = self.sampling.tt_penalties
            orig_upd = tp.update_output_tokens

            def update_output_tokens(new_tokens, _o=orig_upd):
                return _o(ttnn.clone(new_tokens))

            tp.update_output_tokens = update_output_tokens
        if self._lockstep and self.sampling is not None:
            from .serving_state import install_lockstep_sampling

            install_lockstep_sampling(self.sampling, self.mesh)
            self.sampling.seed_manager.salt_duplicate_seeds = False

    def allocate_dsa_caches(self, num_blocks: int):
        """Deferred DSA allocation (allocate_dsa=False): every DSA layer's latent / pooled-key pool for `num_blocks`
        page blocks INCLUDING the model's dummy block (vLLM serving: vLLM's block count + 1, dummy = vLLM's count),
        its prefill constants, then the shared decode inputs. Call once, before any warm-up or trace capture."""
        assert self.inputs is None, "DSA caches are already allocated"
        for d in self.layers:
            if d.attn_kind == "dsa":
                d.attn.allocate_decode_state(self.S, num_blocks=num_blocks)
                d.allocate_decode_state(self.S)  # prefill constants / dispatch / mHC rows (pool already allocated)
        dsa = next((d.attn for d in self.layers if d.attn_kind == "dsa"), None)
        kda = next((d.attn for d in self.layers if d.attn_kind == "kda"), None)
        self.inputs = DeviceDecodeInputs(self.mesh, dsa=dsa, kda=kda, batch=self.S)
        if not self.skip_inactive:
            self._install_inactive_skip()
        if self.sampling is None and hasattr(self, "_lockstep"):  # (early allocation runs before _lockstep is set)
            self._init_sampler()
        return [(d.attn.latent, d.attn.pooled) for d in self.layers if d.attn_kind == "dsa"]

    # --------------------------------------------------------------------------------------------- pieces
    def _install_inactive_skip(self):
        """GLM53_SKIP_INACTIVE=1 (stage 7): decode routes every slot row through the MoE, active or not (S=8 build: 112.8
        ms per step with 1 or 8 active slots; S=1 build: 56.9 ms). The decode MoE path for S < 32 (pairs: one group of
        one row per (row, expert)) reads one expert block per group; groups whose expert id is >= E are skipped by the
        grouped sparse matmul (skip_padding_groups). This sets the expert ids of inactive rows to 65535 inside the
        decode trace, from the per-step active flags of the DSA inputs. Inactive rows then carry unspecified MoE output;
        nothing reads them (no cache / state write, logits unused)."""
        self.skip_inactive = False
        if (
            os.environ.get("GLM53_SKIP_INACTIVE", "1") != "1"
            or self.S < 2  # one slot: nothing to skip
            or self.inputs is None
            or self.inputs.dsa_bufs is None
        ):
            return
        for d in self.layers:
            moe = d.mlp
            if d.mlp_kind != "moe" or not hasattr(moe, "_routed_pairs") or not moe._skip_pad_groups_rule():
                continue
            orig = moe._routed_pairs

            def wrapped(x4, scores, ids, T, _orig=orig):
                act = self.inputs.dsa_bufs["active"]  # [1,1,S,1] fp32, refreshed at the start of every decode step
                if T != self.S or int(act.shape[-2]) != T:
                    return _orig(x4, scores, ids, T)
                idf = ttnn.typecast(ids, ttnn.float32) if ids.dtype != ttnn.float32 else ids
                if idf.layout != ttnn.TILE_LAYOUT:
                    idf = ttnn.to_layout(idf, ttnn.TILE_LAYOUT)
                # ids * act + (1 - act) * 65535: one term is zero, both exact in fp32 (ids < 192)
                keep = ttnn.multiply(idf, act, memory_config=DRAM)
                sent = ttnn.multiply(ttnn.rsub(act, 1.0), 65535.0, memory_config=DRAM)
                ids2 = ttnn.typecast(ttnn.add(keep, sent, memory_config=DRAM), ttnn.uint32)
                for t in (keep, sent) + ((idf,) if idf is not ids else ()):
                    ttnn.deallocate(t)
                out = _orig(x4, scores, ids2, T)
                ttnn.deallocate(ids2)
                return out

            moe._routed_pairs = wrapped
            self.skip_inactive = True
        logger.info(f"inactive-row MoE skip: {self.skip_inactive}")

    def _build_lm_head_ds(self, top_cache, shard):
        """lm_ds: per-die [C, Vl] slice zero-padded to Np = ceil(Vl / 256) * 256 columns, cut into column splits of at
        most LM_SPLIT columns (a single DRAM-sharded matmul over all 38912 columns needs 3.0 MB of L1 per core; the
        tt_transformers Blackhole LM head caps a split near 4000 columns), each DRAM width-sharded over the 8 banks and
        run on the DRAM-sharded decode program over the 32-core width shard of the sharded norm output."""
        from models.demos.blackhole.qwen36.tt import tp_common as tpc

        C, Vl = self.hidden, self.vocab // self.tp
        Np = tpc._roundup(Vl, 32 * tpc.DRAM_CORES)
        step = int(os.environ.get("GLM53_LM_SPLIT", "4096"))
        sizes = [step] * (Np // step) + ([Np % step] if Np % step else [])
        self.lm_cols = (Vl, Np)
        self.lm_splits = []
        full = None
        o = 0
        for j, nj in enumerate(sizes):

            def build(o=o, nj=nj):
                nonlocal full
                if full is None:
                    w = self.rd.top("lm_head.weight").T.contiguous().reshape(C, self.tp, Vl)  # per-die vocab slices
                    full = torch.nn.functional.pad(w, (0, Np - Vl))
                return ttnn.from_torch(
                    full[:, :, o : o + nj].reshape(C, self.tp * nj).contiguous(),
                    dtype=ttnn.bfloat8_b,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.mesh,
                    memory_config=tpc.create_dram_sharded_mem_config(C, nj),
                    mesh_mapper=shard(-1),
                )

            w = cached(top_cache, f"lm_head_ds{step}_{j}", build)
            cores, grid = _ds_grid(C, nj, 32)
            pc = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                in0_block_w=tpc._find_largest_divisor(C // 32 // cores),
                per_core_M=1,
                per_core_N=nj // 32 // cores,
                fused_activation=None,
            )
            valid = max(0, min(nj, Vl - o))
            self.lm_splits.append((w, _width_shard(C, cores, grid), pc, nj, valid))
            o += nj
        full = None
        self.lm_act = self.lm_splits[0][1]
        self.lm_head = None

    def _lm_head(self, hn):
        """hn [1,1,T,C] bf16 (decode: width-sharded or interleaved) -> logits [1,1,T,V/tp] bf16 DRAM."""
        if "lm_ds" not in self.head_opts:
            return ttnn.linear(hn, self.lm_head, compute_kernel_config=self.ckc, memory_config=DRAM)
        T = int(hn.padded_shape[-2])
        if T != 32:  # prefill rows (all_logits): 32-row blocks on the decode program
            parts = []
            for r0 in range(0, T, 32):
                blk = ttnn.slice(hn, (0, 0, r0, 0), (1, 1, r0 + 32, self.hidden), memory_config=DRAM)
                parts.append(self._lm_head(blk))
                ttnn.deallocate(blk)
            lg = ttnn.concat(parts, dim=2, memory_config=DRAM) if len(parts) > 1 else parts[0]
            for t in parts if len(parts) > 1 else []:
                ttnn.deallocate(t)
            return lg
        outs = []
        xs = None
        for w, act, pc, nj, valid in self.lm_splits:
            if xs is None or xs.memory_config() != act:
                if xs is not None and xs is not hn:
                    ttnn.deallocate(xs)
                xs = hn if (hn.is_sharded() and hn.memory_config() == act) else ttnn.to_memory_config(hn, act)
            out = ttnn.linear(
                xs,
                w,
                compute_kernel_config=self.ckc,
                program_config=pc,
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            )
            o = ttnn.to_memory_config(out, DRAM)
            ttnn.deallocate(out)
            if valid < nj:
                o2 = ttnn.slice(o, (0, 0, 0, 0), (1, 1, int(o.shape[-2]), valid), memory_config=DRAM)
                ttnn.deallocate(o)
                o = o2
            if valid > 0:
                outs.append(o)
        if xs is not None and xs is not hn:
            ttnn.deallocate(xs)
        lg = ttnn.concat(outs, dim=-1, memory_config=DRAM)
        for t in outs:
            ttnn.deallocate(t)
        return lg

    def _make_sampler(self):
        if self.tp == 1:
            return None
        from models.common.sampling.generator import SamplingGenerator

        args = SimpleNamespace(
            vocab_size=self.vocab,
            padded_vocab_size=self.vocab,
            cluster_shape=(1, self.tp),
            max_batch_size=self.SR,
            max_top_k=32,
            sampling_dp=1,
            model_config={},
        )
        return SamplingGenerator(args=args, mesh_device=self.mesh, tt_ccl=self.tt_ccl)

    def embed(self, tok_dev):
        """tok_dev uint32 ROW_MAJOR [1, T] (replicated) -> X [1,1,T,4*C/tp] fp32 stream-interleaved."""
        T = tok_dev.shape[-1]
        e = ttnn.embedding(tok_dev, self.embed_w, layout=ttnn.TILE_LAYOUT, memory_config=DRAM)  # [1,T,C/tp]
        e = ttnn.typecast(ttnn.reshape(e, (1, 1, T, e.shape[-1])), ttnn.float32)
        X = ttnn.concat([e] * HC, dim=-1, memory_config=DRAM)
        ttnn.deallocate(e)
        return X

    def head(self, X):
        """X [1,1,T,4*C/tp] -> logits [1,1,T,V/tp] bf16 (vocab-sharded)."""
        T, w = X.shape[-2], X.shape[-1] // HC
        acc = None
        for i in range(HC):
            s = ttnn.slice(X, (0, 0, 0, i * w), (1, 1, T, (i + 1) * w), memory_config=DRAM)
            acc = s if acc is None else ttnn.add(acc, s, memory_config=DRAM)
        h = ttnn.typecast(ttnn.multiply(acc, 1.0 / HC, memory_config=DRAM), ttnn.bfloat16)  # hc_head: stream mean
        ttnn.deallocate(acc)
        if self.tp > 1:
            h = tt_all_gather(
                h, self.mesh, self.tt_ccl, cluster_axis=None, dim=3, topology=TOPOLOGY, dtype=ttnn.bfloat16
            )
        if "norm_sharded" in self.head_opts and int(h.padded_shape[-2]) == 32:
            hn = sharded_rms_norm(
                h, self.norm_w, self.tc["rms_norm_eps"], self.norm_ckc, keep_sharded="lm_ds" in self.head_opts
            )
        else:
            hn = ttnn.rms_norm(
                h,
                epsilon=self.tc["rms_norm_eps"],
                weight=self.norm_w,
                memory_config=DRAM,
                compute_kernel_config=self.norm_ckc,
            )
        ttnn.deallocate(h)
        logits = self._lm_head(hn)
        ttnn.deallocate(hn)
        return logits

    def tokens_dev(self, ids: torch.Tensor):
        return ttnn.from_torch(
            ids.reshape(1, -1).to(torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )

    def logits_to_host(self, logits) -> torch.Tensor:
        if self.tp == 1:
            return ttnn.to_torch(logits).float()
        return ttnn.to_torch(logits, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh, dim=-1)).float()

    # --------------------------------------------------------------------------------------------- paging
    def set_page_table(self, page_table: torch.Tensor):
        """[S, n <= max_context/128] block ids of the DSA latent pool (every DSA layer shares the table)."""
        full = None
        for d in self.layers:
            if d.attn_kind == "dsa":
                d.set_page_table(page_table)
                full = d.attn.page_table_host
        if full is not None:
            self.inputs.set_page_table(full)

    def reset_state(self):
        """Zero every slot's recurrent state and the DSA open-pool accumulators (caches keep their blocks; stale
        cache rows are never attended: positions start again at 0)."""
        for d in self.layers:
            a = d.attn
            bufs = (
                [d.decode_state.recurrent] + list(d.decode_state.conv)
                if d.attn_kind == "kda"
                else [a.acc_m, a.acc_den, a.acc_num]
            )
            for t in bufs:  # zeros (x * 0 would keep a NaN), then free the temporary: the buffers keep their addresses
                z = ttnn.zeros_like(t)
                ttnn.copy(z, t)
                ttnn.deallocate(z)

    # --------------------------------------------------------------------------------------------- prefill
    def chunk_plan(self, L: int, sizes=None):
        """[(n, valid rows)] of an L-token prompt. sizes: forced chunk lengths (warm-up). GLM53_PREFILL_PLAN=cost:
        the multiset of constructed lengths with the least estimated time (PREFILL_COST_MS) that covers L, largest
        first, so every chunk but the last is full (lengths 128 / 512 / 2048: 161 -> one 512 chunk, 600 -> 512 + 128,
        2600 -> 2048 + 512 + 128). GLM53_PREFILL_PLAN=legacy: the largest length, then the smallest that holds the rest
        (2600 -> 2048 + 2048). The estimates ignore the chunk's start position (a deep chunk's sparse path costs more).
        """
        lens = self.prefill_lengths
        if sizes is None and self.prefill_plan == "cost" and len(lens) > 1:
            unit = lens[0]
            nb = -(-L // unit)
            best, choice = [0.0] + [float("inf")] * nb, [None] * (nb + 1)
            for r in range(1, nb + 1):
                for n in lens:
                    c = self.prefill_cost[n] + best[max(0, r - n // unit)]
                    if c < best[r] - 1e-9:
                        best[r], choice[r] = c, n
            sizes, r = [], nb
            while r > 0:
                sizes.append(choice[r])
                r = max(0, r - choice[r] // unit)
            sizes.sort(reverse=True)
        if sizes is not None:
            plan, c0 = [], 0
            for n in sizes:
                ln = min(n, L - c0)
                if ln <= 0 or c0 + n > self.max_context:
                    break
                plan.append((n, ln))
                c0 += ln
            if c0 == L and len(plan) == len(sizes):
                return plan
            logger.warning(f"prefill plan {sizes} does not fit L={L} / max_context={self.max_context}: legacy plan")
        plan, c0, bucket = [], 0, max(lens)  # legacy
        while c0 < L:
            ln = min(bucket, L - c0)
            plan.append((next(b for b in lens if b >= ln), ln))
            c0 += ln
        return plan

    def prefill(
        self, ids: torch.Tensor, slot: int, *, all_logits: bool = False, device_logits: bool = False, sizes=None
    ):
        """One user's prompt ids [L] into decode slot `slot`. Returns host logits [L, V] (all_logits) or [V] (last).
        device_logits=True: the last row's logits stay on the device, copied to all S rows: [1,1,S,V/tp] bf16 (the
        on-device sampler's decode-logits shape; row `slot` is sampled with the slot's params)."""
        L = int(ids.numel())
        if not 0 < L <= self.max_context:
            raise ValueError(f"prompt length {L} outside 1 .. max_seq_len {self.max_context}")
        states = [None] * len(self.layers)
        outs = []
        last = None
        c0 = 0
        for n, ln in self.chunk_plan(L, sizes):
            chunk = torch.zeros(n, dtype=torch.int64)
            chunk[:ln] = ids[c0 : c0 + ln]
            traced = c0 == 0 and n in self._ptraces
            if traced:
                X, tstates = self._prefill_replay(chunk, n, ln, slot)
                states = list(tstates)
            # one input set per layer kind: the chunk inputs depend only on (n, start, length, slot) and the shared
            # page table, never on the layer (DSA: ~17 MB of index rows per 2048-row chunk past position 2050; 11
            # copies were live at once before, stage-6 memory budget)
            shared = {}
            for d in self.layers:
                if not traced and d.attn_kind not in shared:
                    shared[d.attn_kind] = d.prefill_inputs(n, start=c0, length=ln, slot=slot)
            if not traced:
                inputs = [shared[d.attn_kind] for d in self.layers]
                tok = self.tokens_dev(chunk)
                X = self.embed(tok)
                ttnn.deallocate(tok)
                for i, d in enumerate(self.layers):
                    Y, states[i] = d.prefill_forward(X, inputs=inputs[i], attn_state=states[i])
                    ttnn.deallocate(X)
                    X = Y
            if all_logits:
                outs.append(self.logits_to_host(self.head(X))[0, 0, :ln])
            elif c0 + ln == L:  # last row only: one tile of rows through the head
                r0 = ((ln - 1) // 32) * 32
                Xl = ttnn.slice(X, (0, 0, r0, 0), (1, 1, r0 + 32, X.shape[-1]), memory_config=DRAM)
                if device_logits:
                    last = self._rows_of_last(self.head(Xl), ln - 1 - r0)
                else:
                    last = self._row_to_host(self.head(Xl), ln - 1 - r0)
                ttnn.deallocate(Xl)
            if not traced:  # a trace output keeps its address for the next replay
                ttnn.deallocate(X)
            c0 += ln
        for i, d in enumerate(self.layers):
            d.commit_prefill(slot, states[i])
        return torch.cat(outs, dim=0) if all_logits else last

    def choose_moe_split(self):
        """Stage 7 (TTFT): the MoE prefill row split (tt/moe_opt.sorted_rows_cap, GLM53_MOE_SORTED_ROWS, stage-6 memory
        fix) costs ~11 ms per 2048-row MoE layer. It only lowers the prefill peak (whole 2048-row chunk: 417 MiB above
        the layer entry, split: 297 MiB). Run the whole chunk when the largest free DRAM block per die holds the
        unsplit peak with GLM53_MOE_NOSPLIT_FREE_MIB (default 1024) to spare, else keep the split. Call once after the
        caches are allocated and before the warm-up prefills (the chosen path is the one compiled). An explicit
        GLM53_MOE_SORTED_ROWS wins. Bit-identical either way (moe_opt._routed_sorted_split)."""
        if os.environ.get("GLM53_MOE_SPLIT_AUTO", "1") != "1":
            self.moe_split = {"mode": "default", "rows": os.environ.get("GLM53_MOE_SORTED_ROWS", "default")}
            return self.moe_split
        # an explicit GLM53_MOE_SORTED_ROWS wins; one written by an earlier automatic choice (marker) does not
        if "GLM53_MOE_SORTED_ROWS" in os.environ and os.environ.get("_GLM53_MOE_ROWS_AUTO") != "1":
            self.moe_split = {"mode": "env", "rows": os.environ["GLM53_MOE_SORTED_ROWS"]}
            return self.moe_split
        need = int(float(os.environ.get("GLM53_MOE_NOSPLIT_FREE_MIB", "1024")) * 2**20)
        try:
            v = ttnn.get_memory_view(self.mesh, ttnn.BufferType.DRAM)
            free = int(v.largest_contiguous_bytes_free_per_bank) * int(v.num_banks)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"memory view unavailable ({e}): MoE prefill split kept")
            free = 0
        rows = "0" if free >= need else os.environ.get("GLM53_MOE_SORTED_ROWS_SPLIT", "1024")
        os.environ["GLM53_MOE_SORTED_ROWS"] = rows  # read per call by moe_opt.sorted_rows_cap
        os.environ["_GLM53_MOE_ROWS_AUTO"] = "1"
        self.moe_split = {"mode": "auto", "rows": rows, "largest_free_bytes_per_die": free, "need_bytes": need}
        logger.info(f"MoE prefill row split: {self.moe_split}")
        return self.moe_split

    # --------------------------------------------------------------------------------------------- traced prefill
    def _prefill_body(self, tok, shared, zero_state):
        """embed + every layer on one chunk (the captured region of a traced prefill)."""
        X = self.embed(tok)
        states = []
        for d in self.layers:
            st = zero_state if d.attn_kind == "kda" else None
            Y, st = d.prefill_forward(X, inputs=shared[d.attn_kind], attn_state=st)
            ttnn.deallocate(X)
            X = Y
            states.append(st)
        return X, states

    def prepare_prefill_traces(self, slot: int = 0):
        """Allocate the persistent inputs of every traced prefill length (call before ANY trace capture: an allocation
        made after a capture can sit in that trace's freed intermediates and be overwritten by its replays). The
        caches / page table must be in place (DSA chunk page rows are copied per replay)."""
        for n in self.prefill_trace_lengths:
            if n in self._ptr_inputs or n > self.max_context:
                continue
            shared = {}
            for d in self.layers:
                if d.attn_kind not in shared:
                    shared[d.attn_kind] = d.prefill_inputs(n, start=0, length=n, slot=slot)
                    if d.attn_kind == "dsa":
                        shared["dsa"]["attn"]["defer_tail"] = True
            kda = next((d for d in self.layers if d.attn_kind == "kda"), None)
            zero = kda.attn.new_prefill_state() if kda is not None else None  # ttKDA returns new carries: stays zero
            self._ptr_inputs[n] = {
                "tok": self.tokens_dev(torch.zeros(n, dtype=torch.int64)),
                "shared": shared,
                "zero": zero,
            }

    def capture_prefill_traces(self):
        """Capture one prefill trace per prepared length: one eager pass on the persistent inputs (compiles the
        deferred-tail variant), then the capture. Call after every other trace capture and after a device sync."""
        for n, p in self._ptr_inputs.items():
            if n in self._ptraces:
                continue
            X, states = self._prefill_body(p["tok"], p["shared"], p["zero"])
            ttnn.deallocate(X)
            for d in self.layers:
                if d.attn_kind == "dsa":
                    for t in d.attn._tail:
                        ttnn.deallocate(t)
            del states
            ttnn.synchronize_device(self.mesh)
            tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
            try:
                X, states = self._prefill_body(p["tok"], p["shared"], p["zero"])
            finally:
                ttnn.end_trace_capture(self.mesh, tid, cq_id=0)
            tails = [d.attn._tail for d in self.layers if d.attn_kind == "dsa"]
            self._ptraces[n] = {"id": tid, "X": X, "states": states, "tails": tails}
            ttnn.synchronize_device(self.mesh)
            logger.info(f"prefill trace captured: {n} rows")

    def release_prefill_traces(self):
        for p in self._ptraces.values():
            ttnn.release_trace(self.mesh, p["id"])
        self._ptraces = {}

    def _prefill_replay(self, chunk, n, ln, slot):
        """Refresh the persistent inputs of length n for (chunk, valid rows ln, slot), replay, hand the DSA open-pool
        rows to the slot. Returns the trace's (X, states) outputs (valid until the next replay of this trace)."""
        p, tr = self._ptr_inputs[n], self._ptraces[n]
        rep = ttnn.ReplicateTensorToMesh(self.mesh)
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(
                chunk.reshape(1, -1).to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=rep
            ),
            p["tok"],
        )
        dsa = None
        for d in self.layers:
            if d.attn_kind == "kda" and "kda" in p["shared"]:
                a0, a1 = p["shared"]["kda"]["attn"]
                for buf, v in ((a0, 0), (a1, ln)):
                    ttnn.copy_host_to_device_tensor(
                        ttnn.from_torch(
                            torch.tensor([v], dtype=torch.int64),
                            dtype=ttnn.uint32,
                            layout=ttnn.ROW_MAJOR_LAYOUT,
                            mesh_mapper=rep,
                        ),
                        buf,
                    )
                break
        for d in self.layers:
            if d.attn_kind == "dsa":
                dsa = d.attn
                h = dsa.prefill_inputs(n, 0, ln, slot, host=True)
                for key in ("row_idx", "vmask", "chunk_pt"):
                    ttnn.copy_host_to_device_tensor(h[key], p["shared"]["dsa"]["attn"][key])
                break
        ttnn.execute_trace(self.mesh, tr["id"], cq_id=0, blocking=False)
        it = iter(tr["tails"])
        for d in self.layers:
            if d.attn_kind == "dsa":
                d.attn._tail = next(it)
                d.attn.write_prefill_tail(ln, slot)
        return tr["X"], tr["states"]

    def _row_to_host(self, logits, row: int) -> torch.Tensor:
        """One row of logits [1,1,32,V/tp] -> host [V] (stage 7: reading the whole 32-row tile of 154880 logits and
        converting it took ~50 ms of the TTFT; one row-major row is ~0.3 MB)."""
        V = logits.shape[-1]
        one = ttnn.slice(logits, (0, 0, row, 0), (1, 1, row + 1, V), memory_config=DRAM)
        ttnn.deallocate(logits)
        rm = ttnn.to_layout(one, ttnn.ROW_MAJOR_LAYOUT, memory_config=DRAM)
        ttnn.deallocate(one)
        out = self.logits_to_host(rm).reshape(-1)
        ttnn.deallocate(rm)
        return out

    def _rows_of_last(self, logits, row: int):
        """logits [1,1,32,V/tp] -> row `row` copied to S rows [1,1,S,V/tp] (TILE)."""
        V = logits.shape[-1]
        one = ttnn.slice(logits, (0, 0, row, 0), (1, 1, row + 1, V), memory_config=DRAM)
        ttnn.deallocate(logits)
        rm = ttnn.to_layout(one, ttnn.ROW_MAJOR_LAYOUT, memory_config=DRAM)
        ttnn.deallocate(one)
        rep = ttnn.repeat(rm, ttnn.Shape((1, 1, self.SR, 1)), memory_config=DRAM) if self.SR > 1 else rm
        out = ttnn.to_layout(rep, ttnn.TILE_LAYOUT, memory_config=DRAM)
        if rep is not rm:
            ttnn.deallocate(rep)
        ttnn.deallocate(rm)
        return out

    # --------------------------------------------------------------------------------------------- decode
    def decode_step(self, tok_buf):
        """One decode step of all slots (traceable). tok_buf uint32 ROW_MAJOR [1, S]. Returns logits [1,1,S,V/tp]."""
        self.inputs.refresh()
        X = self.embed(tok_buf)
        for d in self.layers:
            bufs = self.inputs.kda_bufs if d.attn_kind == "kda" else self.inputs.dsa_bufs
            Y = d.decode_forward(X, inputs=bufs)
            ttnn.deallocate(X)
            X = Y
        logits = self.head(X)
        if self.SR > self.S:  # sampler rows: zero rows for the padding (never read back)
            padded = ttnn.pad(logits, [(0, 0), (0, 0), (0, self.SR - self.S), (0, 0)], 0.0)
            # no ttnn.deallocate(logits): a pad inside the tile padding returns a VIEW of `logits`, and deallocate
            # (force=True by default) freed the storage the returned tensor - the decode trace's persistent logits
            # output - still uses. The allocator then handed that block to later tensors and every replay wrote the
            # logits into them (stage-6/7 rework 2026-10-18: teacher forcing 14/32/90 whenever the block was reused,
            # e.g. under DRAM pressure; found with ttnn.deallocate(force=False) + an is_allocated() check). Python
            # releases `logits` when it goes out of scope; the shared storage lives as long as `padded`.
            logits = padded
        ttnn.deallocate(X)
        self.inputs.advance()  # pos += 1 for active slots: the next replay decodes the next position
        return logits


def model_from_env(mesh, **kw) -> Glm53Model:
    """GLM53_LAYERS=0,3,4: a reduced model (debug probe); default all 45 layers."""
    env = os.environ.get("GLM53_LAYERS")
    if env and "layer_indices" not in kw:
        kw["layer_indices"] = [int(s) for s in env.split(",")]
    return Glm53Model(mesh, **kw)

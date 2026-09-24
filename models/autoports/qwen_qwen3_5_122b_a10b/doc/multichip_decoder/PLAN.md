# Stage 4 plan: multichip decoder for Qwen/Qwen3.5-122B-A10B on the (1,4) Blackhole mesh (drafted 2026-09-24 07:55 from the code; numbers to be confirmed on hardware)

> Note (2026-09-24, end of stage 4): the DRAM/KV figures below are the planning draft; `doc/context_contract.json` block
> `multichip_decoder_stage4` and README section 2 supersede them (bf16 embedding 0.355 GiB, weights 17.00 GiB, KV pool 11.31 GiB).

Baseline: OptimizedDecoder (SELECTED). Mesh (1,4) P300x2, FABRIC_1D, Ring (ClusterType.P300_X2). Root fixture: parametrize_mesh_tp
(qwen36/tests/test_factory.py:178-203) falls back to (1, len(device_ids)) = (1,4).

## Strategy
- TP=4 for GDN/attention/norms/shared expert via the qwen36 num_devices>1 branch (layer.py:81-99): TPAttention (attention/tp.py:121,
  loader 19-118), TPGatedDeltaNet (gdn/tp.py:161, loader 37-158), DistributedNorm (layer.py:124-164). Needs args.num_devices=4 (from
  mesh in Qwen36ModelArgs.__init__ -> _init_tp_config: n_local_heads 8, n_local_kv_heads 1, kv_replication True (device d holds KV head
  d//2, matching q heads 8d..8d+7), gdn_qkvzab_dim_tp 5152, attn_qkv_fused_dim_tp 4608, attn_out_dim_tp 2048, gdn_value_dim_tp 2048),
  a TT_CCL(mesh_device) (tt_transformers/tt/ccl.py:33) passed as tt_ccl.
- Qwen35MoeModelArgs extensions: policy hook fields consumed by the TP modules instead of env knobs (gdn_state_dtype /
  gdn_decode_high_precision replace QWEN35_GDN_STATE_BF16/QWEN35_GDN_DECODE_BF16 at gdn/tp.py:230,1095; kv_dtype replaces QWEN_SDPA_BF8
  at attention/tp.py:138,710-716; sdpa chunks (TP decode uses 0/0 at attention/tp.py:564-569); gdn_conv1d_max_channels;
  prefill_norm_distributed) + is_distributed_norm(mode) override (base False for dim 3072 on 1D mesh, tt_transformers model_config
  2696-2705) + MoE TP fields (inter_local = 256; moe.py:113 assert holds).
- Stage-3 selections that do NOT transfer (re-decide on TP modules): DS decode weights + DenseDecodeMatmul (TP uses proj_1d_decode
  interleaved + create_matmul_1d_decode_progcfg tuned for 27B; re-sweep cores for attn qkv K=3072/N=4608, wo K=2048/N=3072,
  GDN qkvzab N=5152, out K=2048/N=3072), decode_norm_sharded/attn_norm_out_sharded (DistributedNorm already sharded on
  attn_input_grid = find_grid(96) = 8x2 = 16 cores), one-hot GVA (TP uses repeat_interleave, gdn/tp.py:1069-1070; the recurrent-step
  rewrite is inherited). Mixer dtype hardcoded bfp8 in both TP loaders (= policy).

## Residual contract (layer.py:203-305 TP branch)
Residual x FRACTURED dim=3 ([1,1,B|T,768] per device). attention_norm (DistributedNorm) all-gathers -> replicated [.,.,.,3072] input to
the mixer; mixer returns fractured partial via tt_all_reduce(dim=3) = reduce_scatter_minimal_async on (1,4) (ccl.py:169-191);
h = x + attn (fractured); ff_input = ffn_norm(h) gathers again; feed_forward.forward(ff_input, mode) receives REPLICATED full-dim and
must return FRACTURED [.,.,.,768] partial (output = h + ff_output, layer.py:301) — exactly Qwen36MLP._forward_tp (mlp.py:216-352,
tt_all_reduce at 343-351). Decode: ff norm output arrives L1 WIDTH-sharded on the 16-core grid ([32,192]/core); MoE must
to_memory_config(x, L1) once at entry. Decoder I/O for stacking: fractured [1,1,B,768] / [1,1,T,768]; tests shard at entry
(ShardTensorToMesh dim=-1) and compose at exit (ConcatMeshToTensor dim=-1).
Prefill AGMM trap: with is_distributed_norm(PREFILL)=False the norm output is replicated but TPAttention._fuse_agmm (attention/tp.py:146)
and TPGatedDeltaNet._fuse_agmm (gdn/tp.py:192) gather again -> "K=12288 vs K_w=3072". Option A (default): Ornith gates #2-5
(tpc.prefill_norm_agmm_enabled(args), mlp_gateup_agmm_enabled(nd,args), _fuse_agmm gates, layer.py:50-53,68). Option B: per-norm
distributed attention_norm in prefill (768/die = 24 tiles legal) keeping fused AGMM for the mixers; ffn_norm non-distributed
(MoE needs replicated input). Measure A vs B at T=2048.

## Routed experts: (a) TP on intermediate (RECOMMENDED) vs (b) EP=4
Common: 1.266 GiB/layer bfp4 -> 15.19 GiB/chip either way; router replicated; both end with ONE reduce-scatter dim=3 of
(routed_partial + shared_partial) [B,3072] bf16 (4.6 KB/chip at B=1, 9.4 MB at T=2048); no all-to-all (input replicated).
(a): up_gate [1,256,3072,512] per chip ([up_d 256|gate_d 256], ShardTensorToMesh dim=-1, moe.py:148-157), down [1,256,256,3072]
(dim=-2, moe.py:158-159); routing identical on every chip; gather mode unchanged (ids<256). B=1 decode ~120 us/chip est. (10.6 MB +
16 pairs x ~6 us fixed). Geometry: gate_up N=512 -> 16 tiles -> <=16 cores per_core_N=1 (probe output_tile [32,16] -> 32 cores);
down K=256 -> in0_block_w 8, 48 cores. Balanced, deterministic, small code delta.
(b): up_gate [1,64,3072,2048], down [1,64,1024,3072]; local routing via selector matmul; gather mode impossible (data-dependent local
count); masked mode with nnz=None forces EXPANDED output ([64,32,2048] + [64,32,3072] elementwise ~46 MB) -> ~180 us/chip at B=1;
E[max hits] = 3.5 of 8 (imbalance). Wins at B=32 (~2-2.5x on routed rows) and prefill (~2.5x: G=191 vs 761 pairs) but needs a
SortedDispatch rework (non-local ids to their own zero-mask block; no compact output). Keep as documented optimize-stage candidate
(routed_mode="ep") for batched/TTFT targets.

## Per-device shapes (option a)
attn wqkv_fused [3072,4608] = [q 2048|k 256|v 256|gate 2048] (prepare_attn_qkv_deint, tp_common.py:679-697) bfp8 interleaved;
wo [2048,3072]; q/k norm replicated (1+w folded); KV cache [blocks,1,64,256] bfp8 per chip (6528 B/token/chip), page table
replicated int32 (4096 blocks/user at 262144); GDN qkvzab [3072,5152] = [q 512|k 512|v 2048|z 2048|a 16|b 16] (161 tiles; a/b share
the last tile, gdn/tp.py:438-445); GDN out [2048,3072] (prefill matmul_reduce_scatter fp32, gdn/tp.py:619-629); conv taps 4x[3072],
conv_w1d [3072,1,4] -> 3x1024 channel blocks; A_log/dt_bias [16] sharded, norm_w [128] replicated; recurrent state [B,16,128,128]
bf16 0.5 MiB/user/layer (fp32 1 MiB); conv_states 4x[1,B,3072] + conv_carry [1,3,3072]; router [3072,256] bf16 replicated (+gate op
copy); routed up_gate [1,256,3072,512] / down [1,256,256,3072] bfp4 (324 MiB/layer/chip); shared packed [3072,513] (tile-padded 544),
w2 [256,3072]; decode activations: x frac [1,1,B,768] -> AG -> [1,1,B,3072] L1 width-sharded 16x[32,192] -> mixer -> partial -> RS;
MoE gather-mode fused [1,1,1,8,1,512] -> swiglu [1,8,1,256] -> down [1,8,1,3072] -> weighted sum + shared -> RS. Prefill T=2048:
AG 9.4 MB recv/chip, RS 9.4 MB send/chip (GDN out RS fp32 18.9 MB); sorted dispatch X_disp [1,761,32,3072] 150 MB, out_gu
[1,761,32,512] 25 MB, out_dn 150 MB. Embedding hidden-sharded (tt_transformers/tt/embedding.py:34), LM head vocab-sharded
[3072,62080] (qwen36/tt/model.py:125-140).
Collectives per layer (decode): attention_norm AG, mixer RS, ffn_norm AG, MoE RS = 4 (192/token step), 4.6 KB/chip each at B=1
(latency-bound ~10-20 us each -> verify with tt-perf-report). matmul_reduce_scatter_decode (tp_common.py:467) loses at M=1: keep the
separate RS at decode.

## Context contract on (1,4) (per chip, 31.9 GiB)
routed bfp4 15.19 GiB; attention mixers bfp8 0.245; GDN mixers bfp8 0.826; shared 0.117; routers 0.15; embedding+LM head 0.38
(bfp8) / 0.71 (bf16) -> weights 16.9 / 17.2 GiB. GDN state B=32: 0.58 GiB bf16 (1.15 fp32) + conv 0.63 MiB/user. Trace + CCL
buffers + prefill transients (sorted dispatch peak ~330 MB) 1.0-2.5 GiB -> reserve 3.0 GiB. KV pool 12.0 GiB / 6528 B = 1.98M
tokens: 7 users at 262144, or 32 users x 61,995 tokens. current_supported_context stays 262144. Contract block
multichip_decoder_stage4: mesh [1,4], per_device_weight_bytes, kv_bytes_per_token_per_device 6528, kv_heads_per_device 1
(pair replicated), gdn_state_bytes_per_user_per_device, reserved bytes, kv_pool_bytes, max_users_at_262144 7, tokens_at_batch_32
61995, largest tested.

## GDN conv1d L1 overflow at TP prefill width
TPGatedDeltaNet._conv1d_prefill (gdn/tp.py:288-368): one depthwise ttnn.conv1d over C = qkv_dim_tp channels (HEIGHT_SHARDED,
Conv2dL1FullSliceConfig), taken when self._gdn_conv1d and valid_len is None (gdn/tp.py:495-497). 122B per-device C = 3072 (27B had
1536); Ornith: CB overflow at 2048, 1024 passes (ornith branch gdn/tp.py:36-45 CONV1D_MAX_CHANNELS=1024, conv_w1d_blocks loader
180-195, blocked _conv1d_prefill 363-460 with concat on dim 2). Plan: import the blocked implementation with
args.gdn_conv1d_max_channels (default 1024; blocks [q512|k512], [v1024], [v1024]); verify at T=128/2048/4096; MAC FIR
(_causal_conv1d_fir, _gdn_conv1d=False; already the path for valid_len is not None) as fallback. l1_small_size 24576
(GDN_CONV1D_L1_SMALL_SIZE, qwen36/tt/model_config.py:16).

## Test / evidence plan
1 baseline dump tool (single die OptimizedDecoder, seeded synthetic + real L0/L3, fixed inputs/page tables/positions -> .pt) and mesh
tests loading it (PCC >= 0.995 vs baseline and vs HF); 2 prefill T in {5,37,129,333,2048} + decode per kind (component probes if
< 0.995: norm, qkv, SDPA, wo+RS, router equality across chips, expert outputs, MoE RS); 3 paged KV on mesh per head pair, chunked
2048+2048+333, decode at 4429; 4 non-aligned lengths + 100-token continuation; 5 B=32 (forward_prefill_collect/write_slot,
gdn/tp.py:646-653, 808-828); 6 traced decode (trace region from 64 MB), 3 replays bit-identical; 7 latency: single-chip baseline
re-measured in-session vs mesh, speedup, efficiency = speedup/4 (expected GDN ~0.40-0.45 ms, attention ~0.30-0.35 ms => ~2x, ~50%);
8 tt-perf-report per kind (CCL rows 4/layer, DRAM utilisation of 1D matmuls and 16-core gate_up rows, data movement); 9 fallback
audit (valid_len from_torch boundary only for non-aligned prefill, replicated to mesh); 10 stress 20 cycles; 11 watcher (reset all 4
devices after any failed CCL run); 12 EP rejected-alternative numbers (+ microbench if time).

## Implementation steps
1 shared qwen36 fixes as ONE commit ("AGMM gate + fixes" from ornith-1.5-35b-a3b): tp_common in0_block_w divisor (#1);
prefill_norm_agmm_enabled(args) + mlp_gateup_agmm_enabled(nd,args) (#2); _fuse_agmm gates attention/tp.py:146, gdn/tp.py:192
(#3,#4); layer.py:50-53,68 gates (#5); conv1d channel blocking + args.gdn_conv1d_max_channels (#6); reset_state(batch_size=None) +
TPAttention.reset_cache alias (#7); recurrent_state/fused_conv_state properties (#8); env knobs -> args (gdn_state_dtype,
gdn_decode_high_precision, kv_dtype, sdpa chunks). Skip model.py items until stage 6.
2 tt/model_config.py: extend Qwen35MoeModelArgs (policy attrs, is_distributed_norm override, MoE TP fields).
3 tt/moe.py + tt/optimized_decoder.py: finish num_devices>1 path of Qwen35MoE.forward: entry to_memory_config(x, L1) when sharded;
_route unchanged; routed gather/masked/sorted with Nl=256 (re-derive configs; gather_gate_up_output_tile knob for the [32,16]
probe); out = tt_all_reduce(routed + shared, mesh, tt_ccl, cluster_axis=0, dim=3, topology=args.ccl_topology(),
memory_config=DRAM); shared 1D configs for [3072,513] and [256,3072].
4 tt/multichip_decoder.py: MultichipConfig (routed_mode "tp", prefill_norm_distributed False, dense 1D core counts, gate_up
cores/tile, conv1d max channels, kv/gdn dtypes = SELECTED) + MultichipDecoder(OptimizedDecoder): from_state_dict builds TT_CCL and
passes tt_ccl to LAYER_CLS; allocate_kv_cache per-device [blocks,1,64,256] bfp8 + set_paged_kv_cache; allocate_gdn_state ->
dn.B=batch; dn.reset_state(); dn._stable_state=True (as _allocate_kv_caches_tp, qwen36/tt/model.py:2768-2791) exposing
{"recurrent": dn.rec_state, "conv": dn.conv_states, "carry": dn.conv_carry}; reset_gdn_state -> dn.reset_state_inplace();
prefill_forward ([1,1,T,768] fractured, cos/sin [1,1,T,64] from Qwen36RoPESetup for apply_partial_rope_prefill, one-row page table
pt_row (forward_prefill_paged fills batch_idx=0; layer.py:243-252 passes no user_id), valid_len masking, start_pos>0 continuation
after decode copies conv_states[1..3] into conv_carry first, B>1 via gdn_collect=True + write_slot); decode_forward ([1,1,B,768]
fractured, cos/sin [1,B,1,64] replicated as prepare_decode_inputs_host, qwen36/tt/model.py:3231-3239).
5 tests/test_multichip_decoder.py + tools/baseline_dump.py + mesh _Harness variant (shard/compose at the boundary; tensor cache
under TT_CACHE_PATH with the _mesh1x4 suffix, qwen36/tt/model_config.py:266-283).
6 perf pass: tracy signposted prefill/decode per kind, tools/perf_tables.py reuse, MultichipConfig sweeps (dense 1D cores, shared
cores, gate_up tile/cores, norm mode A vs B).
7 docs: doc/multichip_decoder/README.md, work_log.md, context_contract.json; local commit, log SHAs.

## Risks / fallbacks
conv1d CB overflow -> blocked conv -> MAC FIR; AGMM double gather -> gates (A) / distributed attention norm (B); 16-core gate_up ->
output_tile [32,16] probe, else EP for prefill/large batch; gate op on mesh -> functional router; GDN state API mismatch handled in
multichip_decoder.py (batched GDN prefill kernel cap BH<=32 -> B<=2 for forward_prefill_batched, prefill stays per user); fp32 vs
bf16 state knob re-measured on mesh; CCL hangs -> single TT_CCL, Ring, reset after failures, watcher separate from profiler; trace:
tt_all_reduce allocates inside trace (ok), no host I/O in the step; sparse per-pair fixed cost does not shrink with TP (~50%
efficiency expected; kernel-side limitation to record).

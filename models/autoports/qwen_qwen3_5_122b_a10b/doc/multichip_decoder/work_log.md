# Multichip decoder — work log (stage 4)
Baseline: stage-3 optimized decoder (commit 19aa5d24777). Mesh (1,4) P300x2 (4 Blackhole dies), FABRIC_1D, Ring.
Times below are host-clock; authoritative timestamps are inside the referenced logs (~/qwen35-122b-bringup/logs/04-*.log).

## 2026-09-24
- (host 08:15) mesh smoke: (1,4) opened with `ttnn.set_fabric_config(FABRIC_1D)` + `open_mesh_device(MeshShape(1,4))`; a
  dim=-1 sharded [1,1,32,3072] all-gather through `TT_CCL` semaphores (Ring) reproduces the input (bf16 rounding 0.012).
  9B B=4 paged-decode check re-run after the stage-3 shared GDN kernel edits: PASS (stage-3 review follow-up).
- (host 08:20) shared-code import (reviewed Ornith hunks, artifacts/stage4/ornith_hunks.md): tp_common in0_block_w divisor fix;
  `prefill_norm_agmm_enabled(args)` + `mlp_gateup_agmm_enabled(num_devices, args)` gate (all six call sites: tp_common,
  attention/tp.py, gdn/tp.py, layer.py x2, mlp.py x2); conv1d channel blocking (`CONV1D_MAX_CHANNELS` = 1024 default,
  `args.gdn_conv1d_max_channels` knob; equal blocks with an assert); `TPGatedDeltaNet.reset_state(batch_size=None)` (f-string fix),
  `recurrent_state`/`fused_conv_state` aliases, `TPAttention.reset_cache`; env knobs replaced by args: `gdn_state_dtype`
  (default fp32), `gdn_decode_high_precision` (True), `kv_dtype == bfloat8_b` drives the bf8 SDPA/KV fill cast (was QWEN_SDPA_BF8),
  `sdpa_decode_q_chunk/k_chunk` (default 0/0 as before). Regression: qwen36 TP unit tests (9B, (1,4)) running —
  the 9B per-device conv width 4096 exercises the blocked conv path (4 blocks).
- (host 08:27) TP regression with the import (9B, (1,4), logs/04-gdn-tp.log, 04-attn-tp.log): first runs showed (a) the standalone
  module tests feed a K-fractured input and expect the module to gather — with the AGMM gate off (dim <= 4096) the modules now
  gather explicitly (`ttnn.all_gather`, dim=-1) before the plain column-parallel matmul; `tt_transformers.ccl.tt_all_gather`
  was rejected for that because it frees its input, which the caller owns; (b) `_z_mc` in the GDN in-projection must depend on
  prefill (S > 32), not on the fusion flag. Result: GDN TP 13/13, attention TP 7/7. On the committed HEAD the 9B fused-chunk
  prefill test fails with the conv1d CB overflow (1,684,544 B > L1) that the imported channel blocking fixes.

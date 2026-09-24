# Functional decoder — work log (stage 1)

Host: qb2-120-p11t05 (2x p300c = 4 Blackhole dies, 249 GB RAM). Stage 1 runs on a 1x1 mesh (device 0).
Checkout: branch `qwen3_5_122b_a10b` from main `05875866f20`. Env: `source ~/qwen35-122b-bringup/env.sh`.
HF_MODEL=Qwen/Qwen3.5-122B-A10B (snapshot dc4d348443bc740c68e2d77492492c11606384d5, local cache).

## 2026-09-24
NOTE on times: the HH:MM prefixes below are the operator's running estimates and drift up to ~75 min AHEAD of the host clock;
the authoritative timestamps are the ones inside the referenced log files (host UTC).
- 00:28 tt-smi: 4 dies visible, idle. (1,4) mesh open/close smoke with FABRIC_1D: MESH_SMOKE_OK.
- 00:50 HF book reference (readiness_check.generate, 128+100 tokens, top-k 100) -> bringup/references/qwen35_122b_book_p128_g100.refpt
  (HF top-1 agrees with the true book continuation on 91/100 positions: the mmap-backed bf16 CPU load is sound).
- 00:54 AIME24 chat-template reference -> bringup/references/qwen35_122b_aime24_chat_g100.refpt. Workaround: transformers 5.x
  apply_chat_template(tokenize=True) returns a BatchEncoding; readiness_check.generate iterates it as ids. Patched in
  ~/qwen35-122b-bringup/scripts/hf_reference_aime.py (plugin kept read-only).
- Shared code (qwen36) changes for this autoport: `Qwen36DecoderLayer._build_feed_forward` hook + `feed_forward.forward(x, mode=mode)`;
  `Qwen36MLP.forward(x, mode=None)`; `Qwen36Model.DECODER_LAYER_CLS` and `_layer_state_dict` hooks; `Qwen36GatedDeltaNet.set_external_state`
  accepts a batched `[B, Nv, Dk, Dv]` recurrent buffer. Review of the Ornith-branch shared diff: ~/qwen35-122b-bringup/artifacts/ornith_import_plan.md.
- 01:02 smoke (tools/smoke_functional_decoder.py), REAL weights, T=37, batch 1, bfp8 routed experts, HiFi2:
  L0 (linear_attention+MoE) prefill PCC 0.99825, decode 0.99984; L3 (full_attention+MoE) prefill 0.99955, decode 0.99930.
- 01:12 pytest synthetic seq 5/128/129 (both kinds) + contract + fallback audit: 8 passed (122 s).
- 01:15 pytest synthetic seq333 / batched / traced / determinism / chunked: 7 failed, 7 passed. Root causes and fixes:
  * GDN batched prefill (B=2, B=32) TT_FATAL concat: per-user prefill ran against the batched external state buffer.
    Fix (autoport): `FunctionalDecoder._gdn_prefill` prefills one user from a fresh B=1 state and copies the result into
    row `user_id` of the persistent `[B, ...]` buffers (in place), then re-points the layer at them. Also fixes the traced
    GDN decode (replay had read a stale zero external state: PCC 0.970 -> 0.9988) and determinism (reset now re-points).
  * Full-attention batched decode wrong at B>1 (shared kernel `ttnn_gated_attention.py`): sdpa_decode Q was built with a
    transpose that only works at B=1, and the paged_update_cache shard grid used B cores in ONE row (>11 cores invalid on
    Blackhole). Fix: `permute(2,0,1,3)` / `permute(1,2,0,3)` and `num_cores_to_corerangeset(B, grid, row_wise=True)`.
  * GDN chunk prefill at T=2048 OOM in `ttnn_delta_rule_seq.py` (fp32 output shuffle 64 MiB staged in L1). Fix: L1 only
    when B*T*H*V*4 <= 32 MiB, else DRAM (validated 32-head paths unchanged).
  * GDN decode at B=32 OOM in `ttnn_delta_rule_ops.py` (state 134 MiB fp32 kept in L1). Fix: state and its update stay in
    DRAM when B*H*K*V*4 > 48 MiB (B=1 and the TP per-device state stay in L1).
  * GDN chunk continuation (start_pos>0): the kernel returns a FUSED conv window in chunk mode; the continuation reads the
    external fused conv + recurrent buffers (assert corrected).
- 01:22-01:33 reruns: batched gdn/attn B=2/32, traced gdn, determinism gdn, chunked gdn (T=4429 PCC 0.9976 / decode 0.9974) all PASS.
- 01:41 full synthetic suite (not perf/long/real): 20 passed in 332 s (logs/01-synthetic-full.log).
- 01:44 real-weight tests (-m real_weights): 4 passed. PCC prefill/decode: GDN T=37 0.99824/0.99947, T=300 0.99882/0.99961;
  attention T=37 0.99957/0.99936, T=300 0.99968/0.99988 (logs/01-real-weights.log).
- 01:52 long-context (-m long): GDN layer prefill 262107 tokens in 2048-token chunks = 76.8 s (3415 tok/s single layer), decode at
  pos 262107 = 66.1 ms eager, outputs finite: PASS. Attention layer hit the repo pytest-timeout (300 s) -> @pytest.mark.timeout(5400), rerun.
- 01:58 attention layer long-context: prefill 262107 tokens in 129.3 s (2028 tok/s), eager decode at 262107 = 1050.6 ms (SDPA-decode over
  262K keys, single layer, eager; optimize stages own this), outputs finite: PASS (logs/01-long-attn.log). context_contract.json updated.
- 02:02 watcher run (scripts/01-watcher.sh: TT_METAL_WATCHER=10, APPEND=1, NOINLINE=1, real-weight T=37 both kinds): pytest exit 0, PCC unchanged
  (GDN 0.99824/0.99947, attn 0.99957/0.99936); watcher.log 14488 lines, 0 matches for exception|fatal|sanitiz|overflow; only attach/dump/stack-usage/detach
  lines. Copies: doc/functional_decoder/watcher/{pytest.log,watcher.tail.log}; full log artifacts/01-watcher/generated/watcher/watcher.log.
- 02:10-02:35 perf (scripts/01-perf.sh, Tracy `python -m tracy -r -p -v -m pytest <node id>`; tracy re-splits `-k` strings, so node ids):
  prefill 128 and traced decode captured for both kinds -> doc/functional_decoder/tracy/<kind>/{prefill_128,decode}_{ops.csv,perf_report.csv,perf_report.txt}.
  Device time (Device Time column, us): GDN prefill128 26944 (305 ops), attn prefill128 28644 (132 ops), GDN decode 1257/replay (96 ops),
  attn decode 1368/replay (81 ops); host wall traced decode 1.619 / 1.520 ms/step. SparseMatmul dominates prefill (17.7 / 21.5 ms);
  dense Matmul dominates decode (0.57 / 0.64 ms).
  prefill 2048: Tracy post-processing fails ("Device data missing: Op 1026048 not present in cpp_device_perf_report.csv"), twice, also with
  TT_METAL_PROFILER_MID_RUN_DUMP=1 + ReadDeviceProfiler flush. Fallback TT_METAL_DEVICE_PROFILER=1 + process_ops_logs.py produced an ops
  CSV without op names/DEVICE ID (tt-perf-report rejects it); measured-pass sums: GDN ~228 ms, attn ~251 ms per 2048 tokens (single layer).
  Recorded as a hard-check gap in README.
- 02:55 stage-review (independent subagent): more-work-needed. Findings and actions:
  P1 traced-decode perf window mixed truncated profiler replay sessions (device profiler buffers exhausted after ~5 replays x ~100 ops)
     -> perf tests now capture 3 replays; prefill-2048 profiled as 4 x 512-token start_pos chunks with a ReadDeviceProfiler flush
     between chunks; README numbers re-derived from complete windows (below).
  P1 host work inside prefill (rope get_prefill_rot_mats from_torch per call; GDN reset_state -> host zero init; source-only audit)
     -> cos/sin sliced from the resident device RoPE tables; fresh GDN state is zeroed with device ops (zeros_like+copy / ttnn.zeros)
     and the layer is re-pointed without reset_state(); new `test_runtime_fallback_audit_device_passes` wraps ttnn.from_torch/to_torch
     during an aligned prefill, a chunk continuation and a decode pass (0 transfers, both kinds) and reports the accepted boundary:
     non-aligned GDN prefill builds the valid_len one-hot selector on host (2 from_torch: ttnn_gated_deltanet.py:207,
     ttnn_delta_rule_seq.py:455; attention: 0).
  P2 stage-2 edits in the tree -> stage-1 commit built from the stage-1 file set only (fused-decoder files and the fused-QKV
     attention hunks are left for the stage-2 commit); shared kernel state of the stage-1 logs == committed content.
  P2 no 9B/27B regression evidence -> Qwen3.5-9B downloaded; qwen36 unit/prefill/decode tests run on this branch (01-9b-regression);
     decode-state L1 budget raised to 96 MiB so every existing config (9B: 2 MiB/user, B<=48) keeps its previous L1 path; seq-adapter
     budget 32 MiB == the 9B 2048-chunk footprint (unchanged for chunks <= 2048); attention permute/shard-grid changes are identical at B=1.
  P2 HF parity only to 4429 -> `test_long_hf_parity`: GDN 65573 tokens (chunked, state carried), attention 16421 tokens (CPU eager limit).
  P2 batched decode with one shared position -> `test_synthetic_batched_decode_distinct_positions` (lengths 5/37/129/300, per-user HF refs).
  Other: K/V fill block count from the logical T (+ page-table width assert); start_pos contract asserted (multiple of 128 and of the
     KV block size); page-table slices deallocated; README PCC range / LayerNorm figure corrected; smoke tool import shadowing fixed.
- 03:30 synthetic suite with fixes + new tests: 24 passed (logs/01-synthetic-full2.log).
- 03:45 INFRASTRUCTURE RECOVERY (not a model result): the evidence chain was started twice by mistake (first under the tool's
  10-minute limit, then detached), so two qwen36 9B pytest processes held the same dies concurrently and wedged (SIGTERM/SIGINT
  ignored, tt-smi -ls hung). Recovery per tt-device-usage: SIGKILL both pids, `tt-smi -ls --local` (8 entries), `tt-smi -r`
  (all 4 dies), `tt-smi -ls --local` (8 entries), (1,4) mesh open/close smoke. Chain relaunched once, detached.
- 02:20-03:22 qwen36 Qwen3.5-9B regression on this branch (scripts/01-9b-regression.sh, HF_MODEL=Qwen/Qwen3.5-9B, MESH_DEVICE=P150,
  artifacts/01-9b-regression/*.log): unit/test_gdn.py PASS (1), unit/test_attention.py PASS (1), unit/test_layer.py PASS (3) — these
  exercise the changed GDN decode/prefill kernels and the gated-attention paged-decode path with real 9B weights at the repo's PCC
  thresholds. unit/test_model.py (full 32-layer model) hit the 60-minute `timeout`: py-spy showed the process for >37 min inside
  `Embedding.__init__ -> ttnn.as_tensor` (model.py:70) at 100% CPU with no I/O and no cache writes (first-time host weight-cache build
  of the 2 GB embedding; not a path this autoport touches). tests/test_prefill.py loads the same full model and was stopped to let the
  stage-1 chain continue. Full-model 9B evidence therefore remains OPEN; the shared-code thresholds are set so existing 9B/27B
  configurations keep their previous placement (decode-state L1 budget 96 MiB > 9B's 64 MiB at B=32; seq-adapter L1 budget 32 MiB ==
  9B T=2048; `_seq_memory_config` width rule only affects widths > 4096; attention permute/shard grid identical at B=1).
- 03:22 real-weight rerun after the review fixes: 4 passed, PCC identical to the first run (logs/01-real-weights2.log).
- 03:23 long HF parity (test_long_hf_parity, logs/01-long-parity.log): GDN 65573 tokens (32 chunks of 2048, state carried):
  PCC last-2048 0.99732 / all 0.99741 / decode 0.99790 (HF 26 s, TT 19.4 s); attention 16421 tokens: 0.99931 / 0.99894 / 0.99943
  (HF 20 s, TT 9.7 s). No drift beyond the 3-chunk value. Added a 262107-token GDN parity case (queued after the perf run).
- 03:25-03:28 perf re-collection (scripts/01-perf.sh; decode 3 replays; prefill 2048 as 4x512 chunks with ReadDeviceProfiler flush):
  all six Tracy windows complete (decode replay sessions 115/115/115 GDN, 84/84/84 attention). Device time: GDN prefill128 27.59 ms,
  attn 28.72 ms; GDN prefill2048 411.2 ms (1940 ops), attn 481.1 ms (1248 ops); decode per replay GDN 1.608 ms (115 ops),
  attn 1.439 ms (84 ops); host wall 1.689 / 1.531 ms/step. The earlier device-only "228/251 ms" 2048 estimates were wrong (halves
  not op-aligned) and are withdrawn. README perf table replaced.
- 03:35 GDN HF parity at the full context, 262107 tokens in 128 chunks (state carried): PCC last-2048 0.99744 / all 0.99737 / decode 0.99745
  (HF 107 s, TT 74.1 s; logs/01-long-parity-262k.log). Contract + README updated.
- 03:40 stage-1 checkpoint commit "Qwen3.5-122B-A10B autoport: stage 1 functional decoder" (SHA recorded in ~/qwen35-122b-bringup/STATUS.md and the stage-2 work log; stage-1 files only; fused-decoder files and the fused-QKV attention hunks stay in the working tree for stage 2).
- 03:50 DEFECT found by the stage-2 fused suite: the pre-commit `autoflake` hook removed `from transformers.cache_utils import DynamicCache`
  from tests/reference.py while creating the stage-1 commit (the name was only used as `ref.DynamicCache` from other modules), so every
  full-attention test in the COMMITTED tree raises AttributeError (all stage-1 attention runs pre-date the commit and ran the correct
  helper). Fix: explicit `__all__` re-export + `new_kv_cache()` helper; attention tests (functional + fused) rerun; stage-1 commit amended.
- (host 03:45) DynamicCache re-export verified by rerun: logs/02-fused-attn.log — all 13 functional attention cases PASS with the
  restored helper (the one failure there is a stage-2 fused case, fixed separately).
- (host 03:50) RE-REVIEW P1, a real bug: `prefill_forward` sliced the chunk page table from block 0 for start_pos > 0
  (`ttnn.slice(pt_row, [0, 0], [1, n_blocks])`). `paged_fill_cache` writes input tile i to chunk_page_table[i // block_size], so
  every chunk after the first overwrote the blocks of positions [0, T_pad) and left its own positions zero. Whole-layer PCC with
  synthetic weights could not see it (the attention branch is ~V_std/sqrt(T) against the residual + MoE). Fix:
  `chunk_pt = ttnn.slice(pt_row, [0, start_pos // bs], [1, n_blocks])`. New discriminating tests: `test_chunked_prefill_kv_cache_matches_hf`
  (2048+2048+333 through a random-permutation page table; K and V read back through the page table and compared to HF's DynamicCache
  per 512-position window, PCC >= 0.99) and `test_real_weight_chunked_attention_prefill` (real weights, 2048+333, layer PCC + cache
  content + decode at 2381). Also: attention prefill asserts `valid_len is None`; audit predicate precedence fixed.
  Verification chain (logs/01-evidence-chain2.log): attention chunked/16421-parity/audit/seq333 + the new tests (01-fix2-attn.log),
  capacity attn (01-fix2-capacity-attn.log), fused attention regression (02-fix2-fused-attn.log), prefill-2048 perf both kinds
  (01-perf3.log), watcher incl. the chunked attention case (01-watcher2.log).
- Commit contents: `ttnn_gated_deltanet._seq_memory_config` width rule (in effect for the 03:22-03:28 evidence runs) is now part of the
  stage-1 commit; compact evidence files matched by .gitignore (perf_report csv, stacked csv, console logs, watcher logs) are force-added;
  the raw ops CSVs (0.4-6.4 MB each) stay in the workspace archive ~/qwen35-122b-bringup and doc/functional_decoder/tracy (untracked).
- (host 03:53) fix verified (logs/01-fix2-attn.log, 6 passed): K/V cache windows all 0.99986 (synthetic 4429 and real 2381); attention chunked
  4429 layer PCC 0.99997 (was 0.99888 with the bug), decode 0.99999; attention 16421 parity 0.99999/0.99998/0.99999 (was 0.99931/0.99894/0.99943);
  real-weight chunked 2381: 0.99979 / decode 0.99988; audit attn still 0 host transfers.
- (host 03:55) capacity attn rerun after the fix (logs/01-fix2-capacity-attn.log): 262107 tokens in 117.6 s (2229 tok/s), eager decode at
  262107 = 59.9 ms (the earlier 1050.6 ms included the first-call compile of the decode program), outputs finite: PASS.
- (host 03:58) prefill-2048 perf re-collected after the fix (logs/01-perf3.log): GDN 411.5 ms / 1940 ops, attention 483.7 ms / 1248 ops (within 0.5% of the pre-fix numbers; table updated).
- (host 04:00) watcher rerun on the final code (logs/01-watcher2.log): real-weight T=37 both kinds + chunked attention K/V check, pytest exit 0,
  watcher.log 10,894 lines, 0 exception/fatal/sanitize/overflow matches. Copies refreshed in doc/functional_decoder/watcher/.
- stage-1 commit amended with the P1 fix, new tests, width rule, evidence csv/logs (SHA recorded in STATUS.md / stage-2 work log).

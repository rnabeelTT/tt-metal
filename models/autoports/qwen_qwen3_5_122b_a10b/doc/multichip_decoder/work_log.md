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
- (host 08:30) shared-code import committed: 23d2bca06fa (qwen36 TP fixes; GDN TP 13/13, attention TP 7/7 on the 9B (1,4)).
- (host 08:50) tt/multichip_decoder.py (`MultichipConfig`, `MultichipDecoder`), tools/baseline_dump.py, tests/test_multichip_decoder.py
  written from PLAN.md. Baseline dump (single die, layers 0/3, synthetic + real weights, T in {5,37,129,333,2048} + chunked
  2048+2048+333, K/V per position, GDN state): 4 files under doc/multichip_decoder/baseline (gitignored, 670 MB, 50 s).
  First mesh run (logs/04-mc-first.log): contract test PASS; GDN layer prefill/decode vs HF at T=37 0.99953 (decode), T=129
  0.99944 / 0.99930; attention T=37 0.99988 / 0.99993 — the TP path is correct at the first attempt.
- (host 08:55) first mesh run complete (logs/04-mc-first.log): `test_mc_prefill_decode_against_hf` 10/10 for T in
  {5, 37, 129, 333, 2048} both kinds — GDN T=2048 0.99849 / decode 0.99834, attention T=2048 0.99998 / 0.99999 (synthetic weights
  vs the HF layer). The contract test's expectation of 8 value heads per device was wrong (64 / 4 = 16): fixed in the test and
  the module docstring. Full mesh suite (baseline comparison, chunked K/V per device pair, continuation, B=32, traced decode,
  fallback audit, stress) running (03/04-bringup2).
- (host 09:10) full mesh suite (logs/04-mc-suite.log): 28 passed, 4 failed — (1) contract test: wrong expectation (16 v-heads per
  device), fixed; (2) baseline comparison of the GDN conv window AFTER decode: the single-chip fused window is only written by
  prefill (decode advances the per-tap buffers), so the dump's after-decode window is stale by construction -> the test compares
  the recurrent state after decode and the window after prefill; (3) "fresh T=5 GDN prefill after a long sequence" 0.946: a TEST
  bug — the harness keeps an HF LinearCache across calls and the test reset the HF cache only for attention layers, so the HF
  reference continued the old sequence; fixed (both the suite test and a new reset-history probe); (4) fallback audit counted the
  two `from_torch` of `tp_common._mmrs_prefill_shared_bufs` (persistent CCL intermediate/output buffers created on the first call
  per prefill length) -> the audit warms every length first; documented as setup-time. All other 28 cases pass, incl. K/V per
  device pair through the page table, B=32 batched decode, traced decode (3 bit-identical replays), stress.
- (host 09:25) INFRASTRUCTURE: the first Tracy collection on the mesh (`04-perf.sh`, GDN prefill_128 with the in-session single-chip
  baseline on a (1,1) submesh) stalled after the kernel builds (no output for 20 min, host threads spinning at ~3 cores, no
  compilers running); the process ignored SIGTERM, and the queued probe chain started on top of it when the driver script died
  -> two device jobs (the pitfall recorded in memory). Both SIGKILLed, `tt-smi -r`, mesh smoke re-run. Next: mesh perf without
  the profiler first (host walls, baseline submesh skipped via QWEN35_MC_PERF_BASELINE=0) to isolate a profiler/CCL hang from a
  submesh problem; the queued chains now wait on the pytest PID, not on the driver script.

### 2026-09-24 09:20–09:55 — in-session submesh baseline hangs the parent mesh (profiler exonerated)

- The plain (no profiler) perf run with `QWEN35_MC_PERF_BASELINE=0` passed the GDN `prefill_128` (15.010 ms) and
  `prefill_2048` (245.608 ms, 4×512 chunks) mesh cases, then stalled for 31 minutes in `test_perf_traced_decode[gdn]`.
  That helper still built the `create_submeshes((1,1))` baseline unconditionally. `py-spy dump` of the pytest process:
  the main thread spins in `ttnn.to_torch` inside `_compose` of the first **mesh** prefill after the baseline ran on the
  submesh (host at 300 % CPU, no device output, SIGTERM ignored; same signature as the Tracy stall earlier today).
- Conclusion: after any work on an in-session `(1,1)` submesh of the fabric-enabled `(1,4)` mesh, the parent mesh hangs at
  its first readback. The profiler is not the cause. The mesh-only traced decode (`test_mc_traced_decode`) passes.
- Fix (test side): the single-chip baseline now runs in a separate pytest process on the stage-3 `device` fixture
  (`-m perf_single`, prints `PERF_SINGLE <kind> <case>: X ms`); the mesh cases (`-m perf`) print mesh walls only;
  `tools/perf_join.py` joins both logs into `doc/multichip_decoder/HOST_WALLS.md`. `QWEN35_MC_PERF_BASELINE=1` keeps
  the legacy in-session path for experiments only (documented as hanging on p300x2).
- Recovery: SIGKILL of the pytest pair, `tt-smi -r`, `tt-smi -ls` healthy. The queued probe run started during the
  reset (exit 135, SIGBUS) — queue chains must not proceed after a killed step; the new chain `04-perf-split.sh` runs
  smoke → single baseline → mesh perf → probes in one script.

### 2026-09-24 09:55–10:20 — split perf chain, probes, Tracy on the mesh

- `scripts/04-perf-split.sh`: mesh smoke PASS; single-die baseline (`-m perf_single`, 4 tests, 38 s) GDN prefill_128 18.511 /
  prefill_2048 250.441 / decode 0.867 ms, attention 16.097 / 229.755 / 0.727 ms; mesh (`-m perf`, 6 tests, 71 s) GDN
  14.957 / 240.663 / 0.611 ms, attention 15.103 / 240.908 / 0.482 ms -> decode 1.42x / 1.51x, prefill 0.95-1.24x
  (`HOST_WALLS.md`). Probe run 9/9 PASS (`test_mc_gdn_reset_after_history` x3, contract, audit x2, GDN baseline incl. the
  chunked 4429-token case at PCC 0.99904, continuation x2); details in README section 3/6.
- Tracy on the mesh with the in-session baseline removed (`scripts/04-next-chain.sh` -> `04-perf.sh`): all six cases
  collected, no hang. `tt-perf-report` must run with `--no-merge-devices`: the merged report interleaves the four devices'
  rows (dedups by op id) and reports a 5.4 ms "gap" that is the inter-device offset; the per-device tables are in
  `tracy/<kind>/*_mesh_perf_report.{csv,txt}` and `PERF_TABLES.md` (`tools/perf_tables_mc.py`). Device time per device:
  GDN 8.21 / 50.01 / 0.502 ms, attention 9.03 / 47.20 / 0.446 ms -> 1.97x / 2.48x / 1.65x and 1.68x / 1.88x / 1.47x vs the
  committed stage-3 single-chip tables. Findings in README section 4 (CCL 46-50 us per step; routed gate/up at 114 GB/s on
  16 cores; shared expert at 45 GB/s on 17 cores; prefill wall host-bound: 241 ms wall vs 50 ms device at 2048 tokens).
- Context contract block `multichip_decoder_stage4` written (6528 B/token/device, KV pool 11.39 GiB, 1.87 M tokens,
  7 users at 262144 or 32 x 58 K); README sections 2-7 drafted (watcher line pending).
- Running: `04-sweep.sh` families baseline, norm_mode, dense_1d, gdn_1d, gate_up, down, shared, gdn_state, conv1d, sdpa;
  then the watcher run; then the final full mesh suite (`04-final-suite.sh`).

### 2026-09-24 10:05–10:20 — mesh sweeps (`scripts/04-sweep.sh`, 10 families, 46 candidates, 11 min)

- Every legal candidate has the same PCC vs HF as the baseline (the knobs change program configs, not math).
- `shared_decode_cores=16`: -26 us per decode step on both layer kinds (0.608 -> 0.582 GDN, 0.482 -> 0.456 attention) ->
  adopted into `MULTICHIP_SELECTED`. `dense_1d` 88/33 and 44/44: -9 us (inside 2x noise, left for stage 5). Everything
  else within noise or worse; `dn96` illegal (96-core grid hits the dispatch cores), `norm_B` fails (MoE input fractured),
  `state_fp32` +15 us, `sdpa` 32/256 confirmed at 16 K context (-7 us vs auto). `sweeps/SUMMARY.md`
  (`tools/summarize_sweeps_mc.py`).
- Because the selection changed after the Tracy run, `scripts/04-reperf.sh` (queued after the final suite) re-measures
  the host walls and the Tracy tables with the final config; the final suite itself runs with the final config.

### 2026-09-24 10:16–10:31 — watcher run tripped a kernel assert (open), device left stopped

- `TT_METAL_WATCHER=10` run (`04-next-chain.sh`, 4 tests) aborted 53 s in, during the first test
  `test_mc_real_weight_against_hf[gdn_37]` right after the GDN fused-chunk prefill started: "Watcher detected tripped
  assert and stopped device" (pytest exit 134). The kernel/line were NOT recorded: the watcher names them with
  `log_warning` and the chain ran with `TT_METAL_LOGGER_LEVEL=Error`; the watcher.log dump #6 was cut before the
  tripped core (only device-0 cores were written; the last completed program there was a multi-core matmul).
  The same test is watcher-clean on one die (stage 3), so the trip is mesh-specific (TP GDN prefill path or a CCL).
- Consequence: the queued final suite could not open the mesh (34 errors: "Timed out while waiting for active ethernet
  core 29-25 to become active again" at llrt.cpp:618) — environmental, the device stayed stopped after the watcher abort.
  Killed the queued chains (kill by PID: `pkill -f` on a script name also matches the invoking shell), `tt-smi -r`,
  mesh smoke PASS.
- `scripts/04-tail-chain.sh` (one script, no PID waits): watcher probe of the same test with `TT_METAL_LOGGER_LEVEL=Warning`
  and `TT_METAL_WATCHER=5` -> `tt-smi -r` + smoke -> host walls + Tracy with the final config -> final full suite.

### 2026-09-24 10:31–10:45 — watcher trip root cause: reduce-scatter writer scatter-header setup with one tile per packet

- Probe with `TT_METAL_LOGGER_LEVEL=Warning` (`artifacts/04-watcher-probe/pytest.log`): "Device 0 worker core(x=1,y=8):
  BRISC tripped an assert on line 279 ... Current kernel: .../reduce_scatter_minimal_async/device/kernels/
  ring_reduce_scatter_minimal_async_writer.cpp". Line numbers are reported without the file; the only header in the
  writer's include chain with an `ASSERT` on line 279 is `tt_metal/fabric/hw/inc/api_common.h`:
  `populate_unicast_scatter_write_fields` asserts `NOC_SCATTER_WRITE_MIN_CHUNKS (2) <= chunk_count <= 4`.
- Mechanism (read from `reduce_scatter_minimal_async_program.cpp:509-514` and the writer's `IntermSink<false>::
  set_packet_header_states`): `num_tiles_to_write_per_packet = min(4, fabric_max_payload / page_size)`. The writer
  unconditionally calls `fabric_unicast_noc_scatter_write_set_state(..., chunk_count = num_tiles_to_write_per_packet)`
  (and the fused scatter+atomic-inc variant). With 4 KiB fp32 tile pages that value is 1 -> the assert trips. Every
  actual packet with one tile goes through the unicast path in `send_write_packet`, so the scatter states are never
  used and the data is correct (PCC unchanged) — the assert is real but the failure mode is only visible under the
  watcher. The single-die stage-3 runs never issue a reduce-scatter, which is why they were watcher-clean.
- Which call: the GDN prefill out-projection partial is fp32 (`gdn/tp.py` `_row_proj` non-fused arm: "fp32 [seq,dim]
  output"; `tt_all_reduce` on a (1,4) mesh is `reduce_scatter_minimal_async` and does not cast on that branch).
  Confirmation of the input dtype by `scripts/04-rs-dtype-probe.py` (queued in `04-watcher-final.sh`).
- Fix (shared ttnn kernel, `ring_reduce_scatter_minimal_async_writer.cpp`): wrap both scatter `set_state` calls in
  `if constexpr (num_tiles_to_write_per_packet > 1)`. The Linear-topology sibling
  (`line_reduce_scatter_minimal_async_writer.cpp:259`) has the same pattern; not exercised here (Ring), left for an
  upstream issue. Verification queued: watcher probe of `gdn_37`, then the 4-test watcher set, then reset + smoke.

### 2026-09-24 10:32–10:37 — final config re-measured (`04-tail-chain.sh`, shared_decode_cores=16)

- Host walls (`logs/04-perf-plain-final.log`, `HOST_WALLS.md`): GDN 15.867 / 238.718 / 0.584 ms, attention
  15.356 / 238.668 / 0.458 ms -> decode 1.48x / 1.59x vs one die (was 1.42x / 1.51x).
- Tracy (`logs/04-perf-tracy-final.log`, `tracy/`, `PERF_TABLES.md`): device time per device GDN 8.21 / 49.96 / 0.474 ms,
  attention 9.04 / 47.20 / 0.419 ms -> decode 1.75x / 1.57x (44 % / 39 % efficiency); shared-expert matmul 39 -> 15 us.
  Note: `04-perf.sh` had not been patched for `--no-merge-devices` when the chain ran (the patch sat in a blocked shell
  command); the per-device reports were regenerated from the collected ops CSVs afterwards and the script is fixed.
- Running: final full mesh suite (`logs/04-mc-final-suite.log`), then `04-watcher-final.sh` (RS dtype probe, watcher
  reruns with the fixed writer kernel, reset + smoke).

### 2026-09-24 10:49–10:55 — reduce-scatter inputs, watcher probe with the fixed kernel

- `scripts/04-rs-dtype-probe.py` (synthetic weights, T=37 then decode, both kinds): `reduce_scatter_minimal_async` inputs
  are bf16 `[1,1,128,3072]` (MoE exit, both kinds), bfp8 `[1,1,128,3072]` (attention `wo` partial at prefill) and bf16
  `[1,1,1,3072]` x2 per decode step. The GDN prefill out-projection does NOT appear there: it uses the fused
  `matmul_reduce_scatter_async` (`_fuse_out_mmrs_prefill`, fp32 output "load-bearing for PCC"), whose reduce-scatter
  program is built by the same `reduce_scatter_minimal_async_program.cpp` with the same ring writer kernel. fp32 tiles
  are 4096 B against the 4352 B fabric payload -> one tile per packet -> the 1-chunk scatter `set_state` -> the assert.
  bf16 (2 tiles/packet) and bfp8 (4 tiles/packet) inputs never hit it.
- Note for stage 5 (CCL dtype item): the attention `wo` partial is reduce-scattered as bfp8 (4 quantised partials summed);
  attention PCC vs HF is 0.99998, so it is accepted for now and listed in README section 7.
- Watcher probe with the guarded writer (`artifacts/04-watcher-probe2`): `test_mc_real_weight_against_hf[gdn_37]` PASSED,
  watcher.log 5073 lines, 0 bad, no tripped assert. The process still exits 134, now at TEARDOWN:
  `MetalContext::teardown -> assert_active_ethernet_cores_to_reset`: "Device 0: Timed out while waiting for active
  ethernet core 29-25 to become active again" (llrt.cpp:618). Every non-watcher mesh run tears down cleanly, so this is
  watcher-specific (fabric routers on the eth cores under watcher instrumentation); `tt-smi -r` recovers the board.
  Queued (`04-watcher-set.sh`): the full 4-test watcher set, a reset, a `TT_METAL_WATCHER_DISABLE_ETH=1` diagnostic
  probe (does the teardown timeout go away when the eth cores are not watched?), reset + smoke.

### 2026-09-24 10:52–10:57 — watcher teardown timeout attributed to the eth-core watcher instrumentation

- Full 4-test watcher set in one process (`artifacts/04-watcher`): first test PASSED, watcher.log 2503 lines / 0 bad, then
  the next 5 parametrizations ERRORED at mesh open ("Timed out while waiting for active ethernet core 29-25") — a
  watcher-enabled process can open/close the mesh only once.
- Diagnostic with `TT_METAL_WATCHER_DISABLE_ETH=1` (`artifacts/04-watcher-noeth`, gdn_37): PASSED, exit 0, watcher.log
  11444 lines / 0 bad, clean teardown. So the teardown timeout comes from the watcher instrumentation of the fabric
  router kernels on the ethernet cores (eth core 29-25 does not return to base firmware), not from the model.
  `04-watcher-each.sh`: one process per test (6 tests) with all watcher features on and `tt-smi -r` between runs.

### 2026-09-24 10:58–11:02 — per-test watcher runs (`04-watcher-each.sh`), second (eth-core) watcher report

- Run 1 `gdn_37`: PASSED, watcher.log 2503 lines / 0 bad, exit 134 = the teardown eth timeout.
- Run 2 `attn_37`: test PASSED (PCC 0.99982 / 0.99988; vs single-chip 0.99962 / 0.99978), then at the next kernel launch
  on the ACTIVE ETHERNET core (x=0,y=9) virtual (29,25) of device 0: "subordinate_erisc detected invalid NOC command
  buffer state before starting the next kernel ... Current kernel: tt_metal/fabric/impl/kernels/edm_fabric/
  fabric_erisc_router.cpp". Same eth core as the teardown timeout; the kernel is the fabric router, not model code.
  Control queued (`04-watcher-control.sh`): the plain 20x all-gather smoke (no model code) under the same watcher settings.

### 2026-09-24 10:58–11:09 — watcher evidence complete

- `04-watcher-each.sh`: 6/6 tests PASSED under the watcher (all features), watcher.log 0 bad lines in every run; runs 2
  and 5 (attention) reported the fabric-router NOC-state message on eth core 29-25 AFTER the test passed; the others
  ended with the teardown timeout on the same core. Control (`04-watcher-control.sh`, plain all-gathers, no model
  code): all-gather ok, same teardown timeout, clean watcher.log -> the eth-core reports are independent of the model
  (README section 6). Devices reset and smoke-tested after every run.
- Stage-4 evidence is complete: correctness (final suite 35/35), perf (host walls, Tracy per device, sweeps), context
  contract, watcher, audit, stress. Next: stage-review subagent.

### 2026-09-24 11:12–11:30 — stage review #1: more-work-needed; fixes

Reviewer (independent subagent, read-only) re-derived the suite counts, PCCs, host walls, Tracy sums and the contract
arithmetic and confirmed them; the kernel guard is "correct by inspection". Required work and what was done:

- P2 watcher: the `fabric_erisc_router.cpp` "invalid NOC command buffer state" message appeared only in the two ATTENTION
  runs (after PASSED) and the plain all-gather control reproduced only the teardown timeout -> a control that exercises the
  attention CCL pattern is required. Added `scripts/04-attn-ccl-control.py` (bfp8 `[1,1,128,3072]`, bf16 `[1,1,128,3072]`
  and bf16 `[1,1,1,3072]` `reduce_scatter_minimal_async` via `tt_all_reduce`/`TT_CCL` + all-gathers, 20 rounds) run under the
  same watcher settings (`04-review-fix-chain.sh`); result recorded below.
- P2 contract: `embedding_bf16_lm_head_bfp8_tp4` held the bfp8+bfp8 figure (0.38). Fixed to bf16 embedding 0.355 +
  bfp8 LM head 0.189 = 0.544 GiB; weights 17.08 GiB, KV pool 11.23 GiB, 1.846 M tokens, 7 users at 262144 (1.835 M,
  thin headroom), 32 x 57.7 K. README section 2 mirrored.
- P3 record fixes: traced decode row now says "replays 1-2 vs replay 0 bit-identical" (the eager output is not kept);
  GDN recurrent-state PCC vs single-chip reported as the range 0.99930 (T=5) .. 0.99648 (T>=2048, chunked 4429) against
  the 0.995 bar; conv1d rationale corrected (the overflow evidence is the 9B at 4096 channels per device; 3072 in one
  call works here, 1024 kept as the proven block) in README and the `MultichipConfig` comment; non-aligned GDN prefill
  host transfers = 3 (not 2) in section 7.
- Hard-check gap taken up now: cross-device routing-id equality probe (`scripts/04-routing-probe.py`: `_route` on a
  replicated input, ids/scores composed from all 4 devices and compared bit-exactly, B=1 gate-op row, B=32, T=128).
- Deferred to stage 5 with the reviewer's notes: `dense_1d` 88/33 (-9 us, consistent) to be re-measured with more replays;
  routed gate/up `output_tile [32,16]` probe for 32 cores; option B per-norm variant; shared `TT_CCL` across layers
  (stage 6); bfp8 `wo` partial re-check on real weights at long context.

### 2026-09-24 11:22–11:35 — review #1 device follow-ups

- Routing-id equality probe (`logs/04-routing-probe.log`): PASS — identical ids and scores on all 4 devices for B=1 (gate
  op), B=32 and T=128, 3 repeats each (README section 6).
- Attention-CCL-pattern control under the watcher (20 rounds, then 100 rounds): reduce-scatter sums correct (rel err
  <= 0.0057 for bfp8), watcher.log clean, teardown = eth timeout form; the "invalid NOC command buffer state" form did
  not reproduce. Repeats of `attn_37` (x2), `gdn_37`, traced attention decode under the watcher: all PASSED, all the
  timeout form. Tally 2 of 14 watcher processes show the NOC-state form (both attention, both after PASSED); README
  section 6/7 state the classification and the residual uncertainty.

### 2026-09-24 11:35–11:50 — stage review #2: more-work-needed (documentation only); fixes

- Reviewer confirmed the review-#1 fixes and the routing probe, and derived from the watcher.log dump timestamps that the
  two "invalid NOC command buffer state" processes are exactly the two whose 10 s dump tick fell 1-4 s into teardown
  (aborted "Dump #4 at 33.188s" / "Dump #5 at 43.244s"), while all 12 timeout-form processes (incl. every control and the
  warm-cache repeats) had no tick in that window — verified here over the 14 directories (`watcher/TALLY.md`).
- Fixed: the stale duplicate watcher block in README section 6 removed; classification rewritten with the timing evidence
  (both forms = fabric-router teardown under eth-core watcher instrumentation; not attention/model-specific); section 7
  residual bullet rewritten; control description says `ttnn.all_gather` (the norms use `all_gather_async`); routers
  0.15 -> 0.070 GiB in the contract (bf16 [3072,256] x 48) -> weights 17.00 GiB, KV pool 11.31 GiB, 1.860 M tokens,
  7 users / 32 x 58 K, README section 2 mirrored; PLAN.md notes that the contract JSON supersedes its draft figures;
  watcher evidence copied into `doc/multichip_decoder/watcher/*.txt` (the `.log` files are gitignored) with `TALLY.md`;
  `perf` / `perf_single` markers registered in `tests/conftest.py`.
- Review #3 (more-work-needed, one P3): the `capability_change` string still said "~56K" (1024-based) against
  `tokens_per_user_at_32_users` 58113 -> "~58.1K"; `kv_pool_gib` stated unrounded (11.306) so the token count derives
  from it; the form-(ii) window worded "0-4 s (at or after PASSED)"; the reviewer's firmware-level argument added to README
  section 6 and TALLY.md (`DebugAssertNCriscNOCPacketTagClearedTripped` runs after `kernel_main()` returns on the erisc,
  `active_erisck.cc:51`; the router only returns at mesh close). Duplicate `perf` marker line already removed.

### 2026-09-24 12:00 — stage review #3: clean-pass; committing stage 4

- Reviewer verified the contract block (11.306 GiB -> 1,859,639 tokens -> 7 users / 58113 per user at 32), README section 2
  = JSON, one watcher block with the firmware-level explanation, PLAN supersession note, markers, watcher evidence staged.
  Residual risks recorded: watcher evidence cannot show a clean teardown on this board (upstream report pending); computed
  DRAM budget with thin 7-user headroom; stage-5 inheritances in README section 7. Local commit follows (never pushed).

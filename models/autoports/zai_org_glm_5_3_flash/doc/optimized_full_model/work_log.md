# Stage 7 work log (optimized full model)

All device jobs: `~/glm53-flash-bringup/scripts/run-locked.sh` (shared host lock), launched as `systemd-run --user`
units. Logs in `~/glm53-flash-bringup/logs/`, probe records in `~/glm53-flash-bringup/artifacts/s7/`.

## 2026-10-17

- Read the stage-7 goal (`prompts/model_bringup_multigoal/07-optimized-full-model.txt`), skills full-model, optimize,
  tt-enable-tracing, tt-device-usage; stage-5 / stage-6 READMEs.
- New driver `tools/s7_probe.py` (one build per job): TTFT (dispatch vs total, cold / warm, traced vs eager, MoE split
  A/B), decode (model trace, sampler trace, token-out, generator loop), traced-vs-eager prefill check with repeats,
  Tracy windows, readiness runners and the qualitative / autoregressive suite on the same generator.
- `tt/model.py`: options `GLM53_HEAD` (norm_sharded, lm_ds), `GLM53_PREFILL_LENGTHS`, `GLM53_PREFILL_PLAN` (cost plan),
  `GLM53_PREFILL_TRACE` (traced first chunk), `GLM53_MOE_SPLIT_AUTO`, `GLM53_SKIP_INACTIVE`; `_row_to_host`.
  `tt/dsa.py`: `prefill_inputs(host=True)`, `defer_tail` + `write_prefill_tail` (trace-safe open-pool hand-off).
  `tt/generator.py`: `_warm_plans` (every (length, DSA path)), prefill-trace prepare / capture order, `choose_moe_split`
  before the warm prefills. `tools/full_model_checks.py`: `run_with` (shared generator, output dir).
  `tests/test_generator_vllm_host.py`: warm-plan coverage test (11 passed).
- 12:47 the coordinator's readiness run imported a work-in-progress `lm_ds` default and failed (L1 3.05 MB CBs for the
  single 38,912-column DRAM-sharded LM head). Fix: column splits; and every unvalidated option default-off from then on.
- 12:53 Tracy on the 3-layer model: report failed (device-profiler buffers overflowed). Added profiler flushes and
  `TT_TRACY_SKIP_MISSING_DEVICE_DATA=1`, moved to a 1-layer model (`*_l0`).
- 13:24 reduced (0/3/4) check, S=1: traced prefill bit-identical to eager (logits max |diff| 0, greedy tokens equal) for
  100 / 128 / 161 / 300 tokens. `tests/test_model_probe.py` 3 passed with the stage-7 options on.
- 14:08 reduced bucket costs (S=1, eager, ms): 128: 19, 256: 21, 512: 21, 1024: 36, 2048: 66.
- 14:42 full baseline S=1 (`s7_base_s1.log`): TTFT 128 232 ms, 2048 1182 ms (split) / 1060 (no split), 8192 5134 /
  4657; decode model 56.86, token-out 57.24, loop 57.36 ms.
- 14:04 Tracy l0 before/after: LM head 471 µs interleaved; lm_ds splits 10 x 66 µs -> rejected; final norm 65 -> 7 µs.
- 15:06 candidate run failed at warm-up: `NameError: fold` in `tt/fused_decoder.py` (another agent's edit in
  progress). Rerun with a reduced smoke gate first.
- 17:21 full baseline S=8 (`s7_base_s8.log`): 112.8 ms per step with 1, 4 and 8 active slots.
- 18:34 full candidate S=1 (`s7_cand_s1.log`): traced 128 222 vs eager 232 ms; 512 bucket 375 ms; 1000 tokens 752 ms;
  2048 1033 (no split) vs 1153 (split); 8192 4548 vs 5033; decode model 53.23 ms (stage-5 round-6 defaults), loop
  53.74 ms = 18.6 t/s/u. Traced prefill exact (100 / 500 tokens). Decision: 512 bucket on, prefill traces off, split
  auto on, norm_sharded on.
- 19:48 / 20:04 / 20:31 reduced S=8 repeatability: not reproducible with `ag_persistent=1` (stage-5 default), with or
  without `GLM53_SKIP_INACTIVE`; reproducible with `GLM53_MC=ag_persistent=0`. Skip timing (reduced, S=8): 1 active
  5.74 vs 7.13 ms, 4 active 6.31, 8 active 7.19.
- MTP: checkpoint layer 45 counted (7.25 B expert + 185 M other parameters): ~1.07 GB per die, rejected (README).
- 21:46 final S=1 with stage-5 `DEFAULT_MC` (ag_persistent on): teacher forcing top-1 14 % / top-5 32 % / top-100 90 %
  (prefill check 91 / 100 / 100). 22:03 same with `GLM53_MC=ag_persistent=0`: 92 / 100 / 100 (`s7_tfdiag_agp0.log`).
- 22:51 final S=1 (`s7_final_s1.log`, ag_persistent=0, weight cache): TF 92 / 100 / 100, prefill check 91 / 100 / 100,
  qualitative 6 prompts coherent, autoregressive repeat-identical, degenerate check exit 0; TTFT 128 / 161 / 512 /
  2048 / 8192 = 230 / 376 / 379 / 1035 / 4556 ms; decode 53.69 / 54.08 / 54.18 ms (18.46 t/s/u).
- 23:31 final S=8 (`s7_final_s8.log`, skip on): 83.9 / 95.7 / 114.2 ms at 1 / 4 / 8 active; repeats identical.
- Defaults: GLM53_HEAD=norm_sharded, GLM53_PREFILL_LENGTHS=128,512,2048, GLM53_PREFILL_PLAN=cost,
  GLM53_MOE_SPLIT_AUTO=1, GLM53_SKIP_INACTIVE=1 (>= 2 slots), GLM53_PREFILL_TRACE=none. Not committed (instructions).

## Commits

- ac03689d2bb (2026-10-18): stages 6 + 7, the 2026-10-17 build above.
- de9852c604d (2026-10-20): stage-6/7 review #1 rework (code, evidence, docs).
- b3270061cca (2026-10-20): HF chunk-plan control, `autoregressive/VERDICT.md`, rescored `hf_scores_tt_trajectory.json`.

## 2026-10-18 .. 20: stage-6/7 review #1 rework

The full rework log is in `doc/full_model/work_log.md` ("stage-6 review #1 / stage-7 review #1 rework"). It records
the root cause of the decode corruption (the logits-pad deallocate) and the defects fixed in order. Jobs, all run from
the pinned snapshot `~/glm53-flash-bringup/s67r_final` (HEAD 656c998e3df + rework files), with md5 stamps at start and
end. Logs: `~/glm53-flash-bringup/logs/s67r/`. One-line results: `logs/s67r/status`.

- **Gate:**
  - trace-allocation tracker at 8 slots, replicated / sharded / adapter smoke: 0 violations;
  - reduced probe S = 1 / 8 with `GLM53_DEALLOC_CHECK=1`: pass;
  - press B3 (65,536 tokens + 480 MB ballast): teacher forcing 91 / 100 / 100.
- **j2_s1_default** (default build, 280,576 tokens):
  - readiness 91 / 100 / 100 and 91 / 100 / 100; autoregressive check;
  - TTFT 128 .. 8,192 tokens; page-table change; alternating sampler; chunk plans 513 .. 10,001; decode;
  - deep 32,768 / 100,000 tokens: copy 1.0.
  - Record: `runs/s67r/j2_s1_default.json`.
- **j2b_s1_16k:** the same code at 16,384 tokens. Token-out 51.10 ms and 8K TTFT 4,408 ms, against 56.24 / 6,444 at
  the default: the context-size cost.
- **j3_s8 / j3b_s8_noskip / j4_s32:** batch correctness, bit-identical in every comparison. Skip on vs off: all logits
  bit-identical. Timings are in the README.
- **j6_s1_shard:** block-sharded, `GLM53_PRECISION=none`, 131,072 tokens: teacher forcing 92 / 100 / 100, deep copy
  1.0, sink headroom 65.7 nats.
- **Watcher set:** 5 runs, rc 0, 0 suspicious lines (ETH cores not watched).
- **HF last-row control:** 513 / 600 tokens (`runs/s67r/j2_hf_lastrow.json`).
- **HF rescoring of the AR trajectory:** `autoregressive/hf_scores_tt_trajectory.json`.

## 2026-10-20: stage-7 review #2

Jobs: `~/glm53-flash-bringup/scripts/s7r2_{jobs,agp1,verify}.sh`. Logs: `~/glm53-flash-bringup/logs/s7r2/` (`status` has
one line per job). Records: `runs/s7r2/` and `~/glm53-flash-bringup/artifacts/s7r2/`. Code is pinned to snapshots:
`s7r2_snap` = `git archive b3270061cca`; `s7r2_v` = HEAD 24be34c7cbe + this agent's hunks.

- **P2-1, ag_persistent** (s7r2_snap, `GLM53_PRECISION` unset):
  - det0 / det1 (reduced 0 / 3 / 4, 8 slots, `s7_probe --do ptcheck --ptcheck 100,300`): equal, and repeatable
    with both settings.
  - agp (`scripts/s7r2_agp.py`, full model, 1 slot, 16,384, three AIME24 references), `ag_persistent=0`:
    TF 91 / 96 / 91 (top-5 / top-100 100 %), token-out 50.40 ms.
  - The `=1` build in the same process ran out of DRAM (the first generator's memory was not released). agp1 ran it
    alone: identical TF / prefill check, token-out 49.96 ms.
  - The earlier failure was the logits-pad bug. Decision (stage-7 review #3): not adopted; available via MCOpts.
    The DRAM cost is +71,680 B per bank (det0 vs det1); the reason is the skipped all-gather start barrier with one
    buffer per site reused by 45 layers (a race one run per setting cannot exclude). The same-process OOM was the
    generator teardown leak (fixed in the review-#3 kit).
- **P2-3, sharded first-run gap:**
  - cache: reduced, 16,384 tokens, sharded + replicated.
  - cachefull: full model, sharded, 131,072 tokens, bf16.
  - Both: two fresh 10,001-token prompts and one 2,600-token prompt after the capture, with a traced decode replay
    after each (tracker on). No program compiled after the capture (889 / 890 entries), 0 tracker allocations, first
    run = second run. The j6 gap did not reproduce; not a compile.
  - Added the standalone guard `Glm53Generator.guard_programs` (prefill_forward and the s67r_checks records) as
    protection.
- **Stage-6 review #2 code items** (`tt/generator.py`, `tt/weight_cache.py`, `tools/s67r_checks.py`,
  `tests/test_model_probe.py`, two `tt/moe_opt.py` guard lines):
  - standalone reserve: an explicit `GLM53_DRAM_RESERVE_MIB` wins, else max(data reserve, 1,200 MiB). The
    coordinator's decision keeps the TTFT-favouring standalone default (294,912 on HEAD's profile data); v1 checked
    the serving-workspace sizing (311,296);
  - budget per call;
  - unverified manifest adoption refused when a shard is newer than the cache;
  - flat-distribution seed check;
  - the shared-free check covers every autoport `tt/` module. It found two benign `moe_opt.py` view frees, now
    guarded with `force=False`.
- **Verification** on s7r2_v:
  - v1, the default build at 311,296: TF 92 / 100 / 100, prefill check 92 / 100 / 100, token-out 55.37 ms, seeds
    honoured, 0 programs after capture.
  - v2, reduced 8-slot probe with `GLM53_DEALLOC_CHECK=1`: 7 passed. The first v2 failed on the snapshot's missing
    root `conftest.py`; the second found the two `moe_opt.py` sites.
- **Docs:**
  - README headline = j2 (final default) + j2b (16,384); the 2026-10-17 table is labelled superseded;
  - `perf_summary.json` rebuilt from j2;
  - Tracy and MTP labels; watcher scope; HF reference noise; qualitative scope.

## 2026-10-21: stage-6 / stage-7 review #3

- **ag_persistent:** the decision text now gives the real reason: the skipped all-gather start barrier, and one buffer
  per site reused by 45 layers, a race one run per setting cannot exclude. The DRAM cost is +71,680 B per bank. The
  retest's precision label is fixed (combo_a_kv + kda_in, b3270061cca), and the three `tt/multichip_decoder.py`
  comments are updated.
- **v3_default** (`runs/s7r2/v3_default.json`, pure `git archive 237ec3b8438`, `GLM53_PRECISION` unset):
  - 294,912 replicated; teacher forcing 92 / 95 / 92; token-out 55.04 ms; `program_cache_growth` [];
  - MoE split ENGAGED (largest free block 1,030,724,608 B < 1 GiB): TTFT 2,048 / 8,192 = 1,144 / 7,034 ms;
  - reported to the coordinator, no change made.
- **Teardown leak** (`runs/s7r2/leak.json`, reduced 0 / 3 / 4, kit snapshot):
  - before the fix, `teardown()` + `del` left ~4.24 GB per bank allocated;
  - now three builds in one process (replicated, sharded, replicated): allocation returns from 254 MB to 2.3 - 2.6 MB
    per bank after each teardown, and the model object is collected.

  The same run showed that an earlier build's automatic `GLM53_MOE_SORTED_ROWS=0` blocked the sharded layout's
  setting (512). Fixed afterwards: the auto-marked value is treated as unset (host logic, not re-run on device).

## 2026-10-21: stage-6 / stage-7 review #4

- **MoE split margin:** the standalone replicated reserve is now coupled to the no-split threshold:
  `GLM53_MOE_NOSPLIT_FREE_MIB` (1024) + `NOSPLIT_MARGIN_MIB` (320) = 1,344 MiB. Defaults are 272,384 / 28,672 /
  10,240 / 2,048 at 1 / 8 / 16 / 32 slots (host-tested).
- **`v4_default`** (`doc/optimized_full_model/runs/s7r2/v4_default.json`; pure `git archive 52aff54486a` + the floor):
  - plan 272,384, 1.13 GB free;
  - MoE split OFF (largest free block 1,181,530,112 B);
  - TTFT 2,048 / 8,192 = 1,023 / 5,772 ms; token-out 50.08 ms;
  - TF p0 92 / 100 / 100;
  - 0 post-capture programs.

  It is the headline now; j2 and v3 are history.
- **Code:**
  - `teardown()` pops `_dsa_buckets` after its release and drops `_ballast`;
  - the sharded layout settings are applied via `apply_layout_env` (`GLM53_DSA_SHARD` not exported; prior values
    recorded) and restored by `teardown()`;
  - `_trace_tensors` includes the bucket tensors (`device_tensors()`);
  - `s67r_checks.provenance()` records the snapshot base (`BASE.txt`) as `code_base`.
- **Host tests** (`tests/test_standalone_host.py`): plan_context at 1 / 8 / 16 / 32 slots, the reserve coupling, layout
  env apply / restore, an automatic MoE split choice replaced by the layout's value, and an explicit setting kept.
  `test_generator_vllm_host.py::test_default_precision_uses_serving_budget` now asserts >= 262,144.
- **Docs:** segmented decode (stage 10) in both READMEs, the fallback audit and the contract (`context_size_cost`
  resolved, trace-region note); `full_model.supported_context` = 1,048,576 with a per-layout split.
- **Pending:** v4b (the final code with stage 10's next commit) runs TF on 3 references with the prefill check, an
  autoregressive run, ptchange, sampler alternation, TTFT / token-out, the program guard and the rebinding leak
  check. Script: `~/glm53-flash-bringup/scripts/s7r2_v4b.{py,sh}`.
- **v4b** (2026-10-21; `doc/optimized_full_model/runs/s7r2/v4b_default.json`; pure `git archive 0a2663d0b6b`, the code
  freeze; `code_base` recorded): the final default, now the headline in both READMEs and `perf_summary.json`.
  - Build: 272,384, replicated; segmented decode (`DsaBuckets`); MoE split off (1,234,298,880 B).
  - Readiness: TF 92 / 95 / 92 and prefill check 92 / 95 / 91 (top-1; top-5 / top-100 100 %).
  - Autoregressive: repeat bit-identical, degenerate check exit 0 (`runs/s7r2/v4b_ar/`).
  - ptchange traced = eager 12 / 12; sampler alternation passes with seeds taking effect.
  - TTFT 2,048 / 8,192 = 1,022 / 4,386 ms; token-out 50.10 ms.
  - 0 post-capture programs; 26 `buckets.*` tensors under the address guard.
  - Teardown leaves 2.68 MB per bank (read before `del`); a second full default build in the same process was built,
    captured and generated 8 tokens.

## 2026-10-21: stage-6 review #5 P3s

- AR docs: v4b's trajectory is token-identical to `final_default_h9`, whose HF scoring applies (top-1 94.9 %, top-5
  99.2 %, HF top-1 at positions 100-255, worst rank 21 at position 24).
- Multi-slot: decode is segmented for every DSA build (views always on); at S >= 2 the MoE active-count buckets add
  point traces (freeze-code evidence: stage 10's t15b, 16 slots, 57.9 MiB of decode trace). 32 slots were not re-run
  standalone with MoE buckets (wide serving: stage 9's `s9_final`).
- The leak control is worded as "built, captured and generated 8 tokens". The watcher and tracker sections are scoped
  to pre-stage-10 code (freeze-code tracker: t13 / t14). Prompt lengths p1 / p2: 153 / 131.
- Freeze-code probe (`logs/s7r2/v5_probe_s8.log`; pure 0a2663d0b6b, 8 slots, `GLM53_DEALLOC_CHECK=1`): 6 passed, 1
  failed. `test_no_shared_buffer_frees` reports `moe_opt.py:895 <- model.py:300 <- dsa_buckets.py:387` (8 calls), an
  MoE-bucket force-free of a view-held buffer. Found defect; stage 10 owns the fix; the probe is re-run on the fixed
  HEAD.

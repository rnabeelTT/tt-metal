# Stage 6 work log (full model)

## 2026-10-16

- `tt/model.py` (Glm53Model), `tt/generator.py` (Glm53Generator, `build_generator`) and `tt/decode_inputs.py`
  (per-step DSA / KDA decode inputs computed on the device from the position tensor) written. `tools/readiness.py`
  opens the (1, 4) mesh with fabric, l1_small and a trace region (the stock runners open a bare mesh).
- HF full-model reference on CPU (`tools/hf_full_ref.py`, lazy FP8 experts): every layer 0..44 is bitwise equal to the
  per-layer reference; `readiness_aime24_chat.refpt` generated (161 prompt tokens, 100 generated, top-k 100).
- Reduced probe (`tests/test_model_probe.py`, layers 0, 3, 4): the traced-greedy test hung. Bisect with
  `tools/model_step_probe.py` (one sync per phase) located the hang in a prefill AFTER decode trace replays. Cause:
  prefill builds persistent constants lazily; built after the capture, they landed in freed trace-intermediate memory and
  the next replay overwrote them. Fix: `_warm` prefills every constructed length before any capture, and captures the
  sampler trace right after the model trace. Probe 3 / 3 (`runs/s6_probe4.log`).
- Also fixed: positions now advance on the device at the end of `decode_step`; `reset_state` used `x * 0` (kept NaN,
  leaked one state copy per tensor) and now copies zeros and frees the temporary.

## 2026-10-17

- First full-model readiness: OOM in the 2048-row warm-up prefill (21 MB free per bank; 8 slots, 512 MiB trace
  region). The sampler then rejected 4 slots (`format_sampling_params` needs a multiple of 32): the model now pads the
  logits rows to `SR = ceil32(S)` and the token buffer is SR wide.
- Teacher forcing 93 / 100 / 100 and prefill check 92 / 100 / 100 (4 slots, 256 MiB trace region; `runs/`).
- Autoregressive (256 tokens, HF side precomputed on CPU with the same model) and the 6-prompt qualitative suite pass
  (`autoregressive/VERDICT.md`, `qualitative/VERDICT.md`).
- Memory (`memory_budget.json`, `memory/`): measured weights, KDA state per slot, DSA bytes per token, decode trace
  46 MiB (default region now 64 MiB). The MoE prefill runs its routed experts in row blocks (worst-case dispatch buffer
  183 MB, live twice) and the DSA indexer uses 64 MB blocks: both bit-identical. 1-slot capacity probe: 104,515-token
  prompt, 24 / 24 decode steps correct. `doc/context_contract.json`: supported_context 104,448.
- Weight cache (`weight_cache.md`): the load hang was the driver pin of a file mmap; cache files are staged through a
  memfd. Full model warm load 110 s vs 24 min, byte-identical weights.
- Stage 7 later changed the defaults that this stage measured (chunk plan, MoE split rule, ag_persistent off); the
  final-default readiness is in `doc/optimized_full_model/`.

## 2026-10-18: stage-6 review #1 / stage-7 review #1 rework

Logs: `~/glm53-flash-bringup/logs/s67r/` (every device job stamped with git HEAD + the md5 of the snapshot's
`tt/*.py`; `status` lists every job's one-line result). Device jobs ran from pinned snapshots (`~/glm53-flash-bringup/
s67r_snap*`, `s67r_final` = HEAD 656c998e3df + this rework's files) because other agents edited the shared tree.

Code (tt/generator.py, tt/model.py, tt/decode_inputs.py, tt/dsa.py, tt/dsa_shard.py, tt/weight_cache.py,
tt/functional_decoder.py; guard lines in tt/moe.py / tt/moe_opt.py):
- `build_generator` / `Glm53Generator`: defaults from the DRAM budget (`plan_context`): 1 slot, the largest context of
  the fastest DSA layout that fits; the block-sharded layout is chosen automatically when the request exceeds the
  replicated capacity (bf16 latent only); a request that fits nowhere raises `ValueError` before any weight loads, and
  the standalone generator re-checks the pool against the free DRAM measured after load. Budget source:
  `serving_profiles.json` (stage 9) when its precision id is the build's, else the built-in stage-7 numbers; DSA bytes
  per block from the latent dtype; reserve max(data, 1200 MiB). Trace region default from the data file.
- Decode-overrun guard: positions >= max_seq_len raise on the host (`generate`, `decode_forward`, serving decode) and are
  inactive on the device (`DeviceDecodeInputs.refresh`: no cache row written, open-pool state kept).
- Sampler: every variant (greedy, penalties) compiled before any capture, including the penalty token-count programs
  (the cause of "Cannot load new binaries during trace capture"), and captured back to back after the model trace;
  lockstep sampling (identical per-draw seeds + die-0 token broadcast) on in the standalone generator too.
- Weight cache: per-layer MoE precision exceptions key their own moe directory; checkpoint shard fingerprint (size +
  mtime) in the manifest; `_build_lm_head_ds` in the top digest.
- Prefill/decode transition made allocation-free after the capture: the DSA open-pool hand-off uses request-independent
  programs (one-hot row select + slot mask) instead of slices at (prompt length, slot); `_warm_extra` compiles the
  remaining request-dependent programs (KDA commit per slot, last-row tile per offset, the all-logits head) before the
  capture. Checked by tt-metal's trace allocation tracker (`TT_METAL_TRACE_ALLOC_TRACKING=1`): 0 live post-capture
  allocations at 1 / 8 / 32 slots, replicated and block-sharded, and on the stage-9 adapter smoke.
- Address-stability guard: `Glm53Generator.check_trace_addresses` (after every prefill and before every reset) and an
  `is_allocated` check of every buffer a decode replay reads.

Defects found and fixed (in order):
1. Penalty-sampler capture compiled programs inside the capture (count_tokens path): warm-up now runs it eagerly.
2. Deferred DSA pool (standalone `allocate_dsa=False`) with the sampler built first: traced decode != eager. The sampler
   is now built after the pool (`Glm53Model._init_sampler`); the deferred path passes the reduced probe and the adapter
   smoke's traced == eager check. The standalone default keeps the HEAD allocation order (`GLM53_DEFER_POOL=0`).
3. Penalty update deallocating a view of the token buffer: our call site counts a clone (`Glm53Model._init_sampler`).
4. Request-dependent slice programs compiled after the capture (open-pool hand-off and friends): see above.
5. **The full-model decode corruption (teacher forcing 14 / 32 / 90)**: `Glm53Model.decode_step` padded the logits rows
   to the sampler width with `ttnn.pad` and then `ttnn.deallocate(logits)`. At S < 32 the pad stays inside the tile
   padding and returns a VIEW of `logits`; `ttnn.deallocate` defaults to force=True and freed the storage of the
   returned tensor - the decode trace's persistent logits output. After the capture the allocator handed that block to
   later tensors; every replay then wrote logits into them, and the sampler read logits that other ops overwrote. It
   was latent since stage 6: harmless while nothing was allocated in that block, deterministic once something was (DRAM
   pressure: 65,536 + 480 MB ballast, 102,400 tokens at 1 slot; or any change in allocation order, e.g. the stage-5
   dsa_idx_rep option). Found by elimination: not post-capture allocations (tracker clean), not uninitialized reads
   (NaN-poisoned free DRAM, full and reduced model), not the sampler, not the MoE split, not trace-only (eager decode
   failed too); `ttnn.deallocate(force=False)` made the failing run pass (92 / 100 / 100 vs 14 / 32 / 91 in the same
   session), then a wrapper that records every deallocate leaving a still-referenced buffer named the call site.
   False trails: per-layer host reads / clones and pre-step state snapshots made the failure disappear only because
   they changed the allocation sequence. Fix: no deallocate after the pad. Same pattern (benign) fixed in
   `DeviceDecodeInputs._into`; five benign stage-3 sites in moe.py / moe_opt.py now free with force=False;
   `models/common/sampling/tt_sampling.py:1170` (`sampling_values`, freed after its last use) noted for upstream.
   Permanent check: `GLM53_DEALLOC_CHECK=1` (tests/test_model_probe.py::test_no_shared_buffer_frees).

## 2026-10-20 .. 21: stage-6 review #2 / #3 rework

Commits:
- 675a8b1bed9: stage-6 review #2 P1-1 (the DSA-1M agent). Block-sharded DSA cache with the fp8 latent: 1,048,576 tokens
  at 1 slot (copy 1.0 at 1,048,515 tokens; teacher forcing 91 / 95 / 91 at 16,384; +2.9 ms per token).
- 73c0029ee79: stage 9 G9 probe of profile p300x2-1m (1 x 1,048,576 measured). Contract `supported_context` =
  advertised.
- 237ec3b8438: the stage-6/7 review #2 kit.
  - The ag_persistent re-test (passes on the fixed code; kept off).
  - The standalone post-capture program guard (`Glm53Generator.guard_programs`).
  - Standalone budget: per call; reserve max(data, 1,200 MiB) unless `GLM53_DRAM_RESERVE_MIB`.
  - Weight-cache manifest adoption only when no shard is newer than the cache.
  - A flat-distribution seed check (seeds take effect).
  - `test_no_shared_buffer_frees` covering every autoport `tt/` module (two benign `moe_opt.py` view frees guarded).
  - Evidence: `doc/optimized_full_model/runs/s7r2/`.

Review #3 rework (this entry's kit; logs `~/glm53-flash-bringup/logs/s7r2/`):
- **P2-1, the standalone generator accepts the supported context:**
  - `tt/generator.budget()` gives each layout its own budget from `serving_profiles.json` (sharded:
    `layouts.sharded.budget`, 600 MiB workspace, 292,864 B per block).
  - Every layout is capped at its measured maximum (replicated 325,632, sharded 1,048,576 at 1 slot), so a low
    `GLM53_DRAM_RESERVE_MIB` cannot plan an unmeasured replicated context.
  - `plan_context(1, 1048576)` selects the sharded layout; the default `plan_context(1)` is unchanged (294,912,
    replicated).
  - A sharded plan applies `serving_profiles.layout_env` (`setdefault`) before the build.
  - The `ValueError` names the recipe.
  - Host tests: `tests/test_standalone_host.py`.
- **P2-2, docs and contract:** the defaults bullets and table, the fp8 sharded memory row, the context table
  (p300x2-long 325,632; the 1M recipe), Limitations, the contract's standalone defaults and G9 fields, and
  `serving_profiles.json` (the sharded budget is measured; the done G9 items are removed from `pending`).
  `tools/serving_profiles.py check` passes; `tt-model.yaml` is in sync.
- **P3:**
  - `generate()` sets greedy params itself;
  - the sampled (serving) prefill runs `check_trace_addresses` + `guard_programs`;
  - `generate()` runs the guard once per call;
  - the fallback audit notes that the all-logits head of a 2048-row chunk compiles after the capture;
  - `Glm53Generator.teardown` releases every trace, drops the model references and collects the cycle (a second
    build in one process; the reduced leak check is in `~/glm53-flash-bringup/artifacts/s7r2/leak.json`);
  - `WeightCache.get` re-raises a device OOM instead of rebuilding;
  - a read-only cache root reuses a verified legacy cache, or loads uncached.

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

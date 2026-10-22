# GLM-5.3-Flash (E192) - TTI release run notes (stage 11)

Status: DRAFT (2026-10-21). The release workflow ran to completion. Its acceptance is FAIL with one blocker, the
GPQA CI subset. A full 198-question GPQA run is queued as the resolution path. A loop diagnostic (below) runs
first. The bundle build, fresh-device verification and Hub publish sections are pending.

## Server mode, host, versions
- Topology: external server, no Docker. The autoport vLLM server runs from this tt-metal code. TTI v0.22.0 is a
  client only. The written run spec has `docker_server=false`, `local_server=false` and `service_port=8141`.
- Host: `qb2-120-p11t05`, the p300x2 box (2 x p300c = 4 Blackhole dies, mesh (1,4)). There is no reservation
  container, and no Docker was used by TTI. Each device job ran through the workspace FIFO lock
  (`~/glm53-flash-bringup/scripts/run-locked.sh`), which runs `tt-smi -r` before and after.
- Code:
  - Clean release worktree `~/tt-metal-rel-glm53`, local branch `glm_5_3_flash_rel`.
  - Smokes ran at 0a2663d0b6b. The release run and everything after it ran at the final code freeze
    **0bcf89f7fd6**: the same tree plus the stage-10 S >= 2 fixes, which are not on the 1-slot path.
  - The Python code of the autoport and its whole import closure comes from the worktree (audit: 45 of 45
    `models.*` modules load from the worktree; the server cwd is the worktree).
  - Later code freezes: 3993d5eecef (stage 10: DSA bucket contexts start at 65,536 for S >= 2; S = 1 unchanged,
    so p300x2-1m behaves identically) and the final package HEAD (recorded in the bundle section). The TTI evidence
    on p300x2-1m (1 slot) stays valid for them (lead review).
  - The ttnn binaries and the JIT kernel sources come from `~/tt-metal` (`TT_METAL_HOME`). The chain script checks
    that `ttnn/`, `tt_metal/` and `tt_stl/` there are identical to the worktree commit, and aborts otherwise.
- vLLM 0.26.0 + vllm-tt-plugin 92403f3. TTI `~/tt-inference-server` tag v0.22.0 (9ef9d660, contains #4345 /
  6e396b4). TTI runs from its own worktree `~/glm53-flash-bringup/tti` with the local integration patch, so the
  shared TTI checkout stays clean.
- Serving profile **p300x2-1m** (`serving_profiles.json`): 1 sequence x 1,048,576 tokens, block-sharded fp8 DSA
  cache, profile env `GLM53_DSA_SHARD=1 GLM53_MOE_SORTED_ROWS=512 GLM53_DSA_INDEX_BLOCK_MB=32`, trace region
  64 MiB.
  - This equals `doc/context_contract.json` `supported_context` (1,048,576 = HF-advertised).
  - The served `/v1/models` reports `max_model_len` 1048576 (`release/models.json`).
  - It is the report of record because the release spec must preserve the contract (stage-9 rule).
  - The default customer profile p300x2 (16 x 243,712) appears only as a supplementary benchmark, outside this
    directory.
- Server command and env: `logs/servercmd-release.txt`.
  - Flags: `--block-size 128 --max-num-seqs 1 --max-model-len 1048576 --enable-auto-tool-choice
    --tool-call-parser glm47 --reasoning-parser glm45 --additional-config {"tt": {sample_on_device_mode all,
    trace_region_size 67108864, l1_small_size 24576, FABRIC_1D}}`.
  - Env: `GLM53_PROGRAM_GUARD=1`, weight cache `~/.cache/glm53_flash`, async decode = the adapter default (on).
  - Precision: `combo_a_kv_dense4_g4`, from `selected_precision_config.json`.
- Chain script: `~/glm53-flash-bringup/scripts/11-tti.sh` (smoke | release | evals | spec | bench | loopdiag).
  - It starts and stops only its own server process group, and removes any `.env` that TTI wrote.
  - The HF token is read from the file into the environment and never printed. The copied logs contain no token.

## TTI integration (local patch, `tti_local_integration.patch` = full `git diff` of the TTI worktree)
- `ImplSpec glm53_flash_autoport` with `code_path=models/autoports/zai_org_glm_5_3_flash`.
- Dev-catalog entry (P300X2): `max_concurrency 1`, `max_context 1048576`, block 128, the profile env, parsers
  glm45 / glm47, status FUNCTIONAL (evals enforced).
- `EvalConfig` for zai-org/GLM-5.3-Flash. Both tasks use the chat API (the model's chat template), temperature 1.0,
  top_p 0.95 (generation_config = the E192 card's recipe), seed 42 (TTI default), `capture_reasoning`, and a
  6 h request timeout.
  - `ifeval`: reasoning_effort=low, max_gen_toks 32768.
  - `gpqa_diamond_cot_zeroshot`: reasoning_effort=low, max_gen_toks 65536, published_score 79.8.
- **Effort decision (lead, D2).** Both tasks run at `reasoning_effort=low`. The SHIPPED DEFAULT is the chat
  template's default, **max** effort. Low was chosen for two reasons: it matches the only published E192 GPQA
  number (low / 65,536 tokens), and the E192 card recommends low for chat and MCQ. Low thinking can only lower
  the scores, not raise them. Budget-exhausted or empty answers count as wrong (lm-eval scores `content`; the
  reasoning parser puts the think block in `reasoning`).
- TTI harness patches, with the reason for each:
  1. **Benchmark client venv.** TTI's shared client venv (vllm 0.13, transformers 4.57) cannot load the GLM
     tokenizer (`TokenizersBackend does not exist`). The patch routes this model's sweep to TTI's newer client
     venv (vllm 0.19.1, transformers 5.5.1). Note: the llm driver actually ran `vllm bench serve` from
     `.venv_llm_vllm` (transformers 5.17), which loads the tokenizer.
  2. **Conformance tests read `content`.** An always-thinking model (the template opens `<think>` on every turn)
     returns `content: null` until the think block closes. The patch falls back to `reasoning` (9 call sites).
  3. **`test_non_uniform_seeding` read timeout.** The test fires 32 concurrent requests with a 30 s client read
     timeout. With `max_num_seqs 1`, vLLM queues them; the server log shows "Running: 1, Waiting: 19", and the
     server log is clean. The later requests timed out on the client side. The patch uses `timeout=None` for
     this test, as `test_penalties` already does. The test's checks are unchanged.

## Commands
- Smoke (i), 2026-10-21 14:46, at 0a2663d0b6b:
  `11-tti.sh smoke`
  - health and `/v1/models` (1,048,576)
  - chat "Paris." (6 tokens, effort low)
  - `run.py --dev-mode --model zai-org/GLM-5.3-Flash --tt-device p300x2 --workflow benchmarks --service-port 8141
    --no-auth --skip-system-sw-validation --disable-trace-capture --limit-samples-mode smoke-test` -> exit 0,
    8/8 requests, 0 failed, TTFT 211 ms, TPOT 70.3 ms (`smoke/benchmark_*isl-16_osl-4*.json`)
  - spec_tests -> 21/22 (seeding timeout, see patch 3)
  - evals smoke (ifeval, 3 docs) -> exit 0, 3 real answers
- Smoke (i-b), 15:24, at 0a2663d0b6b: spec_tests **22/22** with patch 3. gpqa eval smoke: 3 docs end to end,
  flexible-extract 3/3, about 30 s per doc (`smoke/`).
- Release (ii), 16:40-21:23, at 0bcf89f7fd6:
  `python run.py --dev-mode --model zai-org/GLM-5.3-Flash --tt-device p300x2 --workflow release --service-port 8141
  --no-auth --skip-system-sw-validation --limit-samples-mode ci-nightly`
  - Log: `logs/tti-release.log`, chain log `logs/tti-release.chain.log`.
  - run.py exit code: 1 (Acceptance FAIL).
  - Report: `release/report_id_glm53-flash-autoport_GLM-5.3-Flash_p300x2_2026-10-21_21-23-04.md`
    (+ `reports_output/release/data/`, run spec `run_specs/runtime_model_spec_release.json`, `release/evals/`,
    `release/benchmarks/`).
  - Wall time 4 h 43 min: evals 3 h 35 min, benchmarks 43 min, spec tests 22 min.
- Runtime estimate, recorded before the release:
  - Unrestricted eval sets at 1 slot (541 ifeval + 198 gpqa documents at the measured ~30 s each): about 6 h.
  - The earlier max-effort estimate was 15-20 h.
  - ci-nightly (limit 0.2 = 109 + 40 docs) was selected for the release run of record. Its accuracy numbers are
    **CI-subset results**, not full-set accuracy. The full GPQA set is run separately (below).

## Results of the release run (CI-subset; NOT full-set accuracy)

| row | result | reference | TTI check |
|---|---|---|---|
| ifeval (effort low, 109 docs) | prompt-level strict **74.31** (81/109); loose 81.7; inst-level strict 84.2, loose 88.7 | none: no IFEval number exists for E192 or the parent; an HF reference of E192 on this host's CPU is not practical (decision (b), 2026-10-18) | NA |
| gpqa_diamond_cot_zeroshot (effort low, 65,536 tokens, 40 docs) | flexible-extract **70.0** (28/40, stderr 7.3); strict-match 0.0 | 79.8, E192 card (effort low, 65,536 tokens) | **FAIL** (ratio 0.877; subset rule needs >= 30/40) |
| spec tests (VLLMParamConformanceTest) | **22/22** | - | PASS |
| benchmarks (12 sweep points, 1 slot) | all requests completed, 0 failed | no perf targets exist | NA (informational) |

- Acceptance: **FAIL, 1 blocker** (gpqa). Nothing is waived.
- The 6 structured-output benchmark tasks that TTI's benchmark config lists did not run: the run produced 12
  sweep blocks only. To be followed up.

Benchmark rows (p300x2-1m, concurrency 1):

| ISL / OSL | TTFT | TPOT |
|---|---|---|
| 128 / 128 | 251 ms | 71.5 ms |
| 128 / 1024 | 248 ms | 73.0 ms |
| 1024 / 128 | 0.82 s | 74.4 ms |
| 2048 / 128 | 1.40 s | 77.5 ms |
| 4096 / 128 | 5.99 s | 77.6 ms |
| 8192 / 128 | 15.1 s | 77.7 ms |
| 8192 / 1024 | 15.1 s | 77.8 ms |
| 10000 / 1024 | 19.6 s | 77.8 ms |
| 16384 / 128 | 33.4 s | 77.7 ms |
| 32768 / 128 | 69.7 s | 77.6 ms |
| 65536 / 128 | 142.7 s | 77.5 ms |
| 131072 / 128 | 288.5 s | 77.3 ms |

### GPQA CI-subset analysis (`release/evals/gpqa_ci_subset_analysis.json`)
- No answer is empty, and none reaches the budget. Generated tokens (reasoning + content): minimum 35, median 356,
  mean 663, maximum 5,023.
- Reasoning tokens: median about 140. At effort=low the model thinks very little and puts most of its working in
  the content.
- No extraction misses. All 12 wrong documents give an explicit different letter, which flexible-extract read
  correctly. 39/40 answers end in `\boxed{X}`.
- strict-match 0.0 is a format artifact: the strict filter wants "The answer is (X)" and the model writes
  `\boxed{X}`.
- 28/40 = 70.0 +/- 7.3 against 79.8 is a 1.3 sigma gap, consistent with sampling noise at n = 40. Resolution path
  (lead decision (b)): the FULL 198-question GPQA Diamond at the same settings, judged with TTI's rule
  (>= 0.95 x 79.8 = 75.81 %, i.e. >= 150-151 of 198). If it falls short, it is reported, not retried.
- Caveats:
  - The card ran NVFP4 weights with its own prompt ("ANSWER: X" after `</think>`). lm-eval's zero-shot CoT prompt
    differs. The full set removes the sampling noise but not that methodology gap.
  - The card says that at low effort "some answers are cut off" at 65K tokens. Here nothing exceeded 5K, so this
    port at low effort thinks far more briefly than the card describes. That may come from the prompt (the card's
    says "Think step by step"); these data cannot attribute it.

### Full 198-question GPQA Diamond (supplementary evidence for the CI-subset blocker)
- First attempt on the report-of-record profile p300x2-1m (`glm53-s11-gpqafull`, from 2026-10-22 01:23): stopped by
  the lead at 70/198 after 3 h 13 min. At 1 slot the long-tail answers (up to the 65,536-token budget) would have
  needed about 6 more hours. lm-eval writes samples only at task end, so no partial samples exist; the planned
  first-70 cross-check between the two profiles was not possible.
- Run of record for this row: profile **p300x2** (16 x 239,616 as served; replicated DSA layout; the same weights,
  precision policy `combo_a_kv_dense4_g4` and code, release worktree 0bcf89f7fd6). It is NOT the report-of-record
  profile; it was chosen for runtime, because 16 concurrent requests hide the long tails.
  - TTI v0.22.0 + the same local patch, in a separate TTI worktree (`~/glm53-flash-bringup/tti-p300x2`, catalog
    max_concurrency 16, max_context 243,712): `run.py --dev-mode --model zai-org/GLM-5.3-Flash --tt-device p300x2
    --workflow evals --service-port 8141 --no-auth --skip-system-sw-validation` with
    `GLM53_EVAL_TASKS=gpqa_diamond_cot_zeroshot`, no limit.
  - Same generation settings: num_concurrent 16, reasoning_effort low, max_gen_toks 65,536, temperature 1.0,
    top_p 0.95, seed 42.
  - 2026-10-22 05:06-07:54 (2 h 49 min).
  - Its raw JSONs are kept outside this directory (`~/glm53-flash-bringup/artifacts/s11/gpqa_full_p300x2/`),
    because the catalog's max_context of 243,712 is a profile value, not the contract.
- **Result: flexible-extract 66.67 % (132/198, stderr 3.4)**; strict-match 3.0 (format, see above).
  - TTI rule against the E192 card's 79.8: ratio 0.835, needs >= 0.95 (>= 150/198) -> **FAIL** (run.py rc 1,
    Acceptance FAIL).
  - The gap is about 3.9 standard errors, so it is not sampling noise.
  - 4/198 answers have empty content: the think block hit the 65,536-token budget (docs 79, 127, 170, 196). These
    are coherent long derivations, not loops (max word run <= 6). They count as wrong.
  - Generated tokens: median 400, mean 3,067, maximum 65,536.
  - The first 40 documents score 31/40 here vs 28/40 in the release CI subset (a different realisation, sampled at
    temperature 1.0).
- Interpretation (no waiver implied):
  - The port scores about 13 points below the card's number under nominally the same effort, budget and sampling.
  - Known methodology differences: the card ran NVFP4 weights on one B200 with its own prompt ("Think step by step,
    then give your final answer as 'ANSWER: X'", extracted after `</think>`). lm-eval's zero-shot CoT prompt and
    flexible extraction differ. On the card's runs at low effort, answers sometimes reached 65K tokens; here the
    median answer is about 400 tokens.
  - Port-side precision: bfp4 experts / projections; stage-8 teacher forcing vs HF, top-1 92/95/92 %.
  - Host analysis of the 198 samples:
    - Effort: the request renders `<|system|>Reasoning Effort: Low` through the template, as intended.
    - Budget: 4/198 cut at the budget, which matches the card's "some answers are cut off".
    - Reasoning tokens: p10 39, median 164, p90 2,839.
    - Accuracy by reasoning length: < 200 tokens 72.3 % (n 112), 200-1K 65.5 % (n 55), 1K-5K 33.3 % (n 15),
      > 5K 62.5 % (n 16).
    - Accuracy by subject: Physics 80.2 % (86), Chemistry 54.8 % (93), Biology 63.2 % (19).
    - Extraction: no misses (192/198 boxed answers, no ambiguous ones).
    - Ranked hypotheses: H1 port precision policy (stage-8 TF top-1 92 %); H2 prompt / methodology; H3 effort
      handling (ruled out); H4 card variance.
    - Proposed decisive test: the same 198 questions with `GLM53_PRECISION=none` (the stage-7 policy). It has
      not been run yet (lead decision).
  - H1 attempt 1 (2026-10-22 15:50-16:45, stage-7 policy file `stage7_fp8kv_h1`, 8 slots, max_model_len 70,016,
    DSA pool 666 blocks): INVALID. The EngineCore died at 16:44 in vllm-tt-plugin
    `_alloc_prefill_state_slots`: "no free device state slot for 4 prefill(s): held=[0..7], capacity=8". vLLM had
    preempted requests when the shared pool filled, and 121/198 requests failed.
    - This is a serving defect: preemption crashes the engine. The shipped p300x2 pool covers about 265K tokens
      for 16 sequences. The defect is handed to the stage-9 owner as a release blocker.
    - Indicative only: on the 77 docs completed before the crash, stage-7 got 61/77 and the shipped policy 55/77
      on the same docs (10 right only with stage 7, 4 right only with the shipped policy).
    - H1 is rerun after the preemption fix (`scripts/11-h1.sh`).
  - H2 (the card's prompt and dataset, fingertap/GPQA-Diamond, "ANSWER: X" after `</think>`):
    `scripts/11-h2.sh` / `11-h2-cardprompt.py`, prepared, not run.
  - These data cannot separate the methodology gap from a port quality gap. The row is classified
    **readiness-fail pending a lead / owner waiver decision**. Per the lead's instruction it was not retried.

### ifeval: two degenerate single-token loops (under diagnosis)
- 2 of 109 answers loop on a single token until the 32,768-token budget:
  - Doc 65 (India/China itinerary table): " boat" from content token 161.
  - Doc 106 (Vietnamese critique): " cả" from content token 14.
- Their think blocks are short and normal. No other ifeval or gpqa output repeats (scan: word runs >= 5, 4-grams
  >= 20). Both count as wrong in 74.31.
- Settings: temperature 1.0, top_p 0.95, seed 42, effort low, device sampler.
- Diagnostic ticket `11-tti.sh loopdiag` (`~/glm53-flash-bringup/scripts/11-loopdiag.py`) replays both requests:
  - A: exact replay
  - B: host-sampled path with logprobs and the full top-5 at the first loop steps
  - C: seeds 1-8
  - D: no seed
  - E: greedy
  - F: top_p 1.0
- Classification: host path does not loop, or p(loop token) well below 0.9 -> sampler defect; B loops with
  p near 1 and C shows similar rates -> model attractor.
- **Result** (`loopdiag/loopdiag.json`, 2026-10-21 21:51-22:17, 0bcf89f7fd6), as looped runs out of total runs:

  | case | doc 65 | doc 106 |
  |---|---|---|
  | A (exact replay, seed 42) | 0/1 | 1/1 (bit-identical to the eval output) |
  | B (host-sampled path, seed 42) | 0/1 | 1/1 |
  | C (seeds 1-8) | 1/8 | 0/8 |
  | D (no seed) | 1/8 | 0/8 |
  | E (greedy) | 0/1 | 0/1 |
  | F (top_p 1.0) | 0/1 | 0/1 |

  - Doc 106 B, the host path: the loop token is the argmax at every onset step. The repeated token " ca" has
    p 0.538 at position 30, then 0.493 / 0.851 / 0.784 at the first repeats, with alternatives " lich" 0.038 and
    " lam" 0.034. It rises to 0.9999 in the loop tail.
  - Classification: **not a device-sampler defect**. The host-sampled path loops the same way, and the device and
    host paths agree. It is a **model repetition attractor** at temperature 1.0, prompt-dependent and rare: about
    1/8 draws on doc 65, and 2/109 in the eval run.
  - Not settled by these data: whether the TT logits at this prefix deviate from the HF reference. That needs an
    HF teacher-forced comparison of positions 30-33, which was not run.
  - Scorecard caveats:
    - 2/109 ifeval answers are single-token loops (model repetition attractor at temperature 1.0).
    - The per-draw rate is about 1/8 on doc 65. The host-sampled path reproduces the loop; greedy and top_p 1.0 do
      not loop.
    - Vietnamese output quality is poor even without loops. This is consistent with the E192 card's lower
      non-English scores (C-Eval 78.3 % vs 89.4 % for E256).
  - HF teacher-forced check of positions 30-33 on the doc-106 prefix: NOT RUN (lead decision: host RAM; allowed
    only after all release device work is done and the host is idle).
  - Reproducibility note: doc 65 at seed 42 replayed differently from the eval run. It diverged at the first
    heading. In the eval run the request followed 64 others in the same session; in the replay it was the first
    request after boot. Doc 106 replayed bit-identically.

## Seeded reproducibility on p300x2-1m (stage-11 finding)
- Symptom: doc 65 at seed 42 gave a different output in the eval than in a replay on a fresh server.
- Controls on a fresh server (`seedreplay-20261021T2317`): C1 = C3 = dbf9c987, but C2 (sent right after C1) =
  9b7f2647.
- History replay (`seedreplay-20261021T2324`): the eval's ifeval requests 0-64 in order, then doc 65, on a fresh
  server: **66/66 bit-identical** to the eval, including doc 65's 32K-token loop. The output is deterministic
  given the server's request history.
- Instrumented diagnostic (`seeddiag/`; build = 0bcf89f7fd6 + an instrumentation-only patch; doc 65, seed 42, 4 runs
  back to back per case):

  | case | outputs |
  |---|---|
  | p300x2-1m (sharded), async on | A B A B |
  | p300x2-1m, async off | A B A B (the same two outputs) |
  | p300x2-long (replicated), async on | 4/4 identical, although the block ids changed between runs |

  - The pushed device-seed / counter streams are identical in every run.
  - On p300x2-1m the A/B split follows the vLLM block ids. A and B diverge at about sequence token 128, where the
    second KV block (block 2 on die 2 vs block 5 on die 1) comes into use.
- Classification: **block-placement numerics of the sharded DSA layout** (die = global block id % 4; stage-9
  placement control: 1-block shift max |delta logit| 0.527). NOT a seed / async defect. Seeded sampling on
  p300x2-1m is reproducible for a given block placement only. The replicated profiles (p300x2, p300x2-long,
  p300x2-wide) are reproducible.
- Effect on this report: none. The eval outputs are one deterministic realisation; the distribution is unchanged.
- Lead decision (c): document the limitation for this release. It goes into the card's Limitations and
  doc/vllm_integration ("Seeded sampling"); the hunks are drafted under
  `~/glm53-flash-bringup/artifacts/s11/hunks/` and committed by the lead.
- Follow-up (a), not in this release: request-relative die placement.
  - Design sketch: the adapter keeps a per-request map from logical block index j to a physical block on die
    j % 4. It needs one free list per die and translates vLLM's block ids into physical ids when it writes the
    page table each step (and on prefill / preemption / free), so vLLM's own block accounting stays unchanged.
    The pool size per die stays the same.
  - Alternative: (b) a placement-invariant cross-die reduction in dsa_shard.py (gather the partials and combine
    them in logical block order), at a decode-time cost on the 1M path.
  - Cost estimate: several days of work, plus re-validation of p300x2-1m (G9 capacity probe, 1M / 300K needles,
    teacher forcing, stage-9 placement control, async stale-input and slot-reuse tests).
- The prepared seed-path patch (`artifacts/s11/seedfix.patch`: no counter re-anchor for async-merged continuing
  slots, plus `tests/test_seed_stream_host.py`) is WITHDRAWN. The seed streams were already identical. It is kept
  only as a hardening note, the same rule as the shared `tt_transformers` generator (#51981).

## Autoport implementation check
- The TTI-written run spec (`run_specs/runtime_model_spec_release.json`) has:
  - `impl.impl_id=glm53_flash_autoport`
  - `impl.code_path=models/autoports/zai_org_glm_5_3_flash`
  - `docker_server=false`, `local_server=false`, `service_port=8141`, `limit_samples_mode=ci-nightly`
  - `device_model_spec.max_context=1048576`, `status=FUNCTIONAL`
- The report metadata has `model_impl: glm53-flash-autoport`.
- The server process ran from `~/tt-metal-rel-glm53` and imported
  `models.autoports.zai_org_glm_5_3_flash.tt.generator_vllm` from there (`EXTRA_MODELS_DIR` = the worktree's
  `vllm_bundle`).
- No `models/tt_transformers` or `models/demos` implementation was launched or benchmarked.
- Autoport implementation check: PASS (evaluated code path models/autoports/zai_org_glm_5_3_flash, matched in the
  copied run spec and report data).

## Recovery and hygiene
- No ARC, ERISC or reset failures. Each ticket ran `tt-smi -r` before and after through run-locked.
- The first smoke attempt did not start: the log directory was missing and run-locked could not open its log.
  It was re-queued.
- Host RAM was logged every minute (`~/glm53-flash-bringup/logs/11/ram-*.log`). During the release, the minimum
  available was about 19 GB, at 15:53, which was before the release started. One earlier OOM on the host
  (13:36) came from another job.
- No leftover server: process-group stop and the trap in the chain script. No tmux sessions. No TTI Docker
  containers.
- Not copied: eval sample dumps (1.1-1.4 MB each, kept in the TTI worktree `workflow_logs`), weights, caches, `.env`.

## Pending (to be filled)
- Loop diagnostic result and classification.
- Full GPQA (198) result.
- Supplementary p300x2 benchmark (outside this directory).
- Bundle build from the final HEAD, fresh-device cold verify, warm 1m verify, Hub publish, Hub verify.
- Stage-11 status line and row classification.

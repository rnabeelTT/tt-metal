# GLM-5.3-Flash (E192) - TTI release run notes (stage 11)

Status: final (2026-10-23). The release workflow ran to completion (acceptance FAIL with one blocker, the GPQA CI
subset). The GPQA row is issue-waived after the full-set and card-protocol runs; the re-aggregated report passes
(see "Stage-11 status"). The bundle is built, verified on a fresh device and published to the Hub, then republished
after the stage-11 review fixes (see "Stage-11 review fixes") and again after the review #2 fixes (Hub rev
0a807cdf; see "Stage-11 review #2 fixes").

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
  - The default customer profile p300x2 (16 x 262,144 in the bundle; 16 x 239,616 as served by the GPQA run below)
    appears only as supplementary evidence: the full GPQA runs and the bundle scorecard.
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

- Acceptance: **FAIL, 1 blocker** (gpqa). Nothing is waived in this run. (Superseded: the GPQA row was later
  issue-waived, and the re-aggregated report passes; see "Release report with the GPQA waiver".)
- The 6 structured-output benchmark tasks that TTI's benchmark config lists: **not applicable**. The TTI v0.22.0
  LLM runner is text-only. It drops every benchmark parameter set that is not a text task with an ISL and an OSL
  (`llm_module/benchmark_configs.py` lines 27-60), so the run produced the 12 text sweep blocks only.

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
  - 4/198 answers have empty content: the think block hit the 65,536-token budget (docs 79, 127, 170, 196). They
    count as wrong. Correction (stage-11 review): docs 196 and 170 are phrase loops. An 8-gram of words repeats
    2,412 times (doc 196) and 679 times (doc 170) in the think block. Doc 79 repeats one 8-gram 37 times; doc 127
    has no repeat by this rule. The earlier "not loops" came from a single-word run metric (max word run <= 6),
    which does not see phrase loops. See "Repetition loops: rate over all saved generations".
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
  - H1 rerun, chemistry only (the user's efficiency decision; the 93 questions whose High-level domain is Chemistry,
    where the shipped policy scored 54.8 %):
    - Code: release worktree 09842218b22, plugin 16febac (the preemption fix).
    - Policy `stage7_fp8kv_h1`, 8 slots, max_model_len 70,016, same settings as the 66.7 run.
    - Layout: block-sharded DSA (pool 3,366 blocks, about 431K tokens). A replicated attempt was pool-starved
      (85K tokens; 37/93 after 65 min) and was stopped.
    - Run 2026-10-22 18:56-23:10. 0 inference errors; the engine survived the full load, which is also field
      evidence for the plugin fix.
    - **Result: 54/93 = 58.1 % (stderr 5.1) vs the shipped policy's 51/93 = 54.8 % on the same questions.**
      Paired: both right 39, stage-7 only 15, shipped only 12 (27 discordant; McNemar not significant).
    - 7/93 hit the 65,536 budget with stage 7 (shipped: in the full set, 4/198 overall).
    - Conclusion (lead's decision rule): the precision policy is ruled out as the main cause of the GPQA gap.
      The shipped precision stays.
  - H2, the card's own protocol (`scripts/11-h2-cardprompt.py`):
    - Dataset fingertap/GPQA-Diamond (198 questions); prompt "Think step by step, then give your final answer as
      'ANSWER: X'"; the letter is taken from the content after `</think>`; no letter counts as wrong.
    - Effort low, 65,536 tokens, temperature 1.0, top_p 0.95, seed 42.
    - Shipped policy, release worktree 09842218b22, plugin 16febac; p300x2, 16 slots, block-sharded DSA (same
      weights; chosen for the pool size), max_model_len 70,016.
    - Run 2026-10-22 23:17 to 2026-10-23 about 02:40.
    - **Result: 147/198 = 74.24 %** (0 errors; 14 answers without an extracted ANSWER, 8 of them budget cuts).
      Against the card's 79.8: -5.6 points (about 1.8 sigma). That is more than the card's stated +/-2-3 points of
      run noise, and it is not explained. bfp4 vs NVFP4 routed experts were not tested.
    - Per subject (card protocol vs the lm-eval run, same questions): Physics 72/86 vs 69/86, Chemistry 63/93 vs
      51/93, Biology 12/19 vs 12/19.
    - Paired: both right 111, card protocol only 36, lm-eval only 21.
    - Conclusion: most of the lm-eval gap (66.7) is the harness prompt / answer format. Chemistry recovers
      12 questions under the card's prompt.
- **GPQA row classification: issue-waived (lead decision, 2026-10-23).** Evidence:
  - lm-eval full set 66.67 %, CI subset 70.0 %.
  - Card protocol 74.24 % against the card's 79.8 % (same effort, budget and sampling; the card ran NVFP4 weights).
  - The stage-8 non-expert precision changes are ruled out (H1 chemistry: stage-7 policy 58.1 % vs shipped
    54.8 %, not significant).
  - Answer extraction, effort rendering and budget handling checked: no misses, template-correct effort,
    budget-cut rate as in the card.
  - The residual -5.6 points (about 1.8 sigma) exceeds the card's stated +/-2-3 points of run noise and is
    unexplained. bfp4 vs NVFP4 routed experts were not tested.
  - No bug hunt (user decision). Per the lead's instruction the row was not retried.

### ifeval: three degenerate loops
- 3 of 109 answers loop until the 32,768-token budget:
  - Doc 65 (India/China itinerary table): " boat" from content token 161.
  - Doc 106 (Vietnamese critique): " cả" from content token 14.
  - Doc 87 (Persian poem): the pair zero-width non-joiner + "ی" from content character 143, 16,295 repeats of the
    16-character unit. The content is garbled Persian before the loop. Correction (stage-11 review #2): the
    whitespace-word scans missed this loop, because the repeated pair contains no space (one "word").
- Their think blocks are short and normal. All three count as wrong in 74.31.
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
  - Correction (stage-11 review): on doc 106, seeds 1 and 6 of case C hit the 1,024-token budget with empty
    content. The diagnostic saved no reasoning text, so these two draws are unclassified (the table counts them as
    not looped). Doc 106 C is therefore 0/6 classified, 2 unclassified.
  - Classification: **not a device-sampler defect**. The host-sampled path loops the same way, and the device and
    host paths agree. It is a **model repetition attractor** at temperature 1.0, prompt-dependent and rare: about
    1/8 draws on doc 65, and 3/109 in the eval run. The rate over all saved eval generations is in "Repetition loops:
    rate over all saved generations".
  - Not settled by these data: whether the TT logits at this prefix deviate from the HF reference. See "HF
    teacher-forced control of a doc-106 loop onset" (stage-11 review).
  - Scorecard caveats:
    - 3/109 ifeval answers are loops (two single-token loops, one two-character loop; model repetition attractor at
      temperature 1.0).
    - The per-draw rate is about 1/8 on doc 65. The host-sampled path reproduces the loop; greedy and top_p 1.0 do
      not loop.
    - Non-English output quality is poor even without loops (doc 106 Vietnamese, docs 18 Kannada, 50 Marathi, 87
      Persian). The HF teacher-forced control and the device precision A/B of the stage-11 review #2 measured the
      cause: see "Non-English output: device precision A/B and HF controls (stage-11 review #2)".
  - HF teacher-forced check of the doc-106 loop onset: not run at that time (lead decision: host RAM). It ran
    after the stage-11 review (see "HF teacher-forced control of a doc-106 loop onset").
  - Reproducibility note: doc 65 at seed 42 replayed differently from the eval run. It diverged at the first
    heading. In the eval run the request followed 64 others in the same session; in the replay it was the first
    request after boot. Doc 106 replayed bit-identically.

### Repetition loops: rate over all saved generations (stage-11 review, P2-2)
- Scanner: `~/glm53-flash-bringup/scripts/11-loopscan.py` -> `artifacts/s11/loopscan/loopscan.json`. Rule: a
  generation loops when one 8-gram of whitespace words occurs >= 20 times, or one word repeats >= 50 times in a row,
  or (character rule, added in the stage-11 review #2) one 16-character substring occurs >= 1,000 times, in the
  reasoning + content text. The earlier scans (word runs >= 5, 4-grams >= 20 on the content only) missed loops in
  the think block, where a looped answer leaves the content empty. The word rules missed loops without whitespace
  (doc 87). Recount with the character rule: `artifacts/s11/loopscan/loopscan_r2.json`.
- Result, every stage-11 eval generation whose full text is saved:

  | set (policy) | loops / generations | looped docs (largest 8-gram count) |
  |---|---:|---|
  | smokes: ifeval 3 + gpqa 3 (shipped, p300x2-1m) | 0 / 6 | - |
  | release ifeval, 109 (shipped, p300x2-1m) | 3 / 109 | 65 (32,516), 106 (32,706): single-token loops; 87 (16-char unit x16,295, character rule only) |
  | release gpqa CI subset, 40 (shipped, p300x2-1m) | 0 / 40 | - |
  | gpqa full set, 198 (shipped, p300x2) | 3 / 198 | 196 (2,412), 170 (679), 79 (37): budget cuts |
  | H1 chemistry, 93 (stage-7 policy, p300x2) | 6 / 93 | 69 (6,980), 121 (322), 89 (316), 47 (131), 36 (56), 125 (23) |

  - Shipped policy: **6 / 353 = 1.7 %** of the generations loop; 5 of them strongly (8-gram or 16-character count
    >= 679).
  - All valid sets: 12 / 446 = 2.7 %. The invalid H1 attempt 1 (engine crash) is not counted (0 / 198 by the rule).
  - Every loop is a budget-exhausted answer (32,768 tokens for ifeval, 65,536 for gpqa). 3 of the 4 gpqa budget
    cuts with the shipped policy are loops by this rule (doc 127 is not).
  - Not scannable: the H2 card-protocol run saved only the last 300 content characters (0 loops in them; 8 budget
    cuts, whose think text is not saved); the loop diagnostic saved no reasoning text.

### Repetition loops: mitigation probe on the bundle (stage-11 review, P2-2)
- `scripts/11-loopprobe.py`, run inside `11-verify-rel.sh` (`LOOPPROBE=...`) on the published bundle (image
  34ec4c80e68b, profile p300x2, warm), 2026-10-23 09:42-10:03 and 10:04-10:23. Prompts: ifeval docs 106 and 65;
  reasoning_effort low, top_p 0.95, 1,024 tokens, seeds 1-32 per prompt and condition, 16 requests in flight.
  Loop rule as in the scan above. Results (`artifacts/s11/loopprobe/loopprobe{,2}.json`):

  | condition | doc 106 | doc 65 | both |
  |---|---:|---:|---:|
  | temperature 1.0 (the eval setting) | 6/32 | 1/32 | 7/64 |
  | temperature 1.0, repetition_penalty 1.05 | 5/32 | 1/32 | 6/64 |
  | temperature 1.0, repetition_penalty 1.10 | 1/32 (+1 unclassified) | 2/32 | 3/64 |
  | temperature 0.6 | 9/32 (+2 unclassified) | 2/32 | 11/64 |

  - "Unclassified" = budget cut with empty content and no loop in the text.
  - repetition_penalty 1.05 does not reduce the rate. 1.10 gives 3/64 vs 7/64 (not significant at this n; doc 65
    goes up). A lower temperature does not help (11/64).
  - So the card recommends no sampling setting. It recommends a `max_tokens` bound, a client-side loop check and a
    retry with another seed (most seeds of a looping prompt do not loop).
  - The per-draw rate on doc 106 (6/32 on p300x2) is higher than in the eval (1 draw, looped) and in the loop
    diagnostic on p300x2-1m (0/6 classified). The loops are strongly prompt-dependent; the eval-wide rate is in the
    scan above.
  - The same sessions repeated the Hub smokes (`SMOKE_RESULT PASS`, burst 16/16) and exercised
    `tt-model pull <repo> --with-weights` with the weights cached: rc 0 in 26 s, "weights -> host HF cache" at
    `~/.cache/huggingface/hub/models--zai-org--GLM-5.3-Flash/snapshots/91b69215...`. The second session used a plain
    `tt-model pull` (no weight fetch); the container found the complete cached snapshot and served.

### HF teacher-forced control of a doc-106 loop onset (stage-11 review, P2-2)
- Tool: `~/glm53-flash-bringup/scripts/11-hfloop.py` (uses `tools/hf_full_ref.py` FullModel, the stage-6 HF CPU
  reference: unmodified HF `Glm5NextTextModel`, E192-masked experts dequantized from the FP8 checkpoint). The
  checkpoint is the parent snapshot @91b69215 + `tt/e192_mask.json` (a host symlink overlay;
  `CheckpointReader` applies the mask). Run 2026-10-23 10:23-10:30 with the device queue empty, under
  `systemd-run -p MemoryMax=40G`: 102 tokens in 389 s, peak RSS 39.6 GB (`--expert-cache-gb 12`).
- Trajectory: the looped doc-106 draw with the earliest single-token onset from the bundle probe (seed 9, base,
  host-sampled because it asked for logprobs): 65 prompt tokens (the HF tokenizer render matches the served
  count) + 23 reasoning tokens + content "Ôi chúa thay ở ai ai ai ...". The run of " ai" starts at completion
  position 30. TT probabilities = the served logprobs (raw model output, top 5). Output:
  `artifacts/s11/loopprobe/hfloop.json`.
- Result at the onset (p of the emitted token, TT / HF; top-1 of each side):

  | pos | token | TT p | HF p (HF rank) | TT top-1 | HF top-1 |
  |---:|---|---:|---:|---|---|
  | 26 | " ch" | 0.471 | 0.071 (1) | " ch" 0.47 | " trời" 0.81 |
  | 27 | "úa" | 0.960 | 0.784 (0) | "úa" | "úa" |
  | 28 | " thay" | 0.0101 | 0.00004 (551) | " " 0.090 | " may" 0.636 |
  | 29 | " ở" | 0.0012 | 0.0004 (63) | " một" 0.48 | " câu" 0.21 |
  | 30 | " ai" | 0.0055 | 0.0002 (104) | " đầu" 0.67 | " đầu" 0.28 |
  | 31 | " ai" | 0.087 | 0.100 (0) | " một" 0.29 | " ai" 0.10 |
  | 32-37 | " ai" | 0.31 / 0.24 / 0.86 / 0.88 / 0.92 / 0.83 | 0.61 / 0.24 / 0.84 / 0.83 / 0.79 / 0.70 (all rank 0) | " ai" | " ai" |

  Over all 38 scored positions: HF top-1 = TT top-1 at 81.6 %, the TT token is the HF argmax at 76 % and in the
  HF top 5 at 87 %, mean |p_TT - p_HF| of the emitted token 0.084.
- Conclusions:
  - The attractor is a model property: once two " ai" are in the context, the HF reference also predicts " ai" as
    its argmax at every repeat (p 0.10 rising to 0.84), as TT does.
  - The entry into the loop is not: it comes from three low-probability samples (positions 28-30). Under the HF
    reference these tokens are 25 to 250 times less likely than under TT. At position 28 the TT distribution is
    flat (top-1 " " 0.09; HF's top-1 " may" 0.64 is not in the TT top 5), and at 26 TT and HF rank " ch" / " trời"
    the other way. So at this prefix the TT logits deviate from the HF reference, and the deviation makes the loop
    entry more likely. The reasoning tokens (positions 0-23) agree well: 22/24 top-1 (the other two have top-1 p 0.28-0.41).
  - One onset only; the cause of the deviation (the bfp4 / LoFi precision policy, or the low-resource language) is
    not separated here. This answers the earlier open point ("whether the TT logits at this prefix deviate from the
    HF reference"): yes, at the loop entry.

### Non-English output: device precision A/B and HF controls (stage-11 review #2)
- Finding of the review: sampled non-English output (temperature 1.0) is often garbled. Doc 106 (Vietnamese): 20 of
  32 bundle draws mix scripts or are word salad (`loopdiag/loopprobe_summary.json`); release ifeval docs 18
  (Kannada), 50 (Marathi) and 87 (Persian, the two-character loop) are broken. English is clean.
- Question: is the cause the stage-8 precision policy (the bfp4 / LoFi non-expert groups or the bfp4 LM head), the
  decode path, or something shared by every policy (the bfp4 routed experts, which every policy needs to fit)?
- Tools (copies in `nonenglish/`): `11r2-tf-device.py` scores fixed token texts on the device under one policy
  (`GLM53_PRECISION`), through prefill (one forward over the whole text) and through traced decode (one forced
  token per step), and saves the top-20 next-token distribution per position. `11r2-tf-hf.py` is the same on the
  HF CPU reference (`tools/hf_full_ref.py`, bf16). `11r2-tf-compare.py` compares them.
- Texts (`nonenglish/texts.json`): the saved seed-9 doc-106 trajectory (65 prompt + 38 tokens, the rows of
  `loopdiag/hfloop.json`), and four hand-written answer paragraphs after a chat prompt (reasoning_effort low,
  `</think>` forced): Vietnamese (127 scored positions), Persian (130), Chinese (84), English control (100).
- Device run (`11r2-tfab.sh`, 2026-10-23 11:10-11:26, ticket glm53-s11-tfab): code = `git archive` of d16b35f3094
  (md5 of `tt/*.py` in `nonenglish/tfab_device.log`); 1 slot, max_seq_len 4,096; the bundle's converted weight
  cache, read only (the cache guard checks the key and the checkpoint fingerprint first; 0 tensors written). The
  policies only re-quantize or change fidelity on top of the cached bfp8 weights, so no new cache was needed.
  Policies: shipped (`combo_a_kv_dense4_g4`); the shipped policy with the LM head at bfp8 (HiFi2 = the stage-7 LM
  head, and LoFi = `lm8_lofi`); `GLM53_PRECISION=none` (the stage-7 policy: every non-expert group bfp8, bf16 DSA
  latent; the highest-precision policy that fits).
- HF reference: `11r2-tf-hf.py`, 2026-10-23 11:26-11:59, device queue empty, `systemd-run -p MemoryMax=40G`, peak
  RSS 39.4 GB.
- Result, top-1 agreement with HF (TT argmax = HF argmax), prefill / traced decode:

  | text (positions) | shipped | LM head bfp8 HiFi2 | LM head bfp8 LoFi | none (stage-7) |
  |---|---|---|---|---|
  | doc 106, positions 24-31 (8) | 0.38 / 0.38 | 0.50 / 0.50 | 0.50 / 0.50 | 0.38 / 0.38 |
  | doc 106, all (38) | 0.74 / 0.82 | 0.76 / 0.74 | 0.76 / 0.76 | 0.79 / 0.79 |
  | Vietnamese paragraph (127) | 0.80 / 0.77 | 0.81 / 0.79 | 0.79 / 0.79 | 0.80 / 0.83 |
  | Persian paragraph (130) | 0.70 / 0.73 | 0.70 / 0.72 | 0.70 / 0.71 | 0.71 / 0.67 |
  | Chinese paragraph (84) | 0.73 / 0.79 | 0.74 / 0.77 | 0.74 / 0.77 | 0.79 / 0.79 |
  | English paragraph (100) | 0.80 / 0.80 | 0.84 / 0.81 | 0.82 / 0.81 | 0.86 / 0.85 |

  - The TT argmax is in the HF top 5 at 0.92-1.00 of the positions for every text and policy (doc 106 window:
    0.62-0.88).
  - Top-5 probability mass, TT (shipped, decode) / HF: doc 106 window 0.73 / 0.77, all doc 106 0.90 / 0.90,
    Vietnamese 0.82 / 0.82, Persian 0.56 / 0.56, Chinese 0.79 / 0.85, English 0.86 / 0.89. On these texts the TT
    distributions are not flatter than HF; only Chinese is slightly flatter.
  - Shipped traced decode = the served bundle on doc 106 (top-1 0.38 / 0.82, the same per-position values), so
    the decode path and the serving path add nothing. Prefill and decode agree within the noise.
  - At doc-106 position 28 every policy misses HF's top-1 " may" (p 0.64): shipped " " 0.09, LM head bfp8
    " " 0.10, none " " 0.69.
  - Decode step at 1 slot (model + sampler trace, `s7_probe.run_decode`): shipped 49.3 ms, LM head bfp8 HiFi2
    49.5 ms, LM head bfp8 LoFi 49.5 ms, none 51.4 ms.
- Interpretation:
  - No policy restores agreement. The highest-precision policy (none) gains about 0.05 top-1 on English and Chinese,
    nothing on Vietnamese and Persian, and nothing at the doc-106 loop entry. The LM head at bfp8 gains at most one
    position of eight at the loop entry (noise at n = 8) and 0.00-0.04 elsewhere.
  - The TT-HF top-1 gap is 0.15-0.30 on these hand-written texts in every language, also English (the stage-8 gate
    texts are HF greedy continuations, where top-1 is 0.91-0.95). English is only somewhat better (0.80 vs
    0.70-0.80).
  - So the deviation comes from a part that every policy shares. The parts outside the stage-8 policy are the bfp4
    LoFi routed experts (bfp8 experts in all layers do not fit), the fp32 KDA / mHC kernels and the DSA kernels.
  - Direct test of the bfp4 experts: the HF reference with every routed expert weight rounded to bfp4 on the port's
    grid (`11r2-tf-hf.py --expert-fmt bfp4`; `tests/hf_ref/bfp.py`, bit-exact to ttnn's packing; everything else
    bf16), 2026-10-23 12:48-13:31, 40 GB cap, expert cache 4 GB. On doc 106 (`nonenglish/cmp_bfp4.json`):
    - HF-bfp4 vs HF: top-1 0.38 at positions 24-31 and 0.79 over all 38, the same as TT vs HF (0.38 / 0.74-0.82).
    - At position 28 HF-bfp4 also predicts " " (0.17; HF " may" 0.64), and at 29 / 30 it predicts ' "' (0.90 /
      0.23), as the TT none policy does (" " 0.69, ' "' 0.67 / 0.16).
    - TT none vs HF-bfp4: top-1 0.88 at positions 24-31 (decode) and 0.92 over all. TT shipped vs HF-bfp4: 0.38 /
      0.82 (the stage-8 bfp4 non-expert groups move this fragile prefix again, but not toward HF).
    - Conclusion (one 38-token Vietnamese trajectory): **the bfp4 routed experts are enough to cause a deviation of
      this size** (bfp4 expert weights alone, with bf16 math, deviate from HF as much as TT, and the TT `none`
      policy tracks HF-bfp4 at 0.92). The shipped policy's bfp4 / LoFi non-expert groups and fp8 DSA latent add a
      second deviation of similar size: TT shipped vs HF-bfp4 is 0.82 / 0.38, the same as vs plain HF. bfp8
      experts in all layers do not fit in DRAM, and the all-bfp8 non-expert policy does not bring TT closer to
      HF, so no precision setting fixes it in this release.
    - Raw per-position rows: `nonenglish/hf_bfp4exp.json` (HF-bfp4), `nonenglish/cmp_bfp4.json` (summary).
    - HF-bfp4 run: `11r2-tf-hf.py --expert-fmt bfp4`, doc 106 only, unbatched, under `systemd-run --user -p
      MemoryMax=40G` (unit glm53-s11-r2hfq2), 2,547 s, peak RSS about 32 GB. A first batched 5-text attempt (unit
      glm53-s11-r2hfb) was OOM-killed at the 40 GB cap.
    - `nonenglish/11r2-langprobe.py` (sampled non-English re-probe) was not run: the shipped precision did not
      change, so a re-probe cannot change the result. The earlier probe evidence stands.
    - The English control on HF-bfp4 was stopped after doc 106 (lead decision: it cannot change the conclusion).
- Decision: **the default precision does not change.** No candidate restores agreement, so a precision change has
  no benefit to pay for.
- What is still true: non-English sampled output at temperature 1.0 can be garbled (mixed scripts, word salad) and
  can enter loops; the TT next-token distribution deviates from the HF reference in every language (top-1 0.70-0.80
  with the shipped precision on hand-written text), more for Persian and Chinese than for English. Chinese (claimed by the E192 card) was
  checked on TT by this teacher forcing only: top-1 0.73-0.79, top-5 mass slightly below HF; sampled Chinese output
  was not probed.

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

## Weights on the host: which commands fetch them (found by the bundle verify; fixed after the stage-11 review)
tt-model (tt-model-manager 3695747, the version used here) puts the pinned weights into the host HF cache only in
these cases (`tt_kernel.container_cli.pull_container` -> `_download_weights` = `snapshot_download(repo, revision)`):
- `tt-model pull <repo> --with-weights`;
- `tt-model serve <repo>` when the bundle is not pulled yet (its automatic pull fetches the weights).
These commands do NOT fetch the weights:
- a plain `tt-model pull <repo>` (`with_weights=False` is the default, `cli.py` pull);
- `tt-model serve` of a pulled bundle or of a staged `tt_kernel_manifest.json`. It only prints "the model will
  download them inside the container at first load".
The download goes to `huggingface_hub.constants.HF_HUB_CACHE` on the host (default `~/.cache/huggingface/hub`). The
container gets the same directory: `container.compose_run` bind-mounts `HF_HOME` read-write at `/hf` with
`HF_HOME=/hf` (plus `/hf-hub` when `HF_HUB_CACHE` is outside `HF_HOME`), and forwards `HF_TOKEN` when the host
sets it. tt-model sets no `HF_HUB_OFFLINE` in the container. The parent repo is public, so no token is needed.

The problem (before the fix):
- `generator_vllm.resolve_checkpoint` read the snapshot with `snapshot_download(..., local_files_only=True)` only.
  After a plain `pull`, vLLM had fetched only the config and tokenizer files into the snapshot. The overlay then
  had no `model.safetensors.index.json`, and the boot failed with an opaque `FileNotFoundError` for
  `/weight-cache/glm53_ckpt_overlay/<rev>/model.safetensors.index.json` (stage-11 cold-verify attempt 2,
  2026-10-23; see "Cold attempts 1 and 2" below for the log status).
- A failed host fetch (attempt 1: a wrong PYTHONPATH broke huggingface_hub's import) is printed by `serve`, but the
  command still exits 0.
- tt-model-manager has no manifest field that makes `pull` fetch the weights by default or marks them as
  required (`WeightsRef` has only repo, revision, allow / ignore patterns and repo type).

The fix (stage-11 review, P2-1):
- `resolve_checkpoint` uses the local snapshot only when it is complete: `model.safetensors.index.json` and
  every shard it names. Otherwise it calls `snapshot_download(name, revision=rev)` (online) into the mounted HF
  cache, with a warning that names `tt-model pull <repo> --with-weights`. If that download fails (offline, no
  disk, network), it raises a `RuntimeError` that names `tt-model pull raahemnabeel/glm-5.3-flash-e192-blackhole
  --with-weights` and `hf download zai-org/GLM-5.3-Flash --revision <rev>`.
- Host test: `tests/test_generator_vllm_host.py::test_resolve_checkpoint_downloads_missing_weights` (mocked
  `snapshot_download`: partial snapshot + failed download -> the RuntimeError; no snapshot -> download + overlay;
  complete snapshot -> no download; index with a missing shard -> download). 26/26 host tests pass.
- The in-container download is slow: no progress in tt-model, and it counts against the engine start limit
  (`VLLM_ENGINE_READY_TIMEOUT_S` 7,200 s). At the measured 113 MB/s it takes about 48 minutes, so the first start
  fits in 2 hours (48 + 23 minutes conversion + kernel compile). The card recommends `pull --with-weights`.
- Not tested on a device: an empty-cache boot through the in-container download. This host has no disk space for a
  second 328 GB copy of the parent snapshot (about 96 GB free), and the cached snapshot cannot be removed. The
  Hub verify after the republish ran with the weights cached (see "Stage-11 review fixes").

## Bundle parent-snapshot path: first exercised by the stage-11 cold verify (bug fixed in da072e3719b)
- The bundle reads the 288-expert parent snapshot @91b69215 through `tt/e192_mask.json`. The env has
  `GLM53_CKPT_OVERLAY_DIR`, no `GLM53_CKPT`; `generator_vllm.resolve_checkpoint` builds a symlink overlay.
- Every earlier device run (stages 1-10, every stage-11 TTI / GPQA / diagnostic server) set `GLM53_CKPT` to the
  pruned E192 build, whose 192 experts are already renumbered (`CheckpointReader.mask` = None). So the parent path
  had never run on a device.
- No earlier `tt-model serve` of this bundle exists: the stage-11 verifies (2026-10-23) are its first container
  boots; logs/11 has no older staged / hub verify logs.
- Host coverage was the fake-config test of the overlay in `test_generator_vllm_host.py`, without a
  layer read.
- Cold-verify attempt 3 (06:13): the container died at the first `layer_state(0)` with
  `KeyError: 0` in `CheckpointReader._expert_name`. The mask has rows for the MoE layers 3-44 only, and dense
  layer 0 was looked up.
- Fix da072e3719b: `layer in self.mask`.
- Host checks on the downloaded parent through the overlay (lead; repeated here): all 46 layers list correctly.
  Dense 0-2 have no experts; layers 3-44 have 192 experts, every mapped name exists, and the router has 192 rows;
  MTP has 288. `build_e192_ckpt.py` copied the tensors verbatim, so the parent path reads the same tensors as the
  pruned build.
- A host regression test for the dense-layer case (`tests/test_ckpt_mask_host.py`) is committed in 4c3619165da
  (from `artifacts/s11/maskfix.patch`). It fails on the old code with KeyError 0 and passes with the fix.

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

## tt-model bundle: build and fresh-device verification
- Build: `scripts/11-package-rel.sh` (via `11-package-watch.sh`, host-RAM guard) from the clean release worktree.
  - Manifest copy with only `source.tt_metal` pointed at the worktree.
  - Final build from da072e3719b with plugin 16febac -> image `tt-model/glm-5.3-flash-e192-blackhole:e279dbf74b34`,
    staged at `~/glm53-flash-bringup/pkgbuild-rel/build/glm-5.3-flash-e192-blackhole/`.
  - The in-image verify steps pass. An earlier build from 09842218b22 (image 51370e9e44ad) is superseded by the
    ckpt fix.
- Disk: deleted the bring-up E192 checkpoint (212 GB), the bring-up weight cache (134 GB) with its symlink, and
  ~/.cache/tt-metal-cache (94 GB), lead-approved. Free went 131 -> 569 GB before the download.
- Cold verify, p300x2 (`scripts/11-verify-rel.sh`, `COLD=1`; attempt 4, 2026-10-23):
  - Fresh device: `tt-smi -r`, no holders. Bundle weight and kernel caches wiped, no host JIT cache.
  - Parent snapshot zai-org/GLM-5.3-Flash @91b69215 fetched the way `tt-model pull` does: 328 GB in 2,859 s
    (attempt 3).
  - First boot to healthy: **1,730 s**, of which the weight conversion took 1,379 s and wrote the 121 GB cache.
  - Served `max_model_len` 262,144 = serving_profiles.json.
  - Smokes PASS:
    - chat and glm47 tool call
    - logprobs, repetition penalty, min_p and seeded requests
    - greedy repeat identical
    - non-aligned prompt lengths 1-4,223
    - page growth
    - burst 16/16 identical to solo (66.6 tok/s)
    - max-effort thinking (haiku, train question)
    - near-limit 262,080-token prompt (218.6 s)
    - `tt-model curl`
  - VERIFY_RESULT PASS.
- Warm verify, p300x2-1m (staged bundle, same weight cache, 2026-10-23 07:0x-07:47):
  - Healthy after 211 s (warm); served `max_model_len` 1,048,576.
  - All smokes pass as in the cold run, and the near-limit **1,040,000-token prompt answers coherently in
    2,291 s** (`SMOKE_LONG 1040000 wall 2291.0s`).
  - `tt-model curl` OK.
  - The script's verdict is `VERIFY_RESULT FAIL staged-p300x2-1m`. The only failing gate is
    `SMOKE_BURST 1 ok, 64 tokens in 4.8s = 13.5 tok/s; 0/1 identical to solo greedy`. That is the documented
    block-placement numerics of the sharded layout (doc/vllm_integration/README.md "Block-sharded layout: greedy
    outputs and block placement"; stage 9 recorded burst 1/2), not a defect.
  - **Classified PASS with this documented exception (lead decision); not rerun.**
  - Attempts 1-3 failed before serving and are recorded above (verify-script PYTHONPATH leak, staged serve does not
    fetch weights, ckpt mask KeyError).
  - Cold attempts 1 and 2: no failure logs exist. Their serve and server logs were overwritten by the later attempts
    (the same file names), and their chain logs (`~/glm53-flash-bringup/logs/11/verify-cold-attempt{1,2}.chain.log`,
    started 05:01 and 05:13) end at `serve exit=0`, without an exit line (the jobs were stopped). The other
    evidence: `logs/11/disk.log` (free 569 / 567 GB before each serve, so no weight download happened), and the
    attempt-3 chain log, which is the first with the separate weight fetch (`weights fetch rc=0 after 2859 s`).
    The causes above are the diagnosis of the stage-11 agent at the time, not a saved error message.

## Bundle (parent-snapshot path) vs stage 9 (pruned build): bit-identical smokes
- The stage-9 served smokes ran on the pruned E192 checkpoint (`GLM53_CKPT`); the bundle reads the parent snapshot
  through the mask. Compare `doc/vllm/evidence/final/p300x2_smoke.log.txt` with `bundle/staged-p300x2-cold-smoke.log`,
  and `doc/vllm/evidence/final/1m_smoke.log.txt` with `bundle/staged-p300x2-1m-smoke.log`.
- Every deterministic smoke line is identical, text and logprobs: chat, tool-call arguments, logprobs (the same
  three logprob values to all printed digits), repetition penalty, min_p, seeded, greedy repeat, non-aligned
  lengths 1-4,223, page growth, max-effort thinking.
- The only differences: the random tool-call ids; the burst size (8 vs 16 on p300x2; 2 vs 1 on p300x2-1m, where the
  burst depends on block placement); the long-prompt length (a different smoke setting).
- So the parent-snapshot path loads the same model as the pruned build (consistent with the host check above:
  `build_e192_ckpt.py` copied the tensors verbatim).

## Hub README: "Runs on p300x2 or p300x2 or ..."
- The README line "Runs on **p300x2** or **p300x2** or **p300x2** or **p300x2** or **p150x4**" is a tt-model
  rendering issue: `tt_kernel/build.py` (tt-model-manager 3695747, line 1044) joins the hardware of every serve
  profile without removing duplicates. The manifest is correct: four profiles run on p300x2. No manifest change.

## Scorecard run on the bundle (`scripts/11-scorecard.sh`, 2026-10-23 07:48-08:50, one ticket, `tt-model serve` per profile)
- Method: the staged bundle (image e279dbf74b34), warm cache. Rows from `10-perf-sweep.py`: greedy, random-token
  prompts as token ids, output length pinned. JSONs in `~/glm53-flash-bringup/artifacts/s11/scorecard/` (kept out
  of this directory: they carry per-profile context values).
- p300x2 (16 x 262,144), healthy after 196 s:

  | input / output, users | TTFT | TPOT | output tok/s |
  |---|---:|---:|---:|
  | 128 / 128, 1 | 259 ms | 75.2 ms | 13.0 |
  | 8,192 / 256, 1 | 4.90 s | 79.0 ms | 10.2 |
  | 131,072 / 256, 1 | 93.8 s | 79.8 ms | 2.2 |
  | 128 / 128, 16 | 3.87 s | 178.0 ms | 77.3 |
  | 1,024 / 256, 16 | 12.2 s | 179.8 ms | 70.6 |

  - CI-style 100 / 100 at 32 concurrency (`vllm bench serve`, 64 prompts, 16 slots): 64/64 succeeded, 74.9 output
    tok/s, mean TTFT 19.8 s (queueing), mean TPOT 177.9 ms.
  - Preemption load test (09-slotleak t2: 16 x 20,000-token prompts x 512 tokens, beyond the pool): PASS. 16/16
    finished with 512 tokens, num_preemptions 0 -> 1, the control prompt is equal to its solo run, 402.8 s.
- p300x2-wide (32 x 159,744), healthy after 241 s:

  | input / output, users | TTFT | TPOT | output tok/s |
  |---|---:|---:|---:|
  | 128 / 128, 1 | 274 ms | 124.5 ms | 7.9 |
  | 128 / 128, 32 | 7.98 s | 165.5 ms | 141.1 |
  | 1,024 / 256, 32 | 24.7 s | 166.0 ms | 122.3 |

  - Smokes PASS, burst 16/16 identical to solo, near-limit 159,680-token prompt in 120.8 s.
  - Served batch-invariance (`tools/vllm_served_invariance.py`): PASS.
- p300x2-long (1 x 344,064), healthy after 226 s: 128 / 128 TTFT 245 ms, TPOT 49.9 ms (20.1 tok/s). Smokes PASS,
  near-limit 344,000-token prompt in 310.8 s.
- p300x2-1m: the TTI release benchmark rows above (128 / 128 TTFT 251 ms, TPOT 71.5 ms), plus the warm verify's
  1,040,000-token prompt in 2,291 s.
- Multi-user TPOT on p300x2 (178 ms at 16 users) is in line with stage 10 (172.6 ms at 16 users, doc/optimized_vllm README).
- The card's Performance, First-start and Accuracy sections are updated from these numbers (hunk
  `artifacts/s11/hunks/tt-model-card-perf.diff`, lead commits).

## Release report with the GPQA waiver (aggregation only)
- TTI v0.22.0 has no standalone `reports` workflow. The release report is assembled in-process from the run's
  blocks.
- The waiver is recorded the TTI way: a `known_issues` entry (workflow_type EVALS, task gpqa_diamond_cot_zeroshot)
  in the GLM dev-catalog entry of `tti_local_integration.patch`. Its reason quotes the RUN_NOTES evidence above.
- The release report was then re-aggregated from its saved report data with TTI's own
  `acceptance_criteria_check` + `build_acceptance_export` + `ReportGenerator` (`11-tti-regen-report.py`). No
  device, no server, and no raw output was re-collected.
- Command: `MODEL_SPECS_ENV=dev PYTHONPATH=<tti> .workflow_venvs/.venv_workflow_run_script/bin/python
  11-tti-regen-report.py workflow_logs/reports_output/release/data/report_data_..._21-23-04.json
  workflow_logs/reports_output/release_waiver`
- Result: EXIT_CODE=0, **Acceptance PASS**.
  - Evals: PASS (gpqa_diamond_cot_zeroshot waived with the quoted reason; ifeval NA, no reference).
  - Spec tests: PASS (22/22). Benchmarks: NA (no targets).
- Report: `release/report_id_glm53-flash-autoport_GLM-5.3-Flash_p300x2_2026-10-21_21-23-04_waiver-regen.md`
  (+ `reports_output/release/data/`). The original Acceptance FAIL report stays next to it.
- Re-run after the stage-11 review (P2-3, 2026-10-23; host only, same command): the `known_issues` reason now says
  "the stage-8 non-expert precision changes are ruled out; the residual -5.6 points (about 1.8 sigma) exceeds the
  card's stated +/-2-3 point run noise and is unexplained; bfp4 vs NVFP4 routed experts were not tested". The
  decision is unchanged. The regen script now also marks the gpqa row "WAIVED (known issue)" in the markdown
  (TTI's renderer keeps the raw FAIL) and lists each waived row with its reason under the acceptance criteria.
  Result: Acceptance PASS, blockers {}.

## Stage-11 status
**release-readiness-ci-subset-pass, with the GPQA row issue-waived.**
- The TTI release workflow (ci-nightly subset, p300x2-1m = the 1,048,576 contract) ran end to end against the
  autoport, with no Docker.
- Accuracy rows are CI-subset results, not full-set accuracy (except the supplementary full GPQA runs):

| row | result | classification |
|---|---|---|
| ifeval (109 docs, effort low) | 74.31 % prompt-strict | no reference exists anywhere; reported as measured (decision (b)); 3/109 loops = model repetition attractor (all saved generations: 6/353 = 1.7 % with the shipped policy) |
| gpqa_diamond_cot_zeroshot | 70.0 % (CI subset); full set 66.67 %; card protocol 74.24 % vs card 79.8 % | issue-waived: the stage-8 non-expert precision changes are ruled out; the residual -5.6 points (about 1.8 sigma) exceeds the card's stated +/-2-3 noise and is unexplained; bfp4 vs NVFP4 experts were not tested |
| spec tests | 22/22 | pass (3 recorded TTI harness patches) |
| benchmarks (12 rows) | all requests completed | informational (no targets) |

- Bundle: the cold fresh-device verify (p300x2), the p300x2-1m warm verify and the scorecard ran on image
  e279dbf74b34, built from da072e3719b (plugin 16febac). The cold verify passed; the p300x2-1m warm verify passed,
  with the documented placement exception; the scorecard passed on every profile. The published image
  34ec4c80e68b is built from 6853aab7971; the `tt/` code is the same as in da072e3719b (the later commits change
  a test, the card and docs only). Only the Hub verify ran on 34ec4c80e68b. After the stage-11 review, the bundle was rebuilt
  and republished (see "Stage-11 review fixes").

## Hub publish
- `scripts/11-publish.sh`: `tt-model push <staged> --repo raahemnabeel/glm-5.3-flash-e192-blackhole --public --publish`
  (lead go, 2026-10-23 09:02). Upload took 35.9 s.
- Result: https://huggingface.co/raahemnabeel/glm-5.3-flash-e192-blackhole rev
  **8b8a71f8428fabeadfd30bb4a48c69c4df743d89**, public, in the tt-model catalog (`tt-model-catalog` tag).
- Bundle: image `tt-model/glm-5.3-flash-e192-blackhole:34ec4c80e68b`, from tt-metal 6853aab7971 + vllm-tt-plugin
  16febac.
- The script's post-push check first reported rc 1. This was a checker bug: it counted the uploader's own
  `.cache/huggingface` metadata as staged files. Corrected recount (checker fixed to skip `.cache/`): 876 staged
  files, 877 on the Hub (+ .gitattributes), 0 missing, 0 LFS sha256 mismatches. Not re-pushed.

## Hub verify (2026-10-23 09:03-09:11, `scripts/11-verify-rel.sh raahemnabeel/glm-5.3-flash-e192-blackhole p300x2`)
- `tt-model pull raahemnabeel/glm-5.3-flash-e192-blackhole`: rc 0 in 35 s. This plain pull fetched no weights
  (correction, stage-11 review: the weights were in the HF cache from the cold verify's fetch; a plain pull does not
  download them, see "Weights on the host").
- Warm serve of the Hub bundle on p300x2: healthy after 231 s; `/v1/models` 262,144.
- Smokes PASS: chat, tool, sampling parameters, non-aligned lengths, burst 16/16 identical to solo (66.9 tok/s),
  max-effort thinking, an 8,192-token prompt.
- `tt-model curl`: "Hello, it's wonderful to see you here!".
- VERIFY_RESULT PASS hub-p300x2. The container was stopped and the device reset by run-locked.

## Stage-11 review fixes (2026-10-23)
- P2-1 weights fetch: `resolve_checkpoint` downloads an incomplete / missing parent snapshot, or raises a clear
  error that names `tt-model pull <repo> --with-weights`; host test; card First start (see "Weights on the host").
- P2-2 loops: corrected GPQA and loop-diagnostic statements; n-gram loop rate over all saved generations; the
  mitigation probe; the HF teacher-forced control; a card Limitations line. Copies of the results in this directory:
  `loopdiag/loopscan.json`, `loopdiag/loopprobe_summary.json` (without token ids), `loopdiag/hfloop.json`.
- P2-3 GPQA waiver wording: RUN_NOTES, the TTI `known_issues` reason, the re-aggregated report (gpqa row shown as
  waived, with the reason).
- P3: stale RUN_NOTES lines, cold attempts 1-2 log status, card performance footnote and 1m prefill, disk units,
  first-start note (`--detach`, `tt-model logs -f`), the yaml comment, "spot-check request", structured-output
  benchmarks (not applicable), the plugin line, RELEASE_CHECKLIST and PLAN, the bit-identical smoke comparison, the
  `reasoning_effort` note, the README hardware line.
- Commit of the fixes: dd6542a592e (code, test, card, docs).
- Rebuild: release worktree `~/tt-metal-rel-glm53` fast-forwarded to dd6542a592e (clean);
  `scripts/11-package-rel.sh` (via `11-package-watch.sh`, MemoryMax 40G) with vllm-tt-plugin 16febac, 10:32-10:39:
  host tests 26 passed, `package exit=0`, image `tt-model/glm-5.3-flash-e192-blackhole:a3925c21fe33`
  (`bundle/package-rel-r2.log`).
- Republish: `scripts/11-publish.sh` (the same flags: `--public --publish`; the checker skips `.cache/`), 10:40:
  Hub rev **4d1085f0b72d8ac696f751d69fd3c5e820d7ef23**, public, tt-model catalog; 876 staged files, 877 on the Hub
  (+ .gitattributes; the old image blobs were dropped by `tt-model push`), 0 missing, 0 LFS mismatches
  (`bundle/publish-r2.log`).
- Hub verify on the device, 10:41-10:49 (`TAGSUF=-r2 LONG=8192 11-verify-rel.sh raahemnabeel/glm-5.3-flash-e192-blackhole
  p300x2`, `bundle/verify-hub-r2.chain.log`):
  - `tt-model pull <repo> --with-weights`: rc 0 in 33 s; "weights -> host HF cache" (cached: metadata check only);
    the image a3925c21fe33 was already loaded from the local build (tt-model compares the digest, so no re-download).
  - Warm serve p300x2: healthy after 245 s; `/v1/models` 262,144; the adapter logged the complete cached snapshot
    (`checkpoint: /hf/hub/models--zai-org--GLM-5.3-Flash/snapshots/91b69215...`), no download.
  - Smokes PASS (chat, tool, sampling parameters, non-aligned lengths, burst 16/16 identical to solo at 66.9 tok/s,
    max-effort thinking, an 8,192-token prompt); `tt-model curl` exit 0. **VERIFY_RESULT PASS hub-p300x2-r2.**
  - Not tested: a true empty-cache start (in-container download, or `--with-weights` downloading 328 GB). This
    host has about 75 GB free, not enough for a second copy of the parent snapshot. By the tt-model code
    (3695747), `pull --with-weights` and the auto-pull of `serve <repo>` call `snapshot_download` into
    `huggingface_hub.constants.HF_HUB_CACHE`, the directory the container mounts at `/hf` (see "Weights on the
    host"); the in-container path is covered by the host test only.

## Stage-11 review #2 fixes (2026-10-23)
- P2 non-English output: device precision A/B and HF controls, see "Non-English output: device precision A/B and
  HF controls (stage-11 review #2)". Cause: the bfp4 routed experts (needed to fit p300x2)
  are a sufficient cause; the shipped bfp4 non-expert groups and fp8 latent add a deviation of similar size (see the
  conclusion in that section; measured on one Vietnamese trajectory). Default precision
  unchanged. Card: a Languages limitation.
- Loop scan: character rule (16-character substring >= 1,000 times); release ifeval 3/109 (doc 87 added), shipped
  policy 6/353 = 1.7 %, all valid 12/446 = 2.7 % (`loopdiag/loopscan.json`). The C-Eval explanation is replaced by
  the control result.
- P3: card loop wording (seen at temperature 1.0 and 0.6), warm-start range 196-245 s, a note that the generated
  README's "several minutes" does not hold for the first start; `resolve_checkpoint` re-points stale or broken
  overlay symlinks, and the snapshot errors name the missing shard files (host tests, 26 passed); the checkpoint
  log lines of the Hub verifies in `bundle/hub-p300x2-r{2,3}-checkpoint.log`; PLAN records reviews #1 and #2.
- Commit 75134275232. Release worktree fast-forwarded (clean). `11-package-watch.sh`, 13:00-13:11: host tests 26
  passed, `package exit=0`, image `tt-model/glm-5.3-flash-e192-blackhole:b9e8208f95ce`, vllm-tt-plugin 16febac
  (`bundle/package-rel-r3.log`).
- Republish (`11-publish.sh`, same flags), 13:12: Hub rev **0a807cdf243c2b081cf4666565913d8d5f567da5**, public,
  tt-model catalog; 876 staged files, 0 missing, 0 LFS mismatches (`bundle/publish-r3.log`).
- Hub verify, 13:31-13:40 (`TAGSUF=-r3 LONG=8192 11-verify-rel.sh <repo> p300x2`, `bundle/verify-hub-r3.chain.log`):
  `pull --with-weights` rc 0 in 31 s; warm serve healthy after 227 s (same weight cache key, no conversion);
  smokes PASS (burst 16/16 identical to solo, 66.5 tok/s); `tt-model curl` exit 0. **VERIFY_RESULT PASS
  hub-p300x2-r3.**
- The precision did not change, so the scorecard and the teacher-forcing gate rows stand. The non-English re-probe
  was skipped (lead decision: the model math is unchanged).
- The card text in this commit adds the bfp4-expert attribution after the publish. The Hub card of rev 0a807cdf
  has the A/B result without the attribution; it is not wrong, so it was not republished (card / docs only).

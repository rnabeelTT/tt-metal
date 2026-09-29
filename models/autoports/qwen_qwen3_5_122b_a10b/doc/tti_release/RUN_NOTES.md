# Qwen3.5-122B-A10B — TTI release run notes (stage 11)

## Server mode and host
- Server: the autoport vLLM server started from this tt-metal checkout by the readiness runner (external-server topology;
  TTI is a client). Command (`~/qwen35-122b-bringup/scripts/11-tti-chain.sh`):
  `python -m readiness_check.run_vllm_server --model-dir models/autoports/qwen_qwen3_5_122b_a10b --hf-model Qwen/Qwen3.5-122B-A10B
  --mesh-device P300x2 --stages serve --max-num-seqs 32 --max-model-len 262144 --block-size 64 --port 8131 --server-timeout 3000
  --tt-config '{"trace_region_size": 805306368, "l1_small_size": 24576, "fabric_config": "FABRIC_1D"}'
  --additional-server-args "--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_xml
  --default-chat-template-kwargs '{\"enable_thinking\": false}'"` (the stage-10 serving configuration + the production
  parsers; thinking OFF by default. The served bundle adopted the same default after run 7 — attempt 9 in the bundle
  section; the earlier bundle attempts 7-8 served the model's default, thinking on). Env: `MESH_DEVICE=P300x2 HF_MODEL=Qwen/Qwen3.5-122B-A10B
  QWEN35_PROGRAM_GUARD=1 QWEN35_TRACE_REGION_MIB=768 QWEN35_DECODE_BUCKETING=1`.
- Host: this p300x2 box (2 x p300c = 4 Blackhole dies, (1,4) mesh; `tt-smi -ls --local` in `logs/11/tt-smi-ls.txt`), devices
  reset with `tt-smi -r` before the server and after the run. No Docker, no tmux; the chain runs under `setsid nohup`
  with the workspace device lock.
- Context: served `max_model_len` 262144 = `doc/context_contract.json` (`GET /v1/models` in the chain log); 32 sequences.
- tt-metal: branch `qwen3_5_122b_a10b`, stage-10 commit 191f7f672f3 (+ the stage-11 working-tree changes: optional
  `readiness_check` import in `tt/generator.py`, `tt-model.yaml`, `vllm_bundle/`, `doc/tti_release/`, `doc/HANDOFF.md`); vLLM 0.26.0 + plugin 5961e48 (+ local registration commit 92403f3).

## TTI checkout and integration
- `~/tt-inference-server` tag v0.22.0 (`9ef9d660`). v0.22.0 gates `--model` on its catalog and looks evals up by HF id, so a
  LOCAL integration patch was applied (not pushed; `tti_local_integration.patch` here = the full `git diff` of the TTI checkout, all revisions folded in: impl, dev catalog
  template, eval config incl. the spec-test matrix, timeout, task / sampling knobs and the ci-nightly subset references): an `ImplSpec`
  `qwen35_122b_autoport` with `code_path=models/autoports/qwen_qwen3_5_122b_a10b`, a dev-catalog template
  (`workflows/model_specs/dev/llm.yaml`: P300X2, max_concurrency 32, max_context 262144, the stage-10 TT config and env,
  parsers qwen3 / qwen3_xml) and an `EvalConfig` for `Qwen/Qwen3.5-122B-A10B` with `ifeval` and `gpqa_diamond_cot_zeroshot`
  (chat API, greedy, max_gen_toks 2048 / 4096, CI_NIGHTLY limit 0.2, published scores 93.4 / 86.6 = the model card's
  thinking-mode numbers; no full-set GPU / HF reference exists, the ci-nightly subsets got same-setting references in run 8b). The `meta_ifeval` / `meta_gpqa_cot` tasks are the Llama-3
  prompt-format ports (EVALS_META venv, `apply_chat_template=False`); for a Qwen chat model the standard lm-eval `ifeval`
  and `gpqa_diamond_cot_zeroshot` tasks through the chat template are the equivalent rows.
- Patch revisions after review round 1 (the copied patch is the final state): the dev-catalog entry's `status` is
  FUNCTIONAL (EXPERIMENTAL had made every eval informational, see the eval rows), both eval tasks carry
  `model_kwargs.timeout = 7200` (lm-eval's 1800 s default produced the run-3 client timeouts), `QWEN35_EVAL_TASKS` can
  restrict a run to named tasks (used by the thinking-mode gpqa run), and the eval-config comment names the real
  acceptance mechanism. Runs 1-4 were made with the earlier revision (status
  EXPERIMENTAL, no timeout override); run 5 onwards with this one.
- Runs use `--dev-mode` (dev catalog) with `--service-port 8131 --no-auth --skip-system-sw-validation`; TTI's written run
  spec (copied here) carries `docker_server=false`, `local_server=false`, `service_port=8131`,
  `impl.code_path=models/autoports/qwen_qwen3_5_122b_a10b`.

## Smoke (external-server, no Docker)
- `curl /health` OK; `/v1/models` -> `Qwen/Qwen3.5-122B-A10B`, max_model_len 262144; chat "The capital of France is Paris."
  (finish stop); tool call through `qwen3_xml`: `get_weather({"city": "Paris"})` (`logs/11/smoke-chat.json`, `smoke-tool.json`).
- TTI: `python3 run.py --dev-mode --model Qwen/Qwen3.5-122B-A10B --device p300x2 --workflow benchmarks --service-port 8131
  --no-auth --skip-system-sw-validation --disable-trace-capture --limit-samples-mode smoke-test` -> exit 0; run spec
  `docker_server=false`, `local_server=false`, `impl.code_path=models/autoports/qwen_qwen3_5_122b_a10b`; benchmark JSON
  `benchmark_..._isl-16_osl-4_maxcon-1_n-8.json`: completed 8, failed 0 (the smoke-test preset = 8 requests of 16 / 4
  tokens), mean TTFT 314.4 ms, mean TPOT 36.48 ms (`disable_trace_capture=True`; ungraded, no perf targets; the copied
  `smoke/benchmark_*.json` is attempt 3's — attempt 2's smoke, 303 / 36.3 ms, was not kept).

## Release workflow
- Command: `python3 run.py --dev-mode --model Qwen/Qwen3.5-122B-A10B --device p300x2 --workflow release --service-port 8131
  --no-auth --skip-system-sw-validation --limit-samples-mode ci-nightly` (log `logs/11/tti-release.log`).
- Runtime estimate for the UNRESTRICTED evals from the stage-10 serving numbers (32 concurrent, ~125 ms per token at width
  32, ~13 s of serialised prefill per 32 admissions): ifeval 541 prompts x ~200 output tokens ~ 17 batches x ~40 s ~ 12 min;
  gpqa_diamond_cot_zeroshot 198 prompts x ~1000 tokens ~ 7 batches x ~130 s ~ 15 min. ci-nightly (limit 0.2 = ~108 and ~40
  prompts) was selected first as the Stage-11 configuration; results are CI-SUBSET results, not full-set accuracy.
- Result: `run.py` EXIT_CODE=0 (release workflow exit code: 0, `logs/11-tti-chain-run2.log` "tti release exit=0"); report
  `release/report_Qwen__Qwen3.5-122B-A10B_2026-09-26T112618+0000.md` (+ `report_data_*.json`, `runtime_model_spec_release.json`):
  **Acceptance: PASS (0 blockers)**, model status EXPERIMENTAL, Benchmarks NA (23 sweep points reported, none graded: no perf
  targets exist for this model), Evals "PASS" (1/2 passed; the gpqa FAIL is informational at status EXPERIMENTAL — see the eval rows), Spec Tests NA
  (see below), Agentic NA.
  Wall time 11:06 -> 13:15 UTC (2 h 9 min: evals 20 min, 23 benchmark points 1 h 49 min).

| exit-code table | command | exit |
|---|---|---|
| server health / chat / tool call | curl (chain) | 0 |
| TTI smoke | `run.py ... --workflow benchmarks --disable-trace-capture --limit-samples-mode smoke-test` | 0 |
| TTI release, run 2 (status EXPERIMENTAL) | `run.py ... --workflow release --limit-samples-mode ci-nightly` | 0 |
| release report acceptance | (in report) | PASS |
| TTI spec tests, thinking-default server (run 3) | `run.py ... --workflow spec_tests` | 1 (harness: `content` is null for reasoning-only responses) |
| TTI spec tests, release configuration (run 4) | `run.py ... --workflow spec_tests` | 0, Acceptance PASS, 22 / 22 |
| TTI evals, thinking mode (run 3) | `run.py ... --workflow evals --limit-samples-mode ci-nightly` | 143 (stopped at the gpqa boundary; ifeval result invalid: client timeouts) |
| TTI evals, FULL SET, release configuration, status FUNCTIONAL (run 5) | `run.py ... --workflow evals` (no limit) | 1 (Acceptance FAIL: both rows below the 0.95 ratio) |
| TTI evals, FULL SET, thinking mode (run 6) | `run.py ... --workflow evals` (no limit; `QWEN35_EVAL_SAMPLING`, 16K / 32K budgets) | 124 (chain's 10 h `timeout`; ifeval done, gpqa 68 / 198, no TTI report) |
| thinking-length probes (run 7) | `scripts/11-think-probe.sh`, `11-loop-precision-chain.sh` | 0 (probes); 3 alternative-policy servers did not come up |
| HF CPU controls (run 7) | `scripts/11-hf-think-ref{,2}.sh` | 0 |
| bundle serve from the staged manifest (attempt 7) | `scripts/11-serve-bundle-run2.sh` | 0 |
| Hub verification rounds 1 / 2 / 3 | `scripts/11-verify-hub{,2,3}.sh` | 0 / 0 (after two false starts, exit 1 each) / 0 |
| thinking probes with the general preset (run 8a) | `scripts/11-think-probe-general.sh` | 0 |
| same-setting HF control (run 8b) | `scripts/11-hf-same-setting-control.sh` (two processes) | 0 (stopped after the union covered all docs) |
| TTI evals, thinking mode, general preset (run 9) | `run.py ... --workflow evals` | 124 (15 h chain budget; ifeval done, gpqa 41 / 198) |
| TTI release, run 10 (FUNCTIONAL, subset references; gpqa reference later found mis-scored) | `run.py ... --workflow release --limit-samples-mode ci-nightly` | 0, Acceptance PASS (superseded) |
| **TTI release, run 11 (FUNCTIONAL, corrected subset references) — the report of record** | `run.py ... --workflow release --limit-samples-mode ci-nightly` | **0, Acceptance PASS, nothing waived** (`logs/11-release11-chain.log`) |
| token-count / rescoring scripts (review rounds 3-4) | `scripts/11-rescore-hf-control.py`, `scripts/11-token-counts.py` in TTI's venv | 0 |
| `11-tti-release.check.sh` | file check | 0 (`~/qwen35-122b-bringup/logs/11/check.log`) |

### Eval rows (CI-SUBSET results, ci-nightly limit 0.2; NOT full-set accuracy; server default thinking OFF in this run)
| task | samples | score | published (thinking mode) | ratio | TTI accuracy check |
|---|---|---|---|---|---|
| ifeval (prompt_level_strict_acc; TTI's `result_keys` lists inst_level_strict too but the score is the first key) | 109 | 92.66 | 93.4 | 0.992 | PASS |
| gpqa_diamond_cot_zeroshot (exact_match, flexible-extract) | 40 | 77.5 | 86.6 | 0.895 | FAIL (informational only because status was EXPERIMENTAL; see below) |

The gpqa row: 31 / 40 correct on the CI subset in NON-thinking greedy mode against a THINKING-mode published number; one item
of a 40-sample subset moves the score by 2.5 points (one standard error ~6.6 points), so the subset cannot resolve a 9-point
gap, and the methodology differs. TTI reported the release as PASS with the row "waived" — but that "waiver" is not row-specific: with no GPU
reference score TTI grades the ratio to the published score (tolerance 0.05, threshold 0.95) and marks the row FAIL
(`report_data ... sections[1].data.accuracy_check = 3`, ratio 0.8949); the dev-catalog `status: EXPERIMENTAL` then makes
every eval informational (`ModelStatusTypes.evals_enforced` is False for EXPERIMENTAL, `acceptance_criteria_metadata
.categories[1].waived = "Accuracy check failed. (informational: model status=EXPERIMENTAL)"`). So the ci-nightly release
report's "Acceptance: PASS (0 blockers)" grades nothing about accuracy; review round 1 flagged this. Consequence: the
dev-catalog status is now FUNCTIONAL (evals enforced) and the row is settled by the same-setting reference (run 8b) in run 11. A third run repeats both evals with the model's default
THINKING mode (server without the default-chat-template override; max_gen_toks 16384 / 32768; same ci-nightly subset) so the
row can be compared with the published methodology (`logs/tti-evals-thinking.log`; outcome: run 3 below — invalid as accuracy evidence).

### Spec tests
TTI's `spec_tests` were a no-op in the release run ("No spec test suites match model='Qwen3.5-122B-A10B' device='p300x2'":
the suite matrix `test_module/test_suites/llm.json` keys on a fixed model list). The local patch now adds the model to the
`VLLMParamConformanceTest` matrix and `server_tests_config.json`; the third run executed `--workflow spec_tests` against the
thinking-default server (run 3 below: harness FAIL on null `content`) and run 4 against the release configuration (22 / 22).

### Autoport implementation check
The evaluated code path is `models/autoports/qwen_qwen3_5_122b_a10b`: the TTI-written run spec (`release/runtime_model_spec_release.json`,
`smoke/runtime_model_spec_smoke.json`) carries `impl.impl_id=qwen35_122b_autoport`, `impl.code_path=models/autoports/qwen_qwen3_5_122b_a10b`,
`docker_server=false`, `local_server=false`, `service_port=8131`; the report metadata says `model_impl: qwen35-122b-autoport`; the served
process imported `models.autoports.qwen_qwen3_5_122b_a10b.tt.generator_vllm` (`logs/vllm-server-release-run.log`, with the per-layer
loading progress and periodic throughput lines filtered out to stay under the repo's 500 KB limit; full copy `~/qwen35-122b-bringup/logs/11/vllm-server.log`: `PROGRAM_GUARD active`,
`decode bucket widths [2, 4, 8, 16, 32]`, `PROGRAM_CACHE 1591 ... new: 0`). No `models/tt_transformers` or `models/demos` implementation
was launched or benchmarked (the `tt_transformers_impl` catalog entries were not selected: the run spec's model_id is
`id_qwen35-122b-autoport_Qwen3.5-122B-A10B_p300x2`).
Autoport implementation check: PASS (evaluated code path models/autoports/qwen_qwen3_5_122b_a10b).

### Recovery and hygiene
- Reset anomaly: before run 4 (spec tests) the chain's first two `tt-smi -r` attempts died with `Bus error`
  (`~/qwen35-122b-bringup/logs/11-spec-chain.log` lines 1-2); the chain's `reset()` retries three times and the third reset
  succeeded (the server came up healthy and ran 22 / 22 conformance tests). Likely cause: the resets overlapped the teardown
  of the failed bundle attempt-1 container / a concurrent manual reset (the same collision is recorded as a bring-up pitfall).
  The eval / probe chains written after this finding verify each reset with a bounded `tt-smi -ls --local`
  (the Hub verification chains check `/dev/tenstorrent` holders instead). No ARC / ERISC failures
  during the TTI runs; `tt-smi -r` before the server and after the chain; no leftover
  `EngineCore` / `api_server` processes (chain log), no tmux sessions, no Docker containers for TTI.
- Two earlier attempts are recorded honestly: attempt 1 (10:23) failed at `--model` (catalog gate, exit 2) before the
  integration patch; attempt 2 (10:41) lost both evals to `HF_HUB_OFFLINE=1` inherited from the workspace env and was stopped
  when an edit to the running chain script broke it (bash reads scripts incrementally); attempt 3 = run2 above is the release
  evidence. Attempt 2's benchmark rows were not kept.
- Server-log line `models.tt_transformers.tt.model_config: Inferring device name: P150x4` under `MESH_DEVICE=P300x2`: the shared
  `ModelArgs.__init__` derives the name from the device count and arch (4 Blackhole dies -> "P150x4"), and the settings that key
  on it in `tt_transformers/tt/model_config.py` are per `base_model_name` tables that do not contain this model; the autoport's
  own model args and the (1,4) mesh plan set every shape / grid, so the label is cosmetic here (present since stage 9).
- Pre-existing tmux sessions `ornith` / `ornith-wrapup` (2026-09-17/18, the earlier Ornith bring-up) and `qwen-bringup`
  (2026-09-24, this bring-up's stage-0 shell, idle) are not stage-11 artifacts; no tmux session was created by the stage-11 chains. A pre-existing `~/tt-inference-server/.env` (64 B, 2026-09-21)
  was neither read nor changed.
- Secrets: the HF token is read from `~/.cache/huggingface/token` into the environment only; the copied logs contain no token
  (`grep -c HF_TOKEN` = 0) and no eval prompt dumps beyond TTI's own progress lines.

## Run 5: FULL-SET evals, release configuration (thinking off, greedy), status FUNCTIONAL — FAIL on both rows
- Command: `python3 run.py --dev-mode --model Qwen/Qwen3.5-122B-A10B --device p300x2 --workflow evals --service-port 8131 --no-auth
  --skip-system-sw-validation` (NO `--limit-samples-mode`: `limit_samples_mode: None`, ifeval 541 / 541 and gpqa 198 / 198
  effective samples), server = the run-2 release configuration (`~/qwen35-122b-bringup/scripts/11-evals-full-chain.sh`; the
  reset was verified with `tt-smi -ls --local`), TTI patch revision 2 (status FUNCTIONAL -> evals enforced, timeout 7200 s).
  Log `logs/tti-evals-full-nothink.log`, report `reports_output/evals_full_nothink/report_*T201731*.md` (+ `data/`, both
  lm-eval `results_*.json`), run spec `run_specs/runtime_model_spec_evals_full_nothink.json` (`docker_server=false`,
  `local_server=false`, `service_port=8131`, `workflow=evals`, `impl.code_path=models/autoports/qwen_qwen3_5_122b_a10b`).
  Wall time 19:04 -> 20:17 UTC (ifeval 24 min, gpqa 49 min; the first 8 minutes overlapped the attempt-8 container build on
  the host CPUs).
- Result: `run.py` rc=1, **Acceptance FAIL (2 blockers)** — with status FUNCTIONAL the accuracy check is enforced and both rows
  miss the 0.95 ratio to the published (thinking-mode) scores:

| task | samples | score (non-thinking, greedy) | published (thinking mode) | ratio | TTI accuracy check |
|---|---|---|---|---|---|
| ifeval (TTI grades prompt_level_strict; inst_level_strict was 91.61) | 541 | 87.99 | 93.4 | 0.942 | FAIL (threshold 0.95) |
| gpqa_diamond_cot_zeroshot (exact_match, flexible-extract; the strict-match filter is 0.0 in every run because the model does not emit the `The answer is (X)` form it expects — TTI grades flexible-extract) | 198 | 68.69 | 86.6 | 0.793 | FAIL |

  The run-2 CI subset (92.66 / 77.5 on 109 / 40 items) was on the lucky side of both full-set numbers, as its standard errors
  allowed. No `__INFERENCE_ERROR__` responses in either task. Token counts (Qwen tokenizer over the decoded device text — the HF side counts generated non-pad tokens, a
  difference of at most one item near the cap; `reports_output/hf_same_setting_control_token_counts.py` -> `token_counts.log`): gpqa median 2852 generated tokens, **88 of 198 answers ran to the 4096-token budget, and 55 of the 62
  wrong answers are those cuts** — the non-thinking CoT score is budget-limited by the stage's own `max_gen_toks 4096`;
  ifeval median 207 tokens, 10 of 541 at the 2048 budget (3 of the 65 wrong).
- Interpretation (not a waiver): both published numbers are the model card's THINKING-mode scores with Qwen's recommended
  sampling; the run-5 server answered WITHOUT thinking and greedily, which is the release configuration the TTI spec tests
  passed on, not the published methodology. Qwen's own non-thinking figures for this family sit far below the thinking-mode
  GPQA number, and 45 % of the gpqa answers were cut at 4096 tokens, so the gap does not by itself indicate a port defect (the
  bring-up's teacher-forcing gates vs the HF model passed at top-1 0.97 / top-5 1.00 / top-100 1.00, stage 8), but it does mean the release-readiness gate is NOT met by
  this configuration. Run 6 repeats both tasks in thinking mode with the model card's decoding (below).

## Run 6: FULL-SET evals, thinking mode with sampling (the model card's coding preset — see the correction below)
- Server: the model default (thinking on, `--reasoning-parser qwen3`, no chat-template override), same TT config;
  `QWEN35_EVAL_SAMPLING` = do_sample, temperature 0.6, top_p 0.95, top_k 20, no presence penalty — which is the model card's
  "thinking mode for precise coding tasks" preset, NOT its "thinking mode for general tasks" preset (temperature 1.0, top_p
  0.95, top_k 20, presence_penalty 1.5; the card says the penalty is what reduces endless repetitions). Review round 2 caught
  this; run 6 therefore does not represent the published methodology, and the general preset is measured in run 8 below.
  Qwen documents that greedy decoding in thinking mode degrades and can loop. `max_gen_toks` 16384 (ifeval) / 32768 (gpqa), lm-eval request timeout 7200 s, full
  sets, status FUNCTIONAL (`scripts/11-evals-thinking-full-chain.sh`, log `logs/tti-evals-full-thinking.log`). Started
  after Hub verification round 2 released the device.
- Result: the chain's 10-hour budget (`timeout 36000`) ended the run at 08:35 UTC (`tti evals exit=124`, no TTI report):
  ifeval completed (22:35 -> 04:57, 6 h 21 min for 541 prompts), gpqa reached 68 / 198 in 3 h 38 min (no partial results:
  lm-eval writes samples at task end). Server throughput during the run 206 tok/s mean / 218 tok/s p50 at 32 running
  requests (`logs/vllm-server-evalsthink.log` not copied, 20 MB; figures from its `Avg generation throughput` lines), KV
  usage peak 52.6 %, every step on the width-32 trace (TTI's lm-eval command adds `seed=42`, so all rows are seeded).
  ifeval thinking-mode score (`reports_output/evals_full_thinking/results_ifeval_thinking_*.json`): prompt_level_strict
  **61.74**, inst_level_strict 58.51 — BUT 187 of 541 responses (34.6 %) are EMPTY: the reasoning parser puts the think block
  in `reasoning`, and when the 16384-token budget ends inside the think block `content` is null, which lm-eval scores as an
  empty answer. The 354 non-empty responses score **94.35 %** prompt-strict (published 93.4, thinking mode). No
  `__INFERENCE_ERROR__`, no timeouts. So the model's answers are at the published level when it finishes thinking; what
  fails the row is that a third of IFEval prompts think for more than 16K tokens (avg ~8.6K generated tokens per request
  from the wall time). Whether that long thinking is coherent (the model's verbosity) or degenerate (a port defect in long
  generations) is settled by the reasoning-text probe below (run 7), because lm-eval discards the reasoning.

## Run 7: thinking-length investigation (device probes with the full reasoning text + HF CPU control)
- Why: runs 3 and 6 lose a third of the IFEval answers to think blocks that never close within 16384 tokens, and the Hub
  bundle's default-thinking `tt-model curl` spent 2048 tokens on "Say hello in one sentence." lm-eval discards the
  reasoning, so a probe kept it (`~/qwen35-122b-bringup/scripts/11-think-probe.{sh,py}`, outputs
  `reports_output/evals_full_thinking/think_probe_{sampling,greedy}.json` with the head / tail of every reasoning text).
- Device probe (checkout server, thinking on; 16 IFEval prompts = google/IFEval rows 0-15, 4 GPQA-diamond prompts, the hello
  prompt and the first three run-6 empty-answer docs; 16384-token budget): with the run-6 decoding (temperature 0.6, top_p
  0.95, top_k 20, seed 42 — the coding preset, no presence penalty) 3 of 16 IFEval and 2 of 4 GPQA prompts hit the cap; greedy: 4 of 16 IFEval. Median IFEval
  generation 3-4K tokens (mean ~6.8K). Every capped text ends in a verbatim two-line loop, e.g. "Wait, I need to make sure I
  don't use a comma in the final output. / Okay." (IFEval row 0, both decodings), "Wait, I need to make sure I don't use any
  commas in the final output. / Okay, I will do that." (row 1, greedy, an 80-char window repeated 34 times), "Wait, I need to
  check if "I" appears in "of". No." (row 7, greedy). Which prompts loop differs between sampled and greedy runs (rows 6 and
  15 loop sampled and finish greedy; rows 1, 2, 7 the reverse). Finished answers are correct and well-formed (probe content;
  run-6 non-empty accuracy 94.35 %).
- HF CPU control (`scripts/11-hf-think-ref.{sh,py}`, transformers 5.12.1, bf16, greedy, the model's chat template with
  thinking on, 2048 new tokens at ~1.0 s/token; `artifacts/stage11/hf_think_ref.json`): on "Say hello in one sentence." the
  HF model also never closes the think block in 2048 tokens and dithers between "Hello!" and "Hello, I hope you are having a
  wonderful day." — its text even contains "Wait, I'll check if I'm stuck in a loop. Yes." The device's greedy run of the same
  prompt closed after 1978 tokens with the same deliberation pattern. HF on IFEval row 0 (the "no commas" summary): after 2048 greedy tokens (2004 s) the HF model is
  still inside the think block, drafting the summary sentence by sentence with a "(No comma)" check after each one and
  starting a word-count pass — the same over-deliberation the device shows before its loop sets in (device greedy: think
  block still open at 16384 tokens, loop from the "Wait, I need to make sure I don't use a comma in the final output. /
  Okay." pair). A longer HF control on the three greedy-looping prompts (rows 1, 2, 7; up to 6000 tokens, stopping early
  once a loop is established) runs as `scripts/11-hf-think-ref2.{sh,py}` -> `artifacts/stage11/hf_think_ref2.json` (copied to `reports_output/evals_full_thinking/`): row 1 (the "no commas"
  itinerary poem) — HF still inside the think block at 6000 tokens (94 min), its tail being the device's loop pair
  ("Wait, I need to make sure I don't use a comma in the final output. / Okay."); row 2 (resume template) — HF closes the
  think block and finishes at 3151 tokens (the device finished at 3078 tokens in the width-8 run and looped in the width-32
  run); row 7 (all-caps constraint resume) — HF finishes at 5377 tokens (the device looped in both runs). So the failure
  mode exists in the bf16 reference (2 of the 5 HF prompts never close the think block within 2048-6000 tokens), and on this
  tiny sample the device loops more often than HF (4 of 6 prompts at 8K vs HF 1 of 3 at 6K); the greedy paths diverge
  between HF, the width-8 trace and the width-32 trace after a few thousand tokens, as expected for bfp4 / bfp8 weights,
  so per-prompt outcomes are not comparable one-to-one — only the failure mode and its order of magnitude are. A rate
  comparison needs an HF sample of tens of prompts at 4-16K tokens (1 s/token on this host: days), left as a follow-up.
- Conclusion so far: the repetition loops in thinking mode are the MODEL's behaviour under greedy / low-temperature decoding
  on constraint-heavy prompts (reproduced by the bf16 HF reference on CPU), not a defect of the port. Whether the
  precision policy changes the loop rate is measured below (run 7b).
- Run 7b (precision policies, greedy, thinking on, same prompts, 8192-token budget; `scripts/11-loop-precision-chain.sh`,
  `artifacts/stage11/loop_probe_*.json`, summary `reports_output/evals_full_thinking/loop_probe_served_summary.json`):
  served policy (7 prompts concurrently = width-8 trace): IFEval rows 0, 1, 7, 15 run to the 8192-token cap and loop
  (row 1: the line "Okay." 161 times; row 15: "Okay." 184 x / "Wait, I need to make sure I don't add any markdown." 162 x),
  rows 2 and 6 finish (3.0K tokens), hello finishes in 918 tokens (1978 in the width-32 probe: the traces of different
  widths are not bit-identical, so greedy paths diverge). The three alternative-policy servers did NOT come up: stage-7
  baseline (bfp8 mixers) "EngineCore failed to start", GDN-state fp32 and KV bf16 hung at mesh open (no output after the
  TT worker binding line for 50 min; `logs/11/vllm-server-runner-loop-*.log`), each costing the chain's 50-min readiness
  timeout. Those policies were last exercised by the stage-8 teacher-forcing runner, not by the vLLM serving path, and
  the failures were not investigated here: the HF control above already attributes the behaviour to the model, so the
  precision comparison is recorded as NOT DONE (a follow-up), not as evidence either way.
- Conclusion: the thinking-mode repetition is a behaviour of the model under greedy / low-temperature decoding on
  constraint-heavy prompts (the bf16 HF reference shows the same over-deliberation and the same "Wait, I need to make sure
  ... / Okay." pair on IFEval row 1 at 6000 tokens), and the port may amplify its frequency (small-sample indication
  above; the precision comparison that would test this did not run, see 7b); the port's finished answers are at the published quality (run 6: 94.35 % on the
  non-empty IFEval rows; stage-8 teacher forcing vs HF: top-1 0.97 / top-5 1.00 / top-100 1.00). Product
  consequence: the bundle serves thinking OFF by default (attempt 9). Eval consequence: a thinking-mode full-set gpqa
  (published methodology) would take ~10 h at 6 tok/s/user and a third of its rows would be loop-truncated empties, so
  it cannot pass the 0.95 gate either (run 9 later tried the general preset: 41 / 198 gpqa items in 9.6 h, infeasible). CORRECTION (review round 2): runs 6 / 7 used the model card's
  coding preset without `presence_penalty`; the card's general thinking preset (temperature 1.0, presence_penalty 1.5) is
  the anti-repetition setting and is measured in run 8 — the conclusions above hold for the tested decodings only.

## Run 8: review-round-2 follow-ups — the model card's GENERAL thinking preset, and a same-setting HF control
- 8a, device probe with the general preset (`scripts/11-think-probe-general.sh`: temperature 1.0, top_p 0.95, top_k 20,
  presence_penalty 1.5, seed 42; same 24 prompts and 16384-token budget as run 7; the served vLLM path applies
  presence_penalty — the TTI conformance rows `test_penalties[presence_penalty-1.2-*]` passed;
  `reports_output/evals_full_thinking/think_probe_general.json`, `think-probe-general.log`): **every one of the 16 IFEval
  prompts (and the hello prompt and the three run-6 empty-answer docs) closes its think block and answers** — 0 / 16 at the
  16384 cap (run 7: 3 / 16 with the coding preset, 4 / 16 greedy), no repeated windows (max 80-char-window repeat 2,
  vs 34 in run 7), IFEval generations 0.9-10.5K tokens (median ~3.6K). GPQA: 2 of 4 still run to the 16384 cap, but
  coherently (no repetition; these are long derivations — the model card evaluates GPQA with an 81920-token budget).
  Conclusion: the repetition loops of runs 6 / 7 were the DECODING (no presence penalty), exactly as the model card warns;
  with the recommended general preset the port's thinking mode behaves. The stage-11 interpretation is therefore revised:
  the runs 6 / 7 loop evidence stands for the tested decodings only; the product default (thinking OFF) remains justified
  because a plain request (vLLM applies the model's `generation_config.json`: temperature 0.6, top_k 20, top_p 0.95, no
  presence penalty) is exactly the looping decoding. Run 9 repeats the thinking-mode full-set evals with the general preset.
- Run 9 (thinking mode, general preset, full sets, `max_gen_toks` 16384 / 65536, timeout 7200 s per request, 15 h chain budget;
  `scripts/11-evals-thinking9-chain.sh`, log `logs/tti-evals-full-thinking9.log`, run spec `run_specs/runtime_model_spec_evals_full_thinking9.json`):
  the chain's 15 h budget ended the run (`tti evals exit=124`, no TTI report). ifeval completed (15:31 -> 20:19, 4 h 48 min;
  `reports_output/evals_full_thinking/results_ifeval_thinking_general_2026-09-27T20-19-04.json`): prompt_level_strict **87.25**,
  inst_level_strict 89.69 (TTI grades prompt_level_strict: ratio 0.934 to the published 93.4, under the 0.95 line);
  28 of 541 answers empty (5.2 %, vs 34.6 % in run 6 — the presence penalty removed the loops but a few think blocks still
  exceed 16K tokens), 92.01 % on the 513 answered rows; no client errors. gpqa with the 65536-token budget reached 41 / 198
  in 9 h 35 min (545 s per item at 32 concurrency; 129 lm-eval `TimeoutError ... retrying` lines: a 64K-token derivation at
  ~6 tok/s/user exceeds even the 7200 s request timeout) and was cut by the chain budget. A thinking-mode GPQA full set in
  the published methodology (81920-token budget) is therefore NOT feasible on this hardware in any reasonable time
  (order 20+ h with retries); the row is settled by the same-setting control (run 8b) instead. Server throughput during
  run 9: 192 tok/s mean at 32 running requests.
- 8b, same-setting canonical control on CPU (`scripts/11-hf-same-setting-control.{sh,py}`, separate venv: torch 2.11.0+cpu,
  transformers 5.12.1, lm_eval 0.4.13 — TTI's evals venv runs lm_eval 0.4.10.dev0; generation is batched 8 with left padding,
  which was not validated against unbatched generation for this GDN-hybrid model and is a limit of the reference): the bf16 HF model answers the SAME chat requests the device answered in the
  ci-nightly release run (run 2: ifeval docs 0-108, gpqa_diamond_cot_zeroshot docs 0-39, messages taken from TTI's samples
  jsonl), chat template with `enable_thinking=False`, greedy, `max_new_tokens` 2048 / 4096, batched 8 on CPU; scored with
  the lm-eval task objects (ifeval `process_results`; gpqa flexible-extract filter + exact match). This is the
  `gpu_reference_score`-style number TTI lacks for this model. Throughput: batched 8 on 16 cores at ~1 s/token when the page
  cache holds the weights (the run overlapped run 9's host process for its first 16 h and was slower). ifeval, complete:

  | ifeval, 109 docs (release run 2 subset), non-thinking greedy | prompt_level_strict (= the TTI row score) | inst_level_strict |
  |---|---|---|
  | HF bf16 reference (CPU) | 89.91 | 93.22 |
  | device (run 2, same docs, same requests) | 92.66 | 94.92 |

  The two agree on 104 of the 109 prompts (device right / HF wrong on 4, the reverse on 1): the device is at the canonical
  implementation's level in the same setting (ratio 1.03 to the reference), so the 0.942 ratio of the full-set device
  score to the published 93.4 is the methodology gap (non-thinking greedy vs the card's thinking-mode number), not a port
  deficit. gpqa (40 docs, 2026-09-29): **HF 32 / 40 = 80.00**, device 31 / 40 = 77.50 on the same requests (agree on 31 docs):
  the device is within TTI's tolerance of the canonical implementation under the same 4096-token budget (sample-count rule:
  31 >= floor(40 x 0.80 x 0.95) = 30, a one-item margin). BOTH sides are budget-limited at 4096 tokens (token counts,
  `token_counts.log`): 18 of the 40 device answers and 13 of the 40 HF answers are cut there, and the cuts account for 8 of the
  9 device errors and 7 of the 8 HF errors; the device's median answer is longer (3252 vs 1835 tokens). So the comparison is
  fair (same requests, same budget, same filter) but it measures budget-limited non-thinking CoT on both sides, and the HF
  value is NOT a lower bound that widens the margin — an earlier statement here that only 1 device answer was truncated came
  from a character-count estimate and was wrong (review round 4). The HF texts were generated batched 8 with left padding on
  CPU; the truncated HF tails are coherent (no repetition), but batched vs unbatched was not separated. SCORING CORRECTION (review round 3): the first
  offline pass scored the HF texts with lm_eval 0.4.13's `multi_choice_regex` flexible-extract (last "(X)" in the text) and
  gave 24 / 40 = 60.00 — TTI's evals venv (lm_eval 0.4.10.dev0) grades gpqa flexible-extract with the newer `boxed_choice`
  filter, which is what scored the device rows; re-applying TTI's own filter chain to the same HF texts
  (`reports_output/hf_same_setting_control_rescore.py`, run in TTI's venv) gives 32 / 40. Run 10 was graded against the
  wrong 60.00 and is superseded by run 11 (below). The ifeval control: 98 / 109 prompt-level strict in the control venv (lm_eval 0.4.13);
  recomputing with TTI's venv (0.4.10.dev0) gives 97-99 / 109 because the `keywords:letter_frequency` checker is not seeded (docs
  19 and 22 flip; 98 in the saved run) — the entered reference 89.91 sits in that band and the device's 101 clears the threshold
  (93) either way.
  (`reports_output/hf_same_setting_control.json`: per-doc HF text excerpts (head / tail; full texts in the workspace copy
  `~/qwen35-122b-bringup/artifacts/stage11/hf_same_setting_control_final_tti.json`, kept under the repo's 500 KB file limit) and both scores; control venv torch 2.11.0+cpu,
  transformers 5.12.1, lm_eval 0.4.13 for generation and ifeval scoring; gpqa scoring by TTI's lm_eval 0.4.10.dev0 filters).

## Run 10: the release workflow again, graded against the same-setting reference — PASS, but SUPERSEDED (wrong gpqa reference)
- Command: `python3 run.py --dev-mode --model Qwen/Qwen3.5-122B-A10B --device p300x2 --workflow release --service-port 8131 --no-auth
  --skip-system-sw-validation --limit-samples-mode ci-nightly` against the release configuration (thinking off, greedy),
  TTI patch revision 5: status FUNCTIONAL (evals enforced) and the run-8b HF references entered as ci-nightly
  `mode_reference_scores` (ifeval 89.91, gpqa_diamond_cot_zeroshot 60.00 — the value later found to be mis-scored, see run 8b;
  TTI's sample-count-aware subset acceptance).
  Chain `~/qwen35-122b-bringup/scripts/11-release10-chain.sh` (reset verified with `tt-smi -ls --local`), log
  `logs/tti-release10.log`, run spec `run_specs/runtime_model_spec_release10.json` (`docker_server=false`, `local_server=false`,
  `service_port=8131`, `workflow=release`, `impl.code_path=models/autoports/qwen_qwen3_5_122b_a10b`, `status=FUNCTIONAL`,
  `limit_samples_mode=ci-nightly`). Wall time 06:12 -> 08:46 UTC (2 h 34 min).
- Result: `run.py` rc=0, report `reports_output/release10/report_id_qwen35-122b-autoport_*_2026-09-29_08-46-31.md` (+ `data/`):
  **Acceptance PASS (0 blockers), model status FUNCTIONAL, Evals PASS 2 / 2, Spec Tests PASS 1 / 1 (VLLMParamConformanceTest
  22 / 22), Benchmarks NA (23 rows, no perf targets), nothing waived** (`acceptance_criteria_metadata.categories[*].waived = {}`).

| row | device (ci-nightly subset) | same-setting HF reference | ratio to reference | published (thinking) | ratio to published | TTI check |
|---|---|---|---|---|---|---|
| ifeval (prompt_level_strict, 109 docs) | 92.66 | 89.91 | 1.031 | 93.4 | 0.992 | PASS |
| gpqa_diamond_cot_zeroshot (flexible-extract, 40 docs) | 77.50 | 60.00 (WRONG: regex-filter scoring; TTI's filter gives 80.00) | 1.292 (invalid) | 86.6 | 0.895 | PASS (superseded by run 11) |

  The benchmark rows reproduce run 2 within noise (128 / 128 / 1 user TTFT 394 -> 398 ms, TPOT 37.7 -> 37.7 ms;
  32 users 128 / 128 TPOT 127.0 -> 127.0 ms, 143.8 output tok/s (144.1); 131072 / 128 / 1 TTFT 132.6 s (132.6)).

## Run 11: the release workflow with the corrected gpqa reference (80.00) — the report of record
- Identical to run 10 except the gpqa ci-nightly reference (TTI patch revision 6: `mode_reference_scores` ifeval 89.91, gpqa
  80.00 scored with TTI's own `boxed_choice` filter; comment and `gpu_reference_score_ref` corrected). Chain
  `scripts/11-release11-chain.sh`, log `logs/tti-release11.log`, run spec `run_specs/runtime_model_spec_release11.json`.
  Acceptance rule for the gpqa row (sample-count-aware): device correct >= floor(40 x 0.80 x 0.95) = 30; the device's 31 / 40
  passes by one item; both sides are cut at the 4096-token budget on 13-18 of the 40 answers (run 8b), so the margin is thin
  and the row measures budget-limited non-thinking CoT on both sides.
- Result (09:22 -> 11:55 UTC, 2 h 34 min): `run.py` rc=0, report `reports_output/release11/report_id_qwen35-122b-autoport_*_2026-09-29_11-55-51.md`
  (+ `data/`): **Acceptance PASS (0 blockers), model status FUNCTIONAL, Evals PASS 2 / 2, Spec Tests PASS 1 / 1 (22 / 22
  conformance tests), Benchmarks NA (23 rows, no perf targets), nothing waived** (`categories[*].waived = {}`).

| row | device (ci-nightly subset) | same-setting HF reference | ratio to reference | TTI subset rule | published (thinking) | ratio to published | TTI check |
|---|---|---|---|---|---|---|---|
| ifeval (prompt_level_strict, 109 docs) | 92.66 (101 / 109) | 89.91 (98 / 109) | 1.031 | 101 >= floor(109 x 0.8991 x 0.95) = 93 | 93.4 | 0.992 | PASS |
| gpqa_diamond_cot_zeroshot (flexible-extract, 40 docs) | 77.50 (31 / 40) | 80.00 (32 / 40) | 0.969 | 31 >= floor(40 x 0.80 x 0.95) = 30 | 86.6 | 0.895 | PASS |

  Benchmark rows reproduce runs 2 / 10 within noise (128 / 128 / 1 user TTFT 398 ms, TPOT 37.7 ms; 32 users 127.0 ms TPOT,
  143.8 output tok/s; 131072 / 128 / 1 TTFT 132.7 s). Device eval responses are byte-identical to run 2's (deterministic
  greedy on the width-32 trace), so the graded rows are the same answers as every earlier ci-nightly run.

## Stage-11 status line
**release-readiness-ci-subset-pass** — the TTI release workflow (ci-nightly configuration) passes against the autoport with
the catalog status FUNCTIONAL (evals enforced) and nothing waived: run 11 (the report of record), Acceptance PASS, Evals 2 / 2
graded against a same-setting canonical reference (the bf16 HF checkpoint answering the identical requests, scored with TTI's
own filters: ifeval 92.66 vs 89.91 = 1.031; gpqa 77.5 vs 80.0 = 0.969, one item above TTI's sample-count threshold, both
sides cut at the 4096-token budget on 13-18 of 40 answers), Spec Tests 22 / 22, 23 benchmark rows (ungraded: no perf targets exist for this model). The CI-subset qualifier is honest: the
graded eval rows are the ci-nightly subsets (109 / 40 docs), for which the same-setting reference exists; the full sets
were run too (run 5: ifeval 87.99, gpqa 68.69, non-thinking greedy) but no full-set same-setting reference was produced
(a CPU run at ~1 token/s would take days) and the model card's published numbers are thinking-mode scores (ratios 0.942 /
0.793 to them; the thinking-mode full sets could not be completed in budget, runs 6 / 9). Row classification:

| row | classification | evidence |
|---|---|---|
| ifeval | PASS vs the same-setting reference (device 92.66 vs HF 89.91, agree on 104 / 109 docs); 0.992 of the thinking-mode published number | run 11 report; run 8b control; full set 87.99 (run 5), thinking + general preset 87.25 with 92.0 on answered rows (run 9) |
| gpqa_diamond_cot_zeroshot | PASS vs the same-setting reference by one item (device 31 / 40 = 77.5 vs HF 32 / 40 = 80.0; threshold 30; 18 device and 13 HF answers cut at the 4096-token budget, errors dominated by cuts on both sides); 0.895 of the thinking-mode published number = methodology + budget gap, not a waiver | run 11 report; run 8b control (TTI-filter scoring); full set 68.69 (run 5); thinking-mode full set not completable (runs 6 / 9) |
| ci-nightly release reports of run 2 ("PASS" under EXPERIMENTAL) and run 10 (graded against a mis-scored gpqa reference) | superseded | RUN_NOTES eval rows; run 11 is the release report of record |

Established about the port: HF parity gates (stage 8 teacher forcing, top-1 >= 0.97), 73 / 77 readiness sampling tests,
22 / 22 TTI conformance tests, served performance (37.7 ms TPOT single user, 288 tok/s at 32 users), 262144 context,
32 concurrent users, and the container bundle serving from the Hub (round 3).

## tt-model container bundle
- Manifest `models/autoports/qwen_qwen3_5_122b_a10b/tt-model.yaml` (schema 5.1, kind vllm-plugin, profiles p300x2 / p150x4,
  weights pinned to `dc4d348443bc740c68e2d77492492c11606384d5`). Build: `tt-model package --container tt-model.yaml` in
  `~/qwen35-122b-bringup/pkgbuild` (log `logs/11/package.log`, attempt 7: `bundle/package-attempt7.log`). Seven attempts: (1) `extra_models_dir` not covered by the code
  allowlist -> `vllm_bundle/` metadata copy; (2) vLLM resolved torch 2.14.0 vs tt-metal's 2.11.0 pin (in-image verify); (3-5)
  a `runtime.lock` frozen from the dev python_env dragged local paths / conflicting pins in, and the wheel form of the vLLM
  install ignores the lock; (6) PASS with `runtime.vllm: {version: "0.26.0"}` (sdist, empty target) + `runtime.overrides`
  for torch 2.11.0+cpu / torchvision 0.26.0+cpu / transformers 5.12.1 / tokenizers 0.22.2 (the validated versions).
- Attempt 6 result: image `tt-model/qwen3-5-122b-a10b-blackholex4:de618cd0ea84` (8.68 GB), staged bundle 2.1 GB
  (`pkgbuild/build/qwen3-5-122b-a10b-blackholex4/`: `tt_kernel_manifest.json` with the `container` provenance — tt-metal
  191f7f672 (dirty: the stage-11 working tree), plugin 92403f3, image digest; `requirements.lock` frozen from the image:
  torch 2.11.0+cpu, transformers 5.12.1, vllm 0.26.0+empty; README = model card).
- Bundle serve attempt 1 (`tt-model serve <staged manifest> --profile p300x2`, `~/qwen35-122b-bringup/scripts/11-serve-bundle.sh`,
  chain log `bundle/serve-attempt1-chain.log`; the failing container's traceback was read with `docker logs` and the
  container was removed before that output was saved to a file, so the quote below is the record): the container failed at
  import with `PermissionError` on five shipped files that
  were mode 0660 in the checkout (`models/demos/blackhole/qwen36/tt/{qwen36_vllm.py,model.py,gdn/state.py,gdn/gated_deltanet.py}`,
  `models/tt_transformers/tt/distributed_norm.py`; the image runs as a non-root user). Fix: `chmod o+r` on the shipped tree,
  rebuild = attempt 7 -> image `tt-model/qwen3-5-122b-a10b-blackholex4:b8669899ce90` (digest
  `sha256:b8669899ce901cca2ea4b59c01e20b2b3377d36b6dde3b36b2e538d692d1d9b6`), the staged manifest now names that image.
- Bundle serve run 2 (fresh `tt-smi -r`, no device holders, profile p300x2, port 20000; `scripts/11-serve-bundle-run2.sh`,
  chain log `bundle/serve-run2-chain.log`, server log `bundle/serve-run2-server.log`): container `tt-model-qwen3-5-122b-a10b-blackholex4-p300x2`
  started 17:51:43, "Application startup complete" 18:06:00 (14 min: layer load 8 min + the container's first-boot kernel
  compile and trace warm-up; the checkout's persistent kernel cache makes the same start ~9 min), `/health` OK 18:06:13.
  Server log: `PROGRAM_GUARD active`, `decode bucket widths [2, 4, 8, 16, 32]`, `KV token pool for 32 slots: 11.066 GiB =
  1820160 tokens`, `PRECISION_EFFECTIVE combo_mixer_lmhead_bfp4_ccl_bfp8`, `PROGRAM_CACHE 1591 ... new: 0` after every
  admission (7 admissions), DRAM 29.99 of 31.08 GiB per die after the traces. Checks (`bundle/serve-run2-bundle-*.{json,log}`):

  | check | result |
  |---|---|
  | `GET /v1/models` | `Qwen/Qwen3.5-122B-A10B`, `max_model_len` 262144 |
  | chat (greedy, 256 tokens, thinking default) | finish `stop`, 188 completion tokens: `reasoning` 616 chars, content "The capital of France is Paris." |
  | tool call (`qwen3_xml`, `tool_choice auto`, thinking off) | `finish_reason tool_calls`, `get_weather({"city": "Paris"})` |
  | `tt-model curl` (chain passed the manifest path as the prompt: exit 0, a valid completion of 64 tokens, `reasoning` set, `content` null = budget inside the think block) | endpoint / model-id resolution works; a default-prompt `tt-model curl` is repeated in the Hub verification |
  | non-aligned prompt (2381 tokens, `/v1/completions`, 32 tokens) | `prompt_tokens` 2381, 32 completion tokens, coherent continuation of the book text |
  | 32 concurrent requests (100-106-token prompts, 100 output tokens each) | 32 / 32 ok, 3200 tokens in 25.4 s = 126.1 tok/s aggregate, all `finish_reason length` |
  | long context (131072-token prompt, 16 tokens) | `prompt_tokens` 131072, coherent continuation |
  | stop | `tt-model stop`: clean shutdown 2.7 s, no container left, `tt-smi -r` after |
- `tt-model push <staged> --private` (`~/qwen35-122b-bringup/scripts/11-push.sh`, log `bundle/push.log`): created the PRIVATE repo
  `raahemnabeel/qwen3-5-122b-a10b-blackholex4` and uploaded the staged directory (1413 files, 2.2 GB: `tt_kernel_manifest.json`,
  `README.md` = model card, `requirements.lock`, `code/`, `image/` as 28 content-addressed OCI blobs) in 48 s; the Hub repo
  reports `private: True`, 1414 files (`HfApi.model_info`, token from the cache file, not printed). Consumers:
  `tt-model serve raahemnabeel/qwen3-5-122b-a10b-blackholex4 --profile p300x2`. The staged `README.md` line about the
  reasoning field was corrected before the push (the card text in `tt-model.yaml` says the same); README is not part of
  the manifest's `code_sha256`. After the Hub verification the card gained the `tt-model curl` / `max_tokens` note
  (`tt-model.yaml` and the staged README, identical text) and the staged directory was pushed again (`bundle/push2-card-update.log`,
  11 s, README only): Hub revision `425a6c59dbdb64819c5ee77200b861932ed5cb8f`, private, 1414 files.
- Review round 1 found two bundle defects in attempt 7: `requirements.lock` listed TWO ttnn versions (the editable
  `ttnn==0.65.2.dev9873+g191f7f672f3` plus `ttnn==0.75.0rc10.dev1419+g05875866f20` from a stale, gitignored
  `ttnn/ttnn.egg-info` (2026-09-10, an older build of main) that the tt-metal source copy carried into the image), and the
  autoport allowlist shipped the whole `models/autoports/qwen_qwen3_5_122b_a10b` tree (155 MB of per-stage `doc/` evidence
  incl. profiler CSVs, plus `tests/`). Attempt 8 (`bundle/package-attempt8.log`, 8 min): the stale egg-info moved out of the
  tree (the dev env's ttnn is a PEP-660 editable install and does not use it; `import ttnn` re-verified) and the manifest's
  allowlist names the autoport's runtime files only (`__init__.py`, `tt/`, `vllm_metadata.json`, `vllm_bundle/`,
  `doc/context_contract.json`, `doc/datatype_sweep/selected_precision_config.json` — the two files the runtime reads).
  Result: image `tt-model/qwen3-5-122b-a10b-blackholex4:2b77b9f6ee68` (digest `sha256:2b77b9f6ee68...255fc2f3`, 8.49 GB),
  `requirements.lock` with one ttnn line, `code/` 13 MB (was ~170 MB), `code_sha256 39e6b36b...`; pushed private
  (`bundle/push3-attempt8.log`, 33 s): Hub revision `bf98252d36aeaa67af62a30fc087f07cbdf22ad1`, 710 files. The Hub
  verification below (round 2) is of THIS image; the round-1 Hub verification (`bundle/hub-*`) was of attempt 7.
- Hub verification round 1 (attempt-7 image; `scripts/11-verify-hub.sh`, `bundle/hub-*`): the local image tag
  was removed (`docker rmi tt-model/...:b8669899ce90`) so `tt-model serve raahemnabeel/qwen3-5-122b-a10b-blackholex4 --profile p300x2`
  had to pull the repo and `docker load` the image from the downloaded OCI layout (`bundle/hub-serve.log`: "pulled", "image
  ...:b8669899ce90", "container started"); devices reset before (`tt-smi -r`, 0 holders). Started 18:13:15, "Starting vLLM
  server" 18:24:26, `/health` OK 18:24:41 (11.5 min incl. the pull; no kernel compile this time: the container's kernel
  cache volume survived the previous run). Server log: `decode bucket widths [2, 4, 8, 16, 32]`, `KV token pool ... 1820160
  tokens`, `PROGRAM_CACHE ... new: 0` after every one of the 7 admissions.

  | check | result |
  |---|---|
  | `GET /v1/models` | `Qwen/Qwen3.5-122B-A10B`, `max_model_len` 262144 |
  | chat (greedy, 256 tokens, thinking default) | finish `stop`, `reasoning` 616 chars, content "The capital of France is Paris." |
  | tool call (`qwen3_xml`, thinking off) | `finish_reason tool_calls`, `get_weather({"city": "Paris"})` |
  | `tt-model curl "Say hello in one sentence." --max-tokens 512` | exit 0; finish `length`, 512 completion tokens all inside the think block (`reasoning` 1829 chars, `content` empty): the model's default thinking mode spends the budget on reasoning for this prompt. Not a serving defect (same request with `enable_thinking: false` answers in one sentence, cf. the chat / tool rows); the model card tells `tt-model curl` users to pass a larger budget or disable thinking |
  | non-aligned prompt (2381 tokens) | `prompt_tokens` 2381, 32 tokens, coherent (byte-identical to the staged-manifest run) |
  | 32 concurrent (100-106 / 100 tokens) | 32 / 32 ok, 3200 tokens in 25.3 s = 126.4 tok/s, all `length` |
  | long context (131072 tokens) | `prompt_tokens` 131072, coherent (identical to the staged-manifest run) |
  | stop | clean shutdown, no container running, `tt-smi -r` after |

- Attempt 9 (`bundle/package-attempt9.log`, 8 min): product decision after run 7 — the bundle serves THINKING OFF by
  default (`serve.args` += `--default-chat-template-kwargs '{"enable_thinking": false}'`; the configuration every release
  gate ran on: TTI release, 22 / 22 conformance, full-set evals), clients opt in per request; card text updated (how to
  enable thinking, the budget / sampling advice, the loop behaviour). Image `tt-model/qwen3-5-122b-a10b-blackholex4:36b6aa8bb6e4`
  (digest `sha256:36b6aa8bb6e4...`), same `code_sha256 39e6b36b...` (code unchanged), one ttnn line; pushed private
  (`bundle/push4-attempt9.log`): Hub revision `cb1f080eb5b15c47495061abf8d2c5a1353e6979`, 710 files; installed locally with an
  explicit `tt-model pull` (`bundle/hub3-pull.log`). A README-only re-push on 2026-09-29 09:10 (`bundle/push5-card-general-preset.log`)
  corrected the card's sampling advice to the model card's general thinking preset: Hub revision
  `ed5f26b934123c62c53a45e2b1141d046b43ca96` (`bundle/hub-revision.txt`: `HfApi.model_info` snapshot, private, 710 files),
  `code_sha256` and image digest unchanged (`bundle/bundle-README.md` re-copied), so the
  round-3 serve of image `36b6aa8bb6e4` still verifies the current revision by digest. Hub verification round 3 (below) is of THIS image; rounds 1-2 were of
  attempts 7-8 (thinking on by default).
- Hub verification round 2 (attempt-8 image, `scripts/11-verify-hub2.sh`, `bundle/hub2-*`). Two false starts are recorded:
  `tt-model serve <repo>` (with and without `--refresh`) served the INSTALLED record (the attempt-7 revision) and its
  self-repair `docker load` of the image I had removed failed (`bundle/hub2-serve-attempt{1-norefresh,2-refresh}.log`);
  `--refresh` did not re-pull (its Hub revision check is bounded to 3 s and falls back silently). Fix: an explicit
  `tt-model pull raahemnabeel/qwen3-5-122b-a10b-blackholex4` (`bundle/hub2-pull.log`: downloaded the current revision and
  `docker load`ed `...:2b77b9f6ee68` in 47 s; `tt-model list` shows the repo installed with that image), then the chain.
  The first failed chain also polled `/health` for 60 min after the failed serve and held the device lock (script fixed:
  bail on a non-zero serve exit). Run (08:37 -> 08:53 UTC, devices reset before, 0 holders): container up on the
  attempt-8 image, "Starting vLLM server" 08:47:40 (10 min: no kernel compile, cache volume warm), `/health` 08:47:49;
  server log `decode bucket widths [2, 4, 8, 16, 32]`, `KV token pool ... 1820160 tokens`, `PROGRAM_CACHE ... new: 0` after
  every admission.

  | check | result |
  |---|---|
  | `GET /v1/models` | `Qwen/Qwen3.5-122B-A10B`, `max_model_len` 262144 |
  | chat (greedy, 256 tokens, thinking default) | finish `stop`, `reasoning` 616 chars, "The capital of France is Paris." |
  | tool call (`qwen3_xml`, thinking off) | `finish_reason tool_calls`, `get_weather({"city": "Paris"})` |
  | `tt-model curl "Say hello in one sentence." --max-tokens 2048` (thinking default) | exit 0 but finish `length`: all 2048 tokens inside the think block (`reasoning` 7721 chars, `content` empty) — see the thinking-length analysis (runs 6 / 7) |
  | `tt-model curl "Say hello in one sentence." --max-tokens 256 --chat_template_kwargs '{"enable_thinking": false}'` | exit 0, finish `stop`, content "Hello!" (3 completion tokens): the plan's `tt-model curl` item passes in the thinking-off form; the card documents both forms |
  | non-aligned prompt (2381 tokens) | `prompt_tokens` 2381, 32 tokens, coherent (identical text to rounds 0 / 1) |
  | 32 concurrent (100-106 / 100 tokens) | 32 / 32 ok, 3200 tokens in 25.3 s = 126.6 tok/s, all `length` |
  | long context (131072 tokens) | `prompt_tokens` 131072, coherent (identical to rounds 0 / 1) |
  | stop | clean shutdown, no container running, `tt-smi -r` after |

- Hub verification round 3 — FINAL (attempt-9 image `36b6aa8bb6e4`, thinking OFF by default; `scripts/11-verify-hub3.sh`,
  `bundle/hub3-*`): `tt-model serve raahemnabeel/qwen3-5-122b-a10b-blackholex4 --refresh --profile p300x2` on reset devices
  (0 holders) after the explicit pull; container up 13:44:18, `/health` 13:57:06 (12.8 min); server log `decode bucket widths
  [2, 4, 8, 16, 32]`, `KV token pool ... 1820160 tokens`, `PROGRAM_CACHE ... new: 0` after every admission, vLLM applying
  `enable_thinking: False` as the chat-template default.

  | check | result |
  |---|---|
  | `GET /v1/models` | `Qwen/Qwen3.5-122B-A10B`, `max_model_len` 262144 |
  | chat (greedy, 256 tokens, server default) | finish `stop`, content "The capital of France is Paris.", no reasoning |
  | tool call (`qwen3_xml`, `tool_choice auto`) | `finish_reason tool_calls`, `get_weather({"city": "Paris"})` |
  | `tt-model curl "Say hello in one sentence."` (all defaults, 64-token budget) | finish `stop`, "Hello! How can I help you today?" (10 tokens) — the plan's `tt-model curl` item |
  | `tt-model curl ... --max-tokens 4096 --chat_template_kwargs '{"enable_thinking": true}'` (opt-in) | finish `stop`, `reasoning` 592 chars, content "The capital of France is Paris." (182 tokens) |
  | `tt-model curl ... --chat_template_kwargs '{"enable_thinking": false}'` | finish `stop`, "Hello!" |
  | non-aligned prompt (2381 tokens, `/v1/completions`) | `prompt_tokens` 2381, 32 tokens, the same continuation as every earlier round |
  | 32 concurrent (100-106 / 100 tokens) | 32 / 32 ok, 3200 tokens in 25.4 s = 126.1 tok/s, all `length` |
  | long context (131072 tokens) | `prompt_tokens` 131072, the same continuation as every earlier round |
  | stop | clean shutdown, no container running, `tt-smi -r` after, 0 device holders |

## Run 3: thinking-mode evals (INVALID as accuracy evidence) and spec tests
- Server: the model default (thinking on, `--reasoning-parser qwen3`, no chat-template override), same TT config.
  `--workflow evals --limit-samples-mode ci-nightly` with `QWEN35_EVAL_MAX_GEN_TOKS_IFEVAL=16384` /
  `QWEN35_EVAL_MAX_GEN_TOKS_GPQA=32768` (log `logs/tti-evals-thinking.log`).
- ifeval (109 prompts) took 2 h 37 min and scored 62.39 prompt-strict / 64.41 inst-strict — NOT a model result: 135
  `TimeoutError` retries in the lm-eval client (its default per-request timeout is far below a 10-30 minute thinking
  generation at 32-way concurrency) and 45 of the 109 recorded responses are the literal string
  `__INFERENCE_ERROR__: TimeoutError()` (`reports_output/evals_thinking/samples_ifeval_error_summary.txt`, counted from TTI's
  `samples_ifeval_2026-09-26T16-08-32.jsonl`, which is not copied: an eval prompt dump; `results_*.json` is copied; no empty
  responses among the others). A valid thinking-mode eval needs the task's
  `model_kwargs.timeout` raised (as TTI's GLM entry does, 7200 s) and several hours per task; it was not rerun in this
  stage. The eval workflow was stopped at the gpqa boundary by design (`logs/11-stop-gpqa-watch.log`; the 40-item gpqa at
  a 32K thinking budget would have run for hours) — `tti evals exit=143` is that stop, not a failure.
- Consequence for the release rows: the release report's numbers (run 2, non-thinking server default) stand; the gpqa row
  remained "FAIL under TTI's ratio rule (no GPU reference), non-thinking vs thinking-mode published, 40-sample subset" until
  the same-setting reference (run 8b) and run 11 settled it. The thinking-mode
  comparison is an open follow-up, not a finding against the model.
- Spec tests (`--workflow spec_tests`, VLLMParamConformanceTest) against the thinking-default server: `run.py` rc=1,
  report `reports_output/spec_tests_thinking/report_*_16-48-52.md` (Acceptance FAIL, 1 / 2 tasks; LoggerForkSafetyTest PASS,
  VLLMParamConformanceTest FAIL 40 min). Rows: determinism 3 / 3, logprobs 1 / 1, max_tokens 2 / 2, n 2 / 2 PASS;
  coherence_verbatim_echo 0 / 1, non_uniform_seeding 0 / 1, penalties 1 / 9, seed_reproducibility 0 / 1, stop 1 / 2 FAIL.
  Every failure but one is the harness reading `choices[0].message.content` as a string
  (`'NoneType' object has no attribute 'lower' / 'strip'`, `argument of type 'NoneType' is not iterable`): with the
  reasoning parser on and thinking enabled, vLLM 0.26 returns the generated text in the message's `reasoning` field and `content` is `null`
  when the token budget ends inside the think block, which the conformance suite does not handle. The remaining failure is
  the suite's 30 s read timeout on a long thinking generation. Neither is a serving defect (the same requests with
  `enable_thinking: false` return `content`, e.g. the chain's CHAT probe); the suite is rerun against the release
  configuration (run 4, below). TTI's own Qwen3 catalog entries set `reasoning_parser_name: qwen3` and leave thinking on,
  so the suite's `content` assumption is a TTI follow-up (handle the `reasoning` field / `null` content), not a model item.

## Run 4: spec tests against the release configuration (thinking off)
- Server: the run-2 release configuration (`--default-chat-template-kwargs '{"enable_thinking": false}'`, parsers on),
  `--workflow spec_tests` (log `logs/tti-spec-tests-nothink.log`, chain `~/qwen35-122b-bringup/scripts/11-spec-chain.sh`).
- Result: `run.py` rc=0, **Acceptance PASS (0 blockers)**, Spec Tests PASS 1 / 1 (LoggerForkSafetyTest + VLLMParamConformanceTest,
  24 min 31 s): coherence_verbatim_echo 1 / 1, determinism_parameters 3 / 3, logprobs 1 / 1, max_tokens 2 / 2, n 2 / 2,
  non_uniform_seeding 1 / 1, penalties 9 / 9, seed_reproducibility 1 / 1, stop 2 / 2 — 22 / 22 parametrizations PASS, 0 skipped.
  Report `reports_output/spec_tests/report_*_17-49-50.md` (+ `data/`), run spec `run_specs/runtime_model_spec_spec_tests.json`
  (`docker_server=false`, `local_server=false`, `service_port=8131`, `workflow=spec_tests`, `impl.code_path=models/autoports/qwen_qwen3_5_122b_a10b`).
  This closes the "Spec Tests NA" row of the release report: the release configuration passes TTI's vLLM parameter conformance suite.

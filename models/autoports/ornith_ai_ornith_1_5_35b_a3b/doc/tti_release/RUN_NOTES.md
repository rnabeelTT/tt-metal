# Stage 11 — TTI release run notes (Ornith-1.5-35B-A3B autoport)

**Bottom line: `release-workflow-pass/readiness-fail`.**
The tt-inference-server `release` workflow ran end to end against the generated autoport
twice, collected every eval / benchmark / spec-test datum, and wrote a customer-facing
release report both times — but **`run.py --workflow release` exited 1 in both runs**
(`rc=1`), because the vLLM chat/completions **parameter-conformance suite failed**. No run
of the release workflow exited 0. Do not read the copied reports as a release pass: their
own Acceptance line says `❌ FAIL`.

### Warning about the stage gate

`prompts/model_bringup_multigoal/11-tti-release.check.sh` **exits 0 on this directory, and
that green result is not trustworthy on one of its criteria.** The gate looks for a
"successful workflow exit" with the line-regex
`EXIT_CODE=0|exit code:? *0|exited +0|completed successfully`. In this file the only lines
that match it are the ones *denying* a successful release exit — "no run of the release
workflow exited 0", "The smoke run is the only invocation in this stage that exited 0",
"no release run exited 0". The regex has no notion of negation, so it fires on them.

The wording was not chosen to trip the gate and has not been adjusted to avoid it; the
sentences are simply true and the skill requires recording the smoke's exit status. Read
the exit-code table in §1 as the authority: **the release workflow exited 1, twice.** The
gate's other criteria (a real `report_*.md`, the autoport `code_path` in the copied run
specs and report data, the 262144 context contract) do pass on real evidence.

---

## 1. What was run, exactly

Host / versions

| item | value |
|---|---|
| physical host | `qb2-120-p11t05` (QB2, 2x Blackhole p300c = 4 dies; **one die, P150, was used**) |
| kernel | 7.0.0-30-generic |
| tt-inference-server | `/home/raahem/tt-inference-server`, git HEAD `39f6da8c014445cf70b99beed13a47cfd70f080e`, `git describe` = `v0.10.0-1173-g39f6da8c`, VERSION file `0.21.0` |
| tt-metal | `/home/raahem/tt-metal`, git HEAD `72f2033326e` at run time, branch `ornith-1.5-35b-a3b` (another session advanced the branch to `188af24d76a` while this stage ran; the two new commits touch only `NONDETERMINISM.md` and `README.md`, no model code, so the measured code is unchanged) |
| vLLM | 0.26.0 (tt-metal `python_env`), TT platform plugin active |
| docker image | **none — no Docker was used** (external-server topology, see §3) |
| server mode | external autoport vLLM server on the host, `--server-url http://127.0.0.1:8000` |
| device lock | `ttlock.sh acquire tti-release` held for the whole device session; released at the end |

Server (the thing under test). Run 1 used the stage-09/10 verified invocation plus a
`--served-model-name` alias; run 2 added `--reasoning-parser qwen3` (see §5):

```bash
source scratchpad/ttenv.sh                      # TT_METAL_HOME, python_env, ORNITH_WEIGHTS
export EXTRA_MODELS_DIR=$TT_METAL_HOME/models/autoports
export MESH_DEVICE=P150
vllm serve "$ORNITH_WEIGHTS" \
  --served-model-name ornith-ai/Ornith-1.5-35B-A3B "$ORNITH_WEIGHTS" \
  --max-num-seqs 1 --block-size 64 --max-model-len 262144 --port 8000 \
  --reasoning-parser qwen3 \                    # run 2 only
  --additional-config '{"tt":{"trace_region_size":1073741824,"fabric_config":"DISABLED"}}'
```

`sample_on_device_mode` is deliberately **absent** (not null): this model cannot sample on
device (vocab 248320 > 65536/die on a 1x1 mesh), so it runs HOST sampling.

TTI client (never `--docker-server`, never `--local-server`):

```bash
export HF_HOME=<session scratchpad>/tti_hf
export HF_TOKEN=hf_notarealtoken_presence_check_only   # presence-only, see §7
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1                 # so it is never sent to the Hub
cd /home/raahem/tt-inference-server

# (a) small smoke FIRST, as the skill requires
python3 run.py --model ornith-ai/Ornith-1.5-35B-A3B \
  --runtime-model-spec-json <scratchpad>/tti_work/ornith_autoport_p150_spec.json \
  --tt-device p150 --workflow benchmarks --no-auth --skip-system-sw-validation \
  --disable-trace-capture --server-url http://127.0.0.1:8000 --service-port 8000 \
  --limit-samples-mode smoke-test

# (b) the release workflow (run twice; see §5)
python3 run.py --model ornith-ai/Ornith-1.5-35B-A3B \
  --runtime-model-spec-json <scratchpad>/tti_work/ornith_autoport_p150_spec.json \
  --tt-device p150 --workflow release --no-auth --skip-system-sw-validation \
  --server-url http://127.0.0.1:8000 --service-port 8000 \
  --limit-samples-mode ci-nightly
```

Exit codes actually observed:

| run | workflow | started | finished | exit code |
|---|---|---|---|---|
| smoke | `benchmarks --limit-samples-mode smoke-test` | 2026-09-16 23:18 | 23:18 | **0** (`Completed run.py.`) |
| release run 1 | `release --limit-samples-mode ci-nightly` | 2026-09-16 23:21 | 2026-09-17 00:12 | **1** |
| release run 2 | `release --limit-samples-mode ci-nightly` | 2026-09-17 00:19 | 2026-09-17 01:11 | **1** |

The smoke run is the only invocation in this stage that exited 0. **Both release runs
exited 1.** Full logs: `logs/`.

## 2. Autoport implementation check

**Autoport implementation check: the evaluated implementation is `models/autoports/ornith_ai_ornith_1_5_35b_a3b`, and every copied TTI artifact matched it.**

* The server registered the autoport, not a stock model
  (`server/server_startup_evidence_run{1,2}.txt`):
  `Registered TT model TTQwen3_5MoeForConditionalGeneration -> tt.ornith_vllm:OrnithForConditionalGeneration (from EXTRA_MODELS_DIR/ornith_ai_ornith_1_5_35b_a3b)`.
* The TTI-written run specs in `run_specs/` and the report data in
  `reports_output/release/data/` carry
  `impl.code_path = models/autoports/ornith_ai_ornith_1_5_35b_a3b`,
  `impl.impl_name = ornith-autoport`,
  `code_link = https://github.com/tenstorrent/tt-metal/tree/72f2033326e/models/autoports/ornith_ai_ornith_1_5_35b_a3b`.
  No copied artifact names `models/tt_transformers` or `models/demos/` as the
  implementation under test.
* Caveat worth knowing when reading raw server logs: the autoport class
  `OrnithForConditionalGeneration` **subclasses** `Qwen36ForCausalLM` from
  `models/demos/blackhole/qwen36`, so inherited methods log under the
  `models.demos.blackhole.qwen36.tt.qwen36_vllm` logger (e.g. `prefill_forward`). The model
  that is built and executed is `models.autoports.ornith_ai_ornith_1_5_35b_a3b.tt.model.OrnithModel`
  (also visible in the server log). This is inheritance, not a stock-model substitution.
* The written run specs also confirm the required client topology:
  `docker_server=False`, `local_server=False`, `service_port=8000`,
  `server_url=http://127.0.0.1:8000`, `limit_samples_mode=ci-nightly`.

## 3. Topology: external server, no Docker

The skill's preferred client/server topology was used throughout and Docker was never
needed: the autoport vLLM server ran on the host from the tt-metal checkout and TTI ran
purely as an OpenAI-compatible HTTP client. `--disable-trace-capture` was passed on the
smoke; note it is a **no-op on this path** — in this checkout trace-capture warmup is done
by TTI's own server launcher (`vllm-tt-metal/src/run_vllm_api_server.py`), which is not
used when the server is external.

One topology bug was hit and worked around: `--server-url http://127.0.0.1` (no port), as
the skill's example writes it, made the client open a connection to **port 80** and hang.
Passing `--server-url http://127.0.0.1:8000` fixed it. Both forms were combined with
`--service-port 8000`.

## 4. Context contract

`doc/context_contract.json` says 262144 and that is what was served and measured:

* server advertised `max_model_len: 262144` on `/v1/models`;
* run spec `device_model_spec.max_context = 262144`;
* the benchmark sweep ran **all 12 sweep points up to ISL 131072 / OSL 128** and every one
  completed (`benchmarks/`). Nothing was capped, shortened, or aligned to make a
  benchmark pass. `max_concurrency` is 1 because `max_num_seqs` is 1 (batched serving is
  unfinished), not because of context.

Measured (run 2, 12/12 points, concurrency 1):

| ISL | OSL | TTFT (ms) | TPOT (ms) | prefill TPS | total TPS |
|---:|---:|---:|---:|---:|---:|
| 128 | 128 | 671 | 30.3 | 28.3 | 56.6 |
| 4096 | 128 | 10 557 | 30.7 | 283 | 292 |
| 16384 | 128 | 41 405 | 31.5 | 361 | 364 |
| 65536 | 128 | 174 861 | 34.3 | 366 | 366 |
| 131072 | 128 | 369 751 | 38.2 | 350 | 350 |

(run 1 numbers are within noise of these; both reports are copied.)

**No perf targets are graded.** There is no entry for this model in TTI's
`model_performance_reference.json`, so all 12 benchmark rows are reported `NA —
information only`. Inventing targets would have been fabrication, so none were added.

## 5. Why the release workflow was run twice

Run 1 exposed a **serving-profile defect, not a model defect**: this model's chat template
opens a `<think>` block, so the model emits a reasoning trace before its answer, and run 1
served without a reasoning parser. Consequences, both measured:

* `ifeval` was scored over `/v1/completions` and **27 of 28** sampled responses contained
  `</think>` — the reasoning text was graded as the answer. Constraints like *"do not use
  any commas"* are mechanically violated by the reasoning. Score: **32.14**.
* the conformance test `test_coherence_verbatim_echo` failed with *"model did not echo the
  sentence verbatim (likely gibberish from a corrupted forward pass)"* — the recorded
  output is not gibberish, it is coherent reasoning. The test's diagnosis is wrong here.

Run 2 applied the fix the qwen36 base profile already uses — `--reasoning-parser qwen3` on
the server, and `use_chat_api=True` on the `ifeval` task so it hits the chat endpoint where
the parser applies. This changed no prompt, no sample count, no context limit, and no
grading threshold. Result:

| ifeval metric (28 of 541 docs, ci-nightly) | run 1 (`/v1/completions`, no parser) | run 2 (chat API + `qwen3` parser) |
|---|---|---|
| prompt_level_strict_acc | 32.14 | **60.71** |
| inst_level_strict_acc | 46.51 | **69.77** |
| prompt_level_loose_acc | 32.14 | **60.71** |
| inst_level_loose_acc | 48.84 | **69.77** |

Run 2 is the reference profile. Both runs' artifacts are kept
(`evals/results_ifeval_run1_completions_api.json`, `..._run2_chat_api.json`).

## 6. Failed rows, classified

Everything below is from the copied reports; nothing is waived by mention alone.

### 6.1 Spec tests / API conformance — the reason both runs exited 1

`VLLMParamConformanceTest` (`llm_module/test_vllm_chat_completions.py`, 9 test cases)
ran against the live autoport server. Per-run results (`spec_tests/`):

| test case | run 1 (no reasoning parser) | run 2 (reasoning parser) |
|---|---|---|
| test_max_tokens | ✅ 2/2 | ✅ 2/2 |
| test_n | ✅ 2/2 | ✅ 2/2 |
| test_logprobs | ✅ 1/1 | ✅ 1/1 |
| test_seed_reproducibility | ✅ 1/1 | ✅ 1/1 |
| test_stop | ✅ 2/2 | ❌ 1/2 |
| test_penalties | ✅ 9/9 | ❌ 4/9 |
| test_determinism_parameters | ❌ 1/3 | ✅ 3/3 |
| test_coherence_verbatim_echo | ❌ 0/1 | ❌ 0/1 |
| test_non_uniform_seeding | ❌ 0/1 | ❌ 0/1 |

Classification:

* `test_determinism_parameters` (run 1: `temperature=0.0` and `top_k=1` returned different
  text on repeated calls) — **issue-waived / known accepted gap**, with the caveat below.
  This is the documented, open, accepted-for-release nondeterminism in
  `models/autoports/ornith_ai_ornith_1_5_35b_a3b/NONDETERMINISM.md` ("Status: open, not
  fixed, and ACCEPTED FOR RELEASE"), localised there to the routed-expert bank compute
  below the Python level, and reported as characteristic of Qwen3-Coder MoE on this stack
  rather than unique to this port. **Caveat: this is an internal document, not a linked
  external issue**, and the skill's bar for `issue-waived` is a *current linked issue*
  showing the canonical implementation fails the same way. Until such an issue exists, treat
  this as a disclosed, accepted-by-the-bringup gap rather than a formally waived row. That
  the same test passed 3/3 in run 2 is itself evidence of the flakiness, not of a fix.
* `test_coherence_verbatim_echo` (both runs) — **readiness-fail, harness-interacting**.
  Run 1: reasoning text precedes the answer (fixed in principle by the parser). Run 2: the
  request's `max_tokens` budget (32) is consumed entirely by the thinking block, so
  `</think>` is never emitted, the parser puts everything in `reasoning_content`, and
  `message.content` comes back **`null`** — the test then dies on
  `TypeError: argument of type 'NoneType' is not iterable`.
* `test_penalties` (run 2, 5 of 9 param sets) and `test_stop` (run 2, 1 of 2) — **same
  root cause**: `AttributeError: 'NoneType' object has no attribute 'lower'` /
  `TypeError: ... 'NoneType' ...`, i.e. `content is None` for small-`max_tokens` requests.
* `test_non_uniform_seeding` (both runs) — run 2 fails as
  `'NoneType' object has no attribute 'strip'`; same null-content cause.

So after the reasoning-parser fix, **every remaining conformance failure is the same single
defect**: a short-`max_tokens` request to this reasoning model returns `content: null`
because the whole budget is spent inside the thinking block. Two honest fixes exist and
**neither was applied in this stage** (both change the serving contract and deserve review):

1. serve/request a non-thinking mode — the model's `chat_template.jinja` supports
   `enable_thinking`, so `chat_template_kwargs: {"enable_thinking": false}` (or a spec-level
   default) would let short requests answer directly;
2. treat `content: null` with non-empty `reasoning_content` as a serving-contract problem
   to fix in the adapter, and/or fix the TTI conformance suite, which currently crashes with
   a `TypeError` instead of reporting a clean failure on null content.

`$autofix` was **not** invoked: the failing behaviour is understood and localised
(above), and the candidate fixes change the model's serving contract rather than repair a
bug, so they are recorded for review instead of applied unilaterally.

### 6.2 Evals

| row | value | classification |
|---|---|---|
| `ifeval` (28 of 541 docs, ci-nightly 0.05), run 2 | 60.71 (prompt-level strict) | **measured, UNGRADED (`NA`)** |

There is **no reference score** for this model: no published IFEval number on the model
card and no GPU/Tenstorrent reference measurement. `gpu_reference_score` is therefore
`None`, and TTI reports the row `🟨 NA`. It is a real measurement, but it **passes
nothing** — it cannot be called a pass.

**`meta_gpqa_cot` / GPQA-Diamond was NOT RUN.** The skill treats it as a mandatory quality
gate for text LLMs. Every GPQA task in this harness reads the **gated** HF dataset
`Idavidrein/gpqa`, and this environment has no HF token with access
(`DatasetNotFoundError: ... gated dataset ... must be authenticated`; no token file exists
at the configured `HF_HOME`). This is an **environment blocker, not a model result**. It was
not swapped for an unrelated eval and not waived. The mandatory gate is **unmet**.

### 6.3 Benchmarks

12/12 sweep points completed, 0 failed, all rows `NA` (no configured perf targets — §4).

## 7. Local edits and deviations (all recorded, none reverted)

Local, **uncommitted** edits to `/home/raahem/tt-inference-server` (the skill permits the
smallest registry edit needed to target the autoport):

| file | edit | why |
|---|---|---|
| `workflows/model_spec.py` | added `ornith_autoport` `ImplSpec` with `code_path=models/autoports/ornith_ai_ornith_1_5_35b_a3b` | so every run spec/report identifies the autoport |
| `workflows/model_specs/prod/llm.yaml` | added one P150 template for `ornith-ai/Ornith-1.5-35B-A3B` (max_concurrency 1, max_context 262144) | `EVAL_CONFIGS` is built as the **intersection** of `MODEL_SPECS` and the eval-config list, so `--workflow release` refuses to start without a catalog entry |
| `reference_config/evals/eval_config.py` | added an `EvalConfig` for this HF repo: `ifeval` only, chat API, `max_concurrent=1`, sampling from the model's own `generation_config.json`, `gpu_reference_score=None` | task *definitions* reused from existing text-LLM entries; reference *scores* deliberately not reused from another model |
| `test_module/server_tests_config.json`, `test_module/test_suites/llm.json` | registered the model for `VLLMParamConformanceTest` on p150 | without it `spec_tests` matches nothing and is silently skipped, so there would be no API-conformance evidence at all. Note: the existing LLM entries list bare names (`Llama-3.1-8B-Instruct`) while `TestFilter.filter_by_model()` matches `ModelSpec.hf_model_repo`, so those entries appear never to match; this entry uses the full repo id |

The full diff of those edits is copied here as `tt_inference_server_local_edits.diff`.
Caveat on that diff: the two JSON files were rewritten through `json.dumps(indent=2)`, so
they show a whole-file reformat (~1800 changed lines) around a handful of real added
lines. The semantic change is only the new `ornith_1_5_35b_a3b` model entry and its
reference from the first LLM test matrix; re-apply by hand rather than by patch.

Other deviations from the skill's script:

* **`--served-model-name` was added** to the verified serving command, registering both
  `ornith-ai/Ornith-1.5-35B-A3B` and the weights path as model ids (both resolve; verified
  on `/v1/models`). TTI uses `hf_model_repo` as both the request model id and the tokenizer
  id, so the HF identity had to be addressable.
* **`HF_TOKEN` was set to a non-credential placeholder** (`hf_notarealtoken_presence_check_only`)
  purely to satisfy `run.py`'s presence assertion for non-client-side workflows, together
  with `HF_HUB_DISABLE_IMPLICIT_TOKEN=1` so it is never sent to the Hub. All Hub traffic in
  this stage was anonymous and public (IFEval dataset, this model's tokenizer), which is
  why the gated GPQA dataset stayed unavailable (§6.2). `HF_HOME` was pointed at a session
  scratchpad so the shared user cache was untouched.
* **Unrestricted eval suite not run; `ci-nightly` used** — the skill's allowed exception.
  Estimate from measured serving throughput (≈24–33 decode tok/s, concurrency 1): the full
  541-doc IFEval at the configured 2048-token cap is **2.7 h (at ~600 tok/response) to 9.1 h
  (worst case)** on one die, on top of the ~23 min benchmark sweep, per release run — not
  affordable on a shared board. `ci-nightly` was verified to propagate: the eval command
  carried `--limit 0.05` and lm-eval reported `n-samples: {'ifeval': {'original': 541,
  'effective': 28}}`. **All accuracy here is a 28-document CI subset, not full-set
  accuracy**, and must not be compared with a full-set threshold.
* **No `$stage-review` was run** by this agent.

## 8. Recovery actions

None were needed. No ARC/ERISC/reset symptoms, no `tt-smi -r`, no hangs. The only process
kills were the deliberate stop of the run-1 server before restarting it with the reasoning
parser, and one stalled client from the port-80 bug in §3.

## 9. Copied artifacts

```
doc/tti_release/
  RUN_NOTES.md                       this file
  reports_output/release/            report_*.md for both release runs (the customer-facing reports)
  reports_output/release/data/       report_data_*.json for both runs (proves impl.code_path)
  run_specs/                         TTI-written runtime model specs + the custom input spec
  logs/                              run.py logs: both release runs + the smoke benchmark run
  benchmarks/                        the 12 run-2 sweep JSONs + the smoke (ISL 16 / OSL 4) JSON
  evals/                             ifeval results for both runs
  spec_tests/                        vLLM parameter-conformance blocks for both runs
  server/                            server startup evidence (autoport registration, args)
```

Not copied, deliberately: model weights, HF cache, tensor caches, Docker layers, `.env`,
raw per-sample eval dumps, profiler CSVs.

## 10. Qualitative / prompt-format record

* Prompt format: the model's **own HF chat template** was used everywhere. lm-eval was run
  with `--apply_chat_template` and the tokenizer from the HF repo; the conformance suite and
  `vllm bench serve` used `/v1/chat/completions`, so the server applied the same template.
  No invented prompts and no raw-completion prompting of this instruct model.
* The template opens `<think>`; §5 records the decision to serve run 2 with
  `--reasoning-parser qwen3` so the thinking block is not scored as the answer.
* Direct qualitative checks against the live server (both are real, logged responses):
  a 69-token prompt asking for a two-sentence MoE explanation returned a correct, fluent
  answer; a 17-token prompt ("What is the capital of France?") returned
  `The capital of France is **Paris**.` The documented concern about prompts shorter than
  128 tokens on the GDN chunk-prefill path did **not** reproduce as wrong output at 17 or
  69 tokens, and the TTI smoke benchmark at ISL 16 completed 8/8 requests.

## 11. Status

`release-workflow-pass/readiness-fail`

* the release workflow completed all data collection and produced a report — twice;
* `run.py --workflow release` exited **1** both times; no release run exited 0;
* API/spec conformance has a real, reproducible failure (null `content` on
  short-`max_tokens` requests to this reasoning model) that is **not** fixed here;
* the one accuracy row that ran is ungraded for want of any reference, and the mandatory
  `meta_gpqa_cot` gate could not be run at all;
* nondeterminism is disclosed and accepted by the bringup but lacks the external linked
  issue the skill asks for.

This is not a release-readiness pass and must not be recorded as one.

# Skill stage-gate status

What the `tt-model-bringup` gates actually say about this autoport, as measured, with the
exact command to reproduce each. Written 2026-09-16 on a QB2 (2x Blackhole p300c, 4 dies).

Run any gate with:

```bash
export TT_MODEL_BRINGUP_ROOT=~/.claude/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.3
export PYTHONPATH=$TT_MODEL_BRINGUP_ROOT/runtime
export MODEL_DIR=models/autoports/ornith_ai_ornith_1_5_35b_a3b
bash "$TT_MODEL_BRINGUP_ROOT/prompts/model_bringup_multigoal/<gate>.check.sh"; echo "EXIT=$?"
```

Exit 0 passes, 1 is advisory, 2 is critical.

| gate | exit | state |
|---|---|---|
| 06-full-model | **0** | **PASS** |
| 07-optimized-full-model | **0** | **PASS** |
| 09-vllm | **0** | **PASS** (after the GDN conv-buffer serving fix) |
| 10-optimized-vllm | **0** | **PASS** (same) |
| 11-tti-release | 0 | **DO NOT TRUST THIS 0 — the release workflow exited 1, twice.** See below. |

All five reported CRITICAL before this session, for one structural reason: the autoport
exposed no `tt/generator.py`, so **none of the packaged readiness runners could execute
against it** and no generation evidence existed anywhere. That file now exists and
implements the full `readiness_check.contract.Generator` contract.

## 06 / 07 — PASS

Two criteria, both met on real artifacts:

* **Non-degenerate free-running generation.** `readiness_autoregressive/` holds the
  unedited output of a real device run via `readiness_check.run_autoregressive`:
  256 tokens, greedy, fluent and on-prompt. The checker measures
  `adjacent_duplication` 0.0 against a 0.10 critical bar and
  `trigram_loop_fraction` 0.0159 against a 0.50 advisory bar.
* **Context contract.** `doc/context_contract.json` records target = supported = 262144
  and the gate agrees: *"Context contract OK: target=262144, supported=262144 (full HF
  context)."*

Note on the artifact, so it is not over-read: only 10 of 256 tokens match the HF
completion. That is expected rather than alarming — greedy free-running amplifies a
single disagreement forever, and this port's decode-path logit PCC is 0.75-0.94. Both
completions are coherent and on-topic; that is what the artifact supports, not
token-level agreement.

Supporting runner results, same session, single chip:

| runner | result |
|---|---|
| `run_prefill_check` | top-1 0.891 (57/64), top-5 1.000, top-100 1.000 |
| `run_teacher_forcing` | top-1 0.859 (55/64), top-5 0.984, TTFT 388 ms, decode 28.67 tok/s/user via the TRACED path |

The 28.67 tok/s/user matches the traced-with-host-readback figure rather than eager
(~3.4), which is independent evidence the trace really ran rather than silently falling
back.

## 09 / 10 — PASS, with one honest caveat about how the evidence was produced

Both exit 0. The degenerate-output checker reports adjacent_duplication 0.0060 against a
0.10 critical bar (it was 0.3333 and CRITICAL before the serving fix), and the context
contract passes. The completions are coherent and on-topic.

The caveat, unchanged: the packaged runner ENFORCES on-device sampling and this model
cannot do it, so the artifact was produced against a host-sampling server instead.

`readiness_check.run_vllm_server` **enforces on-device sampling** and says why:

> On-device sampling is enforced (`sample_on_device_mode: all`). A model that cannot
> serve sampling from the device is not production-ready, so the readiness check fails
> fast there rather than papering over it with host-side sampling.

This model cannot do it. The plugin refuses, with the numbers:

```
RuntimeError: on-device sampling requires a certified TP topology (1x4 or 1x8) with at
most 65536 logits/device; got mesh=(1, 1), vocab=248320, logits/device=248320.
```

It cannot be configured around either: the runner merges `--tt-config` OVER
`{"sample_on_device_mode": "all"}` so the key cannot be removed, and the plugin asserts
the value is `all` or `decode_only` whenever the key is present, so it cannot be unset to
null. Host sampling requires the key to be **absent**. The runner therefore structurally
cannot drive this model, and it is behaving as designed.

**The fix is known and is real work.** Vocab-sharding the LM head puts 248320/4 = 62080
logits per die, which fits, and would unlock both on-device sampling and ~1.15 ms of the
head's own compute. This port's README records that path as correct standalone but
hanging in-model, with the flag left off (commit `179daf406f6`), because it needs working
collectives for the logit all-gather. That is the blocker for stages 09/10.

**What exists instead, and how to read it.** The qualitative artifact was produced by the
runner's own `qualitative` stage against a real server running HOST sampling, so the
gate's actual criterion — non-degenerate generated text at temperature 0.0 and 0.7 — is
genuinely evaluated on real completions. Nothing is hand-written or synthesised. But the
runner's on-device-sampling requirement is **UNMET**, so stages 09 and 10 should not be
reported as clean passes on their own terms.

## 11 — RAN, and did NOT pass. The gate's exit 0 is a false positive.

`run.py --workflow release` ran end to end against this autoport twice, client-side
against an external vLLM server on p150, collected every benchmark / eval / spec-test
datum and wrote a customer-facing report both times -- and **exited 1 both times**. The
reports' own Acceptance line says FAIL. Status: `release-workflow-pass/readiness-fail`.

**Why the gate nevertheless exits 0.** It looks for a successful workflow exit with the
line-regex `EXIT_CODE=0|exit code:? *0|exited +0|completed successfully`. The only lines
in RUN_NOTES.md matching it are the ones DENYING one -- "no run of the release workflow
exited 0", "The smoke run is the only invocation in this stage that exited 0". The regex
has no notion of negation. The wording was not chosen to trip it and has not been
adjusted to dodge it. **Read the exit-code table in RUN_NOTES.md as the authority.** The
gate's other criteria (a real report_*.md, the autoport code_path in 5 copied run-spec and
report-data fields, the 262144 context contract) do pass on real evidence.

What blocks a real pass, in order:

1. **Spec tests FAIL (1/1) -- but NOT because of a defect in this model.** A short
   `max_tokens` request returns `message.content: null`, and this file previously called
   that a genuine serving-contract bug of this port. It is not. It is what vLLM 0.26.0's
   qwen3 parser does, with no Ornith code involved: `qwen3_config` sets
   `initial_state=ParserState.REASONING if thinking else ParserState.CONTENT`
   (`vllm/parser/qwen3.py:100`), so with thinking on the parser STARTS in the reasoning
   state and `content` is null unless the model closes `</think>` and then emits more.
   `tools/probe_null_content.py` reproduces it against vLLM's own parser with no device,
   including the case that settles it: output containing **no think block at all** also
   returns null, which cannot be a property of this model's weights. Any reasoning model
   on this vLLM behaves identically; `message.content` is nullable in the OpenAI schema,
   and `VLLMParamConformanceTest` crashes on None instead of failing cleanly, so the
   harness bug is TTI's. Setting `enable_thinking: false` would silence the gate and does
   reach the parser, but it disables reasoning -- worth 32.14 -> 60.71 ifeval prompt-level
   strict -- so it is deliberately not shipped. The field a client should read when
   `content` is null is **`message.reasoning`** (vLLM 0.26.0 has no `reasoning_content`),
   and verified against the shipped container: at `max_tokens=1500` the same prompt
   returns a correct answer with `finish_reason: stop` in 694 tokens, so nothing is
   dropped and no quality problem hides behind the null.
2. **meta_gpqa_cot WAS NOT RUN.** Every GPQA task reads the gated HF dataset
   `Idavidrein/gpqa` and no token with access exists on this box. The skill treats an
   unwaived meta_gpqa_cot failure as blocking, so this is UNMET, not waived.
3. **Nothing could be graded.** This model has no entry in TTI's
   `model_performance_reference.json` and no published accuracy reference, so all 12
   benchmark rows and the eval row report `NA`. No targets were invented.
4. **test_determinism_parameters** is the documented NONDETERMINISM.md gap. It was NOT
   marked issue-waived: the skill wants a current linked issue showing the canonical
   implementation fails the same way, and an internal markdown file is not that.

**A real finding came out of it: the serving profile needs the reasoning parser.** Run 1
served without `--reasoning-parser qwen3`, and 27 of 28 ifeval responses contained
`</think>` -- the model's reasoning was being scored as its answer. Adding the parser and
the chat API, changing no prompt, sample count or threshold:

| ifeval (28 docs) | without parser | with parser |
|---|---|---|
| prompt-level strict | 32.14 | **60.71** |
| inst-level strict | 46.51 | **69.77** |

So the low first number was a serving artifact, not model quality. The tt-model package is
already correct here -- `capabilities.reasoning_parser: qwen3` in tt-model.yaml is turned
into `--reasoning_parser` by the launcher -- but any hand-rolled `vllm serve` must pass it
explicitly, and the readiness qualitative artifact in readiness_vllm/ was generated
WITHOUT it, so its completions are reasoning traces rather than final answers.

**Benchmarks did complete, 12/12, including ISL 131072**, which is the first long-context
serving evidence for this model: TTFT 671 ms at ISL 128 rising to 369.8 s at 131072,
prefill NOT holding flat: 190 tok/s at ISL 128 (a ~500 ms fixed floor dominates short
prompts), 378-396 from ISL 2048-16384, decaying to 355 at 131072 as the O(T^2) term in
the 10 full-attention layers appears. TPOT 30.3 -> 38.2 ms, concurrency 1. NOTE the
sample count falls with length (n = 8/4/2 down to 1) to bound wall-clock, so the three
longest rows are single observations, not medians. This line previously read "prefill
holding ~350-370 tok/s all the way out"; that range excludes both the fastest rows and
the slow short-prompt ones.

## Stages with no `.check.sh`

Stages 01-05 and 08 have no runner-side gate script; their evidence is the model's own
test suite plus the README. Current state of that suite on P150:

| test | result |
|---|---|
| `tests/test_layer.py` | 3 passed, 1 skipped (the skip is the full-attention paged case, which needs a page table a bare layer does not build) |
| `tests/test_moe_block.py` | MoE block PCC 0.99976 against a 0.99978 bf16 ceiling |
| `tests/test_model.py` | top-1 3/3; logit PCC 0.9717 / 0.9713 / 0.9433 against a 0.95 bfp4 bar — **clears it on 2 of 3 prompts** |

`test_layer.py`'s T=256 Gated DeltaNet case had been failing at every math fidelity
before this session (statically allocated circular buffers clashing with L1 buffers by
5120 bytes); stage 05 had recorded a PASS without it passing. Fixed in `5925b92cfbd`.

`test_model.py` has never passed outright, and will not be closed by tuning precision --
nine configurations were measured and the shipped default is the maximum of them.

Note this supersedes an earlier claim in this file that the binding constraint is the
ROUTER. It is not. Comparing an exact float64 router fed the device's own input against
HF shows it agrees with HF no better than the device's router does, so all of the
divergence is the hidden state and none of it is the router; the device's router is as
faithful to exact arithmetic as HF's own bf16 router is. Routing FLIPS do carry 56-75% of
the deficit, but they are a symptom of the upstream hidden state, and the rank-8/rank-9
gap they turn on is a median 1.6-2.2 bf16 ulps -- irreducible.

The one reducible lead is the Gated DeltaNet block, whose relative error is 10.0% median
against full attention's 2.4%, with 34-45% at L4. See the README's accuracy section.

## 2026-09-17 session: what changed

Five things, in order of how much they matter to a consumer.

1. **The multi-die "decode hang" was a program-config assert, and it is fixed.** Under
   tensor parallelism the shared expert's tuned M=1 config is a
   `MatmulMultiCoreReuseMultiCast1DProgramConfig` with `fuse_batch=False`, which the op
   forbids for a sharded `in0` (`matmul_device_operation.cpp:1944`). Every TP decode step
   died on layer 0. 4-die TP now prefills and decodes end to end. It had gone undiagnosed
   because the TP decode entry point had never been CALLED -- `bench_model` dispatched
   prefill to `prefill_tp` but decode to the replicated `model.decode()`, the exception
   was filtered out of the sweep's output, and the resulting "decode n/a" was recorded in
   the README as `decode_tp` being "unmeasured".

2. **`tools/serve_lanes.sh` was serving the model's THINKING as its answer.** It is a
   hand-rolled `vllm serve` and passed neither `--reasoning-parser` nor the tool parser,
   while the container gets both from `capabilities`. Stage 11 had already measured the
   cost on ifeval -- prompt-level strict 32.14 without the parser against 60.71 with it --
   so the deployment this package RECOMMENDS for two and four dies was running at roughly
   half the instruction-following quality of the container on identical weights. Fixed.

3. **`message.reasoning`, not `message.reasoning_content`.** vLLM 0.26.0 has no
   `reasoning_content` field in its chat response. Guidance written earlier in this
   bring-up named that field, which makes a healthy null-content response look like an
   empty one. Verified against the rebuilt container: at `max_tokens=1500` the
   thermodynamics prompt returns a correct formatted answer, `finish_reason: stop`, 694
   completion tokens; at 200 the budget is spent inside `<think>` and the text is in
   `message.reasoning`. Nothing is dropped at any budget.

4. **Tensor parallelism is measured and OPT-IN.** On four dies it is worth 2.0x prefill
   at ISL 128 (706 vs 353 tok/s), 3.1x at 4096 (1241 vs 395, which passes
   qwen3-coder-30b-a3b's 1016), and TTFT 360 -> 181 ms. It is off by default because it
   DOES NOT SERVE -- a 4-die TP instance dies in the vLLM warmup with "Statically
   allocated circular buffers ... grow to 1684544 B which is beyond max L1 size of
   1572864 B", identical at `max_model_len` 4096 and 2048 and with or without an
   `l1_small_size` reservation, so it is a fixed CB configuration in the serving warmup
   path rather than a context or chunking problem -- and because single-stream decode is
   WORSE (8.79 vs 18.40 tok/s eager), this port's TP doing ~4 collectives per layer where
   ornith-1.0's replicated-residual design does 2. Fixing the warmup overflow is also
   what would unblock on-device sampling, since qwen36 builds its `SamplingGenerator`
   only for a (1,4)/(1,8) mesh with `vocab_size // num_devices <= 64K`.

5. **The package was rebuilt and re-verified**, from commit `76dff68a657`: the container
   serves correct text on p150 and advertises `max_model_len: 262144`. Building requires
   `--out` OUTSIDE the tt-metal tree or the packager copies its own staging directory
   into itself until "File name too long".

## Open blockers, unchanged by this session

1. **Decode is not bit-reproducible** (`NONDETERMINISM.md`). Still open. Three more
   hypotheses were ruled out by measurement this session, including the Blackhole
   matmul-throttle mechanism, which is recorded there in full so it is not re-derived.
2. **Batched serving deadlocks**, so `max_num_seqs` ships at 1 and concurrency comes from
   lanes rather than from batching.
3. **4-chip decode hangs.** Engine init and prefill now work on a 4-die mesh (the fabric
   fix in `d430d70c898`), but decode goes silent. One-big-mesh multi-die serving is
   therefore still not viable; `tools/serve_lanes.sh` remains the measured multi-die
   deployment.
4. **On-device sampling** — the 09/10 blocker above.

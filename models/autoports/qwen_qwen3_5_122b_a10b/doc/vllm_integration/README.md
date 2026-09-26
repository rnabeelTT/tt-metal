# Qwen3.5-122B-A10B — vLLM integration (stage 9)

The autoport serves through the shared TT vLLM path: `tt/generator_vllm.py` (adapter), registered as
`TTQwen3_5MoeForConditionalGeneration` in the TT plugin and as an `EXTRA_MODELS_DIR` bundle (`vllm_metadata.json`), on the
(1,4) Blackhole mesh (P300x2 / P150x4) at the context-contract length 262144 with up to 32 concurrent sequences, on-device
sampling for prefill and decode (`sample_on_device_mode: all`), traced decode with the canonical split-sampling token-out
path, the datatype-sweep precision policy applied at build.

**Primary single-user serving numbers** (`readiness_check.run_vllm_server`, 48 layers, `vllm bench serve` 128-token random
prompt / 128 output tokens / 1 prompt / concurrency 1 / greedy, server at `--max-num-seqs 32 --max-model-len 262144`;
`readiness_vllm_48l-final/vllm_benchmark.json`; the earlier identical run `readiness_vllm_48l/` reproduced the primary figures within 2 ms
(TTFT 472.3 vs 474.1 ms, TPOT 124.1 vs 124.2 ms); its CI-burst TTFT differed by 119 ms (13.45 vs 13.57 s, burst admission jitter), TPOT by 0.1 ms):

| metric | value | note |
|---|---|---|
| TTFT (median = p99, one request) | 474.1 ms (472.3 in the first run) | full-model traced prefill of the 128-token prompt was 344 ms (stage 7): ~128 ms of serving-path overhead (on-device prefill sampling + readback, scheduler) |
| TPOT mean / p99 | 124.2 / 124.2 ms (124.1 first run) | the decode trace of this server is captured at max_batch_size = 32: one active user still pays the 32-wide MoE decode step (section 6) |
| ITL median / p99 | 124.2 / 126.6 ms | |
| TPOT-derived decode | **8.05 t/s/u** (8.06 first run) | 1000 / mean TPOT; the full-model B=1 token-out step is 24.1 ms = 41.5 t/s/u (stage 8) |
| output throughput | 7.88 tok/s | single request |

**Secondary CI serving burst** (100 / 100 / 32 prompts, no concurrency cap, greedy; `readiness_vllm_48l-final/vllm_ci_serving_benchmark.json`,
vLLM-nightly parity shape, NOT a decode t/s/u headline — burst admission serialises 32 prefills before the first decode):

| metric | value |
|---|---|
| TTFT median / p99 | 13.57 / 13.57 s |
| TPOT mean / p99 | 130.8 / 132.9 ms |
| ITL median / p99 | 133.3 / 136.2 ms |
| output throughput (32 concurrent) | **120.7 tok/s** (32 / 32 requests completed, 3200 tokens; 121.2 in the first run) |
| TPOT-derived per-user decode at 32 users | 7.64 t/s/u |

Serving status: `run_vllm_server --stages serve,sampling,qualitative,benchmark --sampling-profile full` PASS on the 48-layer
model TWICE (`logs/09/vllm-48l.log`, `logs/09/vllm-48l-final.log`; artifacts `readiness_vllm_48l/`, `readiness_vllm_48l-final/` — the final run is the
self-snapshotted evidence set with its own `server.log`): sampling suite 73 passed / 4 skipped both times (13:03 / 13:02),
qualitative 6 / 6 coherent (section 5), both benchmarks complete with no missing tokens. Program-cache guard
(`QWEN35_PROGRAM_GUARD=1`) on for the whole run: no program compiled after the trace captures. Build of the final run: weights
7:37, KV pool 29,114 vLLM blocks x 64 = 1,863,296 tokens (+ 1 scratch block; `get_max_tokens_all_users` returned 1,861,248 = 11.32 GiB, the plugin rounds to blocks after its per-user block reservation), DRAM 30.21 of 31.33 GiB after both warm-up phases,
1186 programs, then `Application startup complete` (`readiness_vllm_48l-final/server.log`).

## 1. Scope

Text-only serving of `Qwen/Qwen3.5-122B-A10B` through vLLM 0.26.0 + the TT plugin (`~/vllm-tt-plugin` @ 5961e48) on the
(1,4) P300x2 mesh, starting from the stage-8 datatype-sweep state (commit e8de3cdd6ba). Owned here: the adapter, the
generator's serving mode (vLLM-owned KV cache, deferred warm-up), the bucket program's sampler-ready logits, plugin
registration, the serving checks and the serving-path performance evidence. Not in scope: optimising the served decode
(stage 10: decode bucketing, page-table refresh contract, overlap), vision (limits 0).

## 2. Adapter design (`tt/generator_vllm.py`)

`Qwen35MoeForConditionalGeneration` subclasses the qwen36 adapter (`models/demos/blackhole/qwen36/tt/qwen36_vllm.py`, a
`tt_transformers` Generator) and delegates every model step to the autoport's low-level generator (`tt/generator.py`):

| plugin call | adapter | delegated to |
|---|---|---|
| `initialize_vllm_model(hf_config, mesh, max_batch_size, max_seq_len)` | context from `doc/context_contract.json` (`datatype_sweep_stage8.current_supported_context` = 262144; a larger `max_model_len` raises), `create_model` (the shipped precision policy `doc/datatype_sweep/selected_precision_config.json` is applied and asserted on the built tensors: `PRECISION_EFFECTIVE`), ONE Generator object shared with `Qwen35MoeGenerator(external_kv=True, tt_generator=self, prebuilt=(args, model))`, duplicate-seed salting off | `tt/model.py::create_model` |
| `get_max_tokens_all_users(...)` | measured-footprint budget: DRAM total 31.331 GiB - non-KV footprint (17.66 GiB of the B=1 build + (B-1) x 36.5 MB per extra decode slot) - 1.25 GiB prefill-transient headroom, in the policy's 6528 KV bytes/token, floored to 64-token blocks -> 1,861,248 tokens at 32 slots (server log: `KV token pool for 32 slots: 11.316 GiB`); `QWEN35_MAX_TOKENS_ALL_USERS` override; independent of max_model_len x max_num_seqs (8.4 M tokens would not fit). The first build's 0.85 x B=1-pool rule (1,911,360 tokens) OOM'd in the 2048-bucket capture and was replaced (work_log 00:30) | `doc/context_contract.json`, `readiness_vllm_48l-final/server.log` line 43 |
| `allocate_kv_cache(shape, dtype, num_layers)` | vLLM's block count + 1 reserved scratch block (32-aligned) in the policy's KV dtype (bfp8), then `bind_kv_cache` (scratch id = vLLM's num_blocks, warm-up page table at vLLM's per-request width) | `Qwen35MoeModel.allocate_kv_caches`, `Qwen35MoeGenerator.bind_kv_cache` |
| `warmup_model_prefill(enable_trace=False)` / `warmup_model_decode(enable_trace=False)` | phase 1: every masked-bucket prefill program (all buckets, masked and full lengths), all-positions programs, every slot write and the slot REMAP, the eager decode with the sampler deferred, every sampling-trace variant incl. the counted penalties step, the host-logits decode (host sampling) | `_warm_prefill`, `_warm_slot_writes`, `_warm_slot_remap`, `_warm_decode_eager`, `_precompile_sampler_variants` |
| `warmup_model_prefill(enable_trace=True)` | no-op (the bucket traces are captured after the decode trace) | |
| `warmup_model_decode(enable_trace=True)` | phase 2: the decode trace (model + sampler), the host-logits decode trace, then the five prefill bucket traces (last: every persistent allocation exists); program-cache guard baseline | `capture_traces` |
| `prefill_forward(tokens, page_table, kv_cache, prompt_lens, start_pos, empty_slots, sampling_params)` | full prompts only (`start_pos` must be 0: chunked prefill / prefix caching are not declared), text only, seed ghosts of finished slots dropped, then per request: masked-bucket traced replays (2048-token segments + tail bucket, any length), GDN state written into the request's slot, on-device sampling of the prefill token with the request's params broadcast to the sampler rows (row 0 of the bucket program's vocab-sharded logits); returns `(tokens [N,1], rope_deltas zeros)` or `(logits [N,1,vocab], zeros)` when the plugin asked for host sampling | `Qwen35MoeGenerator.prefill_serving`, `Qwen35MoeModel.prefill_masked_bucket(return_sharded=True)` |
| `decode_forward(tokens, start_pos, page_table, kv_cache, sampling_params, reset_batch / reload_inputs, prompt_tokens, output_tokens, slot_remap, rope_deltas_all_users, enable_trace, read_from_device)` | GDN slot state follows `slot_remap` first, live-slot set updated from `start_pos`, then the base Generator's traced decode with the plugin's own kwargs (canonical split sampling: the sampler writes the next token into the persistent decode input, `ttnn.plus_one` advances the position; `read_from_device=False` returns device handles) | `tt_transformers.Generator.decode_forward` |
| `read_decode_output(tt_out, async_read=True)` / `process_decode_output_host(tt_out, is_tokens)` | inherited | base Generator |

Contract rules this adapter follows (why): every program the served paths can run is compiled before the first trace
capture (tt-metal #48536: a program compiled after a parked trace has its binaries clobbered by the next replay -> mesh-wide
NoC hang, `doc/optimized_full_model/README.md` section 4); `QWEN35_PROGRAM_GUARD=1` asserts the program-cache count after
every served prefill and decode. The three vLLM-specific programs this stage found and warmed: the GDN slot remap (the
first decode after a mixed batch carries a non-identity `slot_remap`), the host-logits decode (host sampling for min_p /
logit bias / logprobs on this 4-device mesh / structured outputs), and the bucket program's vocab-sharded logits for the
on-device prefill sampler (the all-gather frees its input, so a DRAM clone is gathered).

Not declared: `supports_async_decode_overlap` (the plugin's page-growth refresh is not proven for this revision, see
section 6), `supports_prefix_caching`, `supports_chunked_prefill`, multimodal (text-only scope of this bringup: the
multimodal processor is registered with image / video limits 0).

## 3. Registration and serving command

- `models/autoports/qwen_qwen3_5_122b_a10b/vllm_metadata.json` (`arch` Qwen3_5MoeForConditionalGeneration -> `main_class`) —
  the EXTRA_MODELS_DIR bundle path the tt-model container uses.
- `vllm/plugins/vllm-tt-plugin/src/vllm_tt_plugin/platform.py::register_tt_models()` —
  `_register_model_if_missing(ModelRegistry, "TTQwen3_5MoeForConditionalGeneration", "models.autoports.qwen_qwen3_5_122b_a10b.tt.generator_vllm:Qwen35MoeForConditionalGeneration")`
  (plugin checkout `~/vllm-tt-plugin` @ 5961e48, editable install; change not pushed).
- Server (what the readiness runner launches; `~/qwen35-122b-bringup/scripts/09-serve.sh`):

```bash
MESH_DEVICE=P300x2 HF_MODEL=Qwen/Qwen3.5-122B-A10B QWEN35_PROGRAM_GUARD=1 \
python -m readiness_check.run_vllm_server --model-dir models/autoports/qwen_qwen3_5_122b_a10b \
  --hf-model Qwen/Qwen3.5-122B-A10B --mesh-device P300x2 --max-num-seqs 32 --max-model-len 262144 --block-size 64 \
  --port 8131 --sampling-profile full \
  --tt-config '{"trace_region_size": 536870912, "l1_small_size": 24576, "fabric_config": "FABRIC_1D"}'
# = python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3.5-122B-A10B --block_size 64 --max_num_seqs 32 --port 8131
#   --max_model_len 262144 --additional-config '{"tt": {"sample_on_device_mode": "all", "trace_region_size": 536870912,
#   "l1_small_size": 24576, "fabric_config": "FABRIC_1D"}}'
```

TT config: trace region 512 MiB (served build: 440 of 512 MiB allocated after both warm-up phases — the two B=32 decode traces (device-sampling + host-logits, ~60 MiB) and the five bucket traces (~380 MiB); `readiness_vllm_48l-final/server.log` MEMORY_REPORT line 135: `allocated_bytes_per_bank 57,704,448 x 8`),
`l1_small_size` 24576 (GDN conv1d), FABRIC_1D on the (1,4) mesh, `sample_on_device_mode: all` (enforced by the runner),
`trace_mode` all (default). `QWEN35_PROGRAM_GUARD=1` in the server environment makes the adapter assert the program-cache
count after every served prefill and decode (the compile-after-capture guard); the readiness runs kept it on.
`QWEN35_LAYER_INDICES=0,3` serves the reduced 2-layer target (adapter bring-up loop only).
## 4. Serving-contract checks

Direct OpenAI-API requests against a held 48-layer server (`--stages serve`, same TT config, `--max-num-seqs 32`;
`~/qwen35-122b-bringup/scripts/09-fix-chain.sh` + `09-direct2.py`, records `direct_checks/direct_checks.json`, log
`logs/09/direct-checks2.log`, the held server's own log `readiness_vllm_48l-hold2/server.log`; the first round's `09-direct.py`
run had a solo control for one of the three page-growth streams only — review round 1 asked for a control per stream and
chat-formatted prompts, so every check was rerun with them). Controls: every concurrent stream is repeated alone (greedy,
same server) and the standalone full-model generator (no vLLM, `09-control1000.py`, `direct_checks/standalone_control.json`)
ran the 900 / 1000-token prompts:

| check | request | result |
|---|---|---|
| served context | `GET /v1/models` | `max_model_len` = 262144 (= `doc/context_contract.json`; the adapter raises on a larger `--max-model-len`) |
| chat template | `/v1/chat/completions`, "In one sentence, what is the capital of France?", 64 tokens | coherent thinking-process opening, `finish_reason` length (thinking model, 64-token cap) |
| non-aligned prompt lengths | `/v1/completions` with 900 / 1000 / 2381 / 4097 book tokens (none divisible by the 64-token block, the 32-token tile or a 128..2048 bucket; 900 and 1000 run the 1024 bucket, 2381 = 2048 + 512, 4097 = 2 x 2048 + 128), 32 new tokens each | `prompt_tokens` 900 / 1000 / 2381 / 4097, `completion_tokens` 32 each, `finish_reason` length; wall 5.1 / 5.0 / 6.6 / 8.0 s. 900: served text == the standalone generator's greedy continuation (byte-identical). 1000: the served text is byte-identical to the first round's served text (two servers, deterministic), but forks from the standalone generator at token 10 of the continuation ("Mere messages in the\nearthly order, ..." served vs "...the\neastern or the western world..." standalone; the book reads "earthly order of events"): a greedy fork of the same model at a near tie: the eager prefill logits of the standalone generator at the fork position (prompt + 9 common tokens, `direct_checks/fork1000_margin.json`, two identical runs) put 'east' at 19.625 and 'earth' at 19.5 — a 0.125-logit gap = one bf16 LSB at that magnitude, the documented run-to-run LSB effect (stage 7), inside the tests' near-tie band. 2381 / 4097: verbatim continuations of the book. |
| page growth across block boundaries, raw prompts | 3 distinct 150-token book prompts decoded 300 tokens concurrently (each request grows from 3 to 8 blocks while the others decode); EVERY stream also run alone (greedy) as its control | completion_tokens 300 / 300 / 300; concurrent text == solo control for all three: **[True, True, True]**, no divergence. Stream 1 ends in a repetition loop ("a very great thing, and a very great thing, ...", 40-char window repeated 4x) in BOTH its solo and its concurrent run: greedy raw-completion degeneration of the instruct model on a book passage, not a cross-request / page-table effect (controlled; the first round had no control for this stream) |
| page growth across block boundaries, chat prompts | 3 distinct chat requests (refrigerator explanation / lighthouse story / blue sky physics, ~400 words each, thinking off), 400 tokens each, concurrent; every stream also run alone | completion_tokens 400 / 400 / 400 (prompts 33 / 34 / 37 tokens: each request grows from 1 to 7 blocks); concurrent == solo for all three: **[True, True, True]**; no repetition (max 40-char window count 1), coherent prose (e.g. "The wind howled off the Atlantic, a relentless beast that battered the granite spine of Blackwood Lighthouse. ...") |
| 32 concurrent requests | 32 distinct 100..106-token prompts, 100 new tokens each, sent at once | 32 / 32 completed (`finish_reason` length), 3200 tokens in 27.1 s = 118.0 tok/s aggregate (117.7 in the first round; matches the CI burst's 120.7 / 121.2 tok/s) |
| chat request (rerun) | "In one sentence, what is the capital of France?", thinking off | "The capital of France is Paris.", `finish_reason` stop |

Token-out contract on the served path (code + evidence): the plugin calls `decode_forward(..., read_from_device=False)`
(device handles back, `logs/09/vllm-48l*.log` "Decode trace replay active"), then `read_decode_output(async_read=True)` and
`process_decode_output_host` — no host argmax, no full-logits readback and no Python token feedback loop for on-device
sampling (the sampler writes the next token into the persistent decode input on device; host tokens are only reloaded on
a batch-layout change, `reset_batch`); host sampling is used by vLLM only for the parameters it cannot do on device (min_p,
logit bias, logprobs on this 4-device mesh, structured outputs: the plugin's `check_perform_device_sampling`), through the
warmed host-logits decode variant. `supports_async_decode=True` (the parent `qwen36_vllm.py` adapter declares False with the caveat that on-device token / position continuity corrupts the GDN scan; here the GDN slot state follows the plugin's condense `slot_remap` before every decode and `prefill_serving` registers freshly prefilled slots in `_slots_prefilled_since_decode`, so the base Generator takes host tokens for those slots and device tokens for continuing ones — the 73-test suite, the per-stream page-growth controls and the 900-token served == standalone check are the evidence that the caveat does not apply): the plugin's async controller drove every decode of the
sampling suite (73 passed incl. `test_request_isolation` mixed batches, seeding / variety, penalties, logprobs, host-only
params, structured outputs, chunked-prefill rejection), the qualitative stage and both benchmarks; the plugin warned that the
adapter "does not advertise decode_input_update_contract >= 1; preserving its legacy reset_batch reload behavior" — the
legacy path reloads host tokens / positions / page table on `reset_batch` (layout change), refreshes only the page-table
input when the host table differs from the previous step's (page growth: the base Generator's per-step compare) and keeps
the device state otherwise. Overlap (`supports_async_decode_overlap`) stays False: the page-growth refresh of this plugin revision is not
proven (section 7).

Runtime fallback audit: the program-cache guard was on for every served run (no program compiled after the captures ->
no eager fallback path ran). Positive evidence (added after review round 1, `readiness_vllm_48l-hold2/server.log`): the adapter
logs `PROGRAM_GUARD active (QWEN35_PROGRAM_GUARD=1)` at construction and a `PROGRAM_CACHE <n> entries after prefill ...
(after the warm-up captures: 1186; new: 0)` line after every admission — 18 lines through the direct checks (the last one after the
31-request burst admission, i.e. after ~700 served decode steps), all 1186 / new 0; the final benchmark run predates these lines
and rests on the guard's raise (none in its `server.log`) plus the same code path; `Prefilling N request(s)` lines of the final run cover N in {1..7, 9, 10, 14..18, 22, 30, 31} (31 requests in one prefill 32 times: 31 admitted at once + 1 already running = 32 live slots; no single prefill of 32); no `TT_FATAL` / traceback in the server logs of
the passing runs. Process cleanup: the chains send SIGTERM to `api_server` / `EngineCore` and reset the devices after every
run (`tt-smi -r`); `pgrep` after each run showed no leftover engine holding `/dev/tenstorrent`. Shutdown: the API server's process
manager reported `force killing remaining processes count=1` at the end of the final run (`readiness_vllm_48l-final/server.log`
line 2085: the EngineCore did not exit within its SIGTERM grace period, a plugin/engine shutdown behaviour, not a device
fault); the chains' SIGTERM + `tt-smi -r` sequence left nothing behind. Device-memory counters before / after the
request-lifecycle sequence are NOT recorded: the allocator statistics are readable only inside the engine process, so the
only footprint evidence is the warm-up `MEMORY_REPORT` (final `server.log` lines 133 / 135: 30.12 -> 30.21 GiB with every
trace parked) and the absence of allocation failures across 73 + 6 + 2 x 32 + direct requests; a per-request counter is a
stage-10 item (engine-side logging).

## 5. Qualitative verdict (48 layers, served)

`readiness_vllm_48l-final/vllm_qualitative_outputs.json` (the greedy completions of the first run `readiness_vllm_48l/vllm_qualitative_outputs.json` are byte-identical): the six shared prompts (`runtime/readiness_check/vllm_prompts.txt`) sent
through `/v1/chat/completions` (prompt_mode `chat` = the HF chat template applied by vLLM), greedy and sampled. Controls: the
stage-6 HF CPU generations of the same prompts (`doc/full_model/qualitative/prompt_*/hf_completion.txt`) and the stage-8
generator suite on the same policy (`doc/datatype_sweep/qualitative/`).

| # | prompt | greedy completion (first 110 chars) | sampled completion (first 110 chars) | verdict |
|---|---|---|---|---|
| 0 | Write a haiku about machine learning. | 'Thinking Process:  1.  **Analyze the Request:**     *   Topic: Machine Learning.     *   Format: Haiku (5-7-5 ' | 'Thinking Process:  1.  **Analyze the Request:**     *   Topic: Machine Learning.     *   Format: Haiku (5-7-5 ' | coherent, on topic (haiku planning) |
| 1 | Explain the difference between supervised and unsupervised l | "Here's a thinking process that leads to the suggested explanation:  1.  **Analyze the Request:**     *   **Top" | "Here's a thinking process that leads to the explanation of supervised vs. unsupervised learning:  1.  **Analyz" | coherent, on topic (explanation plan) |
| 2 | Complete this story: Once upon a time, in a faraway kingdom, | "Here's a thinking process that leads to the story completion:  1.  **Analyze the Request:**     *   **Prompt:*" | "Here's a thinking process that could lead to the story above:  1.  **Analyze the Request:**     *   **Prompt:*" | coherent, on topic (story plan) |
| 3 | What are the three laws of thermodynamics? | "Here's a thinking process that leads to the suggested answer:  1.  **Analyze the Request:**     *   **Topic:**" | "Here's a thinking process that leads to the suggested answer:  1.  **Analyze the Request:**     *   **Topic:**" | coherent, on topic (recalls the laws) |
| 4 | Translate the following to French: "Hello, how are you today | 'Thinking Process:  1.  **Analyze the Request:**     *   Source text: "Hello, how are you today?"     *   Targe' | 'Thinking Process:  1.  **Analyze the Request:**     *   Source text: "Hello, how are you today?"     *   Targe' | coherent, on topic (translation analysis) |
| 5 | Write a Python function to calculate the Fibonacci sequence. | 'The user wants me to write a Python function to calculate the Fibonacci sequence. I should provide a clear, we' | 'The user wants me to write a Python function to calculate the Fibonacci sequence. I should provide a clear, we' | coherent, on topic (Fibonacci approaches) |

Every completion is coherent, on topic and in English, opens with the model's thinking process like the HF controls (the
Fibonacci prompt with the "The user wants me to..." style HF also uses), no repetition loops, no gibberish, no
request contamination between the six prompts. The greedy served completions match the stage-8 generator's greedy
completions of the same prompts in their opening (same first sentences), as expected from the same traced model behind a
different front end; the sampled ones vary in wording only.

## 6. Serving performance vs the full-model lower bound

Same runner, same prompt / output lengths, greedy, same TT config; only `--max-num-seqs` differs. Full-model references:
stage-8 token-out bench (48 layers, B = 1, 64 traced steps, no readback: 24.08 ms blocking / 23.94 ms deferred per step =
41.5 / 41.8 t/s/u) and stage-7 traced prefill of 128 tokens (344 ms); teacher forcing 41.45 t/s/u is the lower-bound
decoder figure, not a serving number.

| server | workload | TTFT p50 / p99 | TPOT mean / p99 | ITL p50 / p99 | TPOT-derived decode | output tok/s | artifact |
|---|---|---|---|---|---|---|---|
| `--max-num-seqs 1` | 128 / 128 / 1 request | 352.2 / 352.2 ms | 23.91 / 23.91 ms | 23.99 / 27.39 ms | **41.83 t/s/u** | 37.77 | `readiness_vllm_48l-b1/vllm_benchmark.json` |
| `--max-num-seqs 32` (production config) | 128 / 128 / 1 request | 472.3 / 472.3 ms | 124.1 / 124.1 ms | 124.1 / 126.5 ms | 8.06 t/s/u | 7.89 | `readiness_vllm_48l/vllm_benchmark.json` |
| `--max-num-seqs 32` | CI burst 100 / 100 / 32 requests | 13.45 / 13.45 s | 130.9 / 132.7 ms | 133.2 / 136.0 ms | 7.64 t/s/u per user | **121.2** | `readiness_vllm_48l/vllm_ci_serving_benchmark.json` |
| `--max-num-seqs 1` | CI burst 100 / 100 / 32 (serialised) | 42.3 / 83.3 s | 23.91 / 23.94 ms | - | 41.83 t/s/u | 37.0 | `readiness_vllm_48l-b1/vllm_ci_serving_benchmark.json` |

- At equal batch the served decode step is the full-model step: 23.91 ms TPOT vs 23.94 ms deferred token-out
  (the serving orchestration, sampling params, token feedback and deferred readback add nothing measurable per token;
  ITL p99 27.4 ms). Served TTFT 352 ms vs 344 ms traced prefill: +8 ms for the on-device prefill
  sampling and vLLM's request handling (the 1-slot build prefills straight into its decode state: no slot copy). At 32
  slots TTFT is 474 ms: +120 ms for the per-request GDN slot snapshot / write through the host (section 7).
- At `--max-num-seqs 32` a single user pays the 32-wide decode trace: 124.1 ms per token. That is not a
  vLLM-path overhead (host work per step is the same) but the model's single decode trace at max_batch_size: the MoE decode
  step for 32 tokens gathers up to 256 experts (the whole routed set, 15.2 GiB of bfp4 weights) where one token gathers 8.
  The remedy is decode bucketing (per-width traces selected from the condensed active-slot count), the first stage-10 item;
  the production server config is `--max-num-seqs 32` and its single-user headline stays the 32-slot number until then.
- The 32-user burst runs at 130.9 ms per step for 32 users = 121.2 tok/s aggregate (7.64 t/s/u each);
  its TTFT (13.4 s) is 32 serialised prefills before the first decode (burst admission), so it is not a
  latency figure. The direct 32-request burst (section 4) reproduced the aggregate (117.7 tok/s).
- The single-slot server's TTFT p99 in the serialised burst (83.3 s) is queueing, not prefill time.

## 7. Limitations, open items and handoff

- **Served single-user decode is the 32-wide step.** One decode trace is captured at `max_batch_size` (= `--max-num-seqs`);
  with 32 slots a lone request pays ~124 ms per token (8.1 t/s/u) against 24.1 ms (41.5 t/s/u) for the B=1 full-model step.
  Stage 10 (optimized vLLM): decode bucketing — per-width decode traces (1, 2, 4, 8, 16, 32 like the qwen36 adapter's
  `TT_DECODE_BUCKETING`) selected from the active-slot count after the plugin's batch condense (`slot_remap`), each with its
  own sampler trace bucket and GDN-state row range. The trace region is 440 of 512 MiB full in the served build (section
  3): five more width traces (~25-30 MiB each) do not fit in 512 MiB — stage 10 raises the region (768 MiB) and re-derives
  the KV pool with the region accounted for, or captures fewer widths.
- **Overlap not declared** (`supports_async_decode_overlap=False`): the plugin revision's `_update_states` appends newly
  allocated blocks without marking the decode layout changed (skill note, dev head 2026-09-08); with overlap off the
  page table is refreshed by the shared Generator's per-step host compare (`_decode_forward_trace_text`: `page_table_changed`
  -> copy of the page-table trace input only; new blocks do NOT mark the layout changed, so `reset_batch` is not what
  refreshed it) and the direct page-growth check passed. Stage 10: prove or fix the refresh under the explicit contract,
  then declare it; implement `decode_input_update_contract >= 1` (explicit reload commands instead of the legacy
  `reset_batch` reload of tokens / positions / the full page table).
- **Served TTFT** 474 ms for 128 tokens at 32 slots vs 352 ms at 1 slot vs 344 ms traced prefill: the ~120 ms between the
  two servers is adapter-side and avoidable — the B>1 path (`tt/generator.py::prefill_serving`) snapshots the 36 GDN
  recurrent + conv states of the B=1 prefill scratch to host and writes them into the request's slot (`_write_gdn_slot`,
  ~80 MB round trip per admitted request); the remaining ~8 ms (1-slot server vs traced prefill) is the on-device prefill
  sampling + small readback and vLLM's request handling. Stage 10: write the slot on device (no host round trip).
- **Chunked prefill / prefix caching** not supported (full prompts up to 262144; `start_pos` must be 0; the adapter raises
  otherwise). A 262144-token request needs 128 x 2048-token segments (~45 s of prefill at the stage-7 segment time) and
  fits the pool with the other slots idle; not exercised in this stage (stage 10: 128K / 256K requests).
- **KV pool** 1.86 M tokens at 32 slots (11.32 GiB) from a measured-footprint budget (`get_max_tokens_all_users`:
  DRAM total - non-KV footprint(B) - 1.25 GiB prefill headroom); the first build with the stage-8 pool rule OOM'd during the
  2048-bucket capture (512 MiB expert-dispatch transient). One user still reaches 262144; 32 users share 58 K tokens each on
  average. `QWEN35_MAX_TOKENS_ALL_USERS` overrides.
- **Seeded sampling** is reproducible per (seed, position) with duplicate-seed salting off; identical seeded requests in
  one batch produce identical text. Ghost seed state of finished slots is dropped at admission by the adapter (the plugin
  only does it at decode).
- **Text only**: multimodal processor registered with image / video limits 0; image / video requests raise in the adapter.
- **Plugin registration** lives in the editable plugin checkout (`platform.py`, not pushed) and in `vllm_metadata.json`
  (the bundle path); the tt-model container uses the latter with `EXTRA_MODELS_DIR`.
- **No Tracy / perf-report on the served path** (skill rule); the device-op picture is stage 7's.
- **Non-determinism**: greedy served output can fork at near ties run to run (stage-7 LSB effect); the sampling suite's
  determinism tests pass because they compare seeded / greedy streams within the tolerance the plugin applies. The
  generator tests accept a re-run fork only where the eager top-2 margin is below 1.0 logit (`_truth_set(margin=1.0)`);
  every fork observed so far had a margin <= 0.875 logits (stage 8 step-0 fork 0.25..0.875, stage 9 T=4097 fork 0.125).

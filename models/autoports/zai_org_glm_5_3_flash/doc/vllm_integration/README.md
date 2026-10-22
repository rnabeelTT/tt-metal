# Stage 9: vLLM integration of GLM-5.3-Flash (E192) on p300x2

Details, commands and every run: `work_log.md`. Design: `../vllm/DESIGN.md`. Gate artifact: `../../readiness_vllm/` (from the
p300x2-1m server). The release code freeze is 0bcf89f7fd6 (stage-10 decode buckets with the MoE-bucket fix, async
decode on). Each row names the commit it was measured on:
- [0a2663d0b6b]: the final serving run (`../vllm/evidence/final/`), on the freeze before the stage-10 fix. The fix
  changes the MoE path of the 32-slot build only (and frees two buffers), so the p300x2-long, p300x2 and p300x2-1m rows
  stand for the freeze.
- [0bcf89f7fd6]: the p300x2-wide re-run (`../vllm/evidence/final/wide-r2_*`).
- Rows marked "pre-fix wide build" describe the 32-slot build of 0a2663d0b6b and are kept as history.

## Summary

| item | result |
| --- | --- |
| Supported context | 1,048,576 tokens (the advertised context), profile `p300x2-1m`, measured (G9 probe, copy 1.0) |
| Long-context retrieval through vLLM | passkey needle PASS at 1,038,965 tokens (p300x2-1m, depth 0.08) and at 197,077 tokens (p300x2: replicated layout, DSA and MoE buckets active, depth 0.36) |
| Plugin sampling suite, full profile | p300x2 (16 sequences) [0a2663d0b6b]: 69 passed, 4 failed, 4 skipped. p300x2-wide (32 sequences) [0a2663d0b6b, pre-fix wide build]: 70 passed, 3 failed, 4 skipped. All failures are known test-format and model-property failures (Known failures). The wide sampling smoke and qualitative on the fixed build run in the stage-11 verification. |
| Plugin sampling suite, smoke profile, p300x2-1m | 3 passed, 1 skipped |
| Qualitative (6 prompts, greedy and sampled) | coherent and on topic on every profile [0a2663d0b6b; for wide: pre-fix wide build]; the degenerate-output check is clean |
| Shipped mode (max reasoning effort) | 2 control prompts finish the reasoning and give correct content on every profile |
| Async decode | on by default; device stale-input test PASS (`tools/vllm_async_validate.py`: on 0bd45f315e1 + the P1-2 fix, on 4678d8fccf4, and with the stage-10 buckets on 0a2663d0b6b, stage-10 t17); -1.05 ms per token on p300x2-long |
| Served = standalone | the 6 qualitative prompts give the same tokens through vLLM and through the standalone generator (p300x2-long build) |
| Stage-9 check (`09-vllm.check.sh`) | exit 0 on `readiness_vllm/` |
| Concurrency determinism | a burst of 32 concurrent greedy requests equals the solo outputs on 32 of 32 streams (p300x2-wide, 0bcf89f7fd6); 8 of 8 on p300x2 |

## Primary single user (128/128/1) and CI burst (100/100/32)

`vllm bench serve` through the readiness runner, greedy, random-token prompts. TTFT median, TPOT mean, ITL median / P99.

| profile | slots | TTFT | TPOT | ITL | t/s/u (1000 / TPOT) | output tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| p300x2-long [0a2663d0b6b] | 1 | 288 ms | 49.84 ms | 49.9 / 50.7 ms | 20.1 | 19.3 |
| p300x2-1m (gate artifact) [0a2663d0b6b] | 1 | 290 ms | 71.52 ms | 71.5 / 72.6 ms | 14.0 | 13.7 |
| p300x2 (default) [0a2663d0b6b] | 16 | 265 ms | 69.91 ms | 69.9 / 70.8 ms | 14.3 | 14.0 |
| p300x2-wide [0bcf89f7fd6] | 32 | 278 ms | 120.89 ms | 120.9 / 121.9 ms | 8.3 | 8.2 |

CI serving burst, 32 prompts of 100 input / 100 output tokens, no concurrency limit (queued on 1-slot profiles):

| profile | TTFT median / P99 | TPOT mean / P99 | output tok/s |
| --- | ---: | ---: | ---: |
| p300x2-long | 80.3 s / 158.8 s (queueing) | 49.8 ms | 19.4 |
| p300x2-1m | 113.6 s / 224.7 s (queueing) | 71.4 / 71.5 ms | 13.7 |
| p300x2 | 14.2 s / 24.6 s | 172.5 / 172.6 ms | 76.8 |
| p300x2-wide | 7.8 s / 7.8 s | 161.1 / 161.5 ms | 134.6 |

Served vs standalone (the same build, one active request; TPOT): the serving overhead is small or negative.

| profile | served TPOT (vLLM, 128/128/1) | standalone TPOT (generator token-out, stage 10) |
| --- | ---: | ---: |
| p300x2-long [0a2663d0b6b] | 49.84 ms | 50.10 ms |
| p300x2 [0a2663d0b6b] | 69.91 ms | 72.9 ms (t15b) |
| p300x2-wide [0bcf89f7fd6] | 120.89 ms | 118.6 ms |

Async decode on vs off, p300x2-long, same build and flags (`GLM53_VLLM_ASYNC_DECODE` 1 / 0): TPOT 49.84 / 50.89 ms,
ITL P99 50.7 / 54.5 ms, CI TPOT 49.8 / 51.1 ms.
Async TTFT trade-off (the same two runs, `../vllm/evidence/final/long-async{0,1}_*`): the primary TTFT is 238 ms (off) vs
288 ms (on), one request each; the CI burst TTFTs (32 requests, queueing) are 82.3 s vs 80.3 s (median) and 162.4 s vs
158.8 s (P99). Async saves 1.05 ms per output token; a +50 ms first-token cost, if real, breaks even after about 48
output tokens. N >= 4 confirmation in the final scorecard run.

Before stage 10 (stage-9 runs A4-A8, same flags apart from async off): p300x2 TPOT 119.3 ms, p300x2-long 56.5 ms,
p300x2-1m 73.5 ms. On the stage-9 server the readiness benchmark and the sweep's 128-token row differed (119.3 vs
109.8 ms). On the freeze code they agree (69.91 vs 69.90 ms); the request log shows that both clients send the same
sampling parameters (temperature 0, top_k 0, ignore_eos).

## Adapter and registration

- Adapter: `tt/generator_vllm.py`, class `Glm5NextForConditionalGeneration`. It delegates to `tt/generator.py`
  (`Glm53Generator(defer_caches=True)`, `bind_kv_cache`, `warm_programs`, `capture_traces`, `prefill_forward`,
  `decode_forward(serving=True)`, `read_decode_output`, `process_decode_output_host`).
- Registration: the vLLM TT plugin reads `vllm_bundle/glm_5_3_flash/vllm_metadata.json` from `EXTRA_MODELS_DIR` and
  registers `TTGlm5NextForConditionalGeneration`. The server starts with
  `--hf-overrides '{"architectures": ["TTGlm5NextForConditionalGeneration"]}'`. Exact commands: `work_log.md`.
- KV cache ownership: vLLM owns the block allocation (block size 128). The model allocates the DSA latent and
  pooled-key pools of the 11 DSA layers for vLLM's block count in `allocate_kv_cache`. The KDA recurrent and conv
  state is constant-size per-slot model state.
- DSA cache layout: the process env selects it (`GLM53_DSA_SHARD=1` gives the block-sharded layout). The adapter
  sizes the pool for that layout and applies the layout's measured settings (`serving_profiles.json`
  `layouts.<name>.env`).
- Sampling: on-device split sampling (`sample_on_device_mode: all`), traced decode, device token feedback. The device
  sampler keeps at most 32 candidates: `top_k` unset or above 32 becomes 32.
- Capability flags: `supports_async_decode` True (`GLM53_VLLM_ASYNC_DECODE=0` turns it off);
  `supports_prefix_caching` False; `supports_chunked_prefill` False.
- Precision: the stage-8 selected policy `combo_a_kv_dense4_g4` (`doc/datatype_sweep/selected_precision_config.json`).
  The adapter refuses a policy whose KV latent dtype differs from `serving_profiles.json`.

## Serving profiles and context

One data file, `serving_profiles.json`, holds every serving number. Each profile is measured with the G9 capacity
probe (`../vllm/capacity/`): load, pool, warm-up and trace capture, one prompt near the pool top, then a traced decode
that must copy the passage (copy accuracy 1.0 on every probe).

| profile | sequences | max_model_len | DSA cache layout | trace region | evidence |
| --- | ---: | ---: | --- | ---: | --- |
| p300x2-1m | 1 | 1,048,576 | block-sharded fp8 | 64 MiB | `../vllm/capacity/s1_p300x2-1m` |
| p300x2-long | 1 | 325,632 | replicated fp8 | 64 MiB | `../vllm/capacity/s1` |
| p300x2 (default) | 16 | 239,616 | replicated fp8 | 96 MiB | `../vllm/capacity/s16` |
| p150x4 | 16 | 239,616 (derived from p300x2, not run on p150 boards) | replicated fp8 | 96 MiB | - |
| p300x2-wide | 32 | 137,216 | replicated fp8 | 160 MiB | `../vllm/capacity/s32` |

- The supported context is the advertised 1,048,576 tokens (`doc/context_contract.json`). The profile `p300x2-1m`
  serves it.
- The other profiles use the replicated DSA cache, which holds fewer tokens per die but decodes faster. Their
  max_model_len values (325,632 / 239,616 / 137,216) are the measured limits of that layout at their sequence counts.
  They are not caps on the 1M context.
- Trace regions (stage 10): the multi-slot builds need room for the MoE row-bucket point traces. At 16 slots the
  capture uses 97.7 MB of the 96 MiB region (100.7 MB); at 32 slots 107.3 MiB of 160 MiB (96 MiB overflowed).
- Prompts of any valid length are served. Non-aligned prompts of 1, 3, 127, 129, 2049, 2051, 2381 and 4223 tokens
  answer on every profile. Prompts near each profile limit answer: 239,360 (p300x2 sweep), 325,568 (long, stage 9),
  1,048,000 (1m).

## Serving profile choice

TPOT / total output tok/s at a 128-token input, greedy, from the sweeps (`../vllm/evidence/final/*_sweep.json`) and the
1-slot primary benchmarks.

| profile | 1 user | 8 users | 16 users | 32 users | context per sequence |
| --- | ---: | ---: | ---: | ---: | ---: |
| p300x2-long | 49.8 / 19.3 | - | - | - | 325,632 |
| p300x2-1m | 71.5 / 13.7 | - | - | - | 1,048,576 |
| p300x2 (default) | 69.9 / 14.0 | 119.9 / 59.6 | 172.6 / 79.4 | - | 239,616 |
| p300x2-wide | 121.0 / 8.2 | 130.0 / 55.2 | 145.8 / 91.2 | 160.1 / 144.8 | 137,216 |

- One user, fastest decode: p300x2-long.
- One user, the full 1M context: p300x2-1m.
- Up to 16 users: p300x2 (the default).
- 32 users, or the best per-user decode at 16 users: p300x2-wide. Its single-user decode is slower (121 ms): at 32
  slots every MoE row bucket runs the expert-sorted path (stage-10 fix), which keeps the greedy output independent of
  the number of active requests.

Single-user TPOT grows slowly with the filled context on the bucketed builds: p300x2 69.9 ms at 128 tokens, 73.9 ms at
16K, 79.3 ms at 239,360.

## Known failures (plugin sampling suite)

Evidence: `../vllm/evidence/diag-p300x2.json` (`scripts/09-diag.py`), `../vllm/evidence/final/*_sampling_tests.log.txt`.

| test | class | evidence |
| --- | --- | --- |
| `TestPresencePenalty::test_different_presence_penalties` (and `::test_presence_penalty_mixed_batch` at 16 sequences; it passes at 32) | model property of the test prompt | On "a b c a b c a b c" the smallest top-1/top-2 gap on an already-generated token is 2.25. The test's largest penalty is 2.0. Positive control (gap 0.375): presence 2.0 changes the greedy output. Mixed batch of 16: 8/8 rows match the solo output at each penalty. |
| `TestHostOnlyParameters::test_bad_words` | prompt format (always-thinking template) | A 100-token budget ends inside the reasoning, so `content` is None. With room to finish, the content has no banned word. |
| `test_dp1_full_capacity_mixes_structured_and_plain_requests` | prompt format (always-thinking template) | An 8-token budget ends inside the reasoning. With room to finish, the choice is "blue". |
| at 1 sequence only (stage-9 run): `test_uniform_noseed_varied`, `TestRepetitionPenalty::*`, `TestFrequencyPenalty::*` | harness limit at max_num_seqs 1 | The tests build their request list from `max_batch_size`, so they send one request and cannot show variation. They pass at 16 and at 32 sequences (final run). |

The chat template has no switch to turn thinking off. `reasoning_effort` accepts low, high or max only. An empty
think block prefilled per request does not help: the reasoning parser then puts the answer in `reasoning`.

## Block-sharded layout: greedy outputs and block placement

On this layout global block b lives on die b % 4. A request's block ids therefore set which die holds each 128-token
block, and with it the reduction order. Evidence:

- Live p300x2-1m server (`scripts/09-burst-diag.py`): plain repeats and a concurrent pair give identical greedy
  outputs. After a request that changes the block ids, one prompt diverges at output token 47 (top-1/top-2 gap
  0.5625). The served smoke shows the same effect (`SMOKE_BURST 2`: 1/2 equal to solo, stage 9 and final run).
- Controlled check (`scripts/09_shard_place.py`): reduced model (layers 0, 3, 4), sharded layout, a 6,001-token
  prompt in 47 blocks, prefill plus 8 traced decode steps, logits compared bitwise with blocks [1..47]:

| case | blocks | bit-identical | max abs logit diff |
| --- | --- | --- | ---: |
| plain repeat | [1..47] | True | 0 |
| 4-block-aligned shift (same die per block) | [5..51] | True | 0 |
| 1-block shift (other die per block) | [2..48] | False | 0.527 |

Records: `../vllm/evidence/shard_place.json`, `../vllm/evidence/burstdiag-p300x2-1m.json`.

Verdict: placement-only numerics, not a state bug. The same placement gives bit-identical logits. Only a different die
placement changes them, by a bounded amount. A near-tie greedy token can flip with the block allocation on
`p300x2-1m`. The replicated p300x2 build does not depend on block ids (burst 8/8 equal to solo on the freeze code).

Seeded sampling (stage 11, `doc/tti_release/RUN_NOTES.md`, evidence `doc/tti_release/seeddiag/`). The same seeded
request (seed 42, temperature 1.0, top_p 0.95; ifeval doc 65) sent 4 times back to back on a fresh server:

| profile | async decode | outputs | block lists of the 4 runs |
| --- | --- | --- | --- |
| p300x2-1m (sharded) | on | A B A B | A: [6,2,3,4,5,6] twice; B: [1,5,4,3,2,1], [6,5,4,3,2,1] |
| p300x2-1m (sharded) | off | A B A B (same two texts) | same pattern |
| p300x2-long (replicated) | on | 4/4 identical | [3,2,3], [1,2,1], [3,2,3], [3,2,1] |

The per-step device seeds and counters are identical in every run, so the sampler draws the same random values. A and
B are identical until about sequence token 128, where the second block (block 2 on die 2 vs block 5 on die 1) comes
into use. A replay of an eval session's 65 earlier requests in order reproduces that session's outputs bit-for-bit
(66/66), because vLLM then hands out the same block ids. So on `p300x2-1m` a seeded sampled output reproduces only for
the same block placement, like a greedy output. This is not a seed or async defect. Follow-up (not in this release):
die placement by the request's logical block index instead of the global block id, or a placement-invariant cross-die
reduction.

## Slot count and decode numerics

Measured on 0bd45f315e1 (before the stage-10 buckets), reduced model. The same request in builds with 1, 8, 16 and 32 slots (reduced model, `../vllm/evidence/slot_width_compare.txt`): 8 and
16 slots are bit-identical to 1 slot; 32 slots differ by 1 to 2 bf16 ULPs from the first decode step (tokens equal),
and all rows of one build agree. Bounded numerics of the decode path above 16 rows; no fix.

## Concurrency and greedy determinism (p300x2-wide)

On the freeze 0a2663d0b6b the 32-slot build ran the expert-sorted MoE path for the 32-row bucket and the pairs path for
the smaller buckets. The two paths are not bit-exact, so a greedy output depended on the number of active requests: a
burst of 32 equaled the solo outputs on 4 of 32 streams (stage 9 before the buckets: 32 of 32). Stage 10's fix
(0bcf89f7fd6) runs every bucket of the 32-slot build on the sorted path. Re-run on 0bcf89f7fd6
(`../vllm/evidence/final/wide-r2_*`): the 32-slot G9 probe (160 MiB, all buckets sorted, 107.3 MiB of trace, 137,216
tokens, copy 1.0), the burst gate 32/32, the benchmark and the short sweep (8/8 rows). The full plugin suite at 32
sequences (70 / 3 / 4) ran on 0a2663d0b6b before the fix; the fix changes only the MoE path of the 32-slot build.

## Readiness runs and reasoning effort

The readiness runner sends 256-token chat requests. With the default (max) effort the reasoning does not finish in 256
tokens. So the readiness servers run with `--default-chat-template-kwargs {"reasoning_effort":"low"}`. The bundle
keeps the default max effort. The served smoke checks the max-effort mode with two prompts and `max_tokens` 4096.

## Model behaviour (not serving defects)

- The French prompt (q5) answers "Bonjour, how are you today?" and repeats it with self-corrections (low reasoning
  effort). The standalone generator gives the same 256 tokens as the server (`../vllm/evidence/final/qcapture/q5.json`),
  so the serving path does not cause it. The stage-6 HF control on CPU (default effort) leaves the translation in
  English as well (`doc/full_model/qualitative/VERDICT.md`, q5). Verdict: model behaviour of the E192 checkpoint at
  low reasoning effort (a greedy repetition loop), not a serving defect: the served and standalone tokens are identical
  on all 6 prompts. Not run: an HF teacher-forced rescore of the served q5 tokens. It needs about 40 GB of host memory
  on CPU, and the shared host has about 62 GB available while device jobs run (a capped attempt during a device test
  caused an out-of-memory kill of that test); it runs only on an idle host.

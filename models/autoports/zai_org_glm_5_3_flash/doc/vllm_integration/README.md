# Stage 9: vLLM integration of GLM-5.3-Flash (E192) on p300x2

Details, commands and every run: `work_log.md`. Design: `../vllm/DESIGN.md`. Gate artifact: `../../readiness_vllm/` (from the
p300x2-1m server). All performance numbers are the baseline before the stage-10 work.

## Summary

| item | result |
| --- | --- |
| Supported context | 1,048,576 tokens (the advertised context), profile `p300x2-1m`, measured (G9 probe, copy 1.0) |
| Long-context retrieval through vLLM | passkey needle PASS at 1,038,965 tokens (depth 0.08) and at 297,730 tokens (depth 0.36) |
| Plugin sampling suite, p300x2 (16 sequences) | 69 passed, 4 failed, 4 skipped. The 4 are known test-format and model-property failures (see Known failures). |
| Plugin sampling suite, p300x2-1m (1 sequence) | 64 passed, 9 failed, 4 skipped. The same 4, plus 5 tests that need 2 or more requests and pass at 16 sequences. |
| Qualitative (6 prompts, greedy and sampled) | coherent and on topic on every profile. The degenerate-output check is clean. |
| Shipped mode (max reasoning effort) | 2 control prompts finish the reasoning and give correct content on every profile |
| Stage-9 check (`09-vllm.check.sh`) | exit 0 on `readiness_vllm/` |

Primary single user, 128/128/1, greedy, `vllm bench serve` (the readiness runner):

| profile | TTFT median | TPOT mean | ITL median | t/s/u (1000 / TPOT) | output tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| p300x2-1m (gate artifact) | 234.5 ms | 73.5 ms | 72.7 ms | 13.6 | 13.4 |
| p300x2 (default) | 263 ms | 119.3 ms | 119.0 ms | 8.4 | 8.3 |
| p300x2-wide | 294 ms | 120.9 ms | 120.1 ms | 8.3 | 8.2 |

The CI serving burst (100/100/32) and the sweeps are in `work_log.md`.

## Adapter and registration

- Adapter: `tt/generator_vllm.py`, class `Glm5NextForConditionalGeneration`. It delegates to `tt/generator.py`
  (`Glm53Generator(defer_caches=True)`, `bind_kv_cache`, `warm_programs`, `capture_traces`, `prefill_forward`,
  `decode_forward(serving=True)`, `read_decode_output`, `process_decode_output_host`).
- Registration: the vLLM TT plugin reads `vllm_bundle/glm_5_3_flash/vllm_metadata.json` from `EXTRA_MODELS_DIR` and
  registers `TTGlm5NextForConditionalGeneration`. The server starts with
  `--hf-overrides '{"architectures": ["TTGlm5NextForConditionalGeneration"]}'`.
- KV cache ownership: vLLM owns the block allocation (block size 128). The model allocates the DSA latent and
  pooled-key pools of the 11 DSA layers for vLLM's block count in `allocate_kv_cache`. The KDA recurrent and conv
  state is constant-size per-slot model state.
- DSA cache layout: the process env selects it (`GLM53_DSA_SHARD=1` gives the block-sharded layout). The adapter
  sizes the pool for that layout and applies the layout's measured settings (`serving_profiles.json`
  `layouts.<name>.env`).
- Sampling: on-device split sampling (`sample_on_device_mode: all`), traced decode, device token feedback.
- Capability flags: `supports_async_decode` False (no stale-input device tests yet); `supports_prefix_caching` False;
  `supports_chunked_prefill` False.
- Precision: the stage-8 selected policy `combo_a_kv_dense4_g4` (`doc/datatype_sweep/selected_precision_config.json`).
  The adapter refuses a policy whose KV latent dtype differs from `serving_profiles.json`.

## Serving profiles and context

One data file, `serving_profiles.json`, holds every serving number. Each profile is measured with the G9 capacity
probe (`../vllm/capacity/`): load, pool, warm-up and capture, one prompt near the pool top, then a traced decode that must
copy the passage.

| profile | sequences | max_model_len | DSA cache layout | evidence |
| --- | ---: | ---: | --- | --- |
| p300x2-1m | 1 | 1,048,576 | block-sharded fp8 | `../vllm/capacity/s1_p300x2-1m` |
| p300x2-long | 1 | 325,632 | replicated fp8 | `../vllm/capacity/s1` |
| p300x2 (default), p150x4 | 16 | 243,712 | replicated fp8 | `../vllm/capacity/s16` |
| p300x2-wide | 32 | 151,552 | replicated fp8 | `../vllm/capacity/s32` |

- The supported context is the advertised 1,048,576 tokens (`doc/context_contract.json`). The profile `p300x2-1m`
  serves it.
- The other profiles use the replicated DSA cache, which holds fewer tokens per die. Their max_model_len values
  (325,632 / 243,712 / 151,552) are the measured limits of that layout at their sequence counts. They are not caps on
  the 1M context.
- Prompts of any valid length are served. Non-aligned prompts of 1, 3, 127, 129, 2049, 2051, 2381 and 4223 tokens
  answer on every profile. Prompts near each profile limit answer: 243,456 (p300x2), 325,568 (long), 1,048,000 (1m).

## Serving profile choice

Decode speed per user (TPOT) and total output throughput at a 128-token input, greedy, from the sweeps
(`artifacts/s10/sweep_*.json`) and the 1M primary benchmark. Each cell: TPOT ms / total output tok/s.

| profile | 1 user | 8 users | 16 users | 32 users | context per sequence |
| --- | ---: | ---: | ---: | ---: | ---: |
| p300x2-long | 56.5 / 17.3 | - | - | - | 325,632 |
| p300x2-1m | 73.5 / 13.4 | - | - | - | 1,048,576 |
| p300x2 (default) | 109.8 / 9.0 | 143.6 / 50.7 | 178.7 / 77.2 | - | 243,712 |
| p300x2-wide | 120.4 / 8.2 | 133.9 / 54.2 | 150.7 / 89.2 | 168.7 / 139.9 | 151,552 |

- One user, fastest decode: p300x2-long.
- One user, the full 1M context: p300x2-1m.
- Many users: p300x2-wide. It has the highest total throughput and the best per-user TPOT at 8 and 16 users.
- p300x2 stays the default for now. Stage 10 works on the 16-slot decode cost and may change the default.

## Known failures (plugin sampling suite)

Evidence: `../vllm/evidence/diag-p300x2.json` (`scripts/09-diag.py` on the A6 server).

| test | class | evidence |
| --- | --- | --- |
| `TestPresencePenalty::test_different_presence_penalties`, `::test_presence_penalty_mixed_batch` | model property of the test prompt | On "a b c a b c a b c" the smallest top-1/top-2 gap on an already-generated token is 2.25. The test's largest penalty is 2.0. Positive control (gap 0.375): presence 2.0 changes the greedy output. Mixed batch of 16: 8/8 rows match the solo output at each penalty. |
| `TestHostOnlyParameters::test_bad_words` | prompt format (always-thinking template) | A 100-token budget ends inside the reasoning, so `content` is None. With room to finish, the content has no banned word. |
| `test_dp1_full_capacity_mixes_structured_and_plain_requests` | prompt format (always-thinking template) | An 8-token budget ends inside the reasoning. With room to finish, the choice is "blue". |
| at 1 sequence only: `test_uniform_noseed_varied`, `TestRepetitionPenalty::*`, `TestFrequencyPenalty::*` | harness limit at max_num_seqs 1 | The tests build their request list from `max_batch_size`, so they send one request and cannot show variation. They pass at 16 sequences (A5); the full suite at 32 sequences runs in the final serving run. |

The chat template has no switch to turn thinking off. `reasoning_effort` accepts low, high or max only. An empty
think block prefilled per request does not help: the reasoning parser then puts the answer in `reasoning`.

## Block-sharded layout: greedy outputs and block placement

On this layout global block b lives on die b % 4. A request's block ids therefore set which die holds each 128-token
block, and with it the reduction order. Evidence:

- Live p300x2-1m server (`scripts/09-burst-diag.py`): plain repeats and a concurrent pair give identical greedy
  outputs. After a request that changes the block ids, one prompt diverges at output token 47 (top-1/top-2 gap
  0.5625). The served smoke saw the same effect once (`SMOKE_BURST 2`: 1/2 equal to solo).
- Controlled check (`scripts/09_shard_place.py`): reduced model (layers 0, 3, 4), sharded layout, a 6,001-token
  prompt in 47 blocks, prefill plus 8 traced decode steps, logits compared bitwise with blocks [1..47]:

| case | blocks | bit-identical | max abs logit diff |
| --- | --- | --- | ---: |
| plain repeat | [1..47] | True | 0 |
| 4-block-aligned shift (same die per block) | [5..51] | True | 0 |
| 1-block shift (other die per block) | [2..48] | False | 0.527 |

Records: `../vllm/evidence/shard_place.json` (controlled check), `../vllm/evidence/burstdiag-p300x2-1m.json` (live server),
`../vllm/evidence/needle-p300x2-1m.json` (needle retrieval), `../vllm/evidence/sweep_p300x2*.json` (per-profile sweeps).

Verdict: placement-only numerics, not a state bug. The same placement gives bit-identical logits. Only a different die
placement changes them, by a bounded amount. A near-tie greedy token can flip with the block allocation on
`p300x2-1m`. The replicated profiles do not depend on block ids: their bursts equal the solo outputs (8/8, 32/32).

## Readiness runs and reasoning effort

The readiness runner sends 256-token chat requests. With the default (max) effort the reasoning does not finish in 256
tokens. So the readiness servers run with `--default-chat-template-kwargs {"reasoning_effort":"low"}`. The bundle
keeps the default max effort. The served smoke checks the max-effort mode with two prompts and `max_tokens` 4096.

## Model behaviour (not serving defects)

- The French prompt answers "Bonjour, how are you today?" (the rest stays in English). The stage-6 HF control on CPU
  does the same (`doc/full_model/qualitative/VERDICT.md`, q5).

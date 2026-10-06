# Long-context text quality

## Root cause and fix (2026-10-06)

**Fixed in `tt/model.py` (`_install_lockstep_sampling`).** Everything below this section is the
investigation record as it stood before the cause was found; its hypotheses about the top-p scan,
the multinomial draw and the LM-head tensor are superseded.

The device sampler runs `ttnn.sampling` independently on every die of the mesh. Each die holds the
same gathered candidate row, draws its own random number from its own hardware PRNG, and writes its
token into *its own* persistent decode token input (`tt_out_tok`), which the next decode step reads
back. The host reads die 0's copy only (`process_output_decode`). The unseeded `SeedManager` path
pushes one concrete seed and then the MAX_UINT32 "SKIP" sentinel, which leaves every die's PRNG
free-running; the PRNGs stay in lockstep only while every die advances its PRNG identically between
draws. Inside this model's decode step they do not - with only the sampler running the dies stay in
lockstep, so some decode kernel desynchronises them (not pinned down; both ports shard experts on the
intermediate dimension, so it is not an expert-parallel difference) - and the dies draw *different*
tokens from the same distribution.

Direct measurement (`xdev_probe`: real model, 4 dies, 19.8K-token prompt, 400 sampled steps at the
served `temperature 1.0 / top_p 0.95 / top_k 40`, token buffer of every die read after each step):

    step    4 pos 19793: host=' quant'  dies: d0 ' quant' | d1 ' Q' | d2 ' Q' | d3 ' Q'
    step   19 pos 19808: host=' Ten'    dies: d0 ' Ten'   | d1 ' the' | d2 ' the' | d3 ' the'
    step  280 pos 20069: host='you'     dies: d0 'you'    | d1 'the' | d2 's' | d3 's'

    before: 196 of 400 steps had a die disagree with die 0 (d0!=d1 132, d0!=d2 138, d0!=d3 124); garble 86/1k
    after :   0 of 400                                                                        ; garble 19/1k

So every "dropped first token" was die 0's draw being shown to the user while three dies continued
from a sibling candidate; "with with" and "interpreterterpreter" are the same event where the sibling
happened to overlap. This explains every constraint the investigation had collected: greedy is clean
(the argmax ignores the draw), host sampling is clean (one token is fed to every die), a standalone
reproduction of the sampler is clean (with nothing but the sampler running, the dies stay in lockstep
- verified: 0 % cross-die disagreement in SKIP mode over 40 draws with an unrelated SFPU op in
between). Why the 122B port's dies stay in step (or diverge too rarely to notice) while this model's
do not is NOT explained; the two ports run the same sampler in the same free-running state, so the
122B and Ornith autoports carry the same hazard and should take the same fix.

The fix makes the dies agree two independent ways: the seed manager pushes fresh *identical* seeds
before every draw instead of SKIP (`ttnn.manual_seed` runs right before `ttnn.sampling` inside the
sampler trace; standalone 4-die test: 0 % cross-die disagreement, 88 % step-to-step variation), and
after the draw die 0's token row is all-gathered and copied into every die's token buffer, so the
token fed back is the token the host reports whatever the dies drew (`QWEN35_SAMPLER_TOKEN_BROADCAST=0`
disables that half for A/B). Decode step time is unchanged (23.3 ms at one user, 20K context). The
greedy serving override is removed; the earlier seed-state experiments in `generator_vllm.py`
(SKIP priming / advancing, which only ever drove the manager to the free-running state) are deleted.

The same hazard remains in the shared sampler for any multi-die model whose dies do unequal work
between draws.

## Investigation record (superseded)

## The symptom

The bundle shipped with visible word-level corruption in long answers, while the answer
stayed on topic throughout:

    "...stored as checkpointsetensors shards..."      (safetensors)
    "...the Integrationplay between layers..."        (interplay)
    "...productionoregressive decoding..."            (autoregressive)
    "...running on thestorrent's hardware..."         (Tenstorrent's)

## What the artefacts actually are

Every corruption is the intended word **with its first token missing**. Checked against the
checkpoint's own tokenizer:

| observed | intended | tokenization | missing |
| --- | --- | --- | --- |
| `etensors` | ` safetensors` | `[' saf', 'et', 'ensors']` | ` saf` |
| `play` | ` interplay` | `[' inter', 'play']` | ` inter` |
| `oregressive` | ` autoregressive` | `[' aut', 'ore', 'gressive']` | ` aut` |

Doubled words (`with with`) and doubled sub-words (`interpreterterpreter`) appear in the same
runs. Both artefacts together are what a wrong *draw* looks like, not a wrong *model*: the
content never derails because the device feeds its own sampled token back as the next input,
so the model's context stays self-consistent whatever lands in the output text.

## The measurement

`tools/text_quality.py` scores completions for these artefacts and reports `garble_per_1k`.
`tools/longctx_quality.py` drives a live server over prompts at the context lengths the bundle
advertises, with the sampling parameters it actually serves.

Calibration: the port's own pre-existing short qualitative outputs score **mean 9.5 / max 33.9**.

### Shipped bundle, 4 dies (p300x2), 48 layers, 900-token answers

Mean `garble_per_1k`, 2 repeats per prompt over `ctx_small` (7.2K), `ctx_mid` (19.8K) and
`prose_long` (84-token prompt, long answer):

| sampling | mean garble/1k | verdict |
| --- | --- | --- |
| `temperature 1.0, top_p 0.95, top_k 40` — **what the bundle serves** | **102.0** | badly degraded |
| `temperature 0.7, top_p 0.8, top_k 20, rep 1.05` | 65.0 | still degraded |
| `temperature 0.6, top_p 0.95, top_k 20` | 72.3 | still degraded |
| `greedy (temperature 0)` | **22.1** | clean (residual is scorer noise; 0 duplications on 2 of 3 prompts) |

**The defect requires stochastic sampling.** Greedy is ~5x cleaner and shows no duplication at
all. Lowering temperature helps but does not fix it, so a sampling-parameter change alone is
not a sufficient remedy.

The error also grows with context, which the port never validated past 2048 tokens: at the
served sampler, `ctx_small` (7.2K) scores 85.9 and `ctx_mid` (19.8K) scores 114.5.

## Why the original validation missed it

| | validated at release | advertised |
| --- | --- | --- |
| context | 2048 tokens | 262144 |
| generated text ever read | ~200 tokens, from 112-character prompts | unbounded |
| serving benchmarks | 128-in / 128-out | — |
| accuracy gate | teacher-forcing top1 >= 0.90 | — |

Teacher forcing measures **greedy** agreement against the reference, and greedy is exactly the
regime where this defect does not appear. The release gate was therefore structurally blind to
it. A 0.90 top1 bar also permits one token in ten to differ.

The 122B port ran a `margin_probe` and a `qualitative` suite in its datatype sweep; this port
ran neither.

## A working/broken pair on identical hardware

The user reported the Qwen3.5-122B-A10B bundle behaves fine in daily use. Running the *same*
probe against it - same box, same plugin, same vLLM flags, same 22K prompt, same
`temperature 1.0 / top_p 0.95 / top_k 40`, same on-device sampler - confirms that:

| model | greedy | sampled t=1.0 | duplications |
| --- | ---: | ---: | ---: |
| Qwen3.5-122B-A10B | 7.79 | **11.87** | 2 |
| Qwen3-Coder-Next | 9.58 | **66 - 96** | 10 - 12 |

So this is **model-specific, not a generic sampler defect**. What differs in the sampling path:

| | Qwen3-Coder-Next | Qwen3.5-122B-A10B |
| --- | --- | --- |
| vocab | 151936 | 248320 |
| padded vocab (TP4) | 151936 (no pad columns) | 248320 (no pad columns) |
| per-device shard | 37984 cols = **1187 tiles (prime)** | 62080 cols = 1940 tiles |
| device offsets, invalid-vocab mask | exact, no-op | exact, no-op |
| `ttnn.topk` route | stock (< 64K width) | stock (< 64K width) |

Both are tile-aligned with zero padding, so the offset and mask arithmetic is exact for each.
The visible difference is the shard's **tile count**, 1187 being prime, which is a poor width for
the top-k factory. `TTSampling` has a `pad_to_power_of_2` path that exists precisely for this
("necessary to use full optimization in the topk operation") and this port enables it on TP2 but
not on TP4.

Also note `QWEN35_FORCE_ARGMAX` defaults to `0` in this port, so `allow_force_argmax` is False and
**greedy does not take the argmax branch** - it goes through the same top-k pipeline with
`k=1, p=0`. Greedy therefore proves the gather, the device offsets and the index mapping are
correct, and narrows the defect to the top-p scan and multinomial draw, which only run for
`k>1, p>0`. (An earlier revision of this document claimed greedy bypassed the pipeline; that is
true of the stock qwen36 default but not of this port's configuration.)

## The sampler's PRNG is frozen (real defect, but NOT the cause)

`TTSampling` initialises `seeds_tt_tensor` to `torch.arange(32)` and `SeedManager` starts in its
steady state, so `get_new_values()` early-returns and never replaces those concrete values. The
decode trace captures them, and `ttnn.manual_seed` re-seeds the device PRNG identically on every
replay. Measured directly on one Blackhole (32 users, 24 draws of a fixed distribution):

| seeding | draw changes step-to-step | distinct tokens per user |
| --- | ---: | ---: |
| concrete per-user seeds (what ships) | **0.0 %** | **1.00** |
| MAX_UINT32 SKIP sentinel | 96.6 % | 14.84 |

Reproducer: build a 32-user candidate row, call `ttnn.manual_seed(seeds=..., user_ids=...)` then
`ttnn.sampling(...)` with no seed argument, repeatedly, and count distinct results.

**This is worth fixing on its own, but it is not the cause of the word corruption.** Priming the
seed tensor to SKIP before trace capture (`generator_vllm._prime_sampler_seeds_for_trace`, and the
server log confirms `SEED_PRIME wrote SKIP to all 32 slots`) left the garble rate unchanged: 67.24
mean against a 65.68 / 96.00 / 77.50 / 88.97 baseline.

## The sampler's input tensor is identical to a clean reproduction

Dumped from the live server at the sampler's entry point:

    shape=(1,1,32,37984) padded=(1,1,32,37984) dtype=BFLOAT16 layout=TILE
    mem=INTERLEAVED, BufferType::DRAM, shard_spec=nullopt

That is byte-for-byte the descriptor of the `from_torch` tensor used by the standalone mesh
reproduction, which is clean at the same width. So there is no memory-layout difference to correct:
`ttnn.to_memory_config(logits, DRAM)` logged an identical config in and out (a no-op) and the run
still garbled at 34.17. Rules out tile allocation, L1/DRAM placement, page size and padded stride.

## Mitigation in force

`tt-model.yaml` overrides the served generation config to greedy (`temperature 0`), which measures
9.58 and is byte-identical across runs at 22K context. For a code and agentic model that is also
how most callers run it. A request that explicitly sets a non-zero temperature still takes the
sampled path and still shows the corruption.

## Ruled out

- **MTP / speculative decoding** — hard-disabled for this port (`tt/generator_vllm.py`).
- **The decode-perf optimizations** (router large-k top-k route, pairs/gather MoE path,
  down-projection on 64 cores) — all seven A/B variants produce bit-identical 64-token greedy
  streams.
- **Forced-reset frame discards** — zero over the whole run; decode stayed at width 1.
- **Incremental detokenization** — `tokenizers.DecodeStream` round-trips these words exactly.
- **Candidate-set construction in the sampler** — `ttnn.sampling` runs its own bitonic top-k
  over the gathered row, so the concatenated per-device top-32s are re-selected globally.

## What it is

It needs **three things at once**. Remove any one and the output is clean:

1. stochastic sampling (greedy is always clean),
2. the overlapped async-decode fast path (`--async-scheduling`, on by default),
3. enough context (clean below ~2K tokens, intermittent around 8K, consistent by 20K).

Measured at 22K context, 600-token answers, shipped weights, 4 dies:

| condition | mean garble/1k | duplications |
| --- | --- | --- |
| on-device sampling + async decode — **what ships** | 41–94 | 12 |
| greedy (`temperature 0`) | 9.6, three runs, byte-identical | 0 |
| sampling + `logprobs` requested (forces **host** sampling on 4 dies) | 9.8, 17.9 | 0 |
| host sampling (`sample_on_device_mode` unset) | 17–21 | 0 |

Controlled context sweep, same question, filler-padded prompt, sampling on:

| context | garble/1k |
| --- | --- |
| 42 | 17.5, 24.7 |
| 2316 | 12.6, 12.1 |
| 8800 | 107.1, 15.3 |
| 22084 | 67.3, 102.3 |

`logprobs` matters because `vllm_tt_plugin/model_runner.py` refuses device sampling for any
logprobs request unless the mesh has 8 or 32 devices (`if num_devices not in (8, 32): return
False`). On these 4 dies, asking for logprobs silently moves sampling to the host — so that row
is a second measurement of host sampling, not an independent variable.

The tightest constraint on any explanation: **greedy is clean on the same fast path at the same
context**. Greedy is the one mode that never consumes the device RNG.

## A working/broken pair on identical hardware

The user reported the Qwen3.5-122B-A10B bundle behaves fine in daily use. Running the *same*
probe against it - same box, same plugin, same vLLM flags, same 22K prompt, same
`temperature 1.0 / top_p 0.95 / top_k 40`, same on-device sampler - confirms that:

| model | greedy | sampled t=1.0 | duplications |
| --- | ---: | ---: | ---: |
| Qwen3.5-122B-A10B | 7.79 | **11.87** | 2 |
| Qwen3-Coder-Next | 9.58 | **66 - 96** | 10 - 12 |

So this is **model-specific, not a generic sampler defect**. What differs in the sampling path:

| | Qwen3-Coder-Next | Qwen3.5-122B-A10B |
| --- | --- | --- |
| vocab | 151936 | 248320 |
| padded vocab (TP4) | 151936 (no pad columns) | 248320 (no pad columns) |
| per-device shard | 37984 cols = **1187 tiles (prime)** | 62080 cols = 1940 tiles |
| device offsets, invalid-vocab mask | exact, no-op | exact, no-op |
| `ttnn.topk` route | stock (< 64K width) | stock (< 64K width) |

Both are tile-aligned with zero padding, so the offset and mask arithmetic is exact for each.
The visible difference is the shard's **tile count**, 1187 being prime, which is a poor width for
the top-k factory. `TTSampling` has a `pad_to_power_of_2` path that exists precisely for this
("necessary to use full optimization in the topk operation") and this port enables it on TP2 but
not on TP4.

Also note `QWEN35_FORCE_ARGMAX` defaults to `0` in this port, so `allow_force_argmax` is False and
**greedy does not take the argmax branch** - it goes through the same top-k pipeline with
`k=1, p=0`. Greedy therefore proves the gather, the device offsets and the index mapping are
correct, and narrows the defect to the top-p scan and multinomial draw, which only run for
`k>1, p>0`. (An earlier revision of this document claimed greedy bypassed the pipeline; that is
true of the stock qwen36 default but not of this port's configuration.)

## The sampler's PRNG is frozen (real defect, but NOT the cause)

`TTSampling` initialises `seeds_tt_tensor` to `torch.arange(32)` and `SeedManager` starts in its
steady state, so `get_new_values()` early-returns and never replaces those concrete values. The
decode trace captures them, and `ttnn.manual_seed` re-seeds the device PRNG identically on every
replay. Measured directly on one Blackhole (32 users, 24 draws of a fixed distribution):

| seeding | draw changes step-to-step | distinct tokens per user |
| --- | ---: | ---: |
| concrete per-user seeds (what ships) | **0.0 %** | **1.00** |
| MAX_UINT32 SKIP sentinel | 96.6 % | 14.84 |

Reproducer: build a 32-user candidate row, call `ttnn.manual_seed(seeds=..., user_ids=...)` then
`ttnn.sampling(...)` with no seed argument, repeatedly, and count distinct results.

**This is worth fixing on its own, but it is not the cause of the word corruption.** Priming the
seed tensor to SKIP before trace capture (`generator_vllm._prime_sampler_seeds_for_trace`, and the
server log confirms `SEED_PRIME wrote SKIP to all 32 slots`) left the garble rate unchanged: 67.24
mean against a 65.68 / 96.00 / 77.50 / 88.97 baseline.

## The sampler's input tensor is identical to a clean reproduction

Dumped from the live server at the sampler's entry point:

    shape=(1,1,32,37984) padded=(1,1,32,37984) dtype=BFLOAT16 layout=TILE
    mem=INTERLEAVED, BufferType::DRAM, shard_spec=nullopt

That is byte-for-byte the descriptor of the `from_torch` tensor used by the standalone mesh
reproduction, which is clean at the same width. So there is no memory-layout difference to correct:
`ttnn.to_memory_config(logits, DRAM)` logged an identical config in and out (a no-op) and the run
still garbled at 34.17. Rules out tile allocation, L1/DRAM placement, page size and padded stride.

## Mitigation in force

`tt-model.yaml` overrides the served generation config to greedy (`temperature 0`), which measures
9.58 and is byte-identical across runs at 22K context. For a code and agentic model that is also
how most callers run it. A request that explicitly sets a non-zero temperature still takes the
sampled path and still shows the corruption.

## Ruled out

- **MTP / speculative decoding** — hard-disabled for this port (`tt/generator_vllm.py`).
- **The decode-perf optimizations** (router large-k top-k route, pairs/gather MoE path,
  down-projection on 64 cores) — all seven A/B variants produce bit-identical 64-token greedy
  streams.
- **The precision policy / bfp4 LM head** — host sampling is clean on the *same* weights, and
  greedy at 22K context is byte-identical across runs.
- **Chunked prefill / the KV cache** — greedy at 22K context is clean and deterministic.
- **`ttnn.sampling` itself** — a direct unit test on Blackhole with the production-shaped
  candidate row (four concatenated per-device top-32 runs, disjoint per-user id blocks) shows no
  cross-user leak, no draw outside the true global top-k, and a draw distribution matching the
  expected multinomial (total variation distance 0.027).
- **Incremental detokenization** — `tokenizers.DecodeStream` round-trips these words exactly.
- **Forced-reset frame discards** — zero over a full run; decode stayed at width 1.
- **The overlapped async-decode fast path** - `--no-async-scheduling` leaves it unchanged
  (sampled mean 96.0 at 22K context vs 65.7 with async scheduling on; greedy 9.58 either way).
  Single-user decode is 40.8 tok/s with it off vs ~42 on, so it is not a useful lever here.
- **The sampler's own internal trace.** `SamplingGenerator.sample()` captures a trace separate
  from the model's decode trace, and the qwen36 demo warns that exactly this "aliases the model
  trace's buffers -> intermittent garbage". Forcing the sampler to run eagerly instead changed
  nothing: mean 77.5 (36.6 / 112.6 / 91.8 / 69.0) against 66-96 traced.
- **An invalid / padded token id reaching the detokenizer.** vLLM logs `Encountered invalid
  token id` and substitutes an empty string, which would look exactly like a dropped token.
  Zero occurrences across every server run.
- **The per-device shard's awkward top-k width.** Padding each shard up to the next power of two
  before the sampler's `ttnn.topk` (`TTSampling.pad_to_power_of_2`, which this port already enables
  on TP2) made no difference: sampled mean 88.97 with it on, against 66-96 without. 1187 being a
  prime tile count is therefore not the trigger.
- **`top_k`.** Sweeping it at 22K context: k=8 -> 61.3 mean, k=16 -> 102.4. No trend, so neither the
  checkpoint's recommended 40 nor the silent clamp to 32 nor the 32-wide candidate-row boundary is
  implicated.
- **Power-of-two pad columns leaking into the candidate set.** The qwen36 base already sets
  `pad_logits_to_power_of_2 = True`, so each device's shard is padded to 65536 before the top-k:
  37984 -> 65536 here (27552 pad columns, 42 % of the row) against 62080 -> 65536 on the 122B
  (3456, 5 %). `_create_indices_tensors` derives global ids from the UNPADDED width, so a pad
  column reaching the top-k would produce an out-of-range id. Tested on the mesh with all-negative
  logits (the dangerous case): `-sys.float_info.max` casts to `-inf` in bf16 and **0 of 1024**
  top-32 entries came from the pad region, for either model. Not the trigger.

  This also explains why forcing `pad_logits_to_power_of_2` on changed nothing: it was already on.
- **A frozen sampler RNG** — the `SeedManager` state machine does stay in its concrete-seed
  state because the serving path never calls `get_new_values()`, but advancing it every decode
  step changed nothing measurable (134 / 86 / 108 against a 102 baseline), so that deviation is
  not the cause. The change was reverted.

## Where it is localised to

**Every path that consumes the FULL gathered logits is clean. Only the per-shard top-k path is not.**

- greedy: `tt_sampling.py` takes an early `force_argmax` branch -- all-gather the whole vocab, one
  `ttnn.argmax`. Clean, and byte-identical across runs. It never touches the top-k pipeline, which
  is why greedy proves nothing about that pipeline.
- host sampling: vLLM reads the full logits and samples in torch. Clean.
- device sampling: per-device `ttnn.topk(k=32)` over each vocab **shard**, all-gather of values and
  indices, device-offset add, `ttnn.sampling`. This is the only garbling path.

That pipeline was reproduced standalone on the real (1,4) Blackhole mesh at the production vocab
size and found correct -- 0 % cross-user leak, 0 % draws outside the true global top-32, draw
distribution matching the expected multinomial -- under smooth, large-offset, flat, heavily-tied,
peaked and dense-body logit distributions alike.

So the op and the pipeline are correct on a tensor built with `from_torch`. What the standalone test
does **not** reproduce is the real sharded LM-head output: its memory config, sub-core grids,
`sampling_memory_config`, and `_mask_invalid_vocab_logits` over the real padded width. That, and
whatever about it changes with sequence length, is the remaining surface.

## Mitigations

None that is both clean and cheap. Host sampling is correct but ~5x slower (8 tok/s vs 42).
Greedy is correct but is not an acceptable serving default. Lowering temperature reduces the rate
(102 -> 65 at `temperature 0.7`) without fixing it.

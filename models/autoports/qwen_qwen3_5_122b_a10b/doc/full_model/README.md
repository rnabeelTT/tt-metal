# Full model — Qwen/Qwen3.5-122B-A10B on the (1,4) Blackhole mesh (stage 6)

**Headline (full 48-layer model, (1,4) P300x2, B = 1, on-device greedy sampling, top-k candidate strategy, 2026-09-25):**
readiness prefill check top-1 0.98 / top-5 1.00 / top-100 1.00 (AIME24 chat, 161 tokens) and 0.97 / 0.99 / 1.00 (book,
128 tokens); traced teacher-forcing token-out decode 36.5 t/s/u (27.4 ms per token, AIME24; book: 36.5 t/s/u) with top-1
0.97 / top-5 1.00 / top-100 1.00; TTFT 632 ms (161 tokens) / 395 ms (128 tokens); e2e 29.6 t/s/u (AIME24). Token-out
decode includes the sampler trace, the `tt_out_tok` feedback and a 1-token readback per step; the teacher-forcing and
token-out numbers are the same measurement here (the readiness runner forces the next input but takes the token from the
token-out path). The qwen36 force-argmax strategy measured 33.2 t/s/u (30.1 ms) and is kept as an A/B knob. Full context
262144 is supported for one user; the shared KV pool holds ~2.0 M tokens per device (section 2). Batch 32 with mixed
prompt lengths is verified on the reduced model (section 4). Autoregressive and chat-template qualitative checks: section 4.

`tt/model.py` — `Qwen35MoeModel(Qwen36Model)`: embeddings, 48 decoder layers (`Qwen35MoeOptimizedDecoderLayer`, the stage-5
optimized multichip layer with the `MULTICHIP_SELECTED` policy installed once on the args), final `DistributedNorm`,
vocab-sharded bfp8 LM head, on-device sampling (`models/common/sampling` `SamplingGenerator`). `tt/generator.py` —
`build_generator(model_dir, mesh_device, **kw)` -> `Qwen35MoeGenerator` (the `readiness_check.contract.Generator`).
`tools/readiness.py` — runs the readiness runners on a mesh opened with the trace region / small-L1 this model needs.

## 1. Selected implementation and why

`Qwen35MoeModel` subclasses the qwen36 dense model (`models/demos/blackhole/qwen36/tt/model.py`) and swaps the decoder
layer class for the stage-5 layer (`DECODER_LAYER_CLS = Qwen35MoeOptimizedDecoderLayer`). Everything the earlier stages
validated is reused unchanged and only once per process:

- the layer stack: `MultichipDecoder._args_hook(MULTICHIP_SELECTED)` installs the stage-5 policy on the shared args before
  any layer is built (TP4 mixers on 1D decode grids, TP-on-intermediate routed experts bfp4, shared expert on 16 cores,
  bfp8 mixers, bf16 GDN state, bfp8 paged KV, fractured `[1,1,B|T,768]` residual between layers, one reduce-scatter per
  row-parallel boundary and one all-gather per norm); the rejection ledger of stage 5 is not revisited here;
- weights: `StreamingStateDict.layer(i)` hands each layer only its own tensors from the safetensors shards (the base class
  `substate` would read the whole 250 GB checkpoint per layer). There is no ttnn tensor-file cache by default
  (`ttnn.load_tensor` of a 67 MB 4-shard TILE tensor and the reload of the 1.5 GB ROW_MAJOR embedding never return, see
  the work log; `QWEN35_TENSOR_CACHE=1` re-enables it), so every process converts the 48 layers from the shards at start;
- terminal path (qwen36): bf16 replicated embedding, `DistributedNorm` final norm (all-gather of the fractured residual),
  vocab-sharded bfp8 LM head (62080 logits per device), `SamplingGenerator` on-device sampling with the sampler trace
  bound to the persistent token input (`tt_out_tok`);
- decode inputs are device-owned: tokens uint32 `[1,1,1,32]` (the sampler's padded batch shape, so `ttnn.sampling` /
  `ttnn.argmax` write the sampled tokens straight into it; the step slices its B tokens out of a row-major view),
  `current_pos` int32 `[B]`, `rot_idxs` uint32 `[1,B]`, page table int32. RoPE cos/sin are gathered on device from resident tables (`ttnn.embedding(rot_idxs, table)`), and
  `ttnn.plus_one` advances `current_pos` and `rot_idxs` inside the decode trace, so a token-out step needs no host write
  (`_tt_vllm_always_refresh_decode_trace_inputs = False`).

`Qwen35MoeGenerator` implements the readiness `Generator` contract around the stock `models.tt_transformers.tt.generator.
Generator` (`decode_forward` traced, `reset_batch`, `reload_inputs`): `prefill_forward` (masked fixed-bucket prefill in
2048-token segments, last-token logits; `return_all_logits=True` for the prefill check), `decode_forward` (`[B]` sampled
tokens, or `[B, vocab]` logits in the explicit host-sampling compatibility mode), `prefill_logits`, `generate`
(greedy, `next_input` teacher forcing: the host is authoritative only for a forced token or the first step after prefill),
`reset`, `teardown`. Build order is part of the contract (qwen36 rule #48536, work log 17:55): the masked-bucket prefill
programs and their persistent buffers are warmed BEFORE the decode trace is captured, otherwise the first prefill after a
traced decode allocates into the trace's scratch range and the next replay corrupts it (the smoke-8 hang).

The mesh must be opened with a trace region and `l1_small_size >= 24576`; the readiness CLIs open a bare mesh (trace
region 0), so `tools/readiness.py` opens the mesh correctly and calls the runners' programmatic entry points unchanged,
and `build_generator` fails fast with that instruction on a bare mesh (an in-process close + re-open of the FABRIC_1D mesh
hangs).

## 2. Context contract (full stack)

Measured on the 48-layer build (`logs/06-warm-fixed-1.log` MEMORY_REPORT, B = 1, `max_seq_len` = 262144, after the prefill
warm-up; per device = 8 DRAM banks):

| item | per device |
|---|---|
| DRAM total (allocatable) | 4,205,232,512 B x 8 = 31.33 GiB |
| allocated after build + prefill warm-up (weights 17.0 GiB bfp4/bfp8, embedding bf16, LM head bfp8, GDN state, dispatch constants, prefill persistent buffers, KV pool for 1 x 262144) | 2,639,950,208 B x 8 = 19.67 GiB |
| of which the paged KV pool for one user at 262144 (12 attention layers, 1 local KV head, bfp8 K and V, block 64: 4096 blocks) | 1.59 GiB (6,528 B = 6.38 KiB per token per device: 12 layers x K,V x 256 x bfp8) |
| non-KV footprint | 18.08 GiB |
| trace region (decode model trace + sampler trace, B = 1) | 3,579,904 B x 8 = 27.3 MiB used of the 512 MiB reserved (`logs/06-prefill-check-qwen35_122b_aime24_chat_g100.log` MEMORY_REPORT) |
| L1 per core held by the model (shared router-gate buffers) | 10,496 B |

KV pool available for serving = 31.33 - 18.08 - ~0.5 GiB allocator headroom = 12.75 GiB -> ~2,095,000 tokens per device (all
users together; 6,528 B per token). The `total_bytes_per_bank` figure already excludes the trace region (it drops by exactly the
region size when a larger region is reserved: 4,234,838,400 B with 300 MB vs 4,205,232,512 B with 512 MiB), so the region is not
subtracted again. The HF-advertised context 262144 is supported for a single user (1.59 GiB); with 32 concurrent users the pool
is shared (`max_tokens_all_users`) and the GDN recurrent + conv state grows by 19.8 MB per extra user per device (31 x 19.8 MB =
0.57 GiB), i.e. ~2,000,000 pooled tokens = ~62,000 tokens per user on average — the same DRAM-limited statement as the stage-4
contract (`doc/context_contract.json`), now with the measured full-stack footprint instead of the planned one. The vLLM stage
sizes `max_num_batched_tokens` / the block pool from these numbers; no advertised capability is reduced.

## 3. Sampling: choice and rejected alternative

Both common implementations were read before the token-out path was designed:

| | `models/common/sampling` (`SamplingGenerator` + `TTSampling`) — SELECTED | `models/common/modules/sampling/sampling_1d.py` (`Sampling1D`) — rejected |
|---|---|---|
| vocab-sharded logits | native: per-shard top-k (`max_top_k` = 32 candidates), one all-gather of the candidate tiles, then `ttnn.sampling` with (k, p, temp) per user — split greedy is `k=1`, no full-vocab all-gather, no global argmax | expects gathered logits per device; its argmax path is `ttnn.argmax` over the full vocab after an all-gather of `[B, vocab]` (4x the bytes) |
| trace | owns the sampling trace slots keyed by (penalties, log-probs, force-argmax) and `set_trace_bucket` for decode bucketing; `tt_out_tok` bound to the model's persistent token input | no trace ownership; the generator would need its own capture wrapper |
| state | seeds per slot (`SeedManager`), penalties (`TTPenalties`), log-probs, fixed slots / inactive rows — the vLLM contract of stage 9 | seeds/penalties/log-probs partially; no slot bookkeeping |
| mesh | (1,4) TP, `tt_ccl` shared with the model (already integrated in qwen36 on this mesh) | 1D meshes, tuned all-gather configs clamped per submesh; not exercised with this model's `TT_CCL` |

`SamplingGenerator` is the sampler qwen36 already drives on this mesh, it needs no custom code, and its greedy path is
semantically greedy (`GREEDY = SamplingParams(temperature=0.0, top_k=1, top_p=1.0)`, normalised by the sampler to k=1 / temp 1).
Which greedy STRATEGY runs is decided by the qwen36 model config: `SAMPLING_AG_CONFIG["allow_force_argmax"] = True`
(`qwen36/tt/model_config.py`), so every 48-layer run of this stage took the FORCE-ARGMAX path (one all-gather of the full
padded vocab, bf16 `[1,1,32,4 x 62080]`, then `ttnn.argmax`; the logs print "Forcing argmax sampling" at capture) — not the
top-k candidate path (per-shard top-k of 32, one all-gather of the candidates, `ttnn.sampling` k=1) the tracing skill
recommends for vocab-sharded logits. Both are measured on the 2-layer model (`tests/test_generator.py::
test_generator_greedy_sampler_paths`; a Tracy profile of the token-out step is not available — the ops merge fails on the
sampler stack's single-device host reads, `AssertionError: Device data missing: Op ... not present in
cpp_device_perf_report.csv for device 1`, so the 4-way wall breakdown below is the evidence): the force-argmax step measured 5.12 ms = model trace 1.68 + sampler trace 3.34 + token readback 0.05 + host 0.04 ms and the
top-k candidate step 2.42 ms = 1.68 + 0.57 + 0.05 + 0.12 ms on the 2-layer model (`logs/06-generator-tests5.log`), i.e. the
full-vocab all-gather + argmax costs 3.3 ms per step against 0.57 ms for the candidate path — so the TOP-K CANDIDATE PATH IS THE DEFAULT of this port (`create_model` sets `allow_force_argmax=False`;
`QWEN35_FORCE_ARGMAX=1` restores the qwen36 default for A/B). To let `ttnn.sampling` write the sampled tokens back into
the persistent decode token input (it requires a rank-4 `[1,1,1,S]` output, S = the sampler's padded batch of 32), the
token buffer now has that shape (`prepare_decode_inputs_host`) and the decode step takes its B tokens out of it with a
row-major view + slice (`_decode_tokens_view`). Both strategies return the prefill-path ground truth and feed the token
back (`test_generator_greedy_sampler_paths`); the 48-layer numbers in sections 4 and 6 are re-measured on the default.
The host-sampling compatibility mode (`on_device_sampling=False`) runs the decode with the all-gathered logits
(`_lm_head`) and reads one replica back; it exists for shared tests only and is never the measured path. It runs
untraced unless its trace was captured at build (`warm_host_trace=True`): a decode trace captured mid-generation executes
the step twice (compile pass + first replay) and advances the GDN state twice (section 4, mode probe).

## 4. Correctness: prefill check, teacher forcing, autoregressive

References (`bringup/references/`, readiness_v1, k = 100, generated by `readiness_check.generate` from the HF model on the CPU,
2026-09-24 00:52-00:54 UTC): `qwen35_122b_aime24_chat_g100.refpt` (AIME24 problem through the HF chat template, 161 prompt
tokens, 100 generated; sha256 b8b7d0cd0e3650b3c6697b1ae732982b63d8cc728a890d5f04e7179caf6aedbf) and `qwen35_122b_book_p128_g100.refpt` (A Tale of Two Cities, 128 prompt tokens, 100
generated; sha256 a0443856cd85737878d059f73668ab333978e069e38ec64fb1611137f1b9859e) — both produced by `python -m readiness_check.generate --hf-model Qwen/Qwen3.5-122B-A10B ...`
(`--prompt-source aime24 --chat-template --gen-len 100` / `--prompt-source book --prompt-len 128 --gen-len 100`, HF CPU, bf16
checkpoint dtype; `bringup/references/` is gitignored, the hashes pin the files). The
AIME24 reference was produced with the HF tokenizer's chat template (`--chat-template`) and 100 tokens, i.e. the exact
artifact the goal asks for; its metadata (`hf_model_id`, `k`, `tf_prompt_len` 161) is in the file.

Runner: `python -m models.autoports.qwen_qwen3_5_122b_a10b.tools.readiness {prefill-check|teacher-forcing} --reference
bringup/references/<ref>.refpt` (opens the (1,4) mesh with the trace region and small-L1 this model needs, then calls the
packaged `run_prefill_check` / `run_teacher_forcing` unchanged; one process per run, `tt-smi -r` between runs, logs under
the bringup workspace `logs/06-*.log`, reports under `readiness/`).

| check | reference | top-1 | top-5 | top-100 | TTFT / decode | artifact |
|---|---|---|---|---|---|---|
| prefill (all 161 positions through norm + LM head, host argmax vs HF top-k) | AIME24 chat | 0.980 (98/100) | 1.000 | 1.000 | — | `readiness/prefill_check_qwen35_122b_aime24_chat_g100.json` |
| prefill | book p128 | 0.970 (97/100) | 0.990 (99/100) | 1.000 | — | `readiness/prefill_check_qwen35_122b_book_p128_g100.json` |
| teacher forcing (traced decode, on-device sampler token vs HF top-k, next input forced; top-k candidate sampler = the committed default) | AIME24 chat | 0.970 (97/100) | 1.000 | 1.000 | TTFT 632.1 ms (161 tokens), decode 36.45 t/s/u (27.4 ms/token), e2e 29.62 t/s/u — with the force-argmax sampler (first run, `logs/06-teacher-forcing-*`): 0.980 / 1.000 / 1.000, 626.7 ms, 33.24 t/s/u (30.1 ms/token) | `readiness/teacher_forcing_qwen35_122b_aime24_chat_g100.json` |
| teacher forcing (top-k candidate sampler) | book p128 | 0.970 (97/100) | 0.990 (99/100) | 1.000 | TTFT 394.9 ms (128 tokens), decode 36.49 t/s/u (27.4 ms/token), e2e 31.90 t/s/u — force-argmax first run: same top-k, 389.5 ms, 33.05 t/s/u | `readiness/teacher_forcing_qwen35_122b_book_p128_g100.json` |

Bar: top-5 >= 98 %, top-100 = 100 % (the top-1 bar of the datatype-sweep stage is 90 %).

**Autoregressive** (`readiness autoregressive --prompt-file <plugin autoregressive_prompt.txt> --output-dir autoregressive`,
256 new tokens, HF greedy on the CPU first, then the TT generator with on-device greedy sampling):

| | HF (CPU, `from_pretrained` defaults = the bf16 checkpoint dtype, greedy) | TT ((1,4) mesh, bfp4 routed experts / bfp8 mixers, on-device greedy) |
|---|---|---|
| tokens produced | 256 | 256 |
| identical prefix | first 20 tokens (`... the birds were singing`), then `a different tune` (HF) vs `in a rhythm she had never heard before` (TT; identical fork point and prefix on both sampler strategies, `logs/06-autoregressive.log` / `06-autoregressive2.log`) | |
| matching positions after the fork | 24 / 256 positions equal (informational: greedy streams stay apart once forked) | |
| coherence | narrative continues (Elena, the western mountain, the old man, a hidden cave), grammatical, on topic | same opening, then a hermit on the western mountain and the "mountains breathe every hundred years" motif (committed top-k run; the force-argmax run told the old man / Whispering Cave variant), grammatical, on topic, no repetition loops, no language drift |

Verdict: PASS — both completions are coherent continuations of the prompt in English; the TT stream forks from HF at
token 20 on a near-tie continuation (both runs) ("a different tune" / "in a rhythm ...") and stays coherent for the remaining 236
tokens (no duplicated sentences, no trigram loops — machine check `check_degenerate_output --scope autoregressive` in
`06-full-model.check.sh`). Early divergence on a free-running greedy stream is expected with bfp4 routed experts; the
per-position agreement that matters is the teacher-forcing top-k above (98 % / 97 % top-1, 100 % top-100).
Files: `autoregressive/{hf_completion.txt,tt_completion.txt,autoregressive_meta.json}`.

**Qualitative check, shared suite through the chat template** (`qualitative/`, `scripts/06-qualitative.py`, `logs/06-qualitative.log`;
`qualitative_prompt_format.json` records the format): the six readiness prompts (`runtime/readiness_check/vllm_prompts.txt`)
rendered with `apply_chat_template(add_generation_prompt=True)` (no system prompt; the Qwen3.5 template opens a `<think>`
block, so the 128 generated tokens are the model's reasoning trace), HF greedy on the CPU (bf16) as the control, TT greedy
(top-k candidate sampler, the committed default; `logs/06-qualitative2.log` — the first run with the force-argmax sampler,
`06-qualitative.log`, forked at 78 / 9 / 60 / 29 / 36 / 3 and read the same) on the 48-layer model, 128 new tokens each, one process.

| id | prompt | prompt tokens (chat template) | TT tokens | first divergence from HF | equal positions | TT seconds |
|---|---|---|---|---|---|---|
| 0 | Write a haiku about machine learning. | 18 | 128 | 97 | 97 / 128 | 3.8 |
| 1 | Explain the difference between supervised and unsupervised l | 25 | 128 | 84 | 100 / 128 | 3.8 |
| 2 | Complete this story: Once upon a time, in a faraway kingdom, | 33 | 128 | 65 | 66 / 128 | 3.8 |
| 3 | What are the three laws of thermodynamics? | 19 | 128 | 43 | 43 / 128 | 3.8 |
| 4 | Translate the following to French: "Hello, how are you today | 24 | 128 | 36 | 38 / 128 | 3.8 |
| 5 | Write a Python function to calculate the Fibonacci sequence. | 20 | 128 | 15 | 17 / 128 | 3.8 |

Read in full (`qualitative/prompt_<i>/{hf,tt}_completion.txt`): every TT completion is a coherent, on-task reasoning trace
in English with the same structure as HF's ("Analyze the Request / Brainstorming / Drafting"; the Fibonacci prompt closes
its `<think>` block and starts the answer in both), no repetition, no language drift, no format break; the streams fork at a
near-tie token between position 15 (Fibonacci) and 97 (haiku) and stay on topic afterwards. Machine check (`06-full-model.check.sh` -> `check_degenerate_output --scope autoregressive`, over the
seven `autoregressive_meta.json` files): no degenerate output (`logs/06-full-model-check3.log`). Verdict: PASS. The
Elena raw-prompt continuation above is kept as continuation-stress coverage; the instruct-format verdict rests on this suite.

**Generator contract tests** (`tests/test_generator.py`, 2-layer reduced model on the mesh, real weights; run in the
bringup workspace `logs/06-generator-tests2.log`):

| test | what it checks | result (run 5, 9 passed in 5:36) |
|---|---|---|
| `test_generator_cross_request_token_feedback` | after `generate(A, 8)` a request with prompt length len(A)+8 and len(A)+7 decodes its first token from its own prefill (the stock reset merge would otherwise keep the stale device token); token feedback on the second token; determinism | PASS |
| `test_generator_batched_mixed_prompts[2, 8, 32]` | B users with mixed aligned/non-aligned prompt lengths (up to 513) through the warmed per-slot prefill + traced decode with on-device sampling; every user's 4 tokens equal its own B=1 generation | PASS for B = 2, 8, 32 (largest tested batch 32 = the serving target; no hard limit hit) |
| `test_generator_long_prompt_segments[2381, 4097]` | prompts longer than one 2048-token segment through the segmented masked-bucket prefill: the token after the prompt equals the token from prefill(T-1) + one traced decode; T=2381 also equals the single padded all-positions pass (PCC 0.9994) | PASS |
| `test_generator_greedy_sampler_paths[force_argmax, topk_candidates]` | both greedy strategies return the prefill truth, feed the token back (near-tie aware) and are deterministic; 4-way step breakdown | force-argmax 5.119 ms = model trace 1.680 + sampler 3.344 + readback 0.051 + host 0.044; top-k candidates 2.417 ms = 1.681 + 0.567 + 0.052 + 0.117 (2-layer model) |
| `test_generator_all_logits_after_traced_decode` | the all-positions prefill (readiness prefill-check path) is bit-identical before/after traced decodes ran in the process and matches the masked-bucket path (PCC 1.0, same top-2) | PASS |

Near-ties: the test prompt's continuation has two candidates 0.06-0.4 logits apart (8599 / 40527); prefill and decode take
different matmul programs, so a token test accepts the runner-up when the margin is below one logit (`_truth_set`); the
readiness top-k metrics quantify such swaps. Cross-process reproducibility of the prefill logits was checked separately
(`logs/06-det-{a,b,c}.log`: three processes, two trace-region sizes, bit-identical top-5 values); the pytest fixture mesh
gives slightly different logits from the readiness mesh (near-tie margin 0.062 vs 0.375) — same top-2, an environment
(mesh parameter) difference recorded as a hard-check gap.

**Sampler comparison / mode probe** (2-layer reduced model, `logs/06-mode-probe-*.log`): the on-device sampler's token equals
the prefill-path ground truth in both capture orders; the host-logits mode agrees when untraced or when its trace is
captured at build, and disagrees only when its trace is captured mid-generation (section 3).

## 5. Trace evidence (token feedback, positions, page table)

`scripts/06-trace-evidence.py` (bringup workspace; log copied to `trace_evidence.log`) runs on the reduced 2-layer model
(layers 0 and 3, real weights, real KV/page-table/sampler shapes) and reads the PERSISTENT decode-trace inputs back after
each replay:

| check | result (`trace_evidence.log`, re-run 02:45 UTC on the committed rank-4 token buffer and top-k sampler) |
|---|---|
| token feedback | after each replay the persistent device token input holds the token the sampler produced in that replay (steps 1-3: 8599, 43506, 3149); the host wrote tokens only for the first step after prefill |
| position coherence | `current_pos` and `rot_idxs` advance by exactly 1 per replay on device (20 -> 21 -> 22 for a 19-token prompt + first token), no host refresh |
| input buffer identity | the four trace inputs keep their Python/device identity across replays (`same_input_buffers=True`) |
| unchanged page table | no copy: `prev_page_table` object kept, device content equal |
| changed page table | the two trailing (unused) block ids swapped on the host -> the device table shows the new tail `[63, 62]` after the step, and the device token input still holds the sampled token (the page-table copy does not touch the token/position inputs) |
| determinism | two identical 8-token generations are bit-identical; a different prompt gives a different sequence |
| token-out step (2-layer model) | 2.35 ms per step incl. the top-k sampler trace and the 1-token readback (5.15 ms with the force-argmax sampler of the first runs): layer device work ~1.0 ms, model trace replay 1.68 ms, sampler 0.57 ms (section 6) |

The split-sampling contract holds: the model trace ends in sampler-ready vocab-sharded logits, the `SamplingGenerator`
trace consumes them and writes `tt_out_tok` into the model's persistent token input, positions advance on device, and no
host argmax / full-logits readback / untraced sampling is on the token-out path.

## 6. Performance (TTFT, decode t/s/u, layer-stack lower bound)

All numbers: full 48-layer model, B = 1, (1,4) P300x2, on-device greedy sampling, measured by the readiness teacher-forcing
runner (`generate(..., next_input=..., enable_trace=True)`: the token comes from the token-out path, the next input is
forced) and by the trace-evidence script on the reduced model.

| quantity | value | source |
|---|---|---|
| TTFT, 161-token prompt (prefill through the 256-token masked bucket, incl. the first token) | 632.1 ms (626.7 with the force-argmax sampler) | `logs/06-teacher-forcing2-qwen35_122b_aime24_chat_g100.log` |
| TTFT, 128-token prompt | 394.9 ms (389.5 with the force-argmax sampler) | `logs/06-teacher-forcing-qwen35_122b_book_p128_g100.log` |
| traced decode, teacher-forcing token-out (sampler + 1-token readback per step), top-k candidate sampler (committed default) | 36.45 t/s/u = 27.4 ms per step (AIME24), 36.49 t/s/u (book) | `logs/06-teacher-forcing2-*` |
| same, force-argmax sampler (first run) | 33.24 t/s/u = 30.1 ms (AIME24), 33.05 t/s/u (book) | `logs/06-teacher-forcing-*` |
| end-to-end over 100 generated tokens (top-k default) | 29.62 t/s/u (AIME24), 31.90 t/s/u (book) | `logs/06-teacher-forcing2-*` |
| layer-stack lower bound, device time (stage-5 Tracy per layer: 36 GDN x 0.472 ms + 12 attention x 0.406 ms) | 21.9 ms per step | `doc/optimized_multichip_decoder/PERF_TABLES.md` |
| layer-stack traced wall (stage-5 host walls: 36 x 0.583 + 12 x 0.449 ms) | 26.4 ms per step | `doc/optimized_multichip_decoder/HOST_WALLS.md` |
| token-out step on the 2-layer reduced model: force-argmax sampler (first runs; readback wait 4.7 ms = the sampler's full-vocab gather) | 5.15-5.20 ms | `logs/07-tokenout-profile.log` (force-argmax era) |
| token-out step on the 2-layer reduced model: top-k candidate sampler (committed default; 4-way breakdown in section 4) | 2.42 ms | `logs/06-generator-tests5.log` |

Gap analysis: the committed token-out step is 27.4 ms (top-k candidate sampler), 5.5 ms (25 %) above the 21.9 ms
layer-stack device lower bound and 1.0 ms above the sum of the per-layer traced walls. The 2-layer breakdown
(`tests/test_generator.py::test_generator_greedy_sampler_paths`, `logs/06-generator-tests5.log`) isolates the fixed
per-step cost as device time: model trace replay 1.68 ms (1.0 ms of layer device time + ~0.7 ms per-replay dispatch),
sampler trace 0.57 ms (top-k candidates) vs 3.34 ms (force-argmax: the full-vocab all-gather + argmax), token readback
0.05 ms, host 0.04-0.12 ms. Switching the greedy strategy removed 2.7 ms per token at 48 layers (30.1 -> 27.4 ms). What is
left above the lower bound: ~0.7 ms of trace-replay dispatch, the 0.57 ms sampler trace, the terminal norm / LM-head ops
inside the model trace, and the readback; the optimized-full-model stage owns them together with the counting-sort
prefill cost (stage-5 README section 2), traced prefill (TTFT) and the ~1 us per-op dispatch gap on the mesh.

Prefill: 161 tokens run through the 256-token masked bucket in one program per layer (48 x ~ (device 8-9 ms for 128 tokens
per the stage-5 tables, scaled) plus the host dispatch of ~150 ops per layer); the 626.7 ms TTFT is dominated by that host
dispatch — traced prefill is the stage-7 lever (the 512-token chunk trace hang is the open item, stage-5 README section 3).

## 7. Runtime fallback audit, limitations, artifacts

**Fallback audit (model + generator paths)**

| path | owner | host boundary |
|---|---|---|
| embeddings | `Qwen36Model` bf16 replicated `ttnn.embedding` on device | token ids only (decode: the device token input, refreshed from the host only when the caller is authoritative) |
| decoder layers x48 | stage-5 `Qwen35MoeOptimizedDecoderLayer`, fractured residual, no inter-layer gather | none (the stage-4/5 audit: attention 0 host transfers, GDN non-aligned prefill 3 inside the shared kernels) |
| final norm + LM head | `DistributedNorm` (one all-gather) + vocab-sharded bfp8 `ttnn.linear` | none in token-out mode; one replica of the all-gathered logits is read back only in the host-sampling compatibility mode and for `prefill_logits` (the prefill check needs every position) |
| sampling | `SamplingGenerator` trace, `tt_out_tok` = the model's persistent token input | one 32-token readback per step (`read_from_device=True`, the readiness contract needs the token on the host) |
| positions / RoPE | `ttnn.plus_one` on `current_pos` and `rot_idxs` inside the trace; cos/sin gathered on device | none |
| KV cache / page table | generator-owned paged KV (block 64) and page table; the page table is copied only when its content changes | page-table copy on change only |
| GDN recurrent + conv state | model-owned persistent tensors, zeroed in place by `reset()` (trace addresses preserved) | none |
| prefill | masked fixed-bucket programs (128..2048) in 2048-token segments with `chunk_start`; the generator owns padding, masking, cache fill and output slicing | token ids in, last-token logits (or all logits) out |

**Limitations / open items (handed to the optimized-full-model stage)**

- Token-out decode 27.4 ms per step vs a 21.9 ms device lower bound (section 6): ~0.7 ms trace-replay dispatch, the
  0.57 ms sampler trace, the terminal norm / LM-head ops and the 1-token readback; non-blocking replay with a deferred
  readback is not used by the readiness contract (it reads the token every step).
- TTFT is host-dispatch bound (eager masked-bucket prefill, ~150 ops per layer); traced prefill is blocked by the 512-token
  chunk replay hang (stage-5 README section 3).
- The counting-sort dispatch costs ~2.7 ms device per 512-token prefill chunk (stage-5 README section 2); the sentinel rows'
  out-of-range source-token index (inert) is masked with the next dispatch change.
- No ttnn tensor-file cache: each process converts the 48 layers from the safetensors shards (7-8 minutes); the
  `ttnn.load_tensor` reload path of large multi-device tensors never returns (work log, ttnn backlog item).
- `norm_ag_persistent` (persistent decode norm all-gather buffers, -5 us per layer step) is opt-in only until its bit-identity
  loss on the attention traced replays is understood.
- Host-sampling compatibility mode runs untraced unless `warm_host_trace=True` (section 3).
- Serving-contract shape for the vLLM stage: `prefill_forward` maps row u -> decode slot u and `reset()` zeroes every slot
  (no `empty_slots` / per-slot reset yet; qwen36's `prefill_paged_slots` machinery is there for it); the per-slot prefill
  round-trips each user's GDN state through the host (~20 MB per user per device at 48 layers) — correct, a latency item
  for stages 7/9; `prefill_logits` (all-positions pass) is a `max_batch_size == 1` path (used by the readiness prefill check).
- Teacher-forcing top-1 on AIME24 moved 98 -> 97 between the two greedy strategies on identical forced inputs (top-5 and
  top-100 unchanged): a bf16 tie resolved differently by `ttnn.argmax` and `ttnn.sampling`; the flipped position's margin is
  not yet logged (stage-7 check). Logits also differ slightly between the pytest fixture mesh and the readiness mesh (same
  top-2, near-tie margin 0.062 vs 0.375; device-open parameters differ) — recorded, not resolved.
- Batch: the 48-layer readiness numbers are B = 1 (the readiness contract); the generator, KV pool, page table, per-slot
  prefill and the padded decode trace are sized from `max_batch_size` (`QWEN35_MAX_BATCH`), and `tests/test_generator.py`
  exercises B = 2 / 8 / 32 with mixed prompt lengths on the 2-layer model (every user equals its B=1 tokens); 48-layer
  B = 32 end-to-end is the vLLM stage's serving benchmark.

**Artifacts**

- `tt/model.py`, `tt/generator.py`, `tools/readiness.py`; `tt/fused_decoder.py` (`_share_gate_buffers`).
- `readiness/prefill_check_*.json`, `readiness/teacher_forcing_*.json`, `autoregressive/{hf_completion.txt,tt_completion.txt,
  autoregressive_meta.json}`, `trace_evidence.log`, `triage/warm_hang_*.txt` (tt-triage captures of the sort deadlock).
- Bringup workspace logs (`~/qwen35-122b-bringup/logs/06-*.log`): smokes 1-10, mode probe, host-mode checks, trace evidence,
  warm-up reproducers (hang, watcher, MMRS-off, fixed x2), L1 probes, sort stress, readiness runs; scripts `06-*.{py,sh}`.
- `doc/context_contract.json` block `full_model_stage6`.

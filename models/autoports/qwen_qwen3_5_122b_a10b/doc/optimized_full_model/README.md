# Optimized full model — Qwen/Qwen3.5-122B-A10B on the (1,4) Blackhole mesh (stage 7)

**Headline (before -> after, full 48-layer model, (1,4) P300x2, B = 1, on-device greedy sampling, top-k candidate
strategy, 2026-09-25):** token-out decode 27.31 -> **25.06 ms per step (36.6 -> 39.9 t/s/u**, `tools/tokenout_bench.py`,
64 steps, blocking readback = the readiness contract; 24.90 ms with the deferred-readback serving split); traced
teacher-forcing decode 36.45 -> **39.67 t/s/u** (AIME24 chat) and 36.49 -> **39.66 t/s/u** (book), same measurement
family as stage 6 (token-out path, next input forced); warmed batch-1 TTFT 632.1 -> **557.3 ms** (AIME24, 161 tokens)
and 394.9 -> **385.2 ms** (book, 128 tokens) by the teacher-forcing runner (`doc/full_model/readiness/*.json` vs
`readiness/*.json`); the prefill bench measures the same policy's stock host-upload path vs traced replays at
349.6 -> 344.0 ms (128 tokens) and 561.2 -> 555.8 ms (161 tokens) (`prefill_bucket_bench_48layers.json`). Accuracy on the refreshed AIME24 chat / book references:
prefill check 0.97 / 1.00 / 1.00 and 0.97 / 0.99 / 1.00, teacher forcing 0.98 / 1.00 / 1.00 and 0.97 / 1.00 / 1.00
(top-1 / top-5 / top-100; bars top-5 >= 98 %, top-100 = 100 %). What changed: seven decode op-count knobs (GDN decode
layer 103 -> 82 ops, 0.579 -> 0.523 ms; attention 84 -> 77 ops, 0.446 -> 0.430 ms; an eighth, L1 router logits,
measured and dropped for an LSB non-determinism), the expert-sorted prefill dispatch
from bucket 256, and a masked-bucket prefill on persistent device inputs captured once per bucket and replayed for any
real length (bit-identical to the stock path from 1 to 4097 tokens). Token-out decode is 24.36 ms of model trace +
0.55 ms sampler trace + 0.17 ms host: 14.5 % above the stage-7 layer-stack device bound (21.14 ms, Tracy max over
devices 0.454 / 0.399 ms per layer replay) + ~0.75 ms of terminal work (the sampler trace is measured, the final norm / LM head ~0.2 ms is estimated) — section 6. Full context 262144 is unchanged for one user; the
shared KV pool holds ~2.1 M tokens per device on the stage-6 basis (~1.89 M counting the B=1 full-context KV as footprint; section 7).

Numbers in this README are re-derived from the artifacts named next to them (`logs/` = `~/qwen35-122b-bringup/logs`).

## 1. Token-out contract (nonblocking replay, deferred readback)

The decode step already replays its two traces (model, sampler) non-blocking with persistent token / position / RoPE /
page-table inputs (`ttnn.plus_one` advances the positions inside the trace, the sampler writes `tt_out_tok` into the
persistent token input, the page table is copied only when its content changes; stage-6 README section 1). Stage 7 adds
the serving split the vLLM plugin drives (`Qwen35MoeGenerator.decode_forward(read_from_device=False)` ->
`read_decode_output(async_read=True)` -> `process_decode_output_host`, wrapped as `decode_step_async` /
`PendingTokens.result()`) and measures what the host side costs (`tools/tokenout_bench.py`, 64 steps, B = 1):

| model | blocking readback (readiness path) | deferred readback (serving split) | model + sampler replays only | model trace only | sampler trace only | tokens identical |
|---|---|---|---|---|---|---|
| 2 layers (0, 3) | 2.318 ms | 2.308 ms | 2.210 ms | 1.680 ms | 0.541 ms | yes |
| 48 layers (stage-6 code) | 27.309 ms (36.6 t/s/u) | 27.150 ms (36.8 t/s/u) | 27.142 ms | 26.609 ms | 0.542 ms | see section 5 |

Source: `tokenout_before_2layers.json`, `tokenout_before_48layers.json` (`logs/07-tokenout-before-*.log`). The whole
host side of a step (readback + python) is 0.17 ms; deferring the readback recovers 0.16 ms of it. The model trace alone
is 26.6 ms against the 21.9 ms layer-stack device lower bound (stage-6 README section 6): the 4.7 ms gap is INSIDE the
trace, i.e. device idle between programs (~1 us of dispatch per op x ~4700 ops per step), not host work. Section 2.

Probe pitfall (`scripts/07-async-readback.py`): reading the sampler output buffer after the NEXT step was submitted returns
the next token — the sampler writes the persistent feedback buffer in place; the nonblocking read has to be enqueued
before the next submit, which `read_decode_output(async_read=True)` does.

## 2. Where the time goes: op-count ledger and the dispatch gap

Per-layer decode op ledger from the stage-5 Tracy rows (`doc/optimized_multichip_decoder/tracy/{gdn,attn}/decode_mesh_perf_report.csv`,
device 0, one replay = every third row; re-derived by `tools/decode_op_count.py` graph capture, which counts the same
103 / 84 ops):

| layer kind | ops per step | device time | op-to-op gaps | biggest groups |
|---|---|---|---|---|
| GDN (36 layers) | 103 | 471.6 us | 131.0 us | 20 slices, 18 binary, 12 reshapes (flat->head relayouts 3.0/3.3/8.2 us), 7 copies (conv shift 4, repeat_interleave 2, state staging 1), 4 tilize + 3 untilize (repeat_interleave, a/b split), 5 norms, 2 AG + 2 RS |
| attention (12 layers) | 84 | 406.3 us | 86.5 us | 19 slices, 10 binary, 8 sharded->interleaved, 2 rotary (+4 transposes, 2 concat), 2 paged updates, SDPA, 2 AG + 2 RS |
| per 48-layer step | ~4716 | 21.6 ms | ~5.8 ms | (the 27.3 - 21.9 ms token-out gap) |

Both kinds share the MoE block (router matmul -> gate op -> gather-mode sparse gate/up -> swiglu -> sparse down ->
score-weighted sum -> packed shared expert -> reduce-scatter): 38 of the ops, of which 13 are layout traffic around the
1.5 us `generalized_moe_gate` kernel (4 per-step slices of its constant / output buffers, a 1-core DRAM reshape of the
logits, 2 reshards and 2 slices of its outputs) and 6 around the weighted sum (`ttnn.sum` = fill_pad 6.8 us + reduce +
slice, tile-padding reshapes).

## 3. Decode op-count reductions (measured per knob)

Knobs in `MultichipConfig` (default OFF in the dataclass; the stage-7 selected policy turns the measured ones on):

| knob | what it removes | GDN layer 0 traced decode (10 replays) | attention layer 3 | ops per step |
|---|---|---|---|---|
| stage-5 default | - | 0.579-0.580 ms | 0.446 ms | 103 / 84 |
| `gdn_ab_tile_pad` | the a/b (decay, beta) fields end mid-tile in the fused in-projection -> zero-pad each to a tile at load (`.abpad` weight cache, +32 columns): two aligned slices instead of slice + untilize + slice + tilize | 0.561 | - | -3 |
| `gdn_gqa_matmul` | q/k GQA head expansion `repeat_interleave` (untilize + concat + tilize + copy each) -> one exact one-hot matmul per tensor on the flat layout (E in bfp4_b = 0.0 / 1.0 exactly, HiFi4 fp32-acc), v handed to the recurrent kernel in its head-row layout | 0.577 (bf16 E: 0.585 — the 2 MB read per matmul ate the gain) | - | -7 |
| `gdn_state_inplace` | the recurrent-state staging copy: the kernel's fused `mac` writes the persistent fp32 state (`state_out`) | 0.574 | - | -1 |
| all three GDN knobs | | 0.553-0.556 (-24..-26 us, -4.3 %) | - | 103 -> 92 |
| `moe_gate_fast` | the gate op's per-step slices of its 4 constant / output buffers (built once per iteration batch, shared by all layers) and the 2 output reshards (top-k columns sliced straight from the sharded row-major outputs) | 0.571 | 0.435 | -6 |
| `moe_router_l1` | router logits in L1 (the (32,16,16) relayout leaves the 1-core DRAM path) | 0.569 (with gate_fast) | 0.434 | 0 — measured, NOT adopted: together with `moe_gate_fast` the attention decode output stopped being bit-identical across identical prefill+decode cycles (`test_mc_stress_repeated_runs[attn]`: 87 / 3072 elements at 1 bf16 LSB; either knob alone is deterministic; bisect `logs/07-stress-bisect-chain.log`); 2 us per layer was not worth a determinism regression |
| `moe_gather_tail` | `ttnn.sum` over the k expert rows -> `fast_reduce_nc` (no fill_pad; one trailing row trim keeps the logical row) | 0.572 | 0.437 | -1 |
| all MoE knobs | | - | 0.428 (-18 us, -4.0 %) | 84 -> 77 |
| all knobs (round 3, `sweeps/decode_ops_r3.jsonl`) | | 0.545 (-34 us, -5.9 %) | 0.429 (-17 us, -3.8 %) | 103 -> 85 / 84 -> 77 |
| `gdn_norm_fold` (on top of all) | the decode q/k L2 norms as ONE `rms_norm(weight=)` each: the norm scale K**-0.5 and the query scale live in fp32 weight constants instead of two/one trailing multiplies | 0.524 (-21 us) | - | -3 |
| `gdn_conv_l1` (on top of all) | the conv FIR `mac` chain kept in L1 (its DRAM outputs carried 3-5 us op-to-op gaps) | 0.536 (-9 us) | - | 0 |
| all eight knobs (round 4, `sweeps/decode_ops_r4.jsonl`) | | 0.521 (-58 us, -10.0 %) | 0.429 | 103 -> 82 / 84 -> 77 |
| SELECTED POLICY = the seven knobs without `moe_router_l1` (+ `sorted_min_tokens=256` for prefill), measured (`sweeps/final_policy.jsonl`, 10 replays) | | 0.523 (-56 us, -9.7 %) | 0.430 (-16 us, -3.6 %) | 82 / 77 |

PCC vs HF is identical to every printed digit in every row except `gdn_norm_fold` (decode 0.9999631 -> 0.9999618 GDN:
the weighted rms_norm rounds the scale in a different order than rms_norm + multiply; same dtypes and fidelity, a 1.4e-6
PCC change, the traced-replay bit-identity test compares replays with each other and is unaffected); every other knob
moves data, not math (the one-hot matmul and the mask multiplies are exact).
Sources: `sweeps/decode_ops.jsonl`, `sweeps/decode_ops_r2.jsonl`, `sweeps/decode_ops_r3.jsonl`, `decode_op_count.json`,
`logs/07-sweep-*.log`, `logs/07-opcount*.log`. Tried and dropped: `ttnn.view` of the compact gather-mode sparse-matmul
output (volume assert; reshape kept), the gate op on the full persistent buffers (`bias_shape == in_shape` assert -> the
per-iteration buffers are built once instead), bf16 one-hot E (no gain). Not done (candidates, section 7): the GDN conv
shift register (4 copies + 4 elementwise + silu per step), the shared expert as an always-on 257th sparse expert (a dtype
policy change -> datatype-sweep stage), the router matmul on the width-sharded norm output.

## 4. Prefill / TTFT: persistent-input masked buckets and per-bucket traces

**Why the stock prefill is host-bound.** The masked fixed-bucket prefill (128 / 256 / 512 / 1024 / 2048 tokens, 2048-token
segments for longer prompts) dispatches ~150 ops per layer eagerly and uploads its inputs from the host on every call:
tokens, RoPE cos/sin, both page tables, the chunk start, and — per GDN layer — three length masks / one-hots built with
`ttnn.from_torch` (blocking host writes; a "write during trace capture" fatal under a trace). The 48-layer TTFT of
632 ms for 161 tokens (stage 6) is mostly that dispatch: the device time of a 256-token bucket is a fraction of it.

**The stage-5 trace hang was the sort.** `test_perf_prefill_512_traced` (`QWEN35_RUN_PROBES=1`) now captures AND replays
a 512-token chunk of both layer kinds (GDN eager 17.235 ms -> traced 15.390 ms, attention 15.724 -> 14.509 ms, PCC >=
0.999 vs eager; `logs/07-prefill512-traced-*.log`): the replay hang bisected in stage 5 came from the `ttnn.sort`
cross-core-exchange kernel inside the MoE dispatch, replaced by the counting sort in stage 6.

**Persistent inputs (`Qwen35MoeModel.prefill_masked_bucket`, `tt/prefill_len.py`).** Every per-segment input is a
persistent device tensor staged with `copy_host_to_device_tensor` (tokens, cos/sin, full and bucket page tables, chunk
start, and a `PrefillLen` object: `[1,bucket,1]` fp32/bf16 validity masks for the fused-chunk GDN kernel, the `[1,K-1,
bucket+K-1]` one-hot that selects the decode conv window, the `[1,1,1,bucket]` one-hot that selects the next-token row).
The GDN kernels take the object in place of the host int (`fused_chunk.py`, `_causal_conv1d_fir`), the KV fill covers the
whole bucket with the pad rows routed to a scratch block the generator reserves (+1 block; never read: decode overwrites a
position before it attends to it and the causal prefill mask hides pad keys), and the hidden row is selected by the staged
one-hot. The bucket forward is therefore ONE fixed program per bucket: `capture_prefill_bucket_traces` captures it once
(after every prefill program is warm and every persistent allocation exists), and a segment of any real length at any
chunk start is a replay. Text-only; multimodal segments fall back to the stock path.

**Equivalence (2-layer model, real weights, `tools/prefill_bucket_bench.py`, `logs/07-prefill-bench-2l-v3.log` = `prefill_bucket_bench_2layers.json`; the earlier `-2l.log` / `-2l-v2.log` are the hung runs of section 4's hang paragraph):**

| prompt length | path | logits max abs diff vs stock | GDN state (recurrent + conv) | first decode token |
|---|---|---|---|---|
| 1, 17, 128, 129, 161, 500, 2048 | persistent-input eager | 0.0 (PCC 1.0) | identical | identical |
| 1, 17, 128, 129, 161, 500, 2048 | traced replay | 0.0 (PCC 1.0) | identical | identical |
| 2049 (2048 + 1-token tail at chunk_start 2048) | stock / eager / traced, each in isolation (`scripts/07-tail-probe.py`) | same argmax and next token; per-segment walls 119.8 + 28.9 / 98.1 + 32.1 / 98.5 + 30.1 ms | - | identical |
| 2049 (2048 + 1-token tail), 4097 (2 x 2048 + 1) | persistent-input eager AND traced replay (round 6, stock pre-warmed) | 0.0 (PCC 1.0) | identical | identical |
| 2381, 2333, 2560 (tails in bucket 512 at chunk_start 2048), 3000, 3073 (bucket 1024) | persistent-input eager AND traced replay (round 9, `prefill_bucket_bench_2layers_tails.json`) | 0.0 (PCC 1.0) | identical | identical |

**A hang that was not the traced path.** The first bench hung the mesh at its 2049-token case; sequence probes
(`scripts/07-seq-probe.py`) bisected it to `stock 2048 -> traced 2048 replay -> stock 2049` (passes without the first
stock run, or with the persistent-input path in its place), and tt-triage of the hung mesh
(`triage/seq_stock_2048_traced_2048_stock_2049.txt`) shows devices 1-3 "All NOCs hung" with device 0 idle and
`check_binary_integrity` FAILING on every RISC of the running program: the kernel binaries no longer matched their ELF.
That is the compile-after-capture clobber the qwen36 code documents for its decode trace (tt-metal #48536): the stock
host-upload path is no longer the warmed production path, so its first run compiled programs AFTER the bucket traces
were parked, their binaries landed in DRAM the trace treats as free intermediate space, the replay overwrote them and
the next stock run executed corrupted kernels. Production never runs the stock path; the two paths that could still
compile after a capture were fenced: the readiness all-positions prefill now runs on the warmed bucket programs
(`prefill_all_logits`: 2048-token segments + masked tail, every row through norm + LM head, programs compiled at build)
and the bench pre-warms the stock path (`--warm-stock`). Production-like soak (`traced 1 / 2048 / 2049 / 4097 / 161 /
2048 / 500 / 2049`, each with a traced decode step): pass (`logs/07-seq-traced_1_traced_2048_...log`). With the
fixed warm-up and the guard: the mixed soak `all 2049 / traced 2049 / all 4096 / traced 161 / all 1025 / all 17 /
traced 4097 / all 161` (2 layers) holds the program cache at 891 entries from warm-up to the end
(`logs/07-seq-mixed-guard.log`) — no program is compiled after the captures.

**TTFT on the 2-layer model** (`prefill_bucket_bench_2layers.json`, min of 3, host logits consumed):

| prompt | stock host-upload path | persistent-input eager | traced replays |
|---|---|---|---|
| 128 tokens (bucket 128) | 32.8 ms | 33.6 ms | 29.2 ms |
| 161 tokens (bucket 256) | 42.5 ms | 43.1 ms | 38.9 ms |
| 2048 tokens (one full segment) | 94.5 ms | 94.9 ms | 93.0 ms |

With two layers the per-layer host dispatch is a small part of the wall (embedding, all-gathered LM head, logits readback
and the ~8 host->device copies dominate); the traced gain scales with the 48 layers of the full model — section 5.

## 5. Refreshed accuracy evidence (AIME24 chat prefill, teacher forcing, autoregressive) and the 48-layer numbers

All rows: full 48-layer model on the shipped stage-7 policy (seven decode knobs, `sorted_min_tokens=256`, traced prefill;
the prefill checks were re-run on it in review round 2 — `logs/07-prefill-check-*.log`, latest), (1,4) P300x2, B = 1, top-k
candidate greedy sampler (benchmarked against force-argmax on the 2-layer model in the generator suite, `logs/07-generator-tests3.log` lines 314 / 825: token-out step 5.045 ms = model 1.610 + sampler 3.343 ms with force-argmax vs 2.252 ms = 1.611 + 0.543 ms with top-k candidates), `tools/readiness.py` mesh wrapper, references `bringup/references/qwen35_122b_{aime24_chat,book_p128}_g100.refpt`.

| check | prompt | top-1 | top-5 | top-100 | perf | artifact |
|---|---|---|---|---|---|---|
| prefill check (all-positions logits, bucketed) | AIME24 chat, 161 tokens | 0.97 (97/100) | 1.00 | 1.00 | re-run on the shipped policy with `QWEN35_PROGRAM_GUARD=1` (892 programs, none compiled after warm-up), identical to the eight-knob run | `readiness/prefill_check_qwen35_122b_aime24_chat_g100.json`, `logs/07-prefill-check-qwen35_122b_aime24_chat_g100.log` |
| prefill check | book, 128 tokens | 0.97 (97/100) | 0.99 (99/100) | 1.00 | same (guarded re-run identical) | `readiness/prefill_check_qwen35_122b_book_p128_g100.json`, `logs/07-prefill-check-qwen35_122b_book_p128_g100.log` |
| teacher forcing (traced decode, token-out path) | AIME24 chat | 0.98 (98/100) | 1.00 | 1.00 | TTFT 557.3 ms (stage 6: 632.1), decode 39.67 t/s/u (36.45), e2e 32.49 t/s/u (29.62); the eight-knob run: 0.97 / 1.00 / 1.00, 565.7 ms, 39.97 t/s/u | `readiness/teacher_forcing_qwen35_122b_aime24_chat_g100.json`, `logs/07-teacher-forcing-qwen35_122b_aime24_chat_g100.log` |
| teacher forcing | book p128 | 0.97 (97/100; stage 6: 0.97) | 1.00 (stage 6: 0.99) | 1.00 | TTFT 385.2 ms (394.9), decode 39.66 t/s/u (36.49), e2e 34.40 t/s/u (31.90); the eight-knob run: 0.96 / 1.00 / 1.00, 388.3 ms, 39.98 t/s/u | `readiness/teacher_forcing_qwen35_122b_book_p128_g100.json`, `logs/07-teacher-forcing-qwen35_122b_book_p128_g100.log` |
| autoregressive (free-running, read by me) | readiness prompt (Elena, 59 tokens), 256 new tokens | - | - | - | HF and TT both 256 tokens; identical first 20 tokens (22/256 positions equal overall, informational), then a near-tie fork ("singing a different tune" HF / "singing in a rhythm she had never heard before" TT); TT continues a coherent narrative (the western mountain, the old man whittling by the roadside, dialogue about the long shadows), grammatical, English, no repetition loops (`check_degenerate_output` clean) | `autoregressive/{hf,tt}_completion.txt`, `autoregressive_meta.json`, `logs/07-autoregressive.log` |

**Qualitative chat-template suite on the final policy** (`$qualitative-check`, `scripts/07-qualitative.py`: the six
`vllm_prompts.txt` prompts through `apply_chat_template(add_generation_prompt=True)`, 128 new greedy tokens each; HF
controls = the stage-6 CPU generations, same prompt token ids asserted; `qualitative/prompt_0..5/`,
`qualitative/qualitative_prompt_format.json`, `logs/07-qualitative.log`). All six TT completions open the model's
thinking process in English, stay on the asked task, and show no repetition loops or language drift; the greedy streams
fork from HF at near-ties as in stage 6 (bfp4 routed experts). `check_degenerate_output --scope autoregressive` over
these artifacts: clean (`logs/07-check2.log`). Program cache 891 -> 891 entries across the run (no post-capture compile).

| prompt | text | first divergence vs HF (tokens) | equal positions / 128 | TT completion (read by me) |
|---|---|---|---|---|
| 0 | Write a haiku about machine learning. | 98 | 98 | haiku planning (topic, 5-7-5 constraint, imagery, draft attempts) — coherent |
| 1 | Explain the difference between supervised and unsupervised l | 101 | 102 | supervised vs unsupervised: labeled vs unlabeled data, teacher analogy, clusters — correct, simple |
| 2 | Complete this story: Once upon a time, in a faraway kingdom, | 6 | 7 | story continuation plan (young inventor, discovery, fairy-tale tone) — coherent |
| 3 | What are the three laws of thermodynamics? | 43 | 43 | lists zeroth/first/second laws correctly (equilibrium, conservation, entropy) — correct |
| 4 | Translate the following to French: "Hello, how are you today | 36 | 36 | translation analysis of "Hello, how are you today?" into French — on task |
| 5 | Write a Python function to calculate the Fibonacci sequence. | 15 | 30 | Fibonacci: iterative / recursive / memoized / generator options, picks iterative — correct plan |

Bars: top-5 >= 98 %, top-100 = 100 % (stage-6 baseline: prefill 0.98 / 1.00 / 1.00 and 0.97 / 0.99 / 1.00; teacher
forcing 0.97 / 1.00 / 1.00 at 36.5 t/s/u, TTFT 632 / 395 ms).

**Token-out decode, 48 layers** (`tools/tokenout_bench.py`, 64 steps, B = 1, `tokenout_before_48layers.json` /
`tokenout_after_48layers.json`, final policy without `moe_router_l1`; the eight-knob run measured 24.948 / 24.801 / 24.794 / 24.265 ms, `logs/07-tokenout-after-48l.log`):

| step wall | stage-6 code | stage-7 defaults | delta |
|---|---|---|---|
| blocking readback (readiness path) | 27.309 ms (36.6 t/s/u) | 25.058 ms (39.9 t/s/u) | -2.25 ms (-8.2 %) |
| deferred readback (serving split) | 27.150 ms (36.8 t/s/u) | 24.901 ms (40.2 t/s/u) | -2.25 ms |
| model + sampler replays only | 27.142 ms | 24.887 ms | |
| model trace only | 26.609 ms | 24.357 ms | -2.25 ms |
| sampler trace only | 0.542 ms | 0.547 ms | |

Run-to-run: in the final-policy run the blocking stream and its re-run (same prompt, same session) are bit-identical over
all 64 steps, while the deferred-readback stream diverges from them at step 48 on a near-tie; in the eight-knob run the
first stream diverged from its re-run AND from the deferred stream at step 10 (those two agreed). The device path is the
same in all three streams, so this is a rare LSB-level non-determinism surfacing at near-ties (~1 in 64 steps of a
low-entropy list prompt), already present in the stage-6 baseline (its blocking vs deferred streams diverged too) and
independent of the stage-7 knobs (the layer-level stress test is bit-identical with the selected policy). The knobs do
move near-ties: the before and after benches start their streams from a different first token on the same 19-token
prompt (" and" vs " the"; the norm-fold's rounding-order change), while the readiness top-1 stays within noise
(0.97-0.98). Open item in section 7; the readiness top-k metrics quantify the effect.

**TTFT, 48 layers** (`tools/prefill_bucket_bench.py --no-equiv`, min of 3, host logits consumed; `prefill_bucket_bench_48layers.json`):

| prompt | stock host-upload path (same session) | persistent-input eager | traced replays | artifact |
|---|---|---|---|---|
| 128 tokens (bucket 128) | 349.6 ms | 350.0 ms | 344.0 ms | `prefill_bucket_bench_48layers.json` (eight-knob run of 06:02; `moe_router_l1` is a decode-only knob, the prefill path is identical — the shipped-policy traced re-run `_final.json` gives 345.2 ms) |
| 161 tokens (bucket 256, expert-sorted dispatch) | 561.2 ms | 562.7 ms | 555.8 ms | same (`_final.json`: 553.6 ms) |
| 2048 tokens (one segment) | 1932.6 ms | 1935.7 ms | 1934.8 ms | same |
| 128 / 161 / 500 tokens with the sorted dispatch forced at 128 (rejected) | - | - | 455.8 / 551.3 / 749.9 ms | `prefill_bucket_bench_48layers_sorted128.json` |
| 128 / 161 / 500 tokens, masked chunks kept up to 512 (stage-3 threshold) vs the final policy (sorted from 256), same session order, min of 3 | - | - | 344.0 / 571.1 / 752.5 ms vs 345.2 / **553.6** / 753.6 ms | `prefill_bucket_bench_48layers_masked256.json` / `_final.json` |

Same numbers as the table (run of 06:02 with `moe_router_l1` still on — a decode-only knob; `prefill_bucket_bench_48layers.json`,
rows recovered from `logs/07-prefill-bench-48l.log`; the shipped-policy traced rows are `_final.json`: 345.2 / 553.6 / 753.6 ms
at 128 / 161 / 500 tokens): 128 tokens 349.6 (stock) / 350.0 (persistent-input) / **344.0 ms (traced)**;
161 tokens 561.2 / 562.7 / **555.8 ms**; 2048 tokens 1932.6 / 1935.7 / 1934.8 ms. Against the stage-6 headline (632 ms for
161 tokens, 395 ms for 128, measured by the teacher-forcing runner which adds its own per-request overhead) the runner's
own TTFT is in section 5.

At 48 layers the bucket prefill is device-bound (stage-5 Tracy: 8.2 / 9.0 ms per GDN / attention layer at 128 tokens, of
which 4.8 / 6.0 ms are the two 64-token-chunk sparse matmuls of the masked MoE path — every active expert's weights are
read once per chunk), so removing the host dispatch buys 1-3 %; the trace still removes the per-token host work that
matters under concurrency. The lever is the MoE path itself: the expert-sorted dispatch (one weight read per active
expert) below its stage-3 threshold of 512 tokens (`sorted_min_tokens`). Measured on the 48-layer traced path with the
dispatch forced at 128 (`prefill_bucket_bench_48layers_sorted128.json`): 128 tokens 455.8 ms (vs 344.0 masked — the
dispatch's fixed costs, counting sort / gathers / group padding, exceed the two-chunk weight re-read), 161 tokens
551.3 ms, 500 tokens 749.9 ms; so the threshold moves to 256. The like-for-like masked-256 vs sorted-256 numbers at
48 layers (`prefill_bucket_bench_48layers_masked256.json` / `_final.json`, traced path, min of 3): 161 tokens 571.1 ->
553.6 ms (-3.1 %), 128 tokens 344.0 vs 345.2 and 500 tokens 752.5 vs 753.6 (unchanged within noise, both buckets on the
same path in both runs); the layer-harness eager walls point the same way
(`sweeps/prefill_sorted_T{128,256}.jsonl`: T=128 masked 14.0 / 14.3 ms vs sorted 14.9 / 15.9; T=256 masked 24.9 / 27.0
vs sorted 20.8 / 20.0, GDN / attention). What remains is the device time of the short-bucket MoE path itself — the
stage-8 / handoff lever.

Tests on the final policy: generator suite (`tests/test_generator.py`) 10 passed (`logs/07-generator-tests3.log`, with
the sampler-variant test after the token-count precompile fix, section 7; the earlier 9-test run `07-generator-tests2.log`;
the first run had 1 failure, a strict decode-equals-prefill token equality on a random-token prompt with a 0.031-logit
near-tie, now near-tie aware like the rest of the suite — `logs/07-generator-tests.log`, `07-gen-long-segments.log`;
the program-cache count after warm-up rises by the six counted-penalty programs, 892 -> 898 on the 2-layer test model);
decoder suite
(`tests/test_multichip_decoder.py`, non-perf) 40 passed + 2 failed on the first run (`logs/07-decoder-suite.log`: the
fallback audit saw the lazily built stage-7 constants, and the attention stress test caught the `moe_router_l1` LSB
non-determinism, section 3), both closed and re-run green with the traced-decode and dispatch tests (17 passed,
`logs/07-decoder-rerun.log`); `test_mc_traced_decode` re-runs after the LSB-tolerance change: `logs/07-r15-chain.log`; `07-optimized-full-model.check.sh` exit 0 (`logs/07-check3.log`, earlier `07-check.log`, `07-check2.log`).

## 6. Performance tables (tt-perf-report), lower bound vs token-out

Per-layer Tracy windows of the reduced one-layer-per-kind model with the stage-7 defaults: `PERF_TABLES.md`
(`tools/perf_tables_ofm.py`, sources `tracy/<kind>_<case>_mesh_perf_report.csv` + `_ops.csv` + console logs). The full
48-layer model is never profiled with Tracy (device-usage rule); the token-out step is a wall breakdown (section 1).

Lower bound vs token-out (48 layers, B = 1):

| quantity | stage 6 | stage 7 |
|---|---|---|
| layer-stack device lower bound (36 x GDN + 12 x attention device time per traced replay, max over devices) | 21.98 ms (stage-5 Tracy: 0.474 / 0.409 ms) | 21.14 ms (stage-7 Tracy: 0.454 / 0.399 ms per replay, `PERF_TABLES.md`, `perf_summary.json`) |
| model trace replay (48 layers) | 26.61 ms | 24.36 ms |
| sampler trace | 0.54 ms | 0.55 ms |
| host (readback + python) | 0.17 ms | 0.17 ms |
| token-out step (readiness path) | 27.31 ms | 25.06 ms |
| gap above bound + terminal work (sampler 0.55 + final norm / LM head ~0.2) | +4.6 ms (+20.6 % of 22.7 ms) | +3.2 ms (+14.5 % of 21.9 ms) |

The remaining gap is device idle between the ~3880 programs of a step: the stage-7 Tracy rows measure 0.091 ms of
op-to-op gaps per GDN replay and 0.077 ms per attention replay (36 x 0.091 + 12 x 0.077 = 4.2 ms of the 24.36 ms model
trace), i.e. ~1.1 us per program, plus the two all-gathers per layer (12 us each, 3-9 us gaps). Per-op tables (count,
device us, gaps us) of both decode windows are in `PERF_TABLES.md`; the candidates left are in section 7. Prefill device
time per layer is unchanged by the decode knobs (GDN 8.20 / 11.17 / 61.06 ms and attention 9.02 / 10.65 / 58.26 ms at
128 / 256 / 2048 tokens), as expected: the knobs touch the decode path only. These prefill rows are a per-layer profile
of the TEST HARNESS, not of the model's bucket path, and do not add up to the measured full-model TTFT: 36 x 8.20 +
12 x 9.02 = 403 ms vs 344 ms measured at 128 tokens, and the 2048 row is `test_perf_prefill_2048`'s 4 x 512-token
start_pos chunks with a profiler flush between them (2.90 s summed vs 1.93 s measured for the model's single 2048
bucket). The harness feeds random hidden states (~221 unique experts per 64-token chunk, more than real text routes to)
and runs under the device profiler; the model's traced bucket is a different program set. The "device-bound" statement
in section 4 therefore rests on the direct measurement (traced replays save only 1-3 % of the wall vs the eager path),
not on these rows; a Tracy window on the model's own bucket program is the handoff item for the prefill lever.


## 7. Runtime fallback audit, context contract, limitations, artifacts

**Fallback audit (measured full-model path, stage-7 defaults)**

| path | owner | host boundary |
|---|---|---|
| embeddings | bf16 replicated `ttnn.embedding` on device | decode: the device token input (refreshed from the host only when the caller is authoritative); prefill: the bucket's persistent token buffer (one host->device copy per segment) |
| decoder layers x48 | stage-5 `Qwen35MoeOptimizedDecoderLayer` + the stage-7 op-count knobs; fractured residual, no inter-layer gather | none (decode: `tools/decode_op_count.py` graph capture lists device ops only; `test_mc_runtime_fallback_audit` passes with the knobs on — section 5) |
| final norm + LM head | `DistributedNorm` (one all-gather) + vocab-sharded bfp8 `ttnn.linear` | none in token-out mode; one replica of the logits is read back only in the host-sampling compatibility mode and for `prefill_logits` |
| sampling | `SamplingGenerator` trace, `tt_out_tok` = the model's persistent token input | the readiness contract reads the token every step (0.17 ms host per step, section 1); the serving split (`decode_step_async`) enqueues a nonblocking read and resolves it one step later |
| positions / RoPE | `ttnn.plus_one` inside the decode trace; prefill cos/sin staged per segment into the bucket's persistent buffers | none in decode; one host->device copy per prefill segment |
| KV cache / page table | generator-owned paged KV (block 64) + 1 reserved scratch block; decode page table copied on change only; prefill page tables staged per segment (fixed width) | copies on change / per segment |
| GDN recurrent + conv state | model-owned persistent tensors; decode update written in place by the kernel (`gdn_state_inplace`); prefill masks / conv window select / logits row select are persistent device tensors (`tt/prefill_len.py`) | none (the stock path's 3 `from_torch` writes per GDN layer per segment are gone) |
| prefill | masked fixed buckets (128..2048) in 2048-token segments, ONE trace per bucket replayed for any real length and chunk start (section 4) | token ids + 10 small host->device copies per segment in (tokens, cos, sin, two page tables, chunk start, two masks, conv one-hot, logits one-hot), next-token logits out |

**Context contract** (`doc/context_contract.json`, block `optimized_full_model_stage7`; MEMORY_REPORT lines in
`logs/07-prefill-check-qwen35_122b_aime24_chat_g100.log`, 262144-context build, B = 1, every trace parked): DRAM allocated
19.829 GiB per device (stage 6: 19.669; +0.16 GiB = per-bucket persistent prefill inputs and trace outputs, the
all-positions norm / LM-head programs, one KV scratch block); KV pool 13.096 GiB = 2.15 M tokens shared by all users on the stage-6 basis (dram_total - non-KV footprint 18.235 GiB; stage 6: 12.75 GiB / 2.1 M; the conservative figure that keeps the B=1 full-context KV inside the footprint is 11.502 GiB / 1.89 M)
(~56 K per user at 32 with the GDN state), one user still reaches the full 262144; trace region 405.4 MiB of the 512 MiB
reserved (decode 24.9 MiB + the five bucket traces 380.5 MiB — a serving stack must reserve >= 512 MiB or capture fewer
buckets, `QWEN35_PREFILL_TRACE_MAX_BUCKET`); L1 held per core 18.3 KB (gate-op per-iteration buffers built once). No
advertised capability is reduced.

**Limitations / open items**

- Compile-after-capture rule (section 4): any program first compiled after a trace is parked is at risk in the same
  session. The production paths compile nothing after the captures: the masked buckets, the all-positions programs of
  EVERY bucket (the first warm-up warmed bucket b//2 for bucket b and missed 2048 — stage-7 review finding, fixed) and
  all sampling-trace variants are compiled before the bucket captures (the sampler variants in the eager compile pass,
  i.e. before the decode trace exists — `_warm_decode_trace` / `_precompile_sampler_variants`:
  `SamplingGenerator.precompile(all_configs=True)` plus one counted penalties step on a scratch token buffer with the
  trace feedback buffer's spec, because `precompile` runs with `count_tokens=False` and the traced penalties variant
  records the token-count update — the first run of `test_generator_sampler_variants_after_replays` compiled exactly
  those ops inside a live capture, TT_FATAL "Cannot load new binaries during trace capture", `logs/07-gen-sampler-variants.log`;
  a production penalties request would have killed the engine);
  `Qwen35MoeGenerator.program_cache_entries()` / `assert_no_new_programs()` are the guard (program-cache count after
  warm-up vs later; the soak, the qualitative run and `test_generator_sampler_variants_after_replays` assert it, and
  `QWEN35_PROGRAM_GUARD=1` asserts it after every prefill / decode call at runtime). The stock host-upload prefill (bench-only) and
  the exact-length all-positions pass (`_prefill_all_logits_exact`, without a KV scratch block) must not be mixed with
  traced replays; a multimodal or `flex_sdpa=False` fallback segment is not warmed either.
- LSB-level device non-determinism, independent of the stage-7 knobs: (a) 48-layer greedy streams fork at near-ties
  about once in 64 steps (section 5; pre-existing in stage 6); (b) the attention layer's traced-decode replays are not
  always bit-identical — `test_mc_traced_decode[attn]` failed once on the shipped policy in the full suite
  (`logs/07-decoder-suite2.log`) and the repeated-run bisect (`logs/07-r13-chain.log`, 5 runs of traced-decode + stress
  per knob set) passed 5 / 5 on the shipped policy, 5 / 5 with `moe_gather_tail` off, 5 / 5 with `moe_gate_fast` off and
  FAILED 3 / 5 with both MoE knobs off (the stage-5 MoE configuration): the effect is intermittent and memory-layout
  dependent, not introduced by the knobs (which reframes the stage-5 `norm_ag_persistent` rejection, made on this very
  symptom). `test_mc_traced_decode` now reports the differing elements and tolerates an absolute bound of one bf16 LSB at the
  tensor's largest magnitude (max|d| <= max|ref| x 2^-7, plus PCC >= 0.99999; many LSBs for small elements) and keeps
  a strict mode (`QWEN35_STRICT_REPLAYS=1`); after the change 7 / 7 runs were bit-identical incl. the strict one
  (`logs/07-r15-chain.log`, no REPLAY_DIFF line). The run-to-run stress test stays strict (`torch.equal`). Root cause not located (candidates: CCL reduce-scatter accumulation
  order, a read of uninitialised padding); readiness top-5 / top-100 bars unaffected. `moe_router_l1` stays dropped:
  with it the stress test failed deterministically (2 / 2) while the bisect legs without it passed.
- Remaining decode dispatch: after the seven knobs a GDN decode step is still 82 ops (18 elementwise, 15 slices, 11
  relayouts, 4 conv shift copies); candidates with measured value are the conv shift register as one stacked buffer
  (~4 ops), the shared expert as an always-on 257th sparse expert (a dtype policy change: datatype-sweep stage) and the
  router matmul on the width-sharded norm output.
- The traced prefill serves text-only requests; a multimodal segment falls back to the stock path (not warmed here:
  do not mix in a served session before warming it).
- The KV scratch block is reserved by the readiness generator; the vLLM stage must reserve it from the plugin's block
  pool (advertise one block fewer) or route pad rows elsewhere.
- Sentinel rows of the counting-sort dispatch now read token 0 (in range); the `src_tok < T` assert is implicit in the
  masked source index (no device-side assert op).
- Tracy on the token-out step still fails in the ops merge (stage-6 README section 6); per-layer Tracy tables come from
  the reduced model (section 6), the token-out step from wall breakdowns.

**Artifacts**: `README.md`, `work_log.md`, `PERF_TABLES.md`, `perf_summary.json`, `tokenout_before_*.json`,
`tokenout_after_48layers.json`, `prefill_bucket_bench_2layers.json`, `prefill_bucket_bench_48layers.json`,
`decode_op_count.json`, `decode_op_count_r4.json`, `sweeps/decode_ops*.jsonl`, `readiness/*.json`, `autoregressive/`,
`tracy/` (tt-perf-report per-device CSVs + console logs; the raw Tracy `*_ops.csv` provenance files, 40 MB, live in
`~/qwen35-122b-bringup/artifacts/07-tracy-ops/` as in stage 5), `triage/*.txt` (the largest trimmed of its per-core
"in reset" rows); bringup logs `~/qwen35-122b-bringup/logs/07-*.log`, scripts `~/qwen35-122b-bringup/scripts/07-*`.

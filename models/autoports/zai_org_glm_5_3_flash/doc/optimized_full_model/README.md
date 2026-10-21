# GLM-5.3-Flash (E192) optimized full model

Stage 7 of the tt-model-bringup flow. The stage optimizes the full 45-layer model (`tt/model.py`, `tt/generator.py`)
on the (1, 4) mesh of a p300x2 (4 Blackhole dies, FABRIC_1D, TP4). The decoder layers come from stage 5
(`tt/multichip_decoder.py`, `DEFAULT_MC`); this stage does not change them.

## Result (headline): final default

**Final default** (`j2_s1_default`, record `runs/s67r/j2_s1_default.json`, log
`~/glm53-flash-bringup/logs/s67r/j2_s1_default.log`):
- Build: `Glm53Generator(mesh)` with no arguments: 1 slot, replicated DSA cache, `max_seq_len` 280,576 (the largest
  context that fits).
- Precision: stage-8 `combo_a_kv` without the stage-8 `decode_specs.kda_in` decode program. The snapshot predates
  aa104284595, which added kda_in to the default.
- Code: HEAD 656c998e3df plus the stage-6/7 rework files (commits de9852c604d / b3270061cca).
- Prompts: real text (`tools/s67r_checks.text_ids`). Greedy.

**Short-context row** (`j2b_s1_16k`): the same code with `max_seq_len` 16,384.

**Before**: stage-6 code paths on 2026-10-17, stage-7 precision (`GLM53_PRECISION=none` today), `max_seq_len` 16,384,
synthetic prompts (`~/glm53-flash-bringup/artifacts/s7/base_s1.json`).

| Metric | Stage 6 (README) | Before (stage-6 paths, 16K) | **Final default (280,576)** | Same code, 16,384 |
|---|---|---|---|---|
| TTFT, 128-token prompt | - | 232 ms | **228 ms** | 225 ms |
| TTFT, 161-token prompt (readiness prompt length) | 1,252 ms (4 slots) | ~1,182 ms (one 2048 chunk) | **377 ms** (one 512 chunk) | 373 ms |
| TTFT, 512-token prompt | - | ~1,182 ms | **376 ms** | 375 ms |
| TTFT, 2,048-token prompt | - | 1,182 ms | **1,026 ms** | 1,025 ms |
| TTFT, 8,192-token prompt | - | 5,134 ms | **6,444 ms** (regression, see below) | 4,408 ms |
| Decode, model trace only (teacher-forcing boundary), 1 slot | - | 56.86 ms | **55.85 ms** | 50.71 ms |
| Decode, token-out (model + sampler trace, device feedback), 1 slot | - | 57.24 ms | **56.24 ms** | 51.10 ms |
| Decode, generator loop with token readback, 1 slot | 95 ms (4 slots) | 57.36 ms = 17.4 t/s/u | **56.51 ms = 17.7 t/s/u** | 51.26 ms = 19.5 t/s/u |
| Decode, 8-slot build (4,096), 1 / 4 / 8 active (`j3_s8`) | - | 112.8 / 112.8 / 112.8 ms | **81.65 / 93.50 / 112.24 ms** | - |
| Decode, 32-slot build (2,048), 1 / 8 / 32 active (`j4_s32`) | - | - | **115.29 / 117.16 / 114.24 ms** | - |
| Teacher forcing top-1 / top-5 / top-100 (AIME24 chat, 100 positions) | 93 / 100 / 100 % | - | **91 / 100 / 100 %** (runner decode 17.78 t/s/u) | - |
| Prefill check top-1 / top-5 / top-100 | 92 / 100 / 100 % | - | **91 / 100 / 100 %** | - |

How the numbers are measured:
- TTFT is a warm `Glm53Model.prefill` of the prompt to last-row logits on the device, plus a device sync. The
  readiness TTFT (host logits) adds 10 to 20 ms.
- Decode is host wall time per step over 20 back-to-back steps after one warm step, at position 300 (`s7_probe.run_decode`).

The 8K TTFT regression comes from context size, not code. The sparse DSA indexer, in decode and in every prefill chunk
past 2,048 tokens, scores every pool slot allocated to the user. That is `P_alloc`, sized by `max_seq_len`, not the
filled prefix. On the same code, 16,384 vs 280,576 changes token-out decode 51.10 vs 56.24 ms (+10 %) and 8K TTFT
4,408 vs 6,444 ms (+46 %). TTFT up to 2,048 tokens does not change.

The planned fix belongs to stage 10 (`~/glm53-flash-bringup/PLAN.md`, "cap at max position"): score only the user's
filled prefix, bucketed. Prefill is eager, so a bucketed scored length there needs no extra trace. Until then,
short-context deployments should pass `max_seq_len`.

Attribution (stage-7 review #1):
- The decode gain from Before to the 16,384 row (57.36 -> 51.26 ms) comes from stage 5 (round-6 decoder options) and
  stage 8 (combo_a_kv). Stage 7's own decode change is the sharded final RMSNorm (65 -> 7 µs, about 0.06 ms per step).
- Stage 7's own wins are TTFT: the 512-row bucket and the cost chunk plan (161 tokens 1,182 -> 377 ms), the MoE split
  rule and the one-row host logits read.
- The Before 8-slot build ran with `ag_persistent=1` on pre-fix code, so its 112.8 ms is a time baseline only.

### Superseded: the 2026-10-17 stage-7 build

Kept for the record; do not compare later stages against it. Build: stage-7 precision, `max_seq_len` 16,384,
`GLM53_MC=ag_persistent=0`, pre-rework code (the logits-pad lifetime bug was latent), synthetic prompts. Log:
`~/glm53-flash-bringup/logs/s7_final_s1.log`; that run wrote no JSON record.

| Metric | 2026-10-17 build |
|---|---|
| TTFT 128 / 161 / 512 / 2,048 / 8,192 tokens | 230 / 376 / 379 / 1,035 / 4,556 ms |
| Decode model trace / token-out / generator loop | 53.69 / 54.08 / 54.18 ms (18.46 t/s/u) |
| 8-slot build, 1 / 4 / 8 active (`s7_final_s8.log`) | 83.9 / 95.7 / 114.2 ms |
| Teacher forcing / prefill check | 92 / 100 / 100 %, 91 / 100 / 100 % (runner TTFT 423 ms, decode 18.44 t/s/u) |

## What changed (stage-7 defaults)

| Change | Where | Default | Effect (measured) |
|---|---|---|---|
| Cost-based prefill chunk plan: the cheapest multiset of constructed lengths that covers the prompt, largest first | `Glm53Model.chunk_plan`, `GLM53_PREFILL_PLAN=cost`, costs `PREFILL_COST_MS` / `GLM53_PREFILL_COST` | on | a 161-token prompt ran one 2048-row chunk (1.18 s); now one 512-row chunk |
| Prefill lengths 128 / 512 / 2048 (was 128 / 2048) | `PREFILL_LENGTHS_DEFAULT`, `GLM53_PREFILL_LENGTHS` | on | 512-row chunk 378 ms vs 2048-row chunk 1033 ms |
| MoE prefill row split chosen from free DRAM | `Glm53Model.choose_moe_split`, `GLM53_MOE_SPLIT_AUTO=1`, `GLM53_MOE_NOSPLIT_FREE_MIB` (1024) | on | 2048-row chunk 1033 ms unsplit vs 1153 ms split (-10.4 %); 8192 tokens 4548 vs 5033 ms (-9.6 %) |
| Last logits row read alone | `Glm53Model._row_to_host` | on | host-logits TTFT path: ~50 ms -> ~12 ms over the device TTFT |
| Final RMSNorm width-sharded on 32 cores | `GLM53_HEAD=norm_sharded` | on | 65 µs (1 core) -> 7 µs per decode step |
| Warm-up covers every (chunk length, DSA path) a plan can produce | `Glm53Generator._warm_plans` | on | lazily built prefill constants exist before any trace capture (2026-10-16 hang rule) |
| Inactive decode rows skip the routed experts | `GLM53_SKIP_INACTIVE=1`, `Glm53Model._install_inactive_skip` | on (builds with >= 2 slots) | 8-slot build, 1 / 4 / 8 active: 81.65 / 93.50 / 112.24 ms vs 111 ms flat with the skip off (`j3_s8` / `j3b_s8_noskip`); logits bit-identical |
| Traced first prefill chunk | `GLM53_PREFILL_TRACE=128,512` | off (rejected) | 128 rows 222 vs 232 ms; bit-identical |
| DRAM-sharded LM head (column splits) | `GLM53_HEAD=norm_sharded,lm_ds` | off (rejected) | 10 x 66 µs vs 471 µs interleaved |

## Decode breakdown (goal 1)

The numbers in this section are from the 2026-10-17 builds at stage-7 precision. The final default's split is in the
headline: model trace 55.85 ms, sampler 0.39 ms, token-out 56.24 ms, generator loop 56.51 ms. The host adds 0.27 ms
per step. The decode loop has no host input refresh: tokens are fed back on the device, positions advance in the
trace, and the page table is copied only on change.

Layer-stack lower bound (stage 5): 50.7 ms per token at 1 user. That is the sum of the 45 layers' replays at stage-5
reference precision (`GLM53_PRECISION=none`), on the shipped `DEFAULT_MC`
(`doc/optimized_multichip_decoder/README.md`). The earlier 52.3 ms figure was measured with `ag_persistent` on. No
layer-stack bound has been measured at combo_a_kv. The final default's 16,384-token row (model trace 50.71 ms)
already sits at the stage-7-precision layer bound.

The terminal path outside the layers (reduced 1-layer Tracy, `tracy/{before,after}_s1_l0/perf_decode_report.txt`,
device 0..3 union, 2026-10-17):

| Item | Device time | Note |
|---|---|---|
| LM head [32 x 4096] x [4096 x 38720] bfp8, interleaved, 110 cores | 471 µs | 168 MB per die at 357 GB/s; DRAM-bound |
| Sampler trace: TopK (large-indices composite, 110 cores) 205 µs, all-gathers 25 µs, sampling 28 µs, seed 19 µs, casts | 380 µs | split greedy: local top-32 per die, gathered candidates |
| Final RMSNorm | 65 -> 7 µs | `norm_sharded` |
| Stream mean, all-gather of h, embedding, decode-input refresh (35 small ops) | ~250 µs | |

None of these is material at 51 to 56 ms per token; the decoder layers are about 97 % of the step. The DRAM-sharded LM
head lost (below). Force-argmax would gather the full vocabulary (38720 x 32 x 2 B per die); split greedy stays.

Multi-slot decode: without the skip, the step time depends on the built slot count, not on the active slots (8-slot
build, skip off: 111 ms at 1, 4 or 8 active). Every slot row runs the MoE. The decode MoE path for fewer than 32 rows
(pairs: one group per (row, expert)) reads one expert block per group, so 8 rows read 64 expert blocks per layer.

`GLM53_SKIP_INACTIVE` (default on for builds with 2 or more slots) sets the expert ids of inactive rows to 65535 inside
the decode trace, from the per-step active flags. The grouped sparse matmul skips groups with an id >= E
(`skip_padding_groups`), so inactive rows read no expert weights. Final default: 81.65 / 93.50 / 112.24 ms at 1 / 4 / 8
active, all logits bit-identical to the skip off.

The remaining per-row cost at 1 active (81.65 vs 56.24 ms for a 1-slot build) is:
- the pairs glue (row repeat, SwiGLU, k-sum on all 8 x 8 groups);
- the router;
- the 8-row attention.

With 8 active, the skip costs +1.1 % (112.24 vs 110.97 ms): 4 small ops per MoE layer that rebuild the sentinel from the
active flags (`tt/model.py` `_install_inactive_skip`; building it once per step is a stage-10 candidate). The 32-slot
sorted path is unchanged (stage 5); the final default measures it at 114 to 117 ms (`j4_s32`).

## TTFT (goal 2)

Full model, 1 slot, warm, 2026-10-17 candidate build (`cand_s1`, stage-7 precision, ms). The final default's TTFT is in
the headline.

| Prompt | Plan (chunk rows, valid rows) | Eager | Traced first chunk |
|---|---|---|---|
| 128 | (128, 128) | 232 | 222 |
| 161 | (512, 161) | 375 | 383 |
| 512 | (512, 512) | 380 | 375 |
| 1000 | (512, 512), (512, 488) | 757 | 752 |
| 2048 | (2048, 2048) | 1033 | 1033 |
| 8192 | 4 x (2048, 2048) | 4548 | 4550 |

- The 128-row chunk costs 232 ms, about 5 ms of device time per layer (45 layers). The host enqueue is not the limit:
  the traced replay of the same chunk takes 222 ms. A prefill trace per bucket would therefore save 10 ms at 128 rows
  and nothing at 512 rows. It would cost about 30 MB of trace region plus about 40 MB of trace outputs (KDA states, X)
  per bucket, i.e. roughly 6K tokens of 1-slot context per bucket. Rejected; the option (`GLM53_PREFILL_TRACE`) stays.
- The traced path itself is exact: logits and 8-token greedy continuations are bit-identical to eager (100, 128, 161,
  300 and 500-token prompts, reduced and full model, `s7_reduced_check.log`, `s7_cand_s1.log`). It needed the DSA
  open-pool hand-off moved out of the captured ops (`GlmDSA.write_prefill_tail`, `prefill_inputs(host=True)`).
- The 512-row bucket is the TTFT win: prompts of 129 to 512 tokens took a 2048-row chunk (1.03 to 1.18 s); they now
  take one 512-row chunk (375 ms).
- MoE row split (stage-6 memory fix, `GLM53_MOE_SORTED_ROWS=1024`): +120 ms per 2048-row chunk (1,153 vs 1,033 ms) and
  +485 ms at 8,192 tokens. It only lowers the prefill peak (417 -> 297 MiB above the layer entry).
  - `choose_moe_split` runs the whole chunk when the largest free DRAM block per die is >= 1 GiB at warm-up. The
    final default has 1.09 GB free, so there is no split.
  - A build that fills DRAM with KV blocks (vLLM, long contexts) keeps the split.
  - The output is bit-identical either way.

## MTP (goal 4): rejected

The parent checkpoint has one MTP layer: layer 45, kept in `ckpt/glm53_flash_e192_fp8/mtp/`, with
`num_nextn_predict_layers` set to 0 in the E192 config. It is a DSA attention layer with:
- 288 routed experts (the E192 mask covers layers 0-44 only);
- a shared expert;
- `eh_proj` [4096 x 8192];
- `enorm` / `hnorm` / `shared_head.norm`;
- no mHC weights.

| Item | Value |
|---|---|
| Routed expert parameters | 7.25 B (288 x 3 x 4096 x 2048) |
| Other parameters | 185 M |
| Device bytes per die (bfp4 experts, bfp8 rest, TP4) | ~1.07 GB |
| Extra DSA cache per token, per die | 576 B with the fp8 latent (default); 1,088 B bf16 |
| 1-slot context, final default (computed, 2026-10-20) | 280,576 -> ~104K tokens (-63 %): the 2,206-block pool (1.79 GB) less 1.07 GB, over 6,912 B per token |
| 1-slot context, stage-7 precision (2026-10-17, stale) | 104,448 -> ~13.8K tokens (-87 %) |
| 8-slot context (30,720 per slot by default) | 30,720 -> ~11K per slot (computed) |
| Expected speed-up | not measured on this model. It is borrowed from GLM-4.5-Air (26.9 -> 20.5 ms with MTP, <= 1.3x). Here it would be lower: the verify step doubles the MoE expert reads at 1 user, and the 34 KDA layers need a recurrent-state snapshot / rollback for rejected drafts |

The context loss is not a hard physical limit of the model; it is a trade of capability for speed, which the context
contract forbids. MTP is rejected for this model on p300x2. The 2026-10-20 numbers are computed from the default
plan's pool (`runs/s67r/j2_s1_default.json` `plan`), not probed on the device. There is no byte-calculation artifact
beyond this table.

## Rejected candidates

| Candidate | Evidence | Reason |
|---|---|---|
| Traced prefill buckets | 222 vs 232 ms (128 rows), equal at 512 / 2048 rows | device-bound; ~70 MB per bucket of trace region + outputs |
| DRAM-sharded LM head (`lm_ds`) | one split over 38,912 columns overflows L1 (3.05 MB CBs per core); 4096-column splits: 10 x 66 µs = 0.63 ms vs 0.47 ms interleaved | slower; DS decode matmul ran on 8 reader cores at ~270 GB/s |
| Force-argmax sampling | full-vocab gather 38720 x 32 per die | the split greedy sampler is 0.39 ms (0.7 % of the step) |
| Sinkhorn fidelity, 32-user MoE glue, prefill in0_block_w | - | owned by stage 5 (coordinator scope update) |

## Readiness on the final default (goal 5)

Run `j2_s1_default` (headline build). The readiness runners are driven on the same generator.

| Check | top-1 | top-5 | top-100 | Gate | Report |
|---|---|---|---|---|---|
| Teacher forcing (AIME24 #0, chat template, 100 positions) | 91 % | **100 %** | **100 %** | top-5 >= 98 %, top-100 = 100 % | `runs/teacher_forcing.json` |
| Prefill check | 91 % | **100 %** | **100 %** | same | `runs/prefill_check.json` |

Teacher-forcing runner timing: TTFT 407 ms (161 tokens, host logits), decode 17.78 t/s/u.

Exact HEAD default: the HEAD default also includes the stage-8 `kda_in` decode program, which this run did not have.
Stage 8 gated kda_in on 914af5b9594 (`doc/datatype_sweep/`). The first run of the exact HEAD default (rework code +
kda_in) to record teacher forcing and the prefill check is the stage-8 re-selection / stage-9 G9 gate.

Free-running checks on the same build:
- **Autoregressive** (`autoregressive/VERDICT.md`): the stock prompt, 256 tokens.
  - Two runs are bit-identical.
  - `check_degenerate_output.py` exits 0.
  - HF scored over TT's tokens: top-1 93.75 %, top-5 100 %; the phrase loop is HF's own top-1.
- **Qualitative suite:** not rerun in the rework. The j2 autoregressive tokens are bit-identical to stage 8's
  `runs/selected_default_r{1,2}` trajectories, and stage 8 r1 ran the chat suite on combo_a_kv
  (`doc/datatype_sweep/runs/selected_default_r1/checks/`). `qualitative/VERDICT.md` covers the 2026-10-17 build and
  states that mapping.

## Context and memory

The stage-7 changes add no persistent device memory except the 512-row prefill constants (KDA prefill object, DSA pool
/ ape rows, MoE dispatch constants). The decode trace is unchanged; there is no prefill trace.

On 2026-10-17 (stage-7 precision, 1 slot, 16,384) DRAM allocated per die went from 32,410 MB to 32,425 MB (+14.2 MB).

The context the stage ships now comes from the generator's planner and the stage-8 precision:
- 280,576 tokens at 1 slot (standalone default in j2; 294,912 on HEAD's G9 profile data, same 1,200 MiB reserve);
- 311,296 tokens measured on the vLLM path;
- details in `doc/context_contract.json` and `doc/full_model/README.md`.

Free per die after the default build: 1,068,291,584 B. Trace region use: 40.4 MB of the 64 MiB region.

## 2026-10-18 rework (stage-6 review #1 / stage-7 review #1)

Every number in this section comes from one snapshot: HEAD 656c998e3df plus the rework's files (committed as
de9852c604d / b3270061cca). Records are in `runs/s67r/` and logs in `~/glm53-flash-bringup/logs/s67r/`. Each job logged
git HEAD and the md5 of `tt/*.py` at start and end. They matched for every job except j6 (see below).

The decode corruption fixed in this rework was a `ttnn.deallocate` on a view of the trace's logits output
(`doc/full_model/work_log.md`). It gives teacher forcing 14 / 32 / 90 %. "Persistent all-gather" (below) re-tests
`ag_persistent` against that signature.

### Chunk plan vs the legacy plan (1 slot, `j2_s1_default`, 24 greedy tokens after each prefill)

| Prompt | Cost plan | Legacy plan | Prefill cost / legacy (ms) | Last-row logits cost vs legacy | Greedy tokens |
|---|---|---|---|---|---|
| 513 | 512 + 128 (1 row) | 2048 (513 rows) | 566 / 1,034 | PCC 0.674, top-1 equal, top-5 overlap 3 / 5 | equal (24 / 24) |
| 600 | 512 + 128 (88) | 2048 | 620 / 1,033 | PCC 0.991, top-1 equal, top-5 4 / 5 | diverge at token 10 |
| 1,000 | 512 + 512 (488) | 2048 | 768 / 1,032 | PCC 0.998, top-1 equal, top-5 5 / 5 | diverge at token 5 |
| 1,153 | 2048 | 2048 | 1,031 / 1,032 | bit-identical | equal |
| 2,049 | 2048 + 128 (1) | same | 1,190 / 1,190 | bit-identical | equal |
| 2,600 | 2048 + 512 + 128 (40) | 2048 + 2048 (552) | 1,875 / 2,835 | PCC 0.997, top-1 equal, top-5 5 / 5 | equal |
| 10,001 | 5 x 2048 | same | 8,243 / 8,246 | bit-identical | equal |

Where the plans differ, they are different chunkings of the same computation in bfp4 / fp8. At 513 the cost plan
ends with a 1-row chunk. Its last row is computed alone, in a 128-row chunk whose other 127 rows are padding.

HF CPU control: `tools/hf_full_ref.FullModel` (the unmodified HF model on the same E192 checkpoint) on the same ids,
last-row logits (`~/glm53-flash-bringup/scripts/s67r_hf_lastrow.py`, record `runs/s67r/j2_hf_lastrow.json`).

| Prompt | Cost plan vs HF | Legacy plan vs HF |
|---|---|---|
| 513 | PCC 0.705, top-1 equal, top-5 overlap 3 / 5 | PCC 0.713, top-1 equal, top-5 overlap 1 / 5 |
| 600 | PCC 0.964, top-1 equal, top-5 overlap 4 / 5 | PCC 0.957, top-1 equal, top-5 overlap 4 / 5 |

The plan choice does not change accuracy. Both plans are equally far from HF, and the cost plan is no worse on any
metric.

The 513 PCC is low for both plans, and that comes from the row itself. Position 513 is a near-certain token:
- the top-1 logit leads the second by 14 logits, so p(top-1) = 1.000 and the entropy is 0;
- ranks 2 and below are a flat tail (logit std 1.7, against 2.2 at 600 / 1,000);
- on a tail that flat, bfp4 / fp8 noise of a few logits dominates the correlation and reorders ranks 2 to 5;
- those ranks carry no probability mass, so the top-5 overlap there means little.

The 600 / 1,000 / 2,600 rows have real alternatives (p(top-1) 0.54 to 0.86). At 600 the PCC against HF is 0.96 for
both plans. HF was not run at 1,000 or 2,600 (the HF job ran out of memory after 600); there, the cost plan agrees with
the legacy plan at PCC >= 0.997.

### Long context on the final default (1 slot, `max_seq_len` 280,576)

| Prompt | TTFT | Chunks | Traced token-out decode at the deep position | Copy accuracy |
|---|---|---|---|---|
| 32,768 | 28.0 s | 16 | 59.78 ms | 32 / 32 |
| 100,000 | 87.3 s | 49 | 59.86 ms | 32 / 32 |

Block-sharded DSA layout, bf16 latent (`j6_s1_shard`): a non-default configuration.
- `GLM53_PRECISION=none`, 1 slot, `max_seq_len` 131,072, selected by the planner above the replicated 59,392.
- Teacher forcing 92 / 100 / 100 %, prefill check 91 / 100 / 100 %.
- Token-out decode: 56.90 ms.

| Prompt | TTFT | Traced decode at the deep position | Copy accuracy | Sink-weight headroom (min over 11 DSA layers) |
|---|---|---|---|---|
| 32,768 | 28.8 s | 62.58 ms | 32 / 32 | 65.7 nats, no zero weights |
| 100,000 | 89.8 s | 62.74 ms | 32 / 32 | 65.7 nats, no zero weights |

Chunk plans on this layout:
- 2,600 tokens: PCC 0.997, top-1 equal; greedy tokens diverge at token 10. Cost plan 1,981 ms, legacy 2,766 ms.
- 10,001 tokens: bit-identical (both plans are the same chunking). The first run took 11,787 ms against 7,697 ms for
  the second. That gap is not a post-capture compile. Stage-7 review #2 re-ran it on the full 45-layer
  sharded model at 131,072 tokens (b3270061cca, `~/glm53-flash-bringup/artifacts/s7r2/cachefull.json`): two fresh
  10,001-token prefills after the capture, each followed by a traced decode replay.
  - Program cache: 890 entries before and after.
  - Trace-allocation tracker: 0 live allocations at the replays.
  - Times: 7,826 / 7,821 ms.

  The reduced model at 16,384 gives the same: no new programs, first run = second run. The gap did not reproduce,
  and its cause is not identified. Chunk starts 2,048 .. 8,192 are now covered by the tracker. The standalone
  generator counts any later compile (`Glm53Generator.guard_programs`).

j6's end-of-job md5 differs from its start in `tt/generator.py`, `tt/moe.py`, `tt/moe_opt.py` and `tt/weight_cache.py`.
These are comment and black-formatting edits made in the snapshot during the run. Each file's AST is identical, and
the modules were already imported.

### Multi-slot decode (token-out, model + sampler trace, ms per step)

| Build | 1 active | 4 active | 8 active | 32 active | Correctness |
|---|---|---|---|---|---|
| 8 slots, 4,096, inactive-row skip on (`j3_s8`) | 81.65 | 93.50 | 112.24 | - | joint == solo, bit-identical |
| 8 slots, skip off (`j3b_s8_noskip`) | 111.01 | 111.00 | 110.97 | - | all logits bit-identical to skip on |
| 32 slots, 2,048 (`j4_s32`) | 115.29 | - | 117.16 (8) | 114.24 | joint == solo, bit-identical |

## Persistent all-gather (`ag_persistent`)

**Re-tested on the fixed code (stage-7 review #2, 2026-10-21): it passes.**
- Code: b3270061cca snapshot (logits-pad fix in), `GLM53_PRECISION` unset (combo_a_kv_dense4_g4).
- Records: `runs/s7r2/{agp0,agp1,det0,det1}.json`.
- Logs: `~/glm53-flash-bringup/logs/s7r2/{agp,agp1,det0,det1}.log`.

| Check | `ag_persistent=0` | `ag_persistent=1` |
|---|---|---|
| Full model, 1 slot, 16,384: teacher-forcing top-1 / top-5 / top-100, references p0 / p1 / p2 | 91 / 96 / 91 %; 100 %; 100 % | identical |
| Prefill check top-1, p0 / p1 / p2 | 91 / 95 / 91 % | identical |
| Token-out decode (model + sampler trace) | 50.40 ms | **49.96 ms** (-0.44 ms, -0.9 %) |
| Reduced 0 / 3 / 4, 8 slots, prompts 100 / 300: traced == eager, repeat == first (`s7_det_reduced_s8` recipe) | all equal | all equal; tokens identical to `=0` |

The 2026-10-17 failure had the logits-pad bug's exact signature: teacher forcing 14 / 32 / 90, and 8-slot runs that did
not repeat. With that bug fixed, neither symptom appears.

The 2026-10-17 reduced control at `=0` was a constant token (67404). This one is not (91886 / 67404 / 52420), so it can
discriminate.

The gain matches stage 5's per-layer measurement (7 to 15 µs per layer x 45). Caveats:
- one run per setting, in separate processes;
- 8 slots were tested on the reduced model only;
- ttnn still skips the all-gather's start barrier with persistent outputs
  (`all_gather_async_default_program_factory.cpp:752`); no failure was observed.

Decision: **not adopted (capacity re-probe cost); available via MCOpts** (`GLM53_MC=ag_persistent=1` or `MCOpts(ag_persistent=True)`). Quality is identical and decode is
0.44 ms (0.9 %) faster. But the persistent all-gather outputs hold DRAM, and every serving capacity probe (stage 9 G9)
was measured without them, so adopting it means re-probing every serving profile. Its capacity impact is unmeasured.
The re-test also showed DRAM pressure: a second build in the same process ran out of memory. The shipped
`DEFAULT_MC` keeps it off.

The 2026-10-17 record, superseded: the persistent all-gather was taken in stage-5 round 6 and reverted on 2026-10-18
after `s7_final_s1_agp1_TFfail.log` (teacher forcing 14 / 32 / 90) and `s7_det_reduced_s8.log` (8 slots not repeating).

## Tracy

`tracy/{before,after}_s{1,8}_l0/` (2026-10-17, stage-7 precision) holds one decode step (model + sampler replay) and one
128-row prefill of a 1-layer model (layer 0 + embedding, head, sampler), with `perf_{decode,prefill}_report.{txt,csv}`
and advice. These profiles are stale:
- `after_*` was taken with the rejected DRAM-sharded LM head (`GLM53_HEAD=norm_sharded,lm_ds`). The shipped interleaved
  LM head (471 µs) appears only in `before_*`.
- They cover layer 0 (kda_dense) only.

The current full-model profile is stage 8's 3-kind combo_a_kv decode window (`doc/datatype_sweep/profile/combo_a_kv/`).
Stage 10 profiles the final default.

The device-profiler buffers drop ops in these windows (`TT_TRACY_SKIP_MISSING_DEVICE_DATA=1`). Each op appears on at
least one device but not on all four, so per-op times are valid and per-device sums are not.

## Commands and logs

```
# final-default evidence (one generator build per job; scripts pin a snapshot)
GLM53_TP=4 python -m models.autoports.zai_org_glm_5_3_flash.tools.s67r_checks --slots 1 \
    --do ready,ar,ttft,ptchange,sampling,chunkplan,decode,deep --deep 32768,100000 --out j2_s1_default.json
GLM53_TP=4 python -m models.autoports.zai_org_glm_5_3_flash.tools.s67r_checks --slots 1 --seq-len 16384 --do ttft,decode
~/glm53-flash-bringup/scripts/s67r_full.sh j2 j3 j3b j4 j6
# 2026-10-17 builds
~/glm53-flash-bringup/scripts/s7_base.sh base       # stage-6 paths (legacy env)
~/glm53-flash-bringup/scripts/s7_cand.sh            # candidate (prefill traces, 512 bucket)
~/glm53-flash-bringup/scripts/s7_tracy.sh before|after
```

Logs: `~/glm53-flash-bringup/logs/s67r/`, `~/glm53-flash-bringup/logs/s7r2/` (review #2) and
`~/glm53-flash-bringup/logs/s7_*.log` (2026-10-17).

Records: `runs/s67r/*.json` and `~/glm53-flash-bringup/artifacts/s7/{base_s1,cand_s1,...}.json`.

Work log: `work_log.md`.

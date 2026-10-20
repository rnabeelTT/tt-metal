# GLM-5.3-Flash (E192) optimized full model

Stage 7 of the tt-model-bringup flow. The stage optimizes the full 45-layer model (`tt/model.py`, `tt/generator.py`)
on the (1, 4) mesh of a p300x2 (4 Blackhole dies, FABRIC_1D, TP4). The decoder layers come from stage 5
(`tt/multichip_decoder.py`, `DEFAULT_MC`); this stage does not change them.

## Result (headline)

Workload: synthetic prompts (random token ids), greedy, 1 slot (`max_batch_size=1`), max_seq_len 16,384, TP4.
"Before" = stage-6 code paths on the same day (`GLM53_HEAD=legacy GLM53_PREFILL_PLAN=legacy
GLM53_PREFILL_LENGTHS=128,2048 GLM53_MOE_SORTED_ROWS=1024`, stage-5 defaults of 15:00 UTC). "After" = stage-7 defaults
(final run `s7_final_s1.log`, stage-5 defaults of 20:30 UTC with `GLM53_MC=ag_persistent=0`, see "Persistent
all-gather finding"; weight cache `GLM53_CACHE_DIR`). Records: `~/glm53-flash-bringup/artifacts/s7/{base_s1,cand_s1,final_s1}.json`.

| Metric | Stage 6 (README) | Before (same build day) | After (stage-7 defaults) |
|---|---|---|---|
| TTFT, 128-token prompt | - | 232 ms | 230 ms |
| TTFT, 161-token prompt (readiness prompt length) | 1252 ms (4 slots) | ~1182 ms (one 2048 chunk) | 376 ms (one 512 chunk) |
| TTFT, 512-token prompt | - | ~1182 ms (one 2048 chunk) | 379 ms |
| TTFT, 2048-token prompt | - | 1182 ms | 1035 ms (no MoE row split) |
| TTFT, 8192-token prompt | - | 5134 ms | 4556 ms |
| Decode, model trace only (teacher-forcing boundary), 1 slot | - | 56.86 ms | 53.69 ms |
| Decode, token-out (model + sampler trace, device feedback), 1 slot | - | 57.24 ms | 54.08 ms |
| Decode, generator loop with token readback, 1 slot | 95 ms (4 slots) | 57.36 ms = 17.4 t/s/u | 54.18 ms = **18.46 t/s/u** |
| Decode, 8-slot build, 1 / 4 / 8 active slots | - | 112.8 / 112.8 / 112.8 ms (8.8 t/s/u) | 83.9 / 95.7 / 114.2 ms (11.8 / 10.4 / 8.7 t/s/u) |
| Teacher forcing top-1 / top-5 / top-100 (AIME24 chat, 100 positions) | 93 / 100 / 100 % | - | **92 / 100 / 100 %** (decode 18.44 t/s/u in the runner) |
| Prefill check top-1 / top-5 / top-100 | 92 / 100 / 100 % | - | **91 / 100 / 100 %** |

TTFT = warm `Glm53Model.prefill` of the prompt to last-row logits on the device, plus a device sync; the readiness
TTFT (host logits) adds ~12 ms (one logits row to the host, see "TTFT"). Decode numbers are host wall per step over
30 back-to-back steps after a warm step.

Attribution (stage-7 review #1): the decode before/after above (57.36 -> 54.18 ms) comes almost entirely from the
stage-5 round-6 decoder options that landed between the two builds; stage 7's own decode change is the sharded final
RMSNorm (65 -> 7 us, ~0.06 ms per step). The "before" 8-slot build ran with the persistent all-gather that was later
found to break decode (ag_persistent=1), so its 112.8 ms is not a valid baseline for correctness, only for time. Stage 7's
own wins are TTFT: the 512-row bucket and the cost chunk plan (161 tokens 1182 -> 376 ms), the MoE split rule and the
one-row host logits read. The 2026-10-18 rework section below re-measures everything on the rework snapshot.

## What changed (stage-7 defaults)

| Change | Where | Default | Effect (measured) |
|---|---|---|---|
| Cost-based prefill chunk plan: the cheapest multiset of constructed lengths that covers the prompt, largest first | `Glm53Model.chunk_plan`, `GLM53_PREFILL_PLAN=cost`, costs `PREFILL_COST_MS` / `GLM53_PREFILL_COST` | on | a 161-token prompt ran one 2048-row chunk (1.18 s); now one 512-row chunk |
| Prefill lengths 128 / 512 / 2048 (was 128 / 2048) | `PREFILL_LENGTHS_DEFAULT`, `GLM53_PREFILL_LENGTHS` | on | 512-row chunk 378 ms vs 2048-row chunk 1033 ms |
| MoE prefill row split chosen from free DRAM | `Glm53Model.choose_moe_split`, `GLM53_MOE_SPLIT_AUTO=1`, `GLM53_MOE_NOSPLIT_FREE_MIB` (1024) | on | 2048-row chunk 1033 ms unsplit vs 1153 ms split (-10.4 %); 8192 tokens 4548 vs 5033 ms (-9.6 %) |
| Last logits row read alone | `Glm53Model._row_to_host` | on | host-logits TTFT path: ~50 ms -> ~12 ms over the device TTFT |
| Final RMSNorm width-sharded on 32 cores | `GLM53_HEAD=norm_sharded` | on | 65 µs (1 core) -> 7 µs per decode step |
| Warm-up covers every (chunk length, DSA path) a plan can produce | `Glm53Generator._warm_plans` | on | lazily built prefill constants exist before any trace capture (2026-10-16 hang rule) |
| Inactive decode rows skip the routed experts | `GLM53_SKIP_INACTIVE=1`, `Glm53Model._install_inactive_skip` | on (builds with >= 2 slots) | 8-slot build, 1 active: 112.8 -> 83.9 ms; 4 active: 95.7 ms; 8 active: 114.2 ms; tokens exact |
| Traced first prefill chunk | `GLM53_PREFILL_TRACE=128,512` | off (rejected) | 128 rows 222 vs 232 ms; bit-identical |
| DRAM-sharded LM head (column splits) | `GLM53_HEAD=norm_sharded,lm_ds` | off (rejected) | 10 x 66 µs vs 471 µs interleaved |

## Decode breakdown (goal 1)

Full model, 1 slot (`s7_base_s1`, stage-5 defaults of 15:00): model trace 56.86 ms, sampler trace 0.38 ms, token-out
57.24 ms, generator loop (one token readback per step) 57.36 ms. The host adds 0.12 ms per step; the decode loop has no
host input refresh (tokens fed back on the device, positions advanced in the trace, page table copied only on change).

Layer-stack lower bound (stage 5): 3 x 0.895 + 31 x 1.102 + 11 x 1.407 = 52.3 ms (shipped `DEFAULT_MC`, 20:30 UTC). The
stage-7 candidate build on those defaults measured 53.23 ms for the model trace and 53.61 ms token-out
(`cand_s1`): 1.3 ms (2.5 %) over the layer stack. The 1.3 ms beyond the layers (reduced 1-layer Tracy,
`tracy/{before,after}_s1_l0/perf_decode_report.txt`, device 0..3 union):

| Item | Device time | Note |
|---|---|---|
| LM head [32 x 4096] x [4096 x 38720] bfp8, interleaved, 110 cores | 471 µs | 168 MB per die at 357 GB/s; DRAM-bound |
| Sampler trace: TopK (large-indices composite, 110 cores) 205 µs, all-gathers 25 µs, sampling 28 µs, seed 19 µs, casts | 380 µs | split greedy: local top-32 per die, gathered candidates |
| Final RMSNorm | 65 -> 7 µs | `norm_sharded` |
| Stream mean, all-gather of h, embedding, decode-input refresh (35 small ops) | ~250 µs | |

None of these is material at 53 ms per token; the decoder layers are 97.5 % of the step. The DRAM-sharded LM head
lost (below). Force-argmax would gather the full vocabulary (38720 x 32 x 2 B per die); split greedy stays.

Multi-slot decode: the step time depends on the BUILT slot count, not on the active slots (8-slot build: 112.8 ms with
1, 4 or 8 active slots; 1-slot build: 56.9 ms). Every slot row runs the MoE: the decode MoE path for fewer than 32
rows (pairs: one group per (row, expert)) reads one expert block per group, so 8 rows read 64 expert blocks per layer.
`GLM53_SKIP_INACTIVE` (default on for builds with 2 or more slots) sets the expert ids of inactive rows to 65535 inside
the decode trace (from the per-step active flags of the decode inputs); the grouped sparse matmul skips groups with an id
>= E (`skip_padding_groups`), so inactive rows read no expert weights. Final 8-slot build (`s7_final_s8.log`): 83.9 /
95.7 / 114.2 ms per step at 1 / 4 / 8 active slots (before: 112.8 ms at every count). The active slot's tokens are
exact: identical greedy continuations with the skip on and off and over four repeats (reduced model, 8 slots,
`s7_skip_reduced_agp0.log` vs `s7_det_reduced_s8_noagp.log`; full model `s7_final_s8.log`). The remaining per-row
cost at 1 active (83.9 vs 53.7 ms for a 1-slot build) is the pairs glue (row repeat, SwiGLU, k-sum on all 8 x 8 groups),
the router and the 8-row attention; the 8-active step pays 4 small ops per MoE layer for the skip; its 114.2 vs 112.8 ms is not separated from
the stage-5 default change between the two builds (not A/B'd on one build). The 32-slot sorted path is unchanged (stage 5); the 2026-10-18 rework measured it on the full model (`j4_s32`, below).

Single-user serving should build with `GLM53_MAX_BATCH=1` (1-slot build: 53.6 ms token-out = 18.6 t/s/u). The 1-slot
path pads the KDA batch to 2 internally (stage 5) and the sampler rows to 32; it passed the readiness gates below.

## TTFT (goal 2)

Full model, 1 slot, warm (`cand_s1`, ms):

| Prompt | Plan (chunk rows, valid rows) | Eager | Traced first chunk |
|---|---|---|---|
| 128 | (128, 128) | 232 | 222 |
| 161 | (512, 161) | 375 | 383 |
| 512 | (512, 512) | 380 | 375 |
| 1000 | (512, 512), (512, 488) | 757 | 752 |
| 2048 | (2048, 2048) | 1033 | 1033 |
| 8192 | 4 x (2048, 2048) | 4548 | 4550 |

- The 128-row chunk costs 232 ms: ~5 ms of device time per layer (45 layers). The host enqueue is not the limit: the
  traced replay of the same chunk takes 222 ms. A prefill trace per bucket therefore saves 10 ms at 128 rows and nothing
  at 512 rows, and costs ~30 MB of trace region plus ~40 MB of trace outputs (KDA states, X) per bucket, i.e. ~6K
  tokens of 1-slot context per bucket. Rejected; the option (`GLM53_PREFILL_TRACE`) stays.
- The traced path itself is exact: logits and 8-token greedy continuations bit-identical to eager (100, 128, 161, 300,
  500-token prompts, reduced and full model, `s7_reduced_check.log`, `s7_cand_s1.log`). It needed the DSA open-pool
  hand-off out of the captured ops (`GlmDSA.write_prefill_tail`, `prefill_inputs(host=True)`).
- The 512-row bucket is the TTFT win: prompts of 129 to 512 tokens took a 2048-row chunk (1.03 to 1.18 s); they now
  take one 512-row chunk (375 ms).
- MoE row split (stage-6 memory fix, `GLM53_MOE_SORTED_ROWS=1024`): +120 ms per 2048-row chunk (1153 vs 1033 ms) and
  +485 ms at 8192 tokens. It only lowers the prefill peak (417 -> 297 MiB above the layer entry). `choose_moe_split`
  runs the whole chunk when the largest free DRAM block per die is >= 1 GiB at warm-up (1-slot, 16K build: 1.70 GB free,
  no split); a build that fills DRAM with KV blocks (vLLM, long contexts) keeps the split. Bit-identical either way.

## MTP (goal 4): rejected

The parent checkpoint has one MTP layer (layer 45, kept in `ckpt/glm53_flash_e192_fp8/mtp/`, `num_nextn_predict_layers`
set to 0 in the E192 config). It is a DSA attention layer with 288 routed experts (the E192 mask covers layers 0-44
only), a shared expert, `eh_proj` [4096 x 8192] and `enorm` / `hnorm` / `shared_head.norm`, and no mHC weights.

| Item | Value |
|---|---|
| Routed expert parameters | 7.25 B (288 x 3 x 4096 x 2048) |
| Other parameters | 185 M |
| Device bytes per die (bfp4 experts, bfp8 rest, TP4) | ~1.07 GB |
| Extra DSA cache per token | 1088 B per die |
| 1-slot context | 104,448 -> ~13.8K tokens (-87 %) |
| 8-slot context (81,920 per slot) | does not fit |
| Expected speed-up | <= 1.3x (GLM-4.5-Air: 26.9 -> 20.5 ms with MTP); here less: the verify step doubles the MoE expert reads at 1 user, and 34 KDA layers need a recurrent-state snapshot / rollback for rejected drafts |

The context loss is not a hard physical limit of the model; it is a trade of capability for speed, which the context
contract forbids. MTP is rejected for this model on p300x2. It becomes viable with the sharded DSA cache (stage-4
rework) plus a bfp8 latent, which free ~1 GB per die.

## Rejected candidates

| Candidate | Evidence | Reason |
|---|---|---|
| Traced prefill buckets | 222 vs 232 ms (128 rows), equal at 512 / 2048 rows | device-bound; ~70 MB per bucket of trace region + outputs |
| DRAM-sharded LM head (`lm_ds`) | one split over 38,912 columns overflows L1 (3.05 MB CBs per core); 4096-column splits: 10 x 66 µs = 0.63 ms vs 0.47 ms interleaved | slower; DS decode matmul ran on 8 reader cores at ~270 GB/s |
| Force-argmax sampling | full-vocab gather 38720 x 32 per die | the split greedy sampler is 0.38 ms (0.7 % of the step) |
| Sinkhorn fidelity, 32-user MoE glue, prefill in0_block_w | - | owned by stage 5 (coordinator scope update) |

## Readiness on the final defaults (goal 5)

Final run `~/glm53-flash-bringup/logs/s7_final_s1.log`: one 1-slot build (`max_seq_len` 16,384,
`GLM53_MC=ag_persistent=0`), readiness runners driven on the same generator (`tools/s7_probe.py --do ready,checks`).

| Check | top-1 | top-5 | top-100 | Gate | Report |
|---|---|---|---|---|---|
| Teacher forcing (AIME24 #0, chat template, 100 positions) | 92 % | **100 %** | **100 %** | top-5 >= 98 %, top-100 = 100 % | `runs/teacher_forcing.json` |
| Prefill check | 91 % | **100 %** | **100 %** | same | `runs/prefill_check.json` |

Teacher-forcing runner timing: TTFT 423 ms (161 tokens, host logits), decode 18.44 t/s/u, e2e 17.27 t/s/u.

Free-running (`autoregressive/`, `qualitative/`, same build): the stock autoregressive prompt, 256 tokens, two runs
bit-identical; `check_degenerate_output.py --scope autoregressive --missing-artifacts critical`: exit 0, "No degenerate
output detected" (adjacent duplication 0.0, trigram loops 0.054; `autoregressive/degenerate_check.txt`). TT leaves HF's
path at token 1 as in stage 6 (a near-tie inside HF). The 6 chat prompts of the qualitative suite: 4 end on EOS (365 to
1122 tokens), 2 reach the 2048-token budget (q3, q6, as in stage 6); the texts are coherent and on topic
(`qualitative/tt_outputs.json`).

With the stage-5 default `ag_persistent=True` the same 1-slot build failed teacher forcing: top-1 14 %, top-5 32 %,
top-100 90 % (prefill check 91 / 100 / 100 %: prefill does not use the persistent all-gather;
`s7_final_s1_agp1_TFfail.log`). `GLM53_MC=ag_persistent=0` alone restores 92 / 100 / 100 % (`s7_tfdiag_agp0.log`).

## Context and memory

The stage-7 defaults add no persistent device memory except the 512-row prefill constants (KDA prefill object, DSA
pool / ape rows, MoE dispatch constants) and the decode trace is unchanged (no prefill trace). Measured after build and warm-up (1 slot, max_seq_len 16,384, `ttnn.get_memory_view` x 8 banks): DRAM allocated
32,410 MB (stage-6 paths, same day) -> 32,425 MB per die: +14.2 MB. The 1-slot capacity probe of stage 6 (104,448 tokens) left 183 MB free per die at the prefill peak, so
the supported context stays 104,448; at that fill the free-DRAM rule keeps the MoE row split (peak unchanged). Trace
region use 47.6 -> 40.4 MB (stage-5 changes; 64 MiB region unchanged).
`doc/context_contract.json`: unchanged by this stage. Superseded: see "2026-10-18 rework" (stage-8 default, 280,576 / 311,296 tokens).

## 2026-10-18 rework (stage-6 review #1 / stage-7 review #1)

All numbers in this section come from one snapshot: HEAD 656c998e3df plus the rework's files. The defaults are the
current ones: stage-8 precision `combo_a_kv` and `Glm53Generator(mesh)` sizing (see `doc/full_model/README.md`).
Records are in `runs/s67r/` and logs in `~/glm53-flash-bringup/logs/s67r/`. Each job logged git HEAD and the md5 of
`tt/*.py` at start and end; none changed.

The decode corruption fixed in this rework (a `ttnn.deallocate` on a view of the trace's logits output,
`doc/full_model/work_log.md`) gives teacher forcing 14 / 32 / 90 %. That is the same signature this stage attributed
to `ag_persistent=1`. Turning on the persistent all-gather changes the allocation order, which is how that bug shows.
The `ag_persistent` attribution above is therefore suspect. It was not re-tested on the fixed code.

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

The 513 PCC is low for both plans, and that comes from the row itself. Position 513 is a near-certain token: the top-1
logit leads the second by 14 logits, so p(top-1) = 1.000 and the entropy is 0. Ranks 2 and below are a flat tail
(logit std 1.7, against 2.2 at 600 / 1,000). On a tail that flat, bfp4 / fp8 noise of a few logits dominates the
correlation and reorders ranks 2 to 5. Those ranks carry no probability mass, so the top-5 overlap there means
little. The 600 / 1,000 / 2,600 rows have real alternatives (p(top-1) 0.54 to 0.86). At 600 the PCC against HF is 0.96
for both plans. HF was not run at 1,000 or 2,600 (the HF job ran out of memory after 600); there, the cost plan
agrees with the legacy plan at PCC >= 0.997.

### Long context on the final defaults (1 slot, `max_seq_len` 280,576)

| Prompt | TTFT | Chunks | Traced token-out decode at the deep position | Copy accuracy |
|---|---|---|---|---|
| 32,768 | 28.0 s | 16 | 59.78 ms | 32 / 32 |
| 100,000 | 87.3 s | 49 | 59.86 ms | 32 / 32 |

Block-sharded DSA layout, bf16 latent (`j6_s1_shard`: `GLM53_PRECISION=none`, 1 slot, `max_seq_len` 131,072,
selected by the planner above the replicated 59,392). This is a non-default configuration.

- Teacher forcing: 92 / 100 / 100 %. Prefill check: 91 / 100 / 100 %.
- Token-out decode: 56.90 ms.

| Prompt | TTFT | Traced decode at the deep position | Copy accuracy | Sink-weight headroom (min over 11 DSA layers) |
|---|---|---|---|---|
| 32,768 | 28.8 s | 62.58 ms | 32 / 32 | 65.7 nats, no zero weights |
| 100,000 | 89.8 s | 62.74 ms | 32 / 32 | 65.7 nats, no zero weights |

Chunk plans on this layout:
- 2,600 tokens: PCC 0.997, top-1 equal; greedy tokens diverge at token 10. Cost plan 1,981 ms, legacy 2,766 ms.
- 10,001 tokens: bit-identical, as expected, since both plans are the same chunking. The timing (11,787 vs 7,697 ms)
  is a cold first run of the deep-chunk programs, not a plan difference. In j2 the two plans measured within 3 ms.

The run's end-of-job md5 differs from its start in `tt/generator.py`, `tt/moe.py`, `tt/moe_opt.py` and
`tt/weight_cache.py`. These are comment and black-formatting edits made in the snapshot during the run. Each file's
AST is identical, and the modules were already imported.

### Context size vs speed (1 slot, same code)

| `max_seq_len` | Token-out decode | TTFT 128 / 161 / 2,048 / 8,192 tokens | Run |
|---|---|---|---|
| 16,384 | 51.10 ms (19.6 t/s/u) | 225 / 373 / 1,025 / 4,408 ms | `j2b_s1_16k` |
| 280,576 (default) | 56.24 ms (17.8 t/s/u) | 228 / 377 / 1,026 / 6,444 ms | `j2_s1_default` |

The sparse DSA indexer, in both decode and prefill chunks past 2,048 tokens, scores every pool slot allocated to the
user. That is `P_alloc`, sized by `max_seq_len`, not the filled prefix. The default buys 17x the context for +10 %
decode time and +46 % TTFT at 8K. At 16,384 the numbers equal stage 8's (51.14 ms).

### Multi-slot decode (token-out, model + sampler trace, ms per step)

| Build | 1 active | 4 active | 8 active | 32 active | Correctness |
|---|---|---|---|---|---|
| 8 slots, 4,096, inactive-row skip on (`j3_s8`) | 81.65 | 93.50 | 112.24 | - | joint == solo, bit-identical |
| 8 slots, skip off (`j3b_s8_noskip`) | 111.01 | 111.00 | 110.97 | - | all logits bit-identical to skip on |
| 32 slots, 2,048 (`j4_s32`) | 115.29 | - | 117.16 (8) | 114.24 | joint == solo, bit-identical |

Stage 7 measured 83.9 / 95.7 / 114.2 ms on the 8-slot build. At 32 slots the time is flat over 1 / 8 / 32 active slots
(114 to 117 ms). The inactive-row skip targets the S < 32 per-row MoE path (`_install_inactive_skip`).

### Readiness on the final defaults

`j2_s1_default`: teacher forcing 91 / 100 / 100 %, prefill check 91 / 100 / 100 % (`runs/teacher_forcing.json`,
`runs/prefill_check.json`). Teacher-forcing runner: TTFT 407 ms, decode 17.78 t/s/u.

The autoregressive check (`autoregressive/`) ran on the same build:
- two runs are bit-identical;
- `check_degenerate_output.py` exits 0 (adjacent duplication 0.0, trigram loops 0.135);
- the HF trajectory score is in `autoregressive/VERDICT.md`.

## Persistent all-gather finding (for stage 5)

With `DEFAULT_MC.ag_persistent=True` (stage-5 round 6), an 8-slot build is not run-to-run reproducible: four identical
prefill + 8-token greedy runs of the same prompt give different tokens from step 4 to 7 (`s7_det_reduced_s8.log`,
reduced model 0/3/4). With `GLM53_MC=ag_persistent=0` the four runs are identical (`s7_det_reduced_s8_noagp.log`), and
the tokens differ from the persistent run from the first decode step. On the full 45-layer 1-slot build the
persistent all-gather breaks decode outright (teacher forcing top-5 32 % vs 100 %, above). Every stage-7 number "after"
uses `GLM53_MC=ag_persistent=0`; `DEFAULT_MC` (stage 5) must turn it off. The 122B port saw replays lose
reproducibility with persistent all-gather outputs too.

## Tracy

`tracy/{before,after}_s{1,8}_l0/`: one decode step (model + sampler replay) and one 128-row prefill of a 1-layer model
(layer 0 + embedding, head, sampler), `perf_{decode,prefill}_report.{txt,csv}` with advice. Note (stage-7 review #1):
`after_*` was taken with `GLM53_HEAD=norm_sharded,lm_ds`, i.e. the rejected DRAM-sharded LM head (10 x 66 us split rows);
the shipped interleaved LM head row (471 us) appears only in `before_*`. The terminal-path numbers in "Decode breakdown"
use the shipped head. The device-profiler
buffers drop ops in these windows (`TT_TRACY_SKIP_MISSING_DEVICE_DATA=1`): each op appears on at least one device, not
on all four, so per-op times are valid and per-device sums are not.

## Commands and logs

```
# probe (one build per job): readiness, qualitative + autoregressive, TTFT, decode, traced-vs-eager checks, Tracy
GLM53_TP=4 python -m models.autoports.zai_org_glm_5_3_flash.tools.s7_probe --slots 1 --seq-len 16384 \
    --do ready,checks,ttft,decode --prompts 128,161,512,2048,8192 --active 1 --report-dir doc/optimized_full_model/runs
~/glm53-flash-bringup/scripts/s7_base.sh base       # stage-6 paths (legacy env)
~/glm53-flash-bringup/scripts/s7_cand.sh            # candidate (prefill traces, 512 bucket)
~/glm53-flash-bringup/scripts/s7_final.sh           # final defaults
~/glm53-flash-bringup/scripts/s7_tracy.sh before|after
```

Logs `~/glm53-flash-bringup/logs/s7_*.log`; records `~/glm53-flash-bringup/artifacts/s7/*.json`. Work log:
`work_log.md`.

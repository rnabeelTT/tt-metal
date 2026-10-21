# GLM-5.3-Flash (E192) full model

Stage 6 of the tt-model-bringup flow. `tt/model.py` (`Glm53Model`) runs the full text model on the (1, 4) mesh of a
p300x2. `tt/generator.py` (`Glm53Generator`, `build_generator`) is the readiness and serving contract.

## Headline (stage-6/7 rework, 2026-10-18)

Source run: `j2_s1_default` (`tools/s67r_checks.py --slots 1`, log `~/glm53-flash-bringup/logs/s67r/j2_s1_default.log`,
record `doc/optimized_full_model/runs/s67r/j2_s1_default.json`). Code: HEAD 656c998e3df plus this rework's files,
run from a pinned snapshot; git HEAD and the md5 of every `tt/*.py` were logged at the start and end and did not
change. Build: `Glm53Generator(mesh)` with no arguments and no `GLM53_*` setting, i.e. the snapshot's defaults: 1 slot,
replicated DSA cache, `max_seq_len` 280,576, trace region 64 MiB.

Precision: **the snapshot default, combo_a_kv before the stage-8 kda_in program**. The snapshot's `tt/precision.py`
(md5 109284f4) predates stage-8 commit aa104284595. That commit added a KDA input-projection decode program
(`1d:c29:ib4`, layers 1-43) and kept `packer_l1_acc` on the LoFi LM head. Stage 8 measured that delta on the earlier
wrapper (`doc/datatype_sweep/runs/combo_a_kv_g4_kdain_r2`: teacher forcing 91 / 100 / 100). HEAD's exact default is gated by stage 8 (`doc/datatype_sweep/runs/final_default_h9`). That run uses precision
`combo_a_kv_dense4_g4` (the default since d16e6b9c2f0), 1 slot, `max_seq_len` 16,384:
- teacher-forcing top-1 92 / 95 / 92 % on the 3 AIME24 chat references, top-5 and top-100 100 % on all three;
- prefill check 92 / 100 / 100;
- token-out decode 49.94 ms.

The j2 numbers below are the snapshot default at 280,576 tokens. The standalone default keeps the 1,200 MiB reserve,
which favours TTFT (see "Defaults and sizing"). Stage 10's t5 ran the full model at 280,576 (teacher forcing
92 / 100 / 100).

Explicit-override case, `v1_default` (`doc/optimized_full_model/runs/s7r2/v1_default.json`): the reserve was lowered to
the 1,024 MiB serving workspace. Code: HEAD 24be34c7cbe plus this agent's review-#2 hunks, from a step where that was
the rule; precision `combo_a_kv_dense4_g4`. Build: `Glm53Generator(mesh)` sized 311,296 tokens (2,433 blocks), 0.90 GB
free per die after the build. The same build today needs `GLM53_DRAM_RESERVE_MIB=1024`.
- Teacher forcing 92 / 100 / 100; prefill check 92 / 100 / 100.
- Token-out decode 55.37 ms (18.0 t/s/u).
- TTFT 128 / 161 / 512 / 2,048 / 8,192 tokens: 225 / 374 / 376 / 1,149 / 7,173 ms.
- 0 programs compiled after the capture (incl. a 10,001-token chunk plan: 9,196 vs 9,194 ms).

With less than 1 GiB free, `choose_moe_split` keeps the MoE prefill row split. That costs +120 ms per 2048-row chunk
(2,048 tokens: 1,149 vs 1,026 ms at 280,576), on top of the context-size cost below. This is why the standalone
default keeps the 1,200 MiB floor; long contexts belong to the serving profiles (`p300x2-long`, `p300x2-1m`).

| Metric | Value | Workload |
|---|---|---|
| TTFT, 161-token prompt | **377 ms** (397 ms with the host logits row `generate()` reads) | warm, one 512-row chunk |
| TTFT, 128 / 512 / 2,048 / 8,192 tokens | 228 / 376 / 1,026 / 6,444 ms | warm, cost chunk plan |
| Batch-1 decode, model trace only | 55.85 ms per step (17.9 t/s/u) | position 300, 20 replays, no host sync between |
| Batch-1 decode, token-out (model + sampler trace, token fed back on device) | **56.24 ms per step, 17.8 t/s/u** | same; sampler 0.39 ms |
| Batch-1 decode, `generate()` loop (host reads each token) | 56.51 ms per step, 17.7 t/s/u | same |
| Teacher forcing (host feeds the reference token every step) | 17.8 t/s/u decode, top-1 91 %, top-5 100 %, top-100 100 % | AIME24 #0 chat prompt, 161 + 100 tokens |
| Prefill check | top-1 91 %, top-5 100 %, top-100 100 % | same |
| Long prompt, 32,768 / 100,000 tokens | TTFT 28.0 / 87.3 s; decode 59.8 / 59.9 ms per token; copy accuracy 32 / 32 both | copy task, traced token-out decode at the deep position |
| Model build with the weight cache | 132 s | |

Largest full-model batch tested: 32 slots (`j4_s32`, see "Batch > 1"). Limitations: see the last section.

The default context has a cost. The sparse DSA indexer scores every pool slot allocated to the user (`[rows,
P_alloc]`, sized by `max_seq_len`), not only the filled prefix. The same code at `max_seq_len` 16,384 (`j2b_s1_16k`)
gives 51.10 ms token-out decode (19.6 t/s/u) and 4,408 ms TTFT at 8,192 tokens. At 280,576 (the j2 default) these are
56.24 ms and 6,444 ms. The planned fix (stage 10) scores only the filled
prefix. Short-context deployments should pass `max_seq_len`. The TTFT up to 2,048 tokens does not
change (no sparse chunk).

## Structure

- Embedding: the table is hidden-sharded. Die d looks up columns `[d * C/4, (d + 1) * C/4)`. The 4 mHC streams start
  as copies of the embedding, in the stream-interleaved layout of `tt/multichip_decoder.py`.
- 45 `MultichipDecoder` layers (policy `opt`): 34 KDA layers (3 with a dense MLP) and 11 DSA layers.
- Head: stream mean, all-gather, final RMSNorm, LM head (vocab-sharded). The logits stay sharded over the vocab.
- Sampling: `models.common.sampling.SamplingGenerator` on device, in lockstep on the 4 dies (identical per-draw seeds
  and a die-0 token broadcast). The sampler takes a multiple of 32 rows. With fewer slots (S), the model zero-pads the
  logits rows to `SR = ceil32(S)`.
- Prefill: one user per call, in chunks of the constructed lengths (128, 512, 2048; stage-7 cost plan). KDA layers
  carry their state from chunk to chunk. DSA layers write the paged latent cache.
- Decode: all S slots in one traced step. The per-step DSA / KDA inputs come from the position tensor on the device
  (`tt/decode_inputs.py`), and the trace advances the positions. The sampler trace writes the token into the buffer that
  the model trace reads, so steady-state decode needs no host input. A position at or past `max_seq_len` raises on the
  host and is treated as inactive on the device (no cache row written).

## Defaults and sizing

`Glm53Generator(mesh)` and `build_generator()` plan the build before any weight loads (`tt/generator.plan_context`):

- slots default to 1; `max_seq_len` defaults to the largest context of the fastest DSA layout that fits the DRAM
  budget;
- the budget comes from `serving_profiles.json` (stage 9) when its precision id is the build's, else from the
  built-in stage-7 numbers; DSA bytes per 128-token block follow the latent dtype;
- the reserve is `GLM53_DRAM_RESERVE_MIB` when set, else max(data reserve, 1,200 MiB). The data reserve is
  `serving_profiles.json`'s 1,024 MiB serving workspace, so the 1,200 MiB floor wins. On HEAD's profile data that is not
  enough to keep the MoE prefill unsplit: `v3_default` measured a 1,030,724,608 B largest free block at warm-up,
  under the 1 GiB no-split threshold, so the split engaged (TTFT 2,048 tokens 1,144 ms);
- the budget is computed at each plan (`tt/generator.budget()`), so a `GLM53_PRECISION` or reserve set after import
  applies;
- a request above the replicated capacity selects the block-sharded DSA layout automatically, with either latent
  dtype (fp8 rows: 2,288 B per token per die), up to that layout's budget. With a matching `serving_profiles.json`,
  that budget is the layout's own measured one (`layouts.sharded.budget`: 600 MiB workspace, 292,864 B per block),
  not the replicated 1,200 MiB floor. Without a `max_seq_len`, an fp8 build keeps the replicated layout;
- when the plan selects the sharded layout, the generator sets that layout's measured process settings before the
  build (`serving_profiles.layout_env`: `GLM53_MOE_SORTED_ROWS=512`, `GLM53_DSA_INDEX_BLOCK_MB=32`; `setdefault`, as
  the vLLM adapter does);
- no layout plans above its measured maximum (`serving_profiles.max_model_len_for`: replicated 325,632, sharded
  1,048,576 at 1 slot), even with a low `GLM53_DRAM_RESERVE_MIB`;
- a request that fits no layout raises `ValueError` before any weight loads, with the recipe in the message. The
  re-check against the free DRAM measured after the load runs with `GLM53_DEFER_POOL=1` (pool allocated after the
  weights); by default the pool is allocated with the layers.

| Precision | Slots | Default `max_seq_len` | Source |
|---|---|---|---|
| combo_a_kv_dense4_g4 (default since d16e6b9c2f0) | 1 | 294,912 (replicated) | planner on HEAD's `serving_profiles.json` (G9 free-after-load 3,139,928,064 B), 1,200 MiB reserve. Built in `v3_default` (237ec3b8438): teacher forcing 92 / 95 / 92, token-out 55.04 ms, MoE split engaged |
| combo_a_kv_dense4_g4 | 8 | 32,768 (replicated) | planner |
| combo_a_kv_dense4_g4 | 32 | 2,048 (replicated) | planner |
| combo_a_kv_dense4_g4, `max_seq_len=1048576` | 1 | 1,048,576 (block-sharded, selected automatically) | planner, `layouts.sharded.budget`; the serving probe of the same layout is G9 p300x2-1m |
| combo_a_kv_dense4_g4, `GLM53_DRAM_RESERVE_MIB=1024` | 1 / 8 / 32 | 325,632 / 34,816 / 4,096 | planner; `v1_default` built 311,296 on the earlier profile data |
| combo_a_kv (the j2-j4 evidence, before review #2) | 1 / 8 / 32 | 280,576 / 30,720 / 2,048 | j2 / j3 / j4 plans, 1,200 MiB reserve |
| none (bf16 latent) | 1 | 59,392 replicated; 194,560 block-sharded (selected automatically above 59,392) | planner |
| none (bf16 latent) | 8 / 32 | 4,096 replicated, 16,384 sharded / does not fit | planner |

The standalone default trades about 9 % of the serving context for TTFT. On HEAD's profile data the trade is incomplete:
the split still engages (`v3_default`), by about 43 MB of free DRAM. The vLLM serving profiles size with the
measured 1,024 MiB workspace (the stage-8 capacity probe completed with it and failed at 832 MiB) and own the long
contexts. `GLM53_DRAM_RESERVE_MIB=1024` gives the standalone generator the same sizing.

## Warm-up order

`_warm` compiles everything before it captures a trace. After the capture, nothing may allocate in the trace region
(checked with tt-metal's allocation tracker, `TT_METAL_TRACE_ALLOC_TRACKING=1`).
- On the final rework snapshot: 0 live post-capture allocations at 8 slots, for both DSA layouts (`tracker3_rep`,
  `tracker3_shard`) and the stage-9 adapter smoke (`tracker3_adapter`).
- The 1- and 32-slot tracker runs (`tracker4_*`) ran on the morning code (cdab692b0ec), before the final fixes.
- Deep chunks (stage-6 review #2 P2-2, `~/glm53-flash-bringup/logs/s7r2/cachefull.log`, record
  `~/glm53-flash-bringup/artifacts/s7r2/cachefull.json`): full 45-layer model, block-sharded, `GLM53_PRECISION=none`,
  `max_seq_len` 131,072, b3270061cca code. After the capture it prefilled two fresh 10,001-token prompts (chunk starts
  0 .. 8,192) and one 2,600-token prompt, each followed by a traced decode replay.
  - Programs: 890 before and after every prompt.
  - Tracker: 0 live allocations at the replays.
  - The two 10,001-token prefills took 7,826 / 7,821 ms.

  So the deep sparse chunks compile nothing after the capture. j6's 4.1 s first-run gap (11,787 vs 7,697 ms) did not
  reproduce and was not a compile; its cause is not identified. The standalone generator now counts any post-capture
  compile (`Glm53Generator.guard_programs`; `program_cache_growth` in the `s67r_checks` records;
  `GLM53_PROGRAM_GUARD_STANDALONE=raise` raises).

1. One prefill per constructed length. Prefill builds persistent constants lazily. When a prefill first ran after the
   capture, these constants landed in freed trace-intermediate memory. The next decode replay overwrote them, and the
   prefill after the replays hung (2026-10-16, `tools/model_step_probe.py` bisect).
2. `_warm_extra`: the remaining request-dependent programs (KDA commit per slot, last-row tile per offset, the
   all-logits head). The DSA open-pool hand-off uses request-independent programs (one-hot row select and slot mask).
3. One eager decode step with every slot inactive.
4. `_warm_sampler`: every sampler variant (greedy, top-k / top-p / temperature, penalties), including the penalty
   token-count programs.
5. Model trace capture, then one sampler trace per variant on the model trace's logits buffer.

`check_trace_addresses` runs after every prefill and before every reset. It asserts that every buffer a replay reads
is at its capture address and still allocated.

## Decode corruption found in the rework (fixed)

Teacher forcing failed (top-1 / top-5 / top-100 = 14 / 32 / 90 %) under DRAM pressure and after some unrelated
allocation-order changes. The cause was in `Glm53Model.decode_step`: the logits rows were padded with `ttnn.pad`, and
then the code called `ttnn.deallocate(logits)`. At S < 32 the pad returns a view of `logits`. `ttnn.deallocate`
defaults to force=True, so it freed the storage of the trace's persistent logits output. Later tensors were allocated
into that block, and every replay then wrote logits into them. The fix removes the deallocate. `GLM53_DEALLOC_CHECK=1`
(`tests/test_model_probe.py::test_no_shared_buffer_frees`) now fails on any deallocate that leaves a referenced
buffer. The full analysis is in `work_log.md`.

## Readiness (snapshot default, before kda_in; source run `j2_s1_default`)

Reference: `readiness_aime24_chat.refpt`. HF `Glm5NextTextModel` on CPU with the same E192 expert set and lazy FP8
experts (`tools/hf_full_ref.py`). The CPU model matches the per-layer reference bit for bit at all 45 layers. The
prompt is AIME24 #0 with the chat template: 161 prompt tokens, 100 generated.

| Check | top-1 | top-5 | top-100 | Gate | Report |
|---|---|---|---|---|---|
| Teacher forcing | 91 % | **100 %** | **100 %** | top-5 >= 98 %, top-100 = 100 % | `doc/optimized_full_model/runs/teacher_forcing.json` |
| Prefill check | 91 % | **100 %** | **100 %** | same | `doc/optimized_full_model/runs/prefill_check.json` |

HF's own greedy decode agrees with its teacher-forced top-1 at 96 % of the positions, so 91 % at bfp4 projections is
within the expected range. The original stage-6 build (bfp8, 4 slots) gave 93 / 100 / 100 (`runs/`).

Free-running checks:

- Autoregressive (stock prompt, 256 tokens, `doc/optimized_full_model/autoregressive/`):
  - `check_degenerate_output.py` exits 0;
  - two runs give bit-identical tokens;
  - HF scoring of TT's trajectory is in `doc/optimized_full_model/autoregressive/VERDICT.md`.
- Qualitative (6 chat prompts, up to 2,048 tokens): `qualitative/VERDICT.md`. HF scored over TT's q1 and q5 tokens
  covers the spiral window and the "shouldn" token.

## Batch > 1 (full model)

| Run | Slots | What is checked | Result |
|---|---|---|---|
| `j3_s8` | 8 (prompts 161 / 1,153 / 517 in slots 0, 3, 6) | 32 traced greedy steps, token fed back on device. Each slot is compared with its own solo run. The same prompt runs in slots 0 and 6. Slot 6 is prefilled at step 10 while the others decode. | bit-identical in every comparison, all 32 steps |
| `j3b_s8_noskip` | 8, inactive-row MoE skip off | the same logits rows as j3 | all 200 saved logits / token tensors bit-identical to j3 (skip on); same batch checks pass |
| `j4_s32` | 32 (slots 0, 17, 31; prompts 161 / 300 / 517) | the same checks as j3 | bit-identical in every comparison, all 32 steps (late slot 22 steps); default context at 32 slots: 2,048 |
| reduced probe | 1 / 8 | `tests/test_model_probe.py` (`GLM53_PROBE_SLOTS`), 3 layers, all slot paths, final snapshot with `GLM53_DEALLOC_CHECK=1` | 6 passed + 1 skipped / 7 passed (`probe_dchk_s{1,8}.log`); the 32-slot probe (6 passed, `j1_probe_s32.log`) ran on earlier code (d48b7fd06ad) |

## Page table and sampling

- Page-table change (`--do ptchange`): one slot's page table changes between two traced steps, to a new permuted
  block at a 128-token boundary (block 2 to block 2,191). The traced tokens and logits equal the eager decode on the
  same table (min PCC 1.0, max abs diff 0). There is exactly one device copy, and none on the steps where the table
  is unchanged. Pass at 1 slot (slot 0) and 8 slots (slot 1).
- Split sampler with alternating params (`--do sampling`, the serving decode path): greedy, top-k 20 / top-p 0.9 /
  temperature 0.8, and repetition penalty 1.3, alternating over 12 steps:
  - greedy equals the host argmax;
  - sampled tokens are inside the top-k set (the penalty steps are checked on the penalized logits);
  - all 4 dies hold the same token after every step;
  - a fixed seed reproduces.

  Pass at 1 slot and at 8 slots (slots 0 and 7).
- Seed check. In j2 / j3 (top-k 20, top-p 0.9), seeds 1234 and 99 drew the same 6 tokens, which does not discriminate.
  `v1_default` re-ran it on a flat distribution (temperature 1.0, top-k 32, top-p 1.0):
  - seed 1234 twice gives the identical 6 tokens;
  - seed 99 diverges at draw 5 (444 / 10989 vs 26788 / 9529).

  So request seeds take effect.

## Sampler choice

| | `models.common.sampling.SamplingGenerator` (chosen) | `models.common.modules.sampling.sampling_1d.Sampling1D` |
|---|---|---|
| State ownership | owns sampling params, seeds (`SeedManager`) and penalty history (`TTPenalties`) | stateless module: k / p / temperature are per-call arguments; seeds, penalties and state live in separate `*_1d` modules the caller wires up |
| Trace behaviour | one internal trace per variant (penalties x force-argmax), `capture_trace` / `sample(enable_trace=True)`, `precompile(all_configs=True)` | none built in: the caller captures it |
| Mesh support | row- or column-sharded vocab over a 1D or 2D mesh | 1D mesh topologies; needs a `tt_ccl` on more than one device |
| `tt_out_tok` | yes (the decode trace's token buffer) | yes |
| top-k / padding | max top-k 32; pads per-device logits for `ttnn.topk` | the same op path; optional power-of-2 padding |
| Used by | tt_transformers and the vLLM adapters (`SlotSamplingState` reuses its seed and penalty contract) | TTTv2 modules |

SamplingGenerator was chosen because stage 9 serves through the same slot-ordered sampling-params / seed / penalty
contract as the tt_transformers vLLM path, and its per-variant traces are the split-sampler design this model uses.
Sampling1D was rejected because it would need the variant traces, seed management and penalty state rebuilt around
it. The sampler is 0.39 ms of a 56 ms step, so neither has a performance case.

## Runtime fallback audit

| Path | Device or host | Notes |
|---|---|---|
| Prefill, all layers, LM head | device | chunked, cost plan |
| Prefill token in `generate()` | **host** argmax over the last logits row (one vocab row read, about 20 ms of TTFT) | the serving path samples the prefill token on the device (`_prefill_sampled`) |
| `prefill_forward(return_all_logits=True)` / `prefill_logits` | **host**: every row is read back | readiness prefill check only. The all-logits head is warmed only for chunks of <= 512 rows (a 2048-row head holds 158 MB of logits per die, kept out of the warm-up peak). A prompt whose plan has a 2048-row chunk compiles that head after the capture; the post-capture program guard (`Glm53Generator.guard_programs`) reports it. Not a serving path |
| Decode, model step | device, traced | |
| Decode sampling | device, traced, every variant | `on_device_sampling=False` is a host-sampling compatibility mode (host argmax of the logits row); readiness tools only |
| Token feedback | device (`tt_out_tok`) | `generate()` reads one token per step back to the host for its return value |
| Page table | host compares the row; device copy only when it changed | |
| Positions / active flags | device (`decode_inputs.refresh`) | the host writes them only on `reload_inputs` |
| Reset | device: KDA state and DSA open-pool rows zeroed per slot | |
| Weight cache | host disk, out of process (watchdog + memfd staging) | key covers the precision policy, per-layer MoE exceptions and the checkpoint shard size and mtime |

## Watcher

`TT_METAL_WATCHER` on the rework snapshot: every run rc 0 with 0 suspicious lines (`~/glm53-flash-bringup/logs/s67r/status`).
The Ethernet cores were not watched (`TT_METAL_WATCHER_DISABLE_ETH=1`). Earlier runs with them watched aborted at
teardown with "Timed out while waiting for active ethernet core 29-25"; that is a scoped limitation, not a pass on the
ETH cores.

| Run | Path |
|---|---|
| `w_full_s8` | `tools/s67r_checks.py --do batch,chunkplan` on the reduced model (layers 0 / 3 / 4), 8 slots, replicated: multi-slot traced decode, mid-decode prefill, cost-plan chunks (600: 512 dense + 128; 2,600: sparse 512 + 128). The page-table change, the sampler variants and the overrun guard did not run under the watcher |
| `w_full_s8_shard` | the same, block-sharded DSA |
| `shard_b1` | `test_dsa_shard[b1]`, `GLM53_DSA_SHARD=1` |
| `shard_batch32_dsa_moe` | `test_batch32_all_active[dsa_moe]`, `GLM53_DSA_SHARD=1` |
| `shard_lengths_dsa_moe_2051` | `test_lengths[dsa_moe-2051]`, `GLM53_DSA_SHARD=1` |

All ran with `GLM53_WORKER_L1_SIZE=1444864`.

## Memory and context

DRAM is the limit on this model. Measured per die (`memory_budget.json`, `memory/`, `ttnn.get_memory_view`):

| Item | Bytes |
|---|---|
| DRAM before the trace region | 34,178,731,008 |
| Free after the default build (combo_a_kv, 1 slot, 280,576-token pool, trace region) | 1,068,291,584 |
| Decode trace (45 layers + sampler variants) | default region 64 MiB |
| KDA state per slot (34 layers) | 35.65 MB |
| DSA cache per token, replicated, fp8 latent (default) | 6,336 (11 x (512 + 64)) |
| DSA cache per token, replicated, bf16 latent (`GLM53_PRECISION=none`) | 11,968 (11 x (1,024 + 64)) |
| DSA cache per token, block-sharded, bf16 latent | 3,696 (11 x (544 x 2 / 4 + 64)) |
| DSA cache per token, block-sharded, fp8 latent (default precision) | 2,288 (11 x (576 / 4 + 64)); 292,864 B per 128-token block measured |
| 2048-row prefill peak above layer entry (MoE, after the row split) | 297 MiB (was 417 MiB) |

The MoE prefill runs its routed experts in row blocks of `GLM53_MOE_SORTED_ROWS` when the largest free block is under
1 GiB (`choose_moe_split`). The DSA indexer uses blocks of `GLM53_DSA_INDEX_BLOCK_MB` (default 64). Both are pure
memory changes: last-row logits are bit-identical over the settings.

Context per layout (1 slot):

| Layout | Precision | Context | Evidence |
|---|---|---|---|
| replicated | combo_a_kv_dense4_g4 (default) | 325,632 (serving profile `p300x2-long`, 1,024 MiB workspace, G9); standalone default 294,912 (1,200 MiB reserve); j2 ran 280,576 (combo_a_kv) | G9 `doc/vllm/capacity/s1`; stage-8 capacity probe; j2 deep runs at 32,768 and 100,000 (copy 1.0) |
| block-sharded | none (bf16) | 281,344 measured in stage 4 (stage-7 policy, bf16); 194,560 planned with the 1,200 MiB reserve | `doc/multichip_decoder/dsa_shard/capacity/`; `j6_s1_shard` at 131,072: TF 92 / 100 / 100, copy 1.0 at 100,000 tokens |
| replicated | none (bf16) | 104,448 (stage-6 probe) | `memory/capacity_probe_1slot.json` |
| block-sharded | combo_a_kv_dense4_g4 (fp8 latent) | **1,048,576** (advertised; serving profile `p300x2-1m`; standalone `Glm53Generator(mesh, max_seq_len=1048576)`, which selects this layout and its settings; 600 MiB workspace) | `doc/multichip_decoder/dsa_shard/capacity/fp8_s1_1048576/`: copy 1.0 at 1,048,515 tokens; TF 91 / 95 / 91 at 16,384; +2.9 ms per token |

The block-sharded fp8 layout reaches the advertised 1,048,576 tokens at 1 slot. The default stays replicated because
it is faster. The byte math and the evidence are in `doc/context_contract.json` (`dsa_block_sharded_fp8`) and
`doc/multichip_decoder/README.md` ("fp8 rows").

## Limitations

- E192 expert pruning: the model cannot finish some French translations (q5). HF with the same E192 set does the same
  ("shouldn" is HF's top-1), so it is a pruning property, not a TT defect.
- Context: the advertised 1,048,576 tokens are supported at 1 slot with the block-sharded fp8 layout (serving profile
  `p300x2-1m`, G9 measured; standalone `max_seq_len=1048576`), at about +2.9 ms per decode token. The faster replicated
  layout holds 325,632 (`p300x2-long`); the standalone default is 294,912 (TTFT-favouring reserve).
- Largest batch tested on the full model: 32 slots (at 32 slots the default context is short; see
  `serving_profiles.json`).
- Sampler variant capture: the "Cannot load new binaries during trace capture" defect is fixed (`_warm_sampler`).
  Open: one `models/common/sampling/tt_sampling.py` site frees a tensor that may still be referenced
  (`sampling_values`, line 1170). It is noted for upstream, and is harmless at its current last-use position.
- `generate()` takes the prefill token by host argmax (see the fallback audit).

## Commands

```
export GLM53_TP=4      # defaults: 1 slot, 294,912 tokens (replicated), precision combo_a_kv_dense4_g4
python -m models.autoports.zai_org_glm_5_3_flash.tools.readiness teacher-forcing \
  --reference models/autoports/zai_org_glm_5_3_flash/readiness_aime24_chat.refpt
python -m models.autoports.zai_org_glm_5_3_flash.tools.readiness prefill-check \
  --reference models/autoports/zai_org_glm_5_3_flash/readiness_aime24_chat.refpt
python -m models.autoports.zai_org_glm_5_3_flash.tools.s67r_checks --slots 1 \
  --do ready,ar,ttft,ptchange,sampling,chunkplan,decode,deep --out j2.json        # the headline run
python -m models.autoports.zai_org_glm_5_3_flash.tools.s67r_checks --slots 8 --seq-len 4096 \
  --do batch,ptchange,sampling,decode --batch-slots 0,3,6 --batch-lengths 161,1153,517 --out j3.json
GLM53_DEALLOC_CHECK=1 pytest models/autoports/zai_org_glm_5_3_flash/tests/test_model_probe.py   # reduced, 3 layers
```

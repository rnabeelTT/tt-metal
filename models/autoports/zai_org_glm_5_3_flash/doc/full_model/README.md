# GLM-5.3-Flash (E192) full model

Stage 6 of the tt-model-bringup flow. `tt/model.py` (`Glm53Model`) runs the full text model on the (1, 4) mesh of a
p300x2. `tt/generator.py` (`Glm53Generator`, `build_generator`) is the readiness and serving contract.

## Headline (stage-6/7 rework, 2026-10-18)

Source run: `j2_s1_default` (`tools/s67r_checks.py --slots 1`, log `~/glm53-flash-bringup/logs/s67r/j2_s1_default.log`,
record `doc/optimized_full_model/runs/s67r/j2_s1_default.json`). Code: HEAD 656c998e3df plus this rework's files,
run from a pinned snapshot; git HEAD and the md5 of every `tt/*.py` were logged at the start and end and did not
change. Build: `Glm53Generator(mesh)` with no arguments and no `GLM53_*` setting, i.e. the defaults: 1 slot, stage-8
precision `combo_a_kv` (fp8 DSA latent), replicated DSA cache, `max_seq_len` 280,576, trace region 64 MiB.

| Metric | Value | Workload |
|---|---|---|
| TTFT, 161-token prompt | **377 ms** (397 ms with the host logits row `generate()` reads) | warm, one 512-row chunk |
| TTFT, 128 / 512 / 2,048 / 8,192 tokens | 228 / 376 / 1,026 / 6,444 ms | warm, cost chunk plan |
| Batch-1 decode, model trace only | 55.85 ms per step (17.9 t/s/u) | position 300, 32 replays, no host sync between |
| Batch-1 decode, token-out (model + sampler trace, token fed back on device) | **56.24 ms per step, 17.8 t/s/u** | same; sampler 0.39 ms |
| Batch-1 decode, `generate()` loop (host reads each token) | 56.51 ms per step, 17.7 t/s/u | same |
| Teacher forcing (host feeds the reference token every step) | 17.8 t/s/u decode, top-1 91 %, top-5 100 %, top-100 100 % | AIME24 #0 chat prompt, 161 + 100 tokens |
| Prefill check | top-1 91 %, top-5 100 %, top-100 100 % | same |
| Long prompt, 32,768 / 100,000 tokens | TTFT 28.0 / 87.3 s; decode 59.8 / 59.9 ms per token; copy accuracy 32 / 32 both | copy task, traced token-out decode at the deep position |
| Model build with the weight cache | 132 s | |

Largest full-model batch tested: 32 slots (`j4_s32`, see "Batch > 1"). Limitations: see the last section.

The default context has a cost. The sparse DSA indexer scores every pool slot allocated to the user (`[rows,
P_alloc]`, sized by `max_seq_len`), not only the filled prefix. The same code at `max_seq_len` 16,384 (`j2b_s1_16k`)
gives 51.10 ms token-out decode (19.6 t/s/u) and 4,408 ms TTFT at 8,192 tokens. At the 280,576 default these are
56.24 ms and 6,444 ms. Short-context deployments should pass `max_seq_len`. The TTFT up to 2,048 tokens does not
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
  built-in stage-7 numbers; DSA bytes per 128-token block follow the latent dtype; the reserve is
  max(data reserve, `GLM53_DRAM_RESERVE_MIB` = 1,200 MiB);
- with a bf16 latent (`GLM53_PRECISION=none`) a request above the replicated capacity selects the block-sharded DSA
  layout automatically; the fp8 latent has no sharded layout;
- a request that fits no layout raises `ValueError` before any weight loads, and the generator re-checks the pool
  against the free DRAM it measures after the load.

| Precision | Slots | Default `max_seq_len` | Source |
|---|---|---|---|
| combo_a_kv (default) | 1 | 280,576 (replicated) | j2 plan: 2,193 blocks of a 2,206-block budget |
| combo_a_kv | 8 | 30,720 (replicated) | j3 plan |
| combo_a_kv | 32 | 2,048 (replicated) | j4 plan |
| none (bf16 latent) | 1 | 59,392 replicated; 194,560 block-sharded (selected automatically above 59,392) | planner |
| none (bf16 latent) | 8 / 32 | 4,096 replicated, 16,384 sharded / does not fit | planner |

The planner keeps a 1,200 MiB reserve, which is more conservative than the measured prefill workspace: stage 6's
probe ran 104,448 bf16 tokens with a 700 MiB workspace, and stage 8's probe ran 311,296 tokens with 1,024 MiB.
`GLM53_DRAM_RESERVE_MIB` lowers the reserve.

`doc/context_contract.json` records 311,296 tokens as the supported context. That is the stage-8 vLLM capacity probe
(1,024 MiB workspace, 311,235-token prompt, copy accuracy 1.0). The standalone generator keeps a larger reserve
(1,200 MiB), so its default is 280,576.

## Warm-up order

`_warm` compiles everything before it captures a trace. After the capture, nothing may allocate in the trace region
(checked with tt-metal's allocation tracker, `TT_METAL_TRACE_ALLOC_TRACKING=1`: 0 live post-capture allocations at 1,
8 and 32 slots, both DSA layouts, and the stage-9 adapter smoke).

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

## Readiness (current defaults; source run `j2_s1_default`)

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
| reduced probe | 1 / 8 / 32 | `tests/test_model_probe.py` (`GLM53_PROBE_SLOTS`), 3 layers, all slot paths | 6 / 7 / 6 passed |

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

  Pass at 1 slot and at 8 slots (slots 0 and 7). Seeds 1234 and 99 drew the same 6 tokens. The check is informational:
  with top-p 0.9 on this continuation, the top token usually holds more than 0.9 of the mass.

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
| `prefill_forward(return_all_logits=True)` | **host**: every row is read back | readiness prefill check only |
| Decode, model step | device, traced | |
| Decode sampling | device, traced, every variant | `on_device_sampling=False` is a host-sampling compatibility mode (host argmax of the logits row); readiness tools only |
| Token feedback | device (`tt_out_tok`) | `generate()` reads one token per step back to the host for its return value |
| Page table | host compares the row; device copy only when it changed | |
| Positions / active flags | device (`decode_inputs.refresh`) | the host writes them only on `reload_inputs` |
| Reset | device: KDA state and DSA open-pool rows zeroed per slot | |
| Weight cache | host disk, out of process (watchdog + memfd staging) | key covers the precision policy, per-layer MoE exceptions and the checkpoint shard size and mtime |

## Watcher

`TT_METAL_WATCHER` (Ethernet cores excluded with `TT_METAL_WATCHER_DISABLE_ETH=1`) on the rework snapshot, every run
rc 0 with 0 suspicious lines (`~/glm53-flash-bringup/logs/s67r/status`):

| Run | Path |
|---|---|
| `w_full_s8` | reduced probe, 8 slots, replicated (batch, page-table change, sampling, overrun) |
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
| DSA cache per token, block-sharded, bf16 latent only | 3,696 (11 x (544 x 2 / 4 + 64)) |
| 2048-row prefill peak above layer entry (MoE, after the row split) | 297 MiB (was 417 MiB) |

The MoE prefill runs its routed experts in row blocks of `GLM53_MOE_SORTED_ROWS` when the largest free block is under
1 GiB (`choose_moe_split`). The DSA indexer uses blocks of `GLM53_DSA_INDEX_BLOCK_MB` (default 64). Both are pure
memory changes: last-row logits are bit-identical over the settings.

Context per layout (1 slot):

| Layout | Precision | Context | Evidence |
|---|---|---|---|
| replicated | combo_a_kv (default) | 311,296 (vLLM, 1,024 MiB workspace); 280,576 standalone default | stage-8 capacity probe; j2 deep runs at 32,768 and 100,000 (copy 1.0) |
| block-sharded | none (bf16) | 281,344 measured in stage 4 (stage-7 policy, bf16); 194,560 planned with the 1,200 MiB reserve | `doc/multichip_decoder/dsa_shard/capacity/`; `j6_s1_shard` at 131,072: TF 92 / 100 / 100, copy 1.0 at 100,000 tokens |
| replicated | none (bf16) | 104,448 (stage-6 probe) | `memory/capacity_probe_1slot.json` |

The path from these values to the advertised 1,048,576 tokens, with the byte math for each option, is in
`doc/context_contract.json` (`path_to_advertised`).

## Limitations

- E192 expert pruning: the model cannot finish some French translations (q5). HF with the same E192 set does the same
  ("shouldn" is HF's top-1), so it is a pruning property, not a TT defect.
- Context is 280,576 (standalone default) / 311,296 (vLLM) tokens per slot under the default precision, against the
  advertised 1,048,576. The block-sharded layout exists only for the bf16 latent.
- Largest batch tested on the full model: 32 slots (at 32 slots the default context is short; see
  `serving_profiles.json`).
- Sampler variant capture: the "Cannot load new binaries during trace capture" defect is fixed (`_warm_sampler`).
  Open: one `models/common/sampling/tt_sampling.py` site frees a tensor that may still be referenced
  (`sampling_values`, line 1170). It is noted for upstream, and is harmless at its current last-use position.
- `generate()` takes the prefill token by host argmax (see the fallback audit).

## Commands

```
export GLM53_TP=4      # defaults: 1 slot, the largest context that fits, combo_a_kv
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

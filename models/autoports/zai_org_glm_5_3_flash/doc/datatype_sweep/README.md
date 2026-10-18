# GLM-5.3-Flash (E192) datatype sweep

Stage 8 of the tt-model-bringup flow. Full 45-layer model (`tt/model.py`, `tt/generator.py`) on a p300x2 (4 Blackhole
dies, mesh (1, 4), FABRIC_1D, TP4). The sweep selects the weight, fidelity, CCL and KV-cache datatypes of the optimized
full model (stage 7).

## Result

**Selected: `combo_a_kv`** (`selected_precision_config.json`). It is the stage-7 policy plus:

| Group | Stage 7 (shipped) | Selected |
|---|---|---|
| KDA input / output projections (layers 1-43) | bfp8, HiFi4 | **bfp4, LoFi** (decode rows) |
| DSA q_a \| kv_a pack, q_b, o_proj (all 11 DSA layers) | bfp8, HiFi4 | **bfp4, LoFi** (decode rows) |
| Shared expert (layers 3-43) | bfp8, HiFi4 | **bfp4, LoFi** |
| LM head | bfp8, HiFi2 | **bfp4, LoFi** |
| KDA projections, shared expert of layers 0 / 44; dense MLP (0-2) | bfp8, HiFi4 | bfp8, **LoFi** (decode rows) |
| DSA latent pool (KV cache) | bf16 rows | **fp8 e4m3 rows** (6,336 instead of 11,968 B per token per die) |
| Routed experts, router, DSA W_UK / W_UV, indexer, residual, CCL, norms, mHC | unchanged | unchanged |

Acceptance gate (skill default): teacher-forcing top-1 >= 90 %, top-5 >= 98 %, top-100 = 100 %, prefill check the same.
Selection rule (coordinator, on top of the gate): the minimum top-1 over teacher forcing and prefill check on three
AIME24 chat references (p0 = the readiness reference, p1, p2) must be >= 91 %, top-5 / top-100 must hold on all three;
among the configs that pass, prefer context, then decode speed.

| Metric (1 slot, max_seq_len 16,384, TP4) | Shipped (stage 7) | Selected `combo_a_kv` |
|---|---|---|
| Teacher forcing top-1 / top-5 / top-100, p0 (readiness reference) | 92 / 100 / 100 % | 91 / 100 / 100 % |
| Prefill check top-1 / top-5 / top-100, p0 | 91 / 100 / 100 % | 91 / 100 / 100 % |
| Teacher forcing top-1, p1 / p2 | 98 / 91 % | 96 / 91 % |
| Minimum top-1 over p0 / p1 / p2 (TF and prefill check) | 91 % | 91 % |
| Trace-verified teacher-forcing decode (readiness runner, host wall, 1 readback per step) | 18.49 t/s/u | **18.79 t/s/u** |
| Token-out decode (model + sampler trace replays, device feedback, no readback) | 54.07 ms = 18.49 t/s/u | **53.22 ms = 18.79 t/s/u** (-1.6 %) |
| TTFT, warm, device logits: 128 / 161 / 2048 tokens | 230 / 376 / 1035 ms | 228 / 373 / 1026 ms |
| 1-slot context (max_model_len) | 104,448 | **311,296** (measured, fp8 latent; ~340 MB per die still free at the peak) |

Post-selection (normal construction path: `GLM53_PRECISION` unset loads `selected_precision_config.json`; flip
snapshot `~/glm53-flash-bringup/s8_snapshot_flip`, runs `runs/selected_default*`):

| Check | Result |
|---|---|
| Readiness p0 / p1 / p2, TF top-1 (top-5, top-100 = 100 %) | 91 / 96 / 91 % |
| Readiness p0 / p1 / p2, prefill check top-1 | 91 / 95 / 91 % |
| Token-out, warm, no readback, 1 slot (30 replays) | **53.23 ms = 18.79 t/s/u** (model trace 52.85 ms, sampler 0.38 ms) |
| Generate loop with one token readback per step | 53.36 ms = 18.74 t/s/u |
| TTFT 128 / 161 / 2048 tokens (warm, device logits) | 227 / 373 / 1026 ms |
| Non-aligned prompts (100 / 161 / 300 tokens, traced vs eager, repeated) | equal |
| Qualitative suite (6 chat prompts, `runs/selected_default/checks/qualitative/`) | coherent; 5 end on EOS, q3 reaches the 2048-token budget (as in stage 7) |
| Autoregressive degenerate check (`checks/autoregressive/degenerate_check.txt`) | exit 0, no degenerate output |
| Warm model load (all weights from the cache, re-quantized tensors included) | 108 s (stage-7 policy: 89 - 91 s) |
| `GLM53_DSA_SHARD=1` | fp8 ignored with a warning (bf16 sharded pool); readiness 92 / 100 / 100 % |

All performance numbers: p300x2, TP4, 1 slot, max_seq_len 16,384, workload = synthetic prompts for TTFT / token-out,
AIME24 chat references for teacher forcing.

![top-1 vs speed](top1_perf_pareto.png)
![top-5 vs speed](top5_perf_pareto.png)

Pareto charts: x = teacher-forcing accuracy on the readiness reference p0, y = trace-verified teacher-forcing decode
t/s/u (readiness runner). Red = selected. The dotted line is the gate. Every evaluated full-model run is a point;
`_r3` points are the repeat builds of the finalists (three references). top-5 is 100 % for every point except
`combo_a_all` (32 %, listed off-chart). On the top-5 chart every other point ties at 100 %, so the frontier is
the single fastest point (combo_a, rejected by the three-reference top-1 rule).

## Method

- **Code under test.** The sweep ran on a pinned copy of the code, because other stages edited the live tree during
  the sweep: `~/glm53-flash-bringup/s8_snapshot` = `git archive ac03689d2bb models/` + the stage-8 files. Every run
  record has the md5 of each `tt/*.py` file (`code`). The shipped baseline on this code reproduces stage 7 exactly
  (92 / 100 / 100 %, 54.08 ms token-out, 230 ms TTFT at 128 tokens).
- **Candidates.** `tt/precision.py` (`PrecisionOverrides`) applies a candidate JSON (`configs/<id>.json`) at
  construction, through hooks in `MultichipDecoder.from_state_dict` (per layer) and `Glm53Model.__init__` (LM head):
  - `requant`: host round-to-nearest re-quantization of a group's loaded bfp8 tensors to bfp4, per die, same memory
    config. The bfp8 -> bfp4 rounding equals bf16 -> bfp4 except at rounding ties of the bfp8 grid. The device
    typecast is not used (it is not round-to-nearest: stage 3).
  - `fidelity`: compute fidelity of a group's decode-row matmuls (KDA, DSA, dense MLP, LM head) or of all rows (the
    MoE `shared_fidelity` / `routed_fidelity` fields).
  - `moe_layer_exceptions`: per-layer MoE fields; a layer with a dtype exception is built from the checkpoint.
  - `ccl_h_dtype`: the decode all-gathers of the residual in bfp8 (cast before, cast back after).
  - `kv_latent_dtype`: the DSA latent pool in fp8 e4m3 rows.
  No candidate needs its own 116 GB weight cache: every build loads the shipped-policy cache and changes only the
  overridden groups.
- **Propagation check.** Every record has `runtime_policy` (dtypes and compute fidelities read back from the live model
  objects, one entry per layer kind) and `precision.applied` (each change per layer). Example: `runs/combo_a_kv_r3/
  record.json` shows `kda_in` / `kda_out` BFLOAT4_B on layer 4 and BFLOAT8_B on layers 0 and 44, the LM head
  BFLOAT4_B at LoFi, the latent pool `FP8_E4M3 ROW_MAJOR`.
- **Accuracy.** `tools/dtype_sweep.py` builds the generator once per candidate and runs the readiness runners
  (`run_teacher_forcing`, `run_prefill_check`) on the same build. Reference p0: `readiness_aime24_chat.refpt` (AIME24
  prompt 0, chat template, 100 generated tokens, HF CPU reference `tools/hf_full_ref.py`). The finalists also ran on p1
  and p2 (AIME24 prompts 1 and 2, same tool, `refs/`).
- **Trace verification.** The number of `ttnn.execute_trace` calls during teacher forcing is counted: 198 for 99 decode
  steps (model trace + sampler trace per step) in every run (`readiness_trace_replays`).
- **Non-aligned prompts.** Every run prefilled 100-, 161- and 300-token prompts and compared the traced and eager
  greedy continuations (8 tokens, twice each): equal and repeatable in every run (`ptcheck`).
- **Performance.** Token-out = 30 back-to-back replays of the model trace + sampler trace with device token feedback,
  no host readback (the stage-7 benchmark). TTFT = warm prefill to device logits plus a device sync.
- **Context.** The 1-slot capacity of the shipped policy is 818 blocks of 128 tokens at 11,968 B per token per die
  (stage-6 capacity probe). A candidate that frees or adds D bytes per die, or changes the bytes per token, gets
  `floor((818 x 128 x 11,968 - D) / (128 x bytes per token))` blocks, floored to a multiple of 2,048 tokens
  (`tools/dtype_sweep_report.py`). These `~` values are upper bounds: they leave out the DSA constants that grow
  with max_context and the larger workspace a large pool needs (combo_a_kv: estimate 364,544, measured 311,296). The
  selected config was measured directly (below).

## Candidates and results

Single changes on the shipped policy (one build each, reference p0; `runs/<id>/`, logs `~/glm53-flash-bringup/logs/s8/`).
"inner" = layers 1-43 (the skill excludes the first and the last layer from bfp4 by default).

| id | Change | TF top-1 / 5 / 100 | PC top-1 | Token-out ms | vs shipped | Context | Decision |
|---|---|---|---|---|---|---|---|
| shipped | stage-7 policy | 92 / 100 / 100 | 91 | 54.08 | - | 104,448 | baseline |
| kv_fp8 | DSA latent fp8 e4m3 | 92 / 100 / 100 | 91 | 54.09 | +0.01 | ~196,608 | kept (context x 1.9) |
| kda_in4_lofi | KDA in-proj bfp4, LoFi | 91 / 100 / 100 | 90 | 53.71 | -0.37 | ~149,504 | kept in combo_a |
| kda_in4_hifi2 | KDA in-proj bfp4, HiFi2 | 91 / 100 / 100 | 90 | 53.73 | -0.35 | ~149,504 | rejected: same accuracy as LoFi, slower |
| dsa4_lofi | DSA q_a\|kv_a, q_b, o_proj bfp4, LoFi | 94 / 100 / 100 | 95 | 53.79 | -0.29 | ~118,784 | kept |
| kda_out4_lofi | KDA out-proj bfp4, LoFi | 94 / 100 / 100 | 94 | 53.88 | -0.20 | ~114,688 | kept |
| proj8_lofi | bfp8 projections at LoFi (KDA, DSA, dense MLP decode; shared expert) | 93 / 100 / 100 | 95 | 53.93 | -0.15 | 104,448 | kept (dense MLP / layer 0, 44) |
| lm4_lofi | LM head bfp4, LoFi | 93 / 100 / 100 | 91 | 53.94 | -0.14 | ~110,592 | kept |
| shared4_lofi | shared expert bfp4, LoFi | 94 / 100 / 100 | 94 | 54.00 | -0.08 | ~114,688 | kept |
| proj8_hifi2 | same groups as proj8_lofi at HiFi2 | 93 / 100 / 100 | 94 | 54.04 | -0.04 | 104,448 | rejected: LoFi is faster at equal accuracy |
| lm8_lofi | LM head bfp8 at LoFi | 92 / 100 / 100 | 91 | 54.09 | +0.01 | 104,448 | rejected: no gain |
| exp8dn_edge | routed down bfp8 (HiFi2) in MoE layers 3 and 44 | 94 / 100 / 100 | 94 | 54.09 | +0.01 | ~69,632 | rejected: -33 % context |
| routed_hifi3 | routed experts bfp4 at HiFi3 | 93 / 100 / 100 | 94 | 54.43 | +0.35 | 104,448 | rejected: slower, no gate need |
| ccl_h_bfp8 | residual all-gathers in bfp8 | 92 / 100 / 100 | 91 | 54.55 | +0.47 | 104,448 | rejected: slower (2 casts per gather, latency-bound gathers) |

Combinations (p0; the finalists then ran on p0, p1, p2 in repeat builds, `runs/<id>_r3/`):

| id | Content | TF top-1 p0 / p1 / p2 | PC top-1 p0 / p1 / p2 | Min top-1 | Token-out ms | Context | Decision |
|---|---|---|---|---|---|---|---|
| combo_a | kda_in4 + kda_out4 + dsa4 + shared4 + lm4 (inner, LoFi) + bfp8 groups at LoFi | 90 / 94 / 91 | 91 / 95 / 91 | 90 | 53.04 | ~192,512 | rejected: min top-1 90 < 91 |
| **combo_a_kv** | combo_a + fp8 latent | **91 / 96 / 91** | **91 / 95 / 91** | **91** | **53.22** | **311,296 (measured)** | **selected** |
| combo_b | combo_a without kda_in4 | 94 / 93 / 92 | 94 / 95 / 91 | 91 | 53.41 | ~147,456 | passes; less context, slower |
| combo_b_kv | combo_b + fp8 latent | 95 / 94 / 90 | 94 / 95 / 91 | 90 | 53.58 | ~278,528 | rejected: min top-1 90 |
| kv_fp8 | (above) | 92 / 97 / 93 | 91 / 97 / 91 | 91 | 54.09 | ~196,608 | passes; slower, less context |
| shipped | (above) | 92 / 98 / 91 | 91 / 97 / 91 | 91 | 54.07 | 104,448 | passes; slowest of the passing set but one |
| combo_a_all | combo_a on layers 0 and 44 too | **14** / - / - | 90 / - / - | 14 (top-5 32 %) | 53.02 | - | rejected: decode fails |

Top-5 and top-100 are 100 % in every finalist cell. Repeat builds of the same policy measure the same token-out time to
0.02 ms (shipped 54.08 / 54.07, combo_a 53.03 / 53.04, combo_a_kv 53.22 / 53.22), so the speed ranking is real.

**combo_a_all.** Extending the bfp4 groups to the first layer (0: KDA + dense MLP) and the last layer (44: KDA + MoE)
breaks decode: teacher forcing 14 / 32 / 95 %, greedy continuations collapse to a repeated token (154842), while the
prefill check (one prefill per position) stays at 90 / 100 / 100 %. The coordinator suspected the stage-6/7 open-pool hand-off bug (decode corruption
under DRAM pressure). The classification rerun on the fixed rework code crashed in the warm-up prefill inside the
then-current rework copy of `tt/dsa.py` (`logs/s8/combo_a_all_s2trk.log`) and is pending a known-good rework snapshot.
The 1-slot / 16K builds of this sweep keep 1.5+ GB per die free (not the pressure regime), and the rejection stands
on the measured result: combo_a_all is not selected either way.

## Compute-fidelity table (material decode matmul groups)

| Group | Weight | Fidelity | Token-out ms | Accuracy (TF top-1, p0) | Kept | Reason |
|---|---|---|---|---|---|---|
| Routed experts | bfp4 | LoFi | 54.08 (shipped) | 92 | yes | HiFi2 is bit-identical on bfp4 weights (stage 3) |
| Routed experts | bfp4 | HiFi3 | 54.43 | 93 | no | +0.35 ms, the gate does not need it |
| KDA in-proj | bfp4 | LoFi | 53.71 | 91 | yes | |
| KDA in-proj | bfp4 | HiFi2 | 53.73 | 91 | no | same accuracy, slower |
| KDA in / out, DSA, dense MLP, shared expert | bfp8 | HiFi4 (shipped) | 54.08 | 92 | no | |
| same | bfp8 | HiFi2 | 54.04 | 93 | no | |
| same | bfp8 | LoFi | 53.93 | 93 | yes | fastest at equal accuracy |
| KDA out, DSA, shared expert, LM head | bfp4 | LoFi | see table above | 93-94 | yes | |
| LM head | bfp8 | LoFi vs HiFi2 | 54.09 vs 54.08 | 92 | no | no difference |

## Groups not reduced

- **Router** (bf16, HiFi4): routing numerics; not in the bfp4 sweep (skill MoE policy).
- **Dense MLP** (layers 0-2, bfp8): the first layers; bfp4 rejected at layer level in stage 3 (update PCC); its weight
  read is 0.12 GB per die per step (0.3 % of the step). Kept bfp8 at LoFi.
- **DSA W_UK / W_UV** (bfp8) and the indexer projections (bf16, fp32 output): attention-core numerics and index
  scores, 15 + 35 µs per DSA layer.
- **Routed experts above bfp4**: bfp8 in all layers does not fit in DRAM; `exp8dn_edge` measured the accuracy value of
  a partial increase (top-1 +2 points, -33 % context).
- **Residual / activations**: bf16 (the mHC streams); no lower format exists for the residual adds.
- **KV-cache bfp8**: bfp8 is a TILE-only format; the DSA latent pool is a ROW_MAJOR paged row store (`scatter_rows`,
  `sparse_sdpa`). fp8 e4m3 rows are the supported 1-byte format (`tools/kv_fp8_probe.py`: device bf16 -> fp8 cast,
  `scatter_rows` and `sparse_sdpa` FP8_E4M3 all work; attention output PCC 0.99958 vs the bf16 pool). The indexer pool
  keys (64 B per token per layer) stay bf16.

## Long-context check (fp8 latent and bfp4 projections at long range)

`tools/long_ctx_check.py`: one long natural document (the repository's English documentation, 31 files, frozen token
ids `refs/long_doc_ids.json.gz`), the same positions for both builds. For L = 2K, 32K and 100K: prefill L - 200 tokens,
then 200 traced decode steps teacher-forced with the document's tokens; host logits per step. `tools/long_ctx_compare.py`
(`long_context/compare_shipped_vs_combo_a_kv.json`); build max_seq_len 102,400:

| L | Shipped true-token top-1 / top-5 | combo_a_kv true-token top-1 / top-5 | Delta top-1 / top-5 | TT-vs-TT top-1 agreement / top-5 overlap | Prefill s (shipped / selected) |
|---|---|---|---|---|---|
| 2,048 | 0.840 / 0.935 | 0.830 / 0.940 | -1.0 / +0.5 pt | 0.930 / 0.843 | 1.2 / 1.0 |
| 32,768 | 0.555 / 0.825 | 0.540 / 0.815 | -1.5 / -1.0 pt | 0.855 / 0.854 | 23.8 / 21.1 |
| 100,000 | 0.700 / 0.895 | 0.695 / 0.875 | -0.5 / -2.0 pt | 0.875 / 0.825 | 73.7 / 65.4 |

Gate (coordinator): no degradation of more than ~3 points against shipped at 32K / 100K. Result: pass. The deltas stay
within 2 points at every length and the TT-vs-TT top-5 overlap is the same at 100K as at 2K (0.83 - 0.85): the fp8
latent adds no long-range error that the short references would hide. The capacity probe adds a 311K-token check
(copy accuracy 1.0, below).

The shipped run needed `GLM53_DSA_INDEX_BLOCK_MB=32` (indexer score blocks; bit-identical, stage 6): with the default
64 MB, the shipped policy at max_seq_len 102,400 ran out of DRAM in the DSA indexer at the 32K prefill
(`logs/s8/longctx_shipped_try1_oom.log`). The selected policy ran with the default (1 GB more free DRAM).

## Capacity and context contract

`tools/memory_probe.py capacity` (serving path: weights, free DRAM measured, DSA pool sized from it, warm-up + traced
decode capture, one non-aligned copy prompt that fills the pool, 24 traced decode steps at the top) on the selected
policy, 1 slot, 64 MiB trace region (`capacity/combo_a_kv_s1/capacity.json`):

| Item | Stage 7 (shipped) | Selected |
|---|---|---|
| Free DRAM per die after load | 1,988,502,016 B | 3,047,399,936 B |
| DSA cache per token per die | 11,968 B | 6,336 B (fp8 latent 512 B + bf16 pool keys 64 B, x 11 layers) |
| Pool (1 slot) | 818 blocks (104,576 tokens) | 2,433 blocks (311,424 tokens) |
| Copy prompt / prefill time / copy accuracy | 104,515 tokens / 170 s / 1.0 | 311,235 tokens / 408.5 s / 1.0 |
| Free per die after the top-of-pool prefill | 183 MB | 648 MB |
| **max_model_len, 1 slot** | 104,448 | **311,296** |

The first attempt at 364,544 tokens with the 700 MiB workspace of stage 6 ran out of DRAM in the warm-up prefill (the
per-max_context DSA constants grow with T; `logs/s8/cap_combo_a_kv_s1_try1_oom.log`), so the workspace rose to
1,024 MiB. 311,296 is the measured value; ~340 MB per die remain at the peak, so the true limit is somewhat higher.

Other slot counts, derived as in stage 6 (pool minus 43.96 blocks = 35.65 MB of KDA state per extra slot and one block
per user, floored to 2,048; stage 9 measures each serving profile): 4 slots 292,864, 8 slots 270,336, 16 slots
223,232, 32 slots 131,072 tokens. `doc/context_contract.json` section `datatype_sweep` records these values and the
proposed top-level change (supported_context 104,448 -> 311,296) for the flip commit. The block-sharded cache
(`GLM53_DSA_SHARD=1`, 281,344 tokens at 1 slot in stage 4) does not combine with fp8; replicated fp8 now gives more
context without the sharded path's decode cost.

## Default flip

- `tt/precision.py` (flip snapshot version): `GLM53_PRECISION` unset or empty = `selected_precision_config.json` (the
  model default for `build_generator`, the readiness runners, the full model and the vLLM adapter: all of them build
  `Glm53Model` / `MultichipDecoder`); `GLM53_PRECISION=none` (or `shipped`) = the stage-7 policy;
  `GLM53_PRECISION=<json>` = any candidate.
- Weight cache: the shipped-policy cache directory key is unchanged (`opt_s5_pf`); the re-quantized tensors are keyed
  by the re-quantization source digest (`prec-<digest>`), identical for the default and the explicit file path
  (`tests/test_precision_host.py`, host-only, 5 passed).
- With `GLM53_DSA_SHARD=1` the fp8 latent falls back to bf16 with a logged warning; the capacity sizing helper
  `tt/dsa_shard.dsa_bytes_per_token` reads the latent dtype of each layer (1 B fp8, 2 B bf16).
- The flip goes into the live tree with the stage-8 commit (coordinator), followed by the final stage-6/7 gate on the
  new default.

## Limitations

- Accuracy evidence is 300 teacher-forced positions on three AIME24 chat prompts, plus the long-context check. Top-1
  differences of 1-2 points between finalists are at the noise level of 100-position references.
- All perf numbers are 1 slot. Multi-slot decode was not swept per candidate; the precision changes reduce weight
  bytes, which the 8- and 32-slot paths read the same way.
- The sweep ran on the pinned stage-7 code (`ac03689d2bb`); the stage-6/7 rework (open-pool hand-off, lockstep
  sampling) landed after. The final combined gate runs on the merged code.
- fp8 latent and the block-sharded DSA cache (`GLM53_DSA_SHARD=1`) do not combine: the sharded cache keeps bf16.

## Files

- `selected_precision_config.json`: the selected policy (weights, layer ranges, fidelities, activations, CCL, KV
  cache, logits / sampling, flags).
- `configs/*.json`: every candidate. `runs/<run>/record.json`: per-run record (policy readback, applied overrides,
  code md5, DRAM, readiness, non-aligned check, TTFT, decode). `runs/<run>/{teacher_forcing,prefill_check}.json`.
- `sweep_results.json`, `sweep_results.csv`, `top1_perf_pareto.png`, `top5_perf_pareto.png`
  (`tools/dtype_sweep_report.py`).
- `refs/`: references p1, p2 (+ metadata), the long-context document token ids.
- Code: `tt/precision.py`, hooks in `tt/model.py` and `tt/multichip_decoder.py`, fp8 pool in `tt/dsa.py`;
  tools `tools/dtype_sweep.py`, `tools/dtype_sweep_report.py`, `tools/kv_fp8_probe.py`, `tools/long_ctx_check.py`,
  `tools/long_ctx_compare.py`.
- Scripts: `~/glm53-flash-bringup/scripts/s8_*.sh`. Logs: `~/glm53-flash-bringup/logs/s8/`. Work log: `work_log.md`.

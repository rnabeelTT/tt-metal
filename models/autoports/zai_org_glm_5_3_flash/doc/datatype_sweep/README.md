# GLM-5.3-Flash (E192) datatype sweep

Stage 8 of the tt-model-bringup flow. The model is the full 45-layer model (`tt/model.py`, `tt/generator.py`) on a
p300x2 (4 Blackhole dies, mesh (1, 4), FABRIC_1D, TP4). The sweep selects the weight, fidelity, CCL and KV-cache
datatypes of the optimized full model (stage 7).

## Result

**Selected: `combo_a_kv_dense4_g4`** (`selected_precision_config.json`; policy fields equal
`configs/combo_a_kv_dense4_g4.json`). The runs used the file with md5 d96fae9836a8. The committed file (md5 fe43daaeb567)
differs only in an end-of-file newline (pre-commit) and the `base.measured_on` text (review #4); the policy fields are
identical. It is the stage-7 policy plus these changes:

| Group | Stage 7 (`GLM53_PRECISION=none`) | Selected (default) |
|---|---|---|
| KDA input / output projections, layers 1-43 | bfp8, HiFi4 | **bfp4, LoFi** (decode rows) |
| DSA q_a \| kv_a pack, q_b, o_proj (all 11 DSA layers) | bfp8, HiFi4 | **bfp4, LoFi** (decode rows) |
| Shared expert, layers 3-43 | bfp8, HiFi4 | **bfp4, LoFi** |
| Dense MLP gate\|up and down, layers 1-2 | bfp8, HiFi4 | **bfp4, LoFi** (decode rows) |
| LM head | bfp8, HiFi2 | **bfp4, LoFi** |
| KDA projections of layers 0 and 44, shared expert of layer 44, dense MLP layer 0 | bfp8, HiFi4 | bfp8, **LoFi** (decode rows) |
| DSA latent pool (KV cache) | bf16 rows | **fp8 e4m3 rows** |
| Decode programs of the bfp4 groups | DEFAULT_MC (tuned on bfp8) | **bfp4-tuned** (`decode_specs`, section "Decode programs") |
| Routed experts, router, DSA W_UK / W_UV, indexer, residual, CCL, norms, mHC | unchanged | unchanged |

The DSA cache drops from 11,968 to 6,336 bytes per token per die.

**Context, 1 slot:** 325,632 tokens, measured on the final weights by the stage-8 capacity probe (stage 7: 104,448).
337,920 runs out of DRAM. The stage-9 serving probe at the 1,024 MiB workspace governs the served max_model_len
(section "Capacity and context contract").

**Headline: the gate of the default on HEAD (`runs/final_default_h9`).** The stage-6 and stage-7 READMEs are being
updated by their owner to cite this run as the default's TF / PC / decode evidence. At commit eb92932986b they still
name combo_a_kv and 280,576.
- Code: `~/glm53-flash-bringup/s8r3b_snapshot` = `git archive 9fcc17c0dc4 models/` (HEAD, with the logits-pad fix
  46b2a5a601d) plus `tt/precision.py` (md5 6f46fec82f0f: HEAD + the stage-8 review-3 hunks only) and the selected
  config. Every other `tt/*.py` in the record equals 9fcc17c0dc4 (e.g. `model.py` 5fee3c6742a7).
- `GLM53_PRECISION` unset, 1 slot, max_seq_len 16,384, TP4.
- The same-code comparison is the previous default (`kdain_default_h9`: combo_a_kv + the kda_in program only).

| Metric | Previous default (kda_in only) | **Selected `combo_a_kv_dense4_g4`** |
|---|---|---|
| Teacher forcing top-1 p0 / p1 / p2 (top-5, top-100: 100 %) | 91 / 96 / 91 % | **92 / 95 / 92 %** |
| Prefill check top-1 p0 / p1 / p2 (top-5, top-100: 100 %) | 91 / 95 / 91 % | **92 / 95 / 91 %** |
| Trace replays during teacher forcing (99 decode steps + the prefill logits) | 198 | 198 |
| Token-out decode (model + sampler trace, device feedback, no readback, 30 replays) | 50.40 ms | **49.94 ms = 20.02 t/s/u** |
| Trace-verified teacher-forcing decode (readiness runner, 1 readback per step) | 19.85 t/s/u | **20.02 t/s/u** |
| Model trace / generate loop (one token readback per step) | 50.01 / 50.67 ms | 49.56 / 50.10 ms |
| TTFT, warm, device logits: 128 / 161 / 2048 tokens | 226 / 378 / 1024 ms | 226 / 378 / 1024 ms |
| Non-aligned prompts (100 / 161 / 300 tokens): traced vs eager, repeated | equal | equal |
| Model weight load, warm cache | 99.8 s | 98.1 s (a second build: 168 s, host-side variance) |

- The stage-7 policy measured 51.96 ms on the code generation before (914af5b9594, `shipped_r2`); no stage-7 build of
  HEAD was run. The -3.9 % therefore compares two code generations. The review-2 default gives the link: it measured
  50.38 ms on 914af5b9594 (`combo_a_kv_g4_kdain_r2`) and 50.40 ms on HEAD (`kdain_default_h9`), so the two
  generations have the same decode speed for one policy.
- **Performance regime:** every stage-8 speed is at max_seq_len 16,384 and 1 slot. Decode slows by about 10 % at a
  long max_seq_len (the indexer scores every allocated pool slot), and the bfp4 decode programs are measured at 1 slot
  only. Stage 9 re-measures token-out at the served max_model_len and in its 8- and 32-slot gates.

Qualitative and autoregressive checks of the selected default (`runs/final_default_h9/checks/`):
- **Qualitative suite**: 6 chat prompts. 5 end on EOS; q3 reaches the 2,048-token budget while still drafting, as in
  stage 7. The answers are correct and coherent: a 5-7-5 haiku, supervised vs unsupervised learning, the three laws
  of thermodynamics, Fibonacci code.
  - q5 leaves "Bonjour, how are you today?" untranslated. The HF control does the same (doc/full_model/qualitative/VERDICT.md).
  - Adjacent duplication <= 0.034 in every answer; no wrong language or control-token leakage.
- **Autoregressive check**: two runs bit-identical; `check_degenerate_output` exit 0 (adjacent duplication 0.0,
  trigram loop fraction 0.23). TT matches HF's greedy token 0 (" a") and branches at token 1 (" small", HF rank 1, 0.06 logits below " strange").
  From token 24 the free run repeats one sentence about 17 times, as in stage 6. The degenerate check does not catch
  a loop of a long sentence; the loop is accepted by comparison with stage 6, where HF continues the same loop.
  - The same-code previous default (`kdain_default_h9`) branches at token 0 (" something" vs " a"). The difference
    between the two free runs is the policy (dense4 + programs), not the code.
- **Known observation: HF control of the free run.** `tools/hf_full_ref.py --mode score-trajectory` (CPU) teacher-forces
  HF over the TT trajectory (`checks/autoregressive/hf_scores_tt_trajectory.json`):

  | Trajectory | HF top-1 agreement | HF top-5 agreement | Worst TT pick (HF rank, 0-based) |
  |---|---|---|---|
  | stage 6 (stage-7 policy, doc/full_model/autoregressive) | 97.7 % | 99.6 % | 5 |
  | stage 8 previous default (`runs/selected_default_r1`) | 93.75 % | 100 % | 4 |
  | **stage 8 selected (`runs/final_default_h9`)** | **94.9 %** | **99.2 %** | **21** |

  The full-run agreement is high mainly because of the loop tail, where HF continues the loop. Before the loop the
  divergence is larger than in the earlier runs:

  | Trajectory | HF top-1 agreement, first 40 tokens | Largest gap of a TT pick below the HF top-1 (logits) |
  |---|---|---|
  | stage 6 (doc/full_model/autoregressive) | 85 % | 1.875 |
  | stage 7 | 82.5 % | 1.75 |
  | stage 8 previous default (`selected_default_r1`) | 80 % | 1.875 |
  | **stage 8 selected (`final_default_h9`)** | **67.5 %** | **2.625** |

  - Four TT picks are 1.6 logits or more below HF's top 1: position 5 (" broken", rank 4, 2.4), 12 (" watched",
    rank 3, 2.6), 22 (".", rank 5, 1.6) and 24 (" sat", rank 21, 2.2; HF's own top-1 margin there is 0.125).
  - Positions 22 and 24 are the entry to the loop: the rank-5 and rank-21 picks start the repeated sentence.
  - **Chat-prompt control: no regression.** The review-4 reviewer computed the first position where TT diverges from
    HF greedy (128 tokens) on the six qualitative prompts q1-q6: final 38 / 25 / 10 / 15 / 38 / 27 (sum 153);
    `kdain_default_h9` 15 / 25 / 10 / 26 / 53 / 27 (sum 156); stage 6 38 / 75 / 10 / 27 / 31 / 29 (sum 210). The final
    and the previous default are equal within noise.
  - Recorded as a known observation: bfp4 numerics drift in a raw-prompt free run. Teacher forcing, the prefill check
    and the qualitative suite pass. A chat-domain HF score trajectory (q1 / q5, CPU) was not run.

**Acceptance gate** (skill default): teacher-forcing top-1 >= 90 %, top-5 >= 98 % and top-100 = 100 %. The prefill
check must meet the same values. `tools/dtype_sweep_report.py` applies all six to the minimum over every reference of a
run (`pass` in `sweep_results.*`); `pass_p0` is the gate on p0 alone, and `pass_selection_rule` adds the 91 % rule.

**Selection rule** (coordinator, on top of the gate):
1. The minimum top-1 over teacher forcing and prefill check on three AIME24 chat references must be >= 91 %.
   The references are p0 (the readiness reference), p1 and p2. Top-5 and top-100 must hold on all three.
2. Among the configs that pass, prefer context, then decode speed. On HEAD, `combo_a_kv_dense4_g4` is both: it is the
   fastest passing config, and the dense bfp4 layers free 37 MB per die more than the combo_a_kv variants.
3. A context gain below ~5 % that is only estimated (unmeasured) does not override a config with fewer bfp4 groups
   at equal speed. This rule was added after `combo_a_kv_dense4` was measured (review #1). It decided only that case,
   on code with the logits-pad bug (below); on HEAD the dense4 variant is faster, so rule 3 does not apply.

The selection margin is one token: min top-1 = 91 % (prefill check p2), as for every passing config.

![min top-1 vs speed](min_top1_perf.png)
![top-1 vs speed](top1_perf_pareto.png)
![top-5 vs speed](top5_perf_pareto.png)

The charts plot trace-verified teacher-forcing decode t/s/u (readiness runner) on y. Red = selected; the dotted line
is the gate.
- `min_top1_perf.png`: x = the minimum top-1 over p0 / p1 / p2. Only three-reference runs are points. The line is at 91 %.
- `top1_perf_pareto.png` and `top5_perf_pareto.png`: x = accuracy on p0. Every evaluated full-model run is a point.
- On the top-5 chart every point except the old `combo_a_all` / `dense4_g4` collapses ties at 100 %.
- Points from four code generations appear (Method), with one marker and one frontier per generation. A frontier of
  one point has no line: on the `_h9` generation `combo_a_kv_dense4_g4_h9` / `final_default_h9` dominate every other
  `_h9` row, so that frontier is the red point alone. Compare speeds only within one generation:
  - `_h9` runs: HEAD 9fcc17c0dc4 (logits-pad fix). **Only these decide the selection.**
  - `_r2` runs: 914af5b9594; `_r1` runs: b8198430147 (both before the logits-pad fix).
  - All other runs: s8_snapshot (ac03689d2bb).

## Method

### Code under test

| Snapshot | Content | Used for |
|---|---|---|
| `~/glm53-flash-bringup/s8_snapshot` | `git archive ac03689d2bb models/` + the stage-8 files: `tt/precision.py`, the fp8 pool in `tt/dsa.py`, the hooks in `tt/model.py` and `tt/multichip_decoder.py`, `tt/dsa_shard.py` fp8 sizing (`dsa_bytes_per_token`), `tools/dtype_sweep*.py`, `tools/kv_fp8_probe.py`, `tools/long_ctx_*.py`; `tools/memory_probe.py` patched to load the weight cache (`weight_cache="env"`) | the 18-candidate sweep, the three-reference finalists, the first capacity probe, the long-context check |
| `~/glm53-flash-bringup/s8_snapshot2` | s8_snapshot + the then-current stage-6/7 rework files `tt/{dsa, dsa_shard, decode_inputs, functional_decoder, generator, generator_vllm, model, weight_cache, serving_state}.py`, `tt/e192_mask.json`, `tools/{memory_probe, readiness}.py`; dsa_shard.py md5 changed (fp8 sizing line re-added) | the combo_a_all rerun that crashed (below) |
| `~/glm53-flash-bringup/s8_snapshot_flip` | s8_snapshot + the default flip in `tt/precision.py`, the `root_cache` hook, `tests/test_precision_host.py` | the first post-selection runs (`runs/selected_default*`, superseded by `_r1`) |
| `~/glm53-flash-bringup/s8r1_snapshot` | `git archive b8198430147 models/` + the review-#1 `tt/precision.py`, `tools/dtype_sweep.py`, `tools/long_ctx_check.py`, `tests/test_precision_host.py`, `configs/combo_a_kv_dense4.json`; `tools/memory_probe.py` patched to `weight_cache="env"` | the review-#1 gate (`_r1`), the 337,920 / 1,024 MiB capacity probe |
| `~/glm53-flash-bringup/s8r2_snapshot` | `git archive 914af5b9594 models/` + the review-#2 `tt/precision.py`, `tools/{mc_matmul_sweep, dtype_sweep, dtype_sweep_report, long_ctx_check}.py`, `tests/test_precision_host.py`; `tools/memory_probe.py` patched to `weight_cache="env"` | ticket A: the bfp4 geometry sweep, the fp8 profile, the 337,920 / 832 MiB capacity step |
| `~/glm53-flash-bringup/s8r2b_snapshot` | s8r2_snapshot + the LM-head decode-program hook in `tt/precision.py`, `tools/dtype_geometry_summary.py`, the `*_g4*` configs | tickets B and C: the `_r2` gate, the program bisect, the 337,920 step with the 1,024-row MoE split |
| `~/glm53-flash-bringup/s8r3_snapshot` | `git archive 9fcc17c0dc4 models/` (HEAD: logits-pad fix 46b2a5a601d, stage-8 commit aa104284595, stage 6/7 rework de9852c604d) + `configs/combo_a_kv_dense4_kdain.json` + `tt/precision.py` (see below) | the review-#3 reselection (`_h9`: all programs, rest, dense4_g4, dense4 + kda_in, combo_a_all) and the previous default's gate (`kdain_default_h9`) |
| `~/glm53-flash-bringup/s8r3b_snapshot` | s8r3_snapshot + `tt/precision.py` md5 6f46fec82f0f (HEAD + the review-3 hunks only) + `selected_precision_config.json` = combo_a_kv_dense4_g4 (md5 d96fae9836a8) | the final gate (`final_default_h9`, `_warm`), the capacity steps on the final weights |

`tt/precision.py` in the review-3 runs:
- `combo_a_kv_g4_h9`, `combo_a_kv_g4_rest_h9`: md5 e6b14dda53e6 = HEAD + the layer-kind filter of the decode programs.
- `combo_a_kv_dense4_g4_h9`, `combo_a_kv_dense4_kdain_h9`, `combo_a_all_h9`: md5 11a8a3ac06dc = the same + the
  LM-head log fix + another agent's dsa_shard hunk (copied from the live tree; inert on the replicated layout).
- Every difference is in logging or in the block-sharded path. `kdain_default_h9`, `final_default_h9`
and the capacity steps carry 6f46fec82f0f, the version to commit.

Every job after the first baseline ran with its working directory inside its snapshot. `python -m` puts the working
directory before PYTHONPATH. One early baseline ran from the live tree by mistake (6 / 28 / 99 %); it was discarded
(`logs/s8/shipped_LEAKED_livetree.log`).

What each record holds:
- The md5 of each `tt/*.py` (`code.md5`): every `tools/dtype_sweep.py` run from the code pinning on.
- The digest of that md5 map, the base commit, the real argv, the GLM53_* / MESH environment and the calling script:
  the review-#1, #2 and #3 runs (`_r1`, `_r2`, `_h9`) only.
- Older rows in `sweep_results.*` mark their command and script as `[reconstructed]`.
- The long-context records (`long_context/*.json`) predate both. They record the indexer block size
  (`GLM53_DSA_INDEX_BLOCK_MB`: 32 MB for shipped, 64 MB default for combo_a_kv) but no md5 map.
- The capacity probes have a sidecar `provenance.json` with the md5 map, from the 337,920 probes on. The first
  probe (`capacity/combo_a_kv_s1`) has none.
- Kernels and the ttnn build always come from the repo. The live mhc_split_sinkhorn kernel diff against HEAD at the time was inert: only `#ifdef SK_FACES` paths,
which no stage-8 job enabled.

### Candidates

`tt/precision.py` (`PrecisionOverrides`) applies a policy JSON at construction. There are two hooks:
`MultichipDecoder.from_state_dict` (per layer) and `Glm53Model.__init__` (LM head). The JSON fields:
- `requant`: host round-to-nearest re-quantization of a group's loaded bfp8 tensors to bfp4, per die, with the same
  memory config. The rounding from bfp8 equals the rounding from bf16, except at ties of the bfp8 grid. The device
  typecast is not used: it does not round to nearest (stage 3).
- `fidelity`: the compute fidelity of a group's decode-row matmuls (KDA, DSA projections, dense MLP, LM head). For the
  shared expert and the routed experts it covers all rows (the MoE `shared_fidelity` / `routed_fidelity` fields).
- `moe_layer_exceptions`: per-layer MoE fields. A layer with a dtype exception is built from the checkpoint. Its
  re-quantized tensors are not cached either.
- `ccl_h_dtype`: the decode all-gathers of the residual in bfp8 (cast before, cast back after).
- `kv_latent_dtype`: the DSA latent pool in fp8 e4m3 rows.
- `decode_specs`: per-layer MCOpts decode-program specs (and `lm_head`) on the layers where a group is bfp4. DEFAULT_MC
  itself is unchanged.

`GLM53_PRECISION` selects the policy:
- unset or empty: `selected_precision_config.json` (the default);
- `none`, `shipped`, `off` or `0`: the stage-7 policy (standalone generator and tests only);
- `<json>`: a candidate.

The vLLM adapter refuses the stage-7 spellings with a `ValueError` (`tt/generator_vllm.py` `apply_precision_default`):
`serving_profiles.json` budgets the selected policy only. The stage-7 bf16 latent needs about 1.9 times the DSA bytes
per block, and its weights leave less free DRAM. Run stage-7 A/B comparisons with the standalone generator.

No candidate needs its own 116 GB weight cache. Every build loads the shipped-policy cache and changes only the
overridden groups.

### Re-quantized tensor cache

A warm load must not re-quantize. `PrecisionOverrides._rq` stores each re-quantized tensor in the weight cache at
`<root>/<layer_i | top>/prec-<requant source digest>/<group>-<dtype>-<digest>.tensorbin`. The final digest covers:
- the owning component's source digest (`weight_cache.component_digest`: kda / dsa / mlp / moe / top). For the DSA
  q_a|kv_a pack it also covers `MultichipDSA._build_pack`, `DSMatmul`, `IL1DMatmul` and `make_role_mm`. The
  ImportError fallback (host tests) is logged and named `nodigest`;
- the source tensor's memory config, padded shape and dtype.

On a cache hit the loaded tensor's memory config, padded shape and dtype are checked against the source tensor. A
mismatch rebuilds the file. The default and an explicit `GLM53_PRECISION=<selected file>` resolve the same files.

### Propagation check

- `runtime_policy`: dtypes and compute fidelities read back from the live model objects.
- `precision.applied`: every change, per layer. A decode program is logged only on the layers of the kind it drives
  (review #3). In `runs/final_default_h9/record.json`: `spec:kda_in` and `spec:kda_out` on the 32 inner KDA layers,
  `spec:pack_a` and `spec:dsa_o` on the 11 DSA layers, `spec:mlp_gu` on the 2 dense layers 1-2. Before the fix, the
  log also named `spec:kda_in` on the 11 DSA layers, where the spec has no effect.
- `fidelity_seen_one_eager_step`: the fidelity the wrappers passed to each decode matmul group in one eager decode
  step. On HEAD (the selection) it is kda_in LoFi x 34, kda_out LoFi x 34, DSA q_a|kv_a / q_b LoFi x 33, DSA o_proj LoFi x 11,
  dense MLP LoFi x 3. This is every layer of each group (`runs/final_default_h9/record.json`).
- The wrappers keep each module's own `fp32_dest_acc_en`, `packer_l1_acc` and approx mode, and change only the
  fidelity (review #2). Before that, the LM head at LoFi had run without packer L1 accumulation. That affects the
  s8_snapshot rows `lm4_lofi` / `lm8_lofi` and the combinations; the `_r2` gate carries the fixed numerics.

### Accuracy and performance

- **Accuracy.** `tools/dtype_sweep.py` builds the generator once per candidate. It runs the readiness runners
  (`run_teacher_forcing`, `run_prefill_check`) on that build.
  - p0 = `readiness_aime24_chat.refpt`: AIME24 prompt 0, chat template, 100 generated tokens, HF CPU reference
    `tools/hf_full_ref.py`.
  - p1 and p2: AIME24 prompts 1 and 2, made with the same tool (`refs/`).
  - Each reference scores 100 positions: the prefill logits give position 0, and 99 traced decode steps give the rest.
- **Trace verification.** The `ttnn.execute_trace` calls during teacher forcing are counted. The count must equal 2 x
  the decode steps (model trace + sampler trace). It is 198 in every run (`trace_verified`).
- **Non-aligned prompts.** Every run prefilled 100-, 161- and 300-token prompts. It compared the traced and eager
  greedy continuations, 8 tokens each, run twice. They were equal and repeatable in every run (`ptcheck`).
- **Performance.** Token-out = 30 back-to-back replays of model trace + sampler trace, with device token feedback and
  no host readback (the stage-7 benchmark). TTFT = warm prefill to device logits, plus a device sync.
- **Context.** The stage-7 1-slot capacity is 818 blocks of 128 tokens at 11,968 B per token per die (stage-6
  probe). For other candidates the report estimates the context from the build's DRAM delta against the shipped build
  of the same code: `floor((818 x 128 x 11,968 - delta) / (128 x bytes per token))`, floored to 2,048.
  - These `~` values are upper bounds. They leave out the DSA constants that grow with max_context, and the larger
    workspace a large pool needs.
  - Example: combo_a_kv was estimated at 364,544 and measured at 311,296. Only combo_a_kv and the final weights
    (325,632) were measured (section "Capacity and context contract").

## Candidates and results (s8_snapshot code)

Single changes on the stage-7 policy, one build each, reference p0 (`runs/<id>/`, logs `~/glm53-flash-bringup/logs/s8/`).
"inner" = layers 1-43: the skill excludes the first and the last layer from bfp4 by default.

| id | Change | TF top-1 / 5 / 100 | PC top-1 | Token-out ms | vs shipped | Context | Decision |
|---|---|---|---|---|---|---|---|
| shipped | stage-7 policy | 92 / 100 / 100 | 91 | 54.08 | - | 104,448 | baseline |
| kv_fp8 | DSA latent fp8 e4m3 | 92 / 100 / 100 | 91 | 54.09 | +0.01 | ~196,608 | kept (context x 1.9) |
| kda_in4_lofi | KDA in-proj bfp4, LoFi | 91 / 100 / 100 | 90 | 53.71 | -0.37 | ~149,504 | kept in combo_a |
| kda_in4_hifi2 | KDA in-proj bfp4, HiFi2 | 91 / 100 / 100 | 90 | 53.73 | -0.35 | ~149,504 | rejected: same accuracy as LoFi, slower |
| dsa4_lofi | DSA q_a\|kv_a, q_b, o_proj bfp4, LoFi | 94 / 100 / 100 | 95 | 53.79 | -0.29 | ~118,784 | kept |
| kda_out4_lofi | KDA out-proj bfp4, LoFi | 94 / 100 / 100 | 94 | 53.88 | -0.20 | ~114,688 | kept |
| proj8_lofi | bfp8 projections at LoFi (KDA, DSA, dense MLP decode; shared expert) | 93 / 100 / 100 | 95 | 53.93 | -0.15 | 104,448 | kept (dense MLP, layers 0 and 44) |
| lm4_lofi | LM head bfp4, LoFi | 93 / 100 / 100 | 91 | 53.94 | -0.14 | ~110,592 | kept |
| shared4_lofi | shared expert bfp4, LoFi | 94 / 100 / 100 | 94 | 54.00 | -0.08 | ~114,688 | kept |
| proj8_hifi2 | same groups as proj8_lofi at HiFi2 | 93 / 100 / 100 | 94 | 54.04 | -0.04 | 104,448 | rejected: LoFi is faster at equal accuracy |
| lm8_lofi | LM head bfp8 at LoFi | 92 / 100 / 100 | 91 | 54.09 | +0.01 | 104,448 | rejected: no gain |
| exp8dn_edge | routed down bfp8 (HiFi2) in MoE layers 3 and 44 | 94 / 100 / 100 | 94 | 54.09 | +0.01 | ~69,632 | rejected: -33 % context |
| routed_hifi3 | routed experts bfp4 at HiFi3 | 93 / 100 / 100 | 94 | 54.43 | +0.35 | 104,448 | rejected: slower, no gate need |
| ccl_h_bfp8 | residual all-gathers in bfp8 | 92 / 100 / 100 | 91 | 54.55 | +0.47 | 104,448 | rejected: slower (2 casts per gather; the gathers are latency-bound) |

Combinations (p0). The finalists then ran on p0, p1 and p2 in repeat builds (`runs/<id>_r3/`):

| id | Content | TF top-1 p0 / p1 / p2 | PC top-1 p0 / p1 / p2 | Min top-1 | Token-out ms | Context | Decision |
|---|---|---|---|---|---|---|---|
| combo_a | kda_in4 + kda_out4 + dsa4 + shared4 + lm4 (inner, LoFi) + bfp8 groups at LoFi | 90 / 94 / 91 | 91 / 95 / 91 | 90 | 53.04 | ~192,512 | rejected: min top-1 90 < 91 |
| combo_a_kv | combo_a + fp8 latent | 91 / 96 / 91 | 91 / 95 / 91 | 91 | 53.22 | 311,296 (measured) | selected in review #1 (superseded on HEAD) |
| combo_b | combo_a without kda_in4 | 94 / 93 / 92 | 94 / 95 / 91 | 91 | 53.41 | ~147,456 | passes; less context, slower |
| combo_b_kv | combo_b + fp8 latent | 95 / 94 / 90 | 94 / 95 / 91 | 90 | 53.58 | ~278,528 | rejected: min top-1 90 |
| kv_fp8 | (above) | 92 / 97 / 93 | 91 / 97 / 91 | 91 | 54.09 | ~196,608 | passes; slower, less context |
| shipped | (above) | 92 / 98 / 91 | 91 / 97 / 91 | 91 | 54.07 | 104,448 | passes; the slowest but one of the passing set |
| combo_a_all | combo_a on layers 0 and 44 too | 14 / - / - | 90 / - / - | 14 (top-5 32 %) | 53.02 | - | collapse = the logits-pad bug; re-gated on HEAD below |

On the committed code (`_r1`, three references):

| id | Content | TF top-1 p0 / p1 / p2 | PC top-1 p0 / p1 / p2 | Min top-1 | Token-out ms | Context | Decision |
|---|---|---|---|---|---|---|---|
| shipped_r1 | stage-7 policy (`GLM53_PRECISION=none`) | 92 / 98 / 91 | 91 / 97 / 91 | 91 | 52.20 | 104,448 | baseline of this code |
| selected_default_r1 | combo_a_kv (default then) | 91 / 96 / 91 | 91 / 95 / 91 | 91 | 51.14 | 311,296 | selected in review #1 |
| combo_a_kv_dense4_r1 | combo_a_kv + dense MLP bfp4 LoFi on layers 1-2 | 92 / 95 / 92 | 92 / 95 / 91 | 91 | 51.13 | ~+5.8K over combo_a_kv (estimate) | passing alternative, not selected |

**Review-#3 reselection on HEAD (9fcc17c0dc4, with the logits-pad fix), three references each (`runs/*_h9`):**

| id | Content | TF top-1 p0 / p1 / p2 | PC top-1 p0 / p1 / p2 | Min top-1 | Token-out ms | Decision |
|---|---|---|---|---|---|---|
| combo_a_kv_g4_h9 | combo_a_kv + all bfp4 decode programs | 91 / 96 / **89** | 91 / 95 / 91 | 89 | 50.05 | rejected: min top-1 89 |
| combo_a_kv_g4_rest_h9 | combo_a_kv + all programs but kda_in | 91 / 96 / **89** | 91 / 95 / 91 | 89 | 50.77 | rejected: min top-1 89 |
| **combo_a_kv_dense4_g4_h9** | combo_a_kv + dense MLP bfp4 on layers 1-2 + all programs (incl. mlp_gu / mlp_dn) | **92 / 95 / 92** | **92 / 95 / 91** | **91** | **49.95** | **selected: fastest passing** |
| combo_a_kv_dense4_kdain_h9 | combo_a_kv + dense MLP bfp4 on layers 1-2 + the kda_in program | 92 / 95 / 92 | 92 / 95 / 91 | 91 | 50.32 | passes, slower |
| kdain_default_h9 | combo_a_kv + the kda_in program (the previous default, `GLM53_PRECISION` unset) | 91 / 96 / 91 | 91 / 95 / 91 | 91 | 50.40 | passes, slower |
| combo_a_all_h9 | combo_a (no fp8 latent) with bfp4 also on layers 0 / 44 | 91 / 95 / 90 | 90 / 96 / **89** | 89 | 51.04 | rejected: PC p2 89 < the 90 gate |
| **final_default_h9** | the selected config as the default (`GLM53_PRECISION` unset) | **92 / 95 / 92** | **92 / 95 / 91** | **91** | **49.94** | the gate (Result) |

Top-5 and top-100 are 100 % in every `_h9` cell; every run has 198 trace replays and equal non-aligned continuations.

**What the logits-pad fix changed.** Every stage-8 run before review #3 ran `Glm53Model.decode_step` with a
`ttnn.deallocate(logits)` after `ttnn.pad`. At 1 slot the padded path runs every step, and the deallocate freed the
storage of the decode trace's persistent logits output: a deterministic, DRAM-layout-dependent decode corruption
(stage 5, `doc/optimized_multichip_decoder/README.md`; fixed in 46b2a5a601d). Its signature is TF 14 / 32 / 90 with an
intact prefill check.
- **combo_a_all and combo_a_kv_dense4_g4: the collapses were this bug.** On HEAD, combo_a_all reads 91 / 95 / 90 (no
  collapse; it fails the gate by one prefill-check token on p2) and dense4_g4 reads 92 / 95 / 92 (passes).
- **The p2 = 89 shift with all programs (or all but kda_in) on combo_a_kv weights is not the bug**: it reproduces on
  HEAD, so it comes from the numerics of those programs. With the dense bfp4 weights the same programs pass
  (p2 92 / 91). **The exact mechanism is not proven.**
  - At op level each program's output hash equals the current program's, but only on the sweep's random (randn)
    input (`geometry/summary.json`).
  - Probable mechanism (not measured): with fp32 dest and packer-L1 accumulation, a different `in0_block_w` groups
    the fp32 partial K-sums differently. On randn inputs the bf16-rounded output seldom changes; on real activations
    with outliers some outputs change by one bf16 step. Over 45 layers that can move a close top-1 on p2 (one token
    of 100).
  - No layer bisect was run: the selection was gated with all ten programs in place, so the shift does not affect it.
- The classification runs of review #1 (layer 0 only / layer 44 only for combo_a_all) were not needed: the collapse
  is gone, and combo_a_all fails on accuracy alone.

**Earlier code generations (before the fix):**
- **Noise.** Repeat builds of one policy measure the same token-out time to 0.02 ms: shipped 54.08 / 54.07, combo_a
  53.03 / 53.04, combo_a_kv 53.22 / 53.22 (warm rebuild on b8198430147: 51.14 / 51.13), final_default_h9 49.94 /
  49.96 (warm rebuild).
- **combo_a_kv_dense4 (review #1).** It passed at 51.13 vs 51.14 ms. Rule 3 kept combo_a_kv then. Stage 3 had rejected
  dense bfp4 at layer level on layer 0 only, and stage 3 also rejected kda_in4 and mlp8_lofi, which pass here at
  model level.
- **`runs/selected_default_r2`** is the rejected all-programs config run as the default of that time; its
  `precision_env` "unset" refers to that config, not to the final selection.
- **fp8 latent inside the combinations.** It costs +0.17 to +0.19 ms per step in the full model (combo_a 53.03 ->
  combo_a_kv 53.22; combo_b 53.41 -> combo_b_kv 53.58), but only +0.01 ms on the stage-7 policy.
  - A Tracy decode window of combo_a vs combo_a_kv on the reduced model (layers 0, 3, 4; `profile/compare_decode.json`;
    per op the maximum device time over the 4 dies) shows +1.5 µs in `sparse_sdpa` and one +1 µs cast per DSA layer.
  - The total device time is 4,114 vs 4,101 µs. About 0.03 ms of the 0.17-0.19 ms is in the fp8 ops; the rest is not
    in the device op time. The cause is not identified (open, low impact).

## Compute-fidelity table (material decode matmul groups)

| Group | Weight | Fidelity | Token-out ms | TF top-1 (p0) | Kept | Reason |
|---|---|---|---|---|---|---|
| Routed experts | bfp4 | LoFi | 54.08 (shipped) | 92 | yes | HiFi2 is bit-identical on bfp4 weights (stage 3) |
| Routed experts | bfp4 | HiFi3 | 54.43 | 93 | no | +0.35 ms; the gate does not need it |
| KDA in-proj | bfp4 | LoFi | 53.71 | 91 | yes | |
| KDA in-proj | bfp4 | HiFi2 | 53.73 | 91 | no | same accuracy, slower |
| KDA in / out, DSA, dense MLP, shared expert | bfp8 | HiFi4 (shipped) | 54.08 | 92 | no | |
| same | bfp8 | HiFi2 | 54.04 | 93 | no | |
| same | bfp8 | LoFi | 53.93 | 93 | yes | fastest at equal accuracy |
| KDA out, DSA, shared expert, LM head | bfp4 | LoFi | see the table above | 93-94 | yes | |
| LM head | bfp8 | LoFi vs HiFi2 | 54.09 vs 54.08 | 92 | no | no difference |

bfp4 at HiFi2 was compared against bfp4 at LoFi at model level only for the KDA input projection, the largest bfp4
group. For the other bfp4 groups only LoFi was run at model level. The op-level geometry sweep (next section) ran
HiFi2 on each family's best program (`geometry/summary.json`, `hifi2_trials`).

## Decode programs for bfp4 weights

The DEFAULT_MC decode programs were tuned on bfp8 weights in stage 5. The bfp4 groups saved only about 35-40 % of the
bandwidth-bound expectation, so `tools/mc_matmul_sweep.py --wdtype bfp4 --fidelity LoFi --alt-fidelity HiFi2` swept the
decode geometries again on bfp4 weights (one die, 100 traced replays per candidate). It covered in0 blocks 4-64,
1D core counts 16-110, DRAM-sharded activation cores and readers, and the LM head [4096 x 38720].
Results: `geometry/bfp4_lofi_decode.jsonl`, summary `geometry/summary.json` (`tools/dtype_geometry_summary.py`).

| Group | Current program | µs | Fastest buildable within noise | µs | Output vs current | Adopted |
|---|---|---|---|---|---|---|
| kda_in | ttnn default | 151.0 (sweep) / 90.2 (in model) | `1d:c29:ib4` | 74.4 | hash identical | yes |
| kda_out | `1d:c16:ib4` | 28.6 | `1d:c16:ib16` | 25.0 | hash identical | yes |
| dsa_o | `1d:c16:ib4` | 47.2 | `1d:c16:ib16` | 39.6 | hash identical | yes |
| dsa_qb | `1d:c16:ib4` | 23.8 | `1d:c16:ib16` | 21.1 | hash identical | yes |
| pack_a (q_a\|kv_a) | `1d:c32:ib8` | 29.5 | `1d:c16:ib32` | 25.3 | hash identical | yes |
| shared_gu | `1d:c32:ib16` | 22.8 | `1d:c16:ib64` | 18.7 | hash identical | yes |
| shared_dn | `1d:c16:ib4` | 14.4 | `1d:c16:ib16` | 13.3 | hash identical | yes |
| mlp_gu (layers 1-2) | `1d:c16:ib4` | 61.3 | `1d:c16:ib8` | 55.7 | hash identical | yes |
| mlp_dn (layers 1-2) | `1d:c16:ib4` | 37.9 | `1d:c16:ib16` | 33.0 | hash identical | yes |
| LM head | ttnn default | 355.3 (sweep) / 307.6 (in model) | `1d:c110:ib4` | 330.8 | hash identical | yes (see below) |

"Buildable" means `make_role_mm` builds the same core count: the 1D core count divides the N tiles. "Within noise"
means the error against fp64 is within 5 % of the current program's error. "Hash identical" means the output's sha256
on the sweep's random input equals the current program's. It does not prove equal outputs on real activations (see
the p2 = 89 shift in "Candidates and results").

**The sweep's baseline for the two "ttnn default" roles is not the in-model program.** The sweep times "default" with
`packer_l1_acc=False` on a 2-D weight. In the model (Tracy, `profile/combo_a_kv/perf_decode_report.csv`):
- kda_in runs at 90.2 µs on 87 cores (in0 block 2), not 151 µs. The in-model gain per layer is therefore about 16 µs,
  not 77 µs; that explains the 0.68 ms realized vs 2.4 ms predicted on the `_r2` code.
- The LM head runs at 307.6 µs on 110 cores (in0 block 2, `packer_l1_acc=True`). The sweep's program runs at
  330.8 µs in a harness that times the default at 355.3 µs. The sign of its in-model effect is therefore unknown;
  the bound is about +/- 0.05 ms per token (once per step). It is adopted with the selection as measured end to end
  (the full selected config is the fastest run). Not measured: a `combo_a_kv_dense4_g4` run without the `lm_head`
  spec, or a Tracy decode window of the final config. Either settles it.

Model-level adoption (rule (b) of the coordinator: the 3-reference gate with the programs in place):
- On the `_r2` code (before the logits-pad fix): all programs and all-but-kda_in gave p2 TF 89 % on combo_a_kv
  weights; kda_in alone passed (50.38 ms) and became the review-#2 default.
- On HEAD (`_h9`, section "Candidates and results"): the same p2 = 89 shift reproduces on combo_a_kv weights, so on
  those weights the programs other than kda_in stay unadopted. With the dense bfp4 weights (`combo_a_kv_dense4_g4`)
  all programs pass (92 / 95 / 92, PC 92 / 95 / 91) and give the fastest passing config: **all ten programs are
  adopted with the dense4 selection**.
- The kda_in program at 8 and 32 slots (real rows through `IL1DMatmul`) is not measured here; stage 9's 8- and 32-slot
  gates cover it.

## Groups not reduced

- **Router** (bf16, HiFi4): routing numerics. It is not in the bfp4 sweep (skill MoE policy).
- **Dense MLP layer 0** (bfp8): the first layer, excluded from bfp4 by the skill's default. Layers 1-2 are bfp4 in the
  selection.
- **DSA W_UK / W_UV** (bfp8) and the indexer projections (bf16, fp32 output): attention-core numerics and index
  scores, 15 + 35 µs per DSA layer.
- **Routed experts above bfp4**: bfp8 in all layers does not fit in DRAM. `exp8dn_edge` measured the value of a
  partial increase: top-1 +2 points, -33 % context.
- **Residual / activations**: bf16 (the mHC streams). No lower format exists for the residual adds.
- **KV cache in bfp8**: bfp8 is a TILE-only format, but the DSA latent pool is a ROW_MAJOR paged row store
  (`scatter_rows`, `sparse_sdpa`). fp8 e4m3 rows are the supported 1-byte format.
  - `tools/kv_fp8_probe.py` (`evidence/kv_fp8_probe.json`): the device bf16 -> fp8 cast, `scatter_rows` and
    `sparse_sdpa` FP8_E4M3 all work. The attention output PCC is 0.99958 against the bf16 pool.
- **Indexer pool keys in bfp8**: not tried. `ttnn.experimental.dsa_decode_scores` requires bf16 keys
  (`dsa_decode_scores_device_operation.cpp:37`). The saving would be 30 B per token per layer, 5 % of the fp8 cache.

## Long-context check

`tools/long_ctx_check.py` checks the fp8 latent and the bfp4 projections at long range:
- Document: the repository's English documentation, 31 files, frozen token ids `refs/long_doc_ids.json.gz`.
- Both builds see the same positions, at max_seq_len 102,400.
- For L = 2K, 32K and 100K: prefill L - 200 tokens, then run 199 traced decode steps teacher-forced with the
  document's tokens. The prefill logits and the 199 decode steps give 200 scored positions (host logits).
- The check compares combo_a_kv with the stage-7 policy on the s8_snapshot code (before the logits-pad fix). The final
  selection adds the dense bfp4 layers 1-2 and the decode programs; it was not re-run at long context (not done in
  review #4; a 32K / 100K rerun can go into stage 9). The risk is low: the added changes are weight and program
  changes that do not depend on the position, and the KV-cache dtype is unchanged (fp8 latent in both).

`tools/long_ctx_compare.py` compares the two builds (`long_context/compare_shipped_vs_combo_a_kv.json`):

| L | Shipped true-token top-1 / top-5 | combo_a_kv true-token top-1 / top-5 | Delta top-1 / top-5 | TT-vs-TT top-1 agreement / top-5 overlap | Prefill s (shipped / selected) |
|---|---|---|---|---|---|
| 2,048 | 0.840 / 0.935 | 0.830 / 0.940 | -1.0 / +0.5 pt | 0.930 / 0.843 | 1.2 / 1.0 |
| 32,768 | 0.555 / 0.825 | 0.540 / 0.815 | -1.5 / -1.0 pt | 0.855 / 0.854 | 23.8 / 21.1 |
| 100,000 | 0.700 / 0.895 | 0.695 / 0.875 | -0.5 / -2.0 pt | 0.875 / 0.825 | 73.7 / 65.4 |

The gate (coordinator) allowed no degradation beyond ~3 points against shipped at 32K and 100K. **Result: pass.**
- The deltas stay within 2 points at every length.
- The TT-vs-TT top-5 overlap at 100K equals the overlap at 2K (0.83-0.85).

This check ran on the s8_snapshot code. There is **no accuracy evidence beyond 100K tokens**. The capacity probes below
(311K and 325K tokens) check function (copy accuracy), not next-token accuracy.

The shipped run needed `GLM53_DSA_INDEX_BLOCK_MB=32` (indexer score blocks; bit-identical, stage 6). With the default
64 MB, the shipped policy at max_seq_len 102,400 ran out of DRAM in the DSA indexer at the 32K prefill
(`evidence/longctx_shipped_try1_oom.excerpt.txt`). The selected policy ran with the default; it has 1 GB more free
DRAM.

## Capacity and context contract

`tools/memory_probe.py capacity` follows the serving path:
1. load the weights and measure free DRAM;
2. size the DSA pool from it;
3. warm up and capture the traced decode;
4. prefill one non-aligned copy prompt that fills the pool;
5. run 24 traced decode steps at the top.

Settings: 1 slot, 64 MiB trace region, `GLM53_PRECISION` unset. The probe sizes the pool as
`(free after load - workspace) // bytes per block`, capped at the `--max-context` tokens. A larger step therefore
needs a smaller workspace.

**Final weights (`combo_a_kv_dense4_g4`, review #3).** Code: `s8r3b_snapshot` (HEAD 9fcc17c0dc4 + `tt/precision.py`
6f46fec82f0f). Free DRAM after load: 3,139,928,064 B per die. DSA cache: 6,336 B per token per die (811,008 B per
block).

| Step | `--max-context` / workspace | Options | Pool | Result |
|---|---|---|---|---|
| F1 | 337,920 / 832 MiB | MoE row split 512 | 2,641 blocks | **out of DRAM** in the warm-up prefill, DSA indexer `_select_pools` (43 MB) (`evidence/h9_cap_337920_moe512.excerpt.txt`) |
| F2 | 337,920 / 832 MiB | MoE row split 512, indexer score blocks 32 MB | 2,641 blocks | warm-up and capture pass; **out of DRAM** in the 337,859-token prefill, MoE `_routed_sorted_split` (82.6 MB, 10.3 MB per bank; largest free block 8.6 MB per bank) (`evidence/h9c_cap_337920_ix32.excerpt.txt`) |
| F3 | 325,632 / 1,000 MiB | MoE row split 512, indexer score blocks 32 MB | 2,545 blocks | **pass**: 325,571-token prompt, prefill 520.4 s, copy accuracy 1.0 (24 / 24); 224 MB free at the prefill peak |

- The options are `GLM53_MOE_SORTED_ROWS=512` and `GLM53_DSA_INDEX_BLOCK_MB=32`. Both are bit-identical row / block
  splits (stage 6), but they are not the code defaults (1,024 rows, 64 MB).
- The bracket: **325,632 passes and 337,920 fails** (3.7 % apart). The 2,048-token steps between them (327,680 to
  335,872) were not probed. A 256-row MoE split at 337,920 was not probed either (not done in review #4): its buffer is
  (2,048 + 31 x 192 + 32) rows x 8,192 B = 8.22 MB per bank, just below F2's largest free block of 8.55 MB per bank.
- The code-default auto split picks 1,024 rows (`tt/model.py` `choose_moe_split`), so the default path at 325,632 is
  not proven here; the stage-9 probe at code defaults decides it. A possible change (not made): let the auto split
  choose 512 rows and 32 MB indexer blocks when the pool is large. A step must be a multiple of 2,048 (the prefill bucket); a 324,608-token try was rejected at construction.
- Files: `capacity/final_s1_337920_ws832_moe512`, `final_s1_337920_ws832_moe512_ix32`,
  `final_s1_325632_ws1000_moe512_ix32` (each `capacity.json` + `provenance.json` with the md5 map).

**combo_a_kv weights (review #1 and #2, before the final selection).** Free DRAM after load 3,047,399,936 B per die.

| Step | Code | `--max-context` / workspace | Pool | Result |
|---|---|---|---|---|
| stage 6 (stage-7 policy) | stage 6 | 131,072 / 700 MiB | 818 blocks (free DRAM) | 104,515-token prompt, 170 s, copy 1.0; 183 MB free at the prefill peak |
| 1 | s8_snapshot | 364,544 / 700 MiB | 2,849 blocks | **out of DRAM** in the warm-up prefill (`evidence/cap_combo_a_kv_s1_try1_oom.excerpt.txt`) |
| 2 | s8_snapshot | 311,296 / 1,024 MiB | 2,433 blocks (free DRAM) | **pass**: 311,235-token prompt, 408.5 s, copy 1.0; 648 MB free after the prefill |
| 3 | b8198430147 | 337,920 / 1,024 MiB | 2,433 blocks (free DRAM) | pass, same pool as step 2 (the larger step was not reached); 836 MB free after the prefill |
| 4 | 914af5b9594 | 337,920 / 832 MiB | 2,641 blocks | **out of DRAM** in the warm-up prefill, MoE `_routed_sorted_split` (the default 1,024-row split, called from `_routed_masked`) (`evidence/r2_cap_337920.excerpt.txt`) |
| 5 | 914af5b9594 | 337,920 / 832 MiB, `GLM53_MOE_SORTED_ROWS=1024` | 2,641 blocks | the same configuration as step 4 (1,024 is the default): the same failing allocation (`evidence/r2_cap_337920_split.excerpt.txt`) |

Files: `capacity/combo_a_kv_s1*/`. Review #2 reported steps 4 and 5 as "without" and "with" the row split; that was
wrong, because step 4 already used the default 1,024-row split.

### One 1-slot number and one sizing rule

There is one sizing rule for the replicated layout. `tt/serving_profiles.py` (data `serving_profiles.json`) and
`tt/generator.plan_context` use the same formula:
- pool blocks = floor((free after load - (KDA slots - 2) x 35,651,584 B - reserve) / 811,008 B), or the measured pool
  when a capacity probe for that slot count is in `serving_profiles.json` (`measured`);
- 1-slot max_model_len = the largest multiple of 2,048 with ceil(L / 128) + 2 <= pool blocks.

With the final weights (free 3,139,928,064 B) and the 1,024 MiB serving reserve, the rule gives 2,547 blocks and
**325,632 tokens at 1 slot**. The stage-8 probe (F3) measured the same value.

| Number | Value | What it is | Governs |
|---|---|---|---|
| Stage-8 standalone capacity probe (F3) | 325,632 | measured, 1,000 MiB workspace, MoE split 512 and indexer blocks 32 MB | evidence of the largest feasible value of the replicated fp8 layout on the final weights |
| Stage-9 serving capacity (A2, `doc/vllm/capacity/s1`) | pending | measured on the serving path at the 1,024 MiB workspace, code-default options | **serving** (vLLM max_model_len, `serving_profiles.json` `measured["1"]` after `tools/serving_profiles.py ingest`) |
| Standalone generator default (`Glm53Generator` / `build_generator()`, no max_seq_len) | **280,576** now | computed by the code: the rule on the `serving_profiles.json` budget (still the combo_a_kv free DRAM, 3,047,399,936 B) with reserve max(1,024 MiB, `GLM53_DRAM_RESERVE_MIB`, default 1,200 MiB) | the standalone generator only (176 MiB more headroom than serving). It changes when stage 9 ingests the final-weights budget (then 294,912 with the 1,200 MiB reserve); `GLM53_DRAM_RESERVE_MIB=1024` gives the serving value |

- Until stage 9 ingests its probe, `serving_profiles.json` still holds the combo_a_kv budget (free 3,047,399,936 B):
  its rule gives 309,248 at 1 slot. This is conservative (the final weights leave 92 MB more per die). Stage 9 owns
  that file.
- The generator uses the `serving_profiles.json` budget only when its `precision.id` equals the default policy id;
  otherwise it falls back to the stage-7 budget (112,640 tokens at 1 slot). The stage-8 commit changed the default id
  and broke this (review #4 P1); d16e6b9c2f0 sets `precision.id` = `combo_a_kv_dense4_g4`. Host test
  `tests/test_generator_vllm_host.py::test_default_precision_uses_serving_budget` now guards it: the default must use
  the data-file budget, the 1-slot default must be >= 280,576, and at the serving reserve the generator must accept
  the data file's 1-slot value.
- If the stage-9 probe at code defaults fails at 325,632, the serving value is lower than the stage-8 number. The
  stage-8 number then stays the largest feasible value with the two split options set.

All stage-8 decode speeds are at max_seq_len 16,384. The indexer scores every allocated pool slot, so decode at the
served max_model_len is about 10 % slower (`doc/context_contract.json` `full_model.context_size_cost`). Stage 9
re-measures token-out at the served max_model_len.

The other slot counts follow the same rule (stage 9 measures each serving profile).

`doc/context_contract.json` (stage-8 sections: top-level `supported_context`, `capacity_evidence`,
`limiting_reason_detail`, and `datatype_sweep`) records these values. The `capability_reduction` section and the
block-sharded layout are owned by the DSA-1M work (another agent) and are not changed here.

**Block-sharded layout.** `GLM53_DSA_SHARD=1` with the committed `tt/dsa_shard.py` keeps a bf16 latent (3,696 B per
token per die; a build that asks for fp8 logs a warning; readiness 92 / 100 / 100 %). Another agent is
re-implementing the sharded layout with an fp8 latent, with its own capacity probe; its 1-slot capacity is not
measured here.

## Default and test policy

- `GLM53_PRECISION` unset = `selected_precision_config.json`. It is consumed by every path that builds `Glm53Model` /
  `MultichipDecoder`: `build_generator`, the readiness runners and the full model. The vLLM adapter
  (`tt/generator_vllm.py`, committed with the stage-9 host files in de9852c604d) sets `GLM53_PRECISION` to the
  selected file when unset; that it serves the same policy is stage 9's propagation check.
- Decode programs (`decode_specs`) apply only to layers of the kind they drive (e.g. `kda_in` only on KDA layers), and
  `precision.applied` logs only those.
- With `GLM53_HEAD=lm_ds`, the LM head requant is skipped with a warning (the column-split weights are not
  re-quantized).
- Host test `tests/test_precision_host.py` (10 tests) covers:
  - the default resolves to the selected file;
  - `none` / `shipped` / `off` / `0` resolve to stage 7;
  - the default and explicit-file paths give the same cache files (real `WeightCache` path layout);
  - the cache name changes with memory config, padded shape, dtypes and component digest;
  - exception layers are not cached;
  - the dsa_shard fallback of the committed `tt/precision.py` (fp8 request -> bf16 latent with a warning). The DSA-1M
    work (stage 6, another agent) replaces that branch with an fp8 sharded latent and owns its test.

| Suite / tool | Policy |
|---|---|
| tests/test_multichip_decoder.py, test_dsa_shard.py, test_perf_decoder.py | stage 7 through a module-scoped autouse fixture (`GLM53_PRECISION` = the caller's value or `none`, no leak into other modules): they compare against stage-3/5/7 references |
| tools/mc_decode_ab.py, dsa_shard_probe.py, s5_sparse_probe.py, baseline_dump.py | stage 7 (`GLM53_PRECISION` defaults to `none` at the top of the tool) |
| tests/test_model_probe.py, tools/model_step_probe.py, tools/memory_probe.py, readiness / full-model / vLLM paths | the default (`combo_a_kv_dense4_g4`; the vLLM path refuses `none`): the model-level gates and capacity test the shipping policy |
| tools/dtype_sweep.py, tools/long_ctx_check.py | whatever `GLM53_PRECISION` says (recorded) |

## Limitations

- Accuracy evidence is 300 teacher-forced positions on three AIME24 chat prompts, plus the long-context check to 100K
  tokens. There is no accuracy evidence beyond 100K tokens; the capacity probes check function (copy accuracy).
- Top-1 differences of 1-2 points between finalists are at the noise level of 100-position references. The selection
  margin is one token (min top-1 91 %).
- `teacher_forcing.json` stores counts, not per-position predictions, so "equal results" between two builds means
  equal counts, not proven bit-identical outputs.
- The 18-candidate sweep and the long-context check ran on the s8_snapshot code. The selection and its gate ran on
  HEAD 9fcc17c0dc4 (`_h9`). Speeds compare only within one code generation.
- The long-context check covers combo_a_kv, not the dense bfp4 layers and decode programs of the final selection.
- The 1-slot capacity is bracketed, not exact: 325,632 passes, 337,920 fails; the steps between were not probed. The
  passing probe sets two non-default split options (MoE rows 512, indexer score blocks 32 MB). The serving value at
  code defaults is the stage-9 probe.
- The free-run HF control of the selection diverges more before the loop than earlier runs (first 40 tokens: 67.5 %
  HF top-1 agreement vs 80-85 %; the loop is entered through rank-5 and rank-21 picks). The chat-prompt control shows
  no regression against the previous default (153 vs 156; Result).
- Not done (recorded for stage 9 or later): the 256-row MoE split and the steps between 325,632 and 337,920; a long-
  context rerun of the final selection; a split of the LM-head program's own cost; `combo_a_all` with the fp8 latent
  (optional: combo_a_all fails the gate and is slower); a chat-domain HF score trajectory.
- All performance numbers are for 1 slot at max_seq_len 16,384. Multi-slot decode and the decode programs at 8 / 32
  slots were not measured per candidate (stage 9).
- The fp8 +0.17-0.19 ms inside the combinations is mostly not in the device op time; the cause is open.

## Files

- `selected_precision_config.json`: the selected policy (weights, layer ranges, fidelities, decode programs,
  activations, CCL, KV cache, logits / sampling, flags). Read at runtime from `doc/` by `tt/precision.py`: the
  stage-11 bundle must ship it.
- `configs/*.json`: every candidate.
- `runs/<run>/record.json`: per-run record (policy readback, applied overrides, fidelity counts, code md5 and
  provenance, DRAM, readiness, non-aligned check, TTFT, decode).
- `runs/<run>/{teacher_forcing,prefill_check}.json` and `runs/<run>/<ref>/`: the readiness reports.
  `runs/final_default_h9/checks/`: the qualitative and autoregressive checks of the selection (with the HF control).
- Review-#3 runs (HEAD 9fcc17c0dc4): `runs/final_default_h9/` (the gate of the default), `runs/final_default_h9_warm/`
  (warm rebuild, token-out), `runs/kdain_default_h9/` (the review-2 default on the same code), and the reselection
  `runs/{combo_a_kv_g4,combo_a_kv_g4_rest,combo_a_kv_dense4_g4,combo_a_kv_dense4_kdain,combo_a_all}_h9/`; config
  `configs/combo_a_kv_dense4_kdain.json`.
- Review-#3 capacity: `capacity/final_s1_337920_ws832_moe512/`, `capacity/final_s1_337920_ws832_moe512_ix32/`,
  `capacity/final_s1_325632_ws1000_moe512_ix32/`; excerpts `evidence/h9_cap_337920_moe512.excerpt.txt`,
  `evidence/h9c_cap_337920_ix32.excerpt.txt`.
- `sweep_results.json`, `sweep_results.csv`, `min_top1_perf.png`, `top1_perf_pareto.png`, `top5_perf_pareto.png`
  (`tools/dtype_sweep_report.py --selected final_default_h9 --code-ref HEAD`; it warns when the selected run's md5 map
  differs from `tt/*.py` at that git ref, or at the working tree without `--code-ref`).
- `capacity/`, `long_context/`, `refs/` (p1, p2 references and metadata, the long-context document ids).
- `geometry/` (the bfp4 decode-program sweep and its summary), `profile/` (the combo_a vs combo_a_kv Tracy windows:
  perf reports and `compare_decode.json`). The raw Tracy op CSVs (3.2 MB each) are not in the repo:
  `~/glm53-flash-bringup/logs/s8/{combo_a,combo_a_kv}_ops.csv.gz`.
- `evidence/`: `kv_fp8_probe.json` and the OOM / crash log excerpts named in the text.
- Code: `tt/precision.py`; the hooks in `tt/model.py` and `tt/multichip_decoder.py`; the fp8 pool in `tt/dsa.py`;
  `tt/dsa_shard.dsa_bytes_per_token`.
- Tools: `tools/dtype_sweep.py`, `tools/dtype_sweep_report.py`, `tools/dtype_geometry_summary.py`,
  `tools/kv_fp8_probe.py`, `tools/long_ctx_check.py`, `tools/long_ctx_compare.py`; the stage-8 options of
  `tools/mc_matmul_sweep.py` (`--wdtype`, `--fidelity`, `--alt-fidelity`, `out_sha`, role `lm_head`). Test:
  `tests/test_precision_host.py`.
- Scripts: `~/glm53-flash-bringup/scripts/s8_*.sh`, `s8r1_inner.sh`, `s8r2{a,b,c}_inner.sh`, `s8r3{a,b,c}_inner.sh`,
  `s8_hfscore*.sh`. Logs: `~/glm53-flash-bringup/logs/s8/`. Work log: `work_log.md`.

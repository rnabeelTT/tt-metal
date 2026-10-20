# GLM-5.3-Flash (E192) datatype sweep: work log (stage 8)

All times UTC, 2026-10-18. Hardware: p300x2 (4 Blackhole dies), mesh (1, 4), FABRIC_1D, TP4. Branch `glm_5_3_flash`
(HEAD ac03689d2bb plus the uncommitted stage-8 files). Device jobs ran through `~/glm53-flash-bringup/scripts/run-locked.sh`
(one shared host lock with the stage-3 rework jobs that ran at the same time).

## 01:30 start

- Read the `datatype-sweep` skill and the stage-8 goal prompt. The plugin has no `08-*.check.sh`; the goal prompt's
  completion list is the gate.
- Starting state: shipped policy `tt/optimized_decoder.SELECTED = opt_s5_pf`; full-model readiness on it (stage 7):
  teacher forcing 92 / 100 / 100 %, prefill check 91 / 100 / 100 %, 1-slot context 104,448 tokens.
- Weight cache: the shipped cache directory had the key `opt_s5` (written before SELECTED was renamed to
  `opt_s5_pf`). Both names have the same weight-dtype digest (56b50afe6f), and at TP4 they build identical weights
  (`single_die_mc` and `prefill_cfg` act only at tp == 1). A build with `opt_s5_pf` would have written a second 116 GB
  directory. Action: symlink `~/.cache/glm53_flash/v1-mesh1x4-opt_s5_pf-56b50afe6f-ckpt5bbd7a14bd` ->
  `v1-mesh1x4-opt_s5-56b50afe6f-ckpt5bbd7a14bd`. No candidate writes a new cache directory (see below).

## 01:40 candidate mechanism (no new weight cache per candidate)

`tt/precision.py` `PrecisionOverrides`, selected with `GLM53_PRECISION=<json>` (unset = shipped policy, no change):

- `requant`: host round-to-nearest bfp4 of the loaded (cached, bfp8) weights of a group, per die, same memory config.
- `fidelity` / `fidelity_layers`: compute fidelity of a group's decode-row matmuls (KDA in / out, DSA q_a|kv_a, q_b,
  o_proj, dense MLP, LM head) or of all rows (MoE `shared_fidelity`, `routed_fidelity`, which are OptConfig fields).
- `moe_layer_exceptions`: per-layer MoE OptimizationConfig fields; a layer with a dtype exception is built from the
  checkpoint (no weight cache); every other layer loads from the shipped-policy cache.
- `ccl_h_dtype`: the decode residual all-gathers in bfp8.
- `kv_latent_dtype`: the DSA latent pool in fp8 e4m3 rows (`tt/dsa.py`).

Hooks: `tt/multichip_decoder.MultichipDecoder.from_state_dict` (per layer) and `tt/model.Glm53Model.__init__` (LM head,
one line). Every candidate is a json file in `configs/`.

## 01:45 KV-cache dtype probe (`tools/kv_fp8_probe.py`, `~/glm53-flash-bringup/logs/s8/kv_fp8_probe.log`)

BFP8 is a TILE-only format; the DSA latent pool is a ROW_MAJOR paged row store (`scatter_rows`, `sparse_sdpa`), so a
bfp8 pool does not fit the contract. `sparse_sdpa` accepts FP8_E4M3 rows. One die, GLM geometry (K = V = 512,
2048 selected keys):

| Step | Result |
|---|---|
| typecast TILE bf16 -> fp8_e4m3 | TT_FATAL (fp8 is ROW_MAJOR only) |
| typecast ROW_MAJOR bf16 -> fp8_e4m3 | ok, PCC 0.99961 vs bf16, equals torch float8_e4m3fn within 1 ulp |
| scatter_rows of fp8 rows into an fp8 pool | ok |
| sparse_sdpa fp8 pool vs bf16 pool, 32 rows / 128 rows | PCC 0.99958 / 0.99958 (bf16 pool vs torch 0.99981) |

Implemented as candidate `kv_fp8` (row-major cast before `scatter_rows`, kv_format FP8_E4M3).

## 01:55 reduced-model smoke (layers 0, 3, 4; `logs/s8/smoke_all.log`, `smoke_shipped.log`)

One candidate with every override at once (bfp4 KDA in / out, DSA, shared expert, LM head; LoFi / HiFi2 / HiFi3
fidelities; bfp8 down exception on layer 3; bfp8 CCL; fp8 latent): it builds, the runtime readback shows every applied
dtype and fidelity (`S8 POLICY`, `S8 APPLIED`), traced and eager prefills match for 100- and 161-token prompts, decode
traces replay. The same layers on the shipped policy read back the shipped dtypes with no override applied.

## 02:05 code pinning (other agents edit the live tree during the sweep)

The stage-3 and stage-6/7 rework agents changed `tt/*.py` and the mhc_split_sinkhorn C++ while the sweep was queued.
To keep every candidate on identical code, the sweep runs from `~/glm53-flash-bringup/s8_snapshot` =
`git archive ac03689d2bb models/` + the stage-8 files (`tt/precision.py`, the `tt/dsa.py` fp8-pool edit, the
`tt/model.py` and `tt/multichip_decoder.py` hooks, `tools/dtype_sweep*.py`, `tools/kv_fp8_probe.py`). Kernels and the
ttnn build come from the repo (`TT_METAL_HOME`); the live mhc_split_sinkhorn kernel diff against HEAD is inert here
(only `#ifdef SK_FACES` paths, enabled by `MHC_SINKHORN_FACES=1`, which no stage-8 job sets).

First attempt (02:22, `logs/s8/shipped_LEAKED_livetree.log`): the job ran `python -m` with cwd = the live tree, so
`models` resolved to the live tree, not the snapshot (`python -m` puts the cwd before PYTHONPATH). Teacher forcing
6 / 28 / 99 %, TTFT 541 ms at 128 tokens: the uncommitted stage-6/7 rework edits of that moment, not the shipped code.
Fix: `env -C <snapshot>` for every job, `S8_REQUIRE_PKG` (the job exits if the package resolves elsewhere) and the md5
of every `tt/*.py` in each record (`code`).

## 02:50 shipped baseline on the pinned code (`logs/s8/shipped.log`, `runs/shipped/`)

Teacher forcing 92 / 100 / 100 % (198 trace replays during the 99 decode steps: model + sampler trace per step),
prefill check 91 / 100 / 100 %, TTFT 230 / 375 / 1035 ms (128 / 161 / 2048 tokens, warm, device logits), model trace
53.69 ms, token-out 54.08 ms, generate 54.18 ms (18.46 t/s/u). Identical to stage 7 (`doc/optimized_full_model`).

## 03:00 - 07:25 single-group candidates (`logs/s8/<id>.log`, `runs/<id>/`)

Device queue: other stages' jobs held the device for long stretches; from 05:00 the stage-8 candidates ran as one
device-queue ticket (`scripts/s8_batch.sh`: several candidates back to back, `tt-smi -r` between them). Every run passes
the top-1 / top-5 / top-100 gate (90 / 98 / 100 %); traced and eager prefill continuations are equal for the 100-, 161-
and 300-token prompts in every run (non-aligned lengths). Token-out ms per step (model + sampler trace, 1 slot):

| id | TF top-1 / 5 / 100 | prefill check top-1 | token-out ms | vs shipped |
|---|---|---|---|---|
| shipped | 92 / 100 / 100 | 91 | 54.08 | - |
| kv_fp8 | 92 / 100 / 100 | 91 | 54.09 | +0.01 |
| kda_in4_lofi | 91 / 100 / 100 | 90 | 53.71 | -0.37 |
| kda_in4_hifi2 (07:35) | 91 / 100 / 100 | 90 | 53.73 | -0.35 |
| dsa4_lofi | 94 / 100 / 100 | 95 | 53.79 | -0.29 |
| kda_out4_lofi | 94 / 100 / 100 | 94 | 53.88 | -0.20 |
| proj8_lofi | 93 / 100 / 100 | 95 | 53.93 | -0.15 |
| lm4_lofi | 93 / 100 / 100 | 91 | 53.94 | -0.14 |
| shared4_lofi | 94 / 100 / 100 | 94 | 54.00 | -0.08 |
| proj8_hifi2 | 93 / 100 / 100 | 94 | 54.04 | -0.04 |
| lm8_lofi | 92 / 100 / 100 | 91 | 54.09 | +0.01 |
| exp8dn_edge | 94 / 100 / 100 | 94 | 54.09 | +0.01 (context 69,632) |
| routed_hifi3 | 93 / 100 / 100 | 94 | 54.43 | +0.35 |
| ccl_h_bfp8 | 92 / 100 / 100 | 91 | 54.55 | +0.47 |

Run-to-run noise between builds of numerically equal decode programs: shipped / lm8_lofi / kv_fp8 / exp8dn_edge read
54.08 - 54.09 ms (about 0.02 ms).

## 07:30 - 08:40 combinations (one ticket)

| id | TF top-1 / 5 / 100 | prefill check | token-out ms |
|---|---|---|---|
| combo_a (kda_in4 + kda_out4 + dsa4 + shared4 + lm4, inner layers, LoFi, dense MLP LoFi) | 90 / 100 / 100 | 91 / 100 / 100 | 53.03 |
| combo_a_kv (combo_a + fp8 latent) | 91 / 100 / 100 | 91 / 100 / 100 | 53.22 |
| combo_b (combo_a without kda_in4) | 94 / 100 / 100 | 94 / 100 / 100 | 53.41 |
| combo_a_all (combo_a on layers 0 and 44 too) | **14 / 32 / 95** | 90 / 100 / 100 | 53.02 |

combo_a_all breaks decode (greedy continuations degenerate into token 154842 loops) while prefill holds: bfp4 in the
first / last layer is rejected, as the skill's default exclusion expects. combo_a sits exactly on the top-1 gate; two
further AIME24 chat references (prompt indices 1 and 2, `tools/hf_full_ref.py --mode readiness`, CPU) are generated to
separate combo_a / combo_a_kv / combo_b / shipped on 300 positions before selection.

## 08:40 - 10:40 three references for the finalists

The p0 gap between the finalists (90 - 94 % top-1) is 1 - 4 tokens of 100. Two more AIME24 chat references were made on
the CPU HF model (`tools/hf_full_ref.py --mode readiness --aime24-prompt-index 1 / 2 --gen-len 100`, 8 threads,
~28 min each; `refs/readiness_aime24_chat_p{1,2}.refpt`). The finalists ran again in one ticket (`runs/<id>_r3/`):
teacher forcing and prefill check on p0, p1, p2 on the same build (`tools/dtype_sweep.py --do ...,extra`).

| id | TF top-1 p0 / p1 / p2 | PC top-1 p0 / p1 / p2 | min top-1 | token-out ms |
|---|---|---|---|---|
| shipped | 92 / 98 / 91 | 91 / 97 / 91 | 91 | 54.07 |
| kv_fp8 | 92 / 97 / 93 | 91 / 97 / 91 | 91 | 54.09 |
| combo_b | 94 / 93 / 92 | 94 / 95 / 91 | 91 | 53.41 |
| combo_b_kv | 95 / 94 / 90 | 94 / 95 / 91 | 90 | 53.58 |
| combo_a | 90 / 94 / 91 | 91 / 95 / 91 | 90 | 53.04 |
| combo_a_kv | 91 / 96 / 91 | 91 / 95 / 91 | 91 | 53.22 |

Selection rule (coordinator): min top-1 >= 91 over the three references, top-5 / top-100 on all three (all 100 %), then
context, then speed. Passing: combo_a_kv, combo_b, kv_fp8, shipped. combo_a_kv has the most context and is the fastest
of them: **selected**.

combo_a_all on the stage-6/7 rework files (s8_snapshot2, `TT_METAL_TRACE_ALLOC_TRACKING=1`): crashed in the warm-up
prefill (`dsa.py _set_open_pool`, "Input Tensor is not allocated") before any measurement, in the then-current rework
copy of `tt/dsa.py`; reported to the coordinator. The classification run waits for a known-good rework snapshot.

## 10:30 - 12:40 capacity, long context, post-selection

- Re-quantized tensors are now cached (`tt/precision.py _rq`): `<cache root>/layer_<i>/prec-<requant source digest>/
  <group>-<dtype>.tensorbin` (and `top/` for the LM head). Warm load of the selected default: 108 s (shipped 89 - 91 s).
- Capacity probe, try 1 (`logs/s8/cap_combo_a_kv_s1_try1_oom.log`): `--max-context 364544 --workspace-mib 700`.
  After load 3.05 GB free per die (shipped: 1.99 GB); the pool took 2,849 blocks; the warm-up prefill then ran out of
  DRAM in the MoE (`_routed_masked`, 82.5 MB buffer, 15 MB free per bank): the per-max_context DSA constants grow
  with T, so 700 MiB of workspace is too small at 364K. Try 2: `--max-context 311296 --workspace-mib 1024`.
- Long-context check (`tools/long_ctx_check.py`, `long_context/`): the shipped run at max_seq_len 102,400 ran out of
  DRAM in the DSA indexer (`_select_pools`) at the 32K prefill with the default `GLM53_DSA_INDEX_BLOCK_MB=64`
  (`logs/s8/longctx_shipped_try1_oom.log`, reported to the coordinator); rerun with 32 MB (bit-identical blocks,
  stage 6).
- Post-selection on the flip snapshot (`~/glm53-flash-bringup/s8_snapshot_flip`, `GLM53_PRECISION` unset =
  `selected_precision_config.json`; `runs/selected_default*`): readiness p0 / p1 / p2 TF 91 / 96 / 91, PC 91 / 95 / 91
  (top-5 / top-100 100 %), non-aligned check equal, TTFT 227 / 373 / 1026 ms, token-out 53.23 ms, qualitative suite
  coherent (5 of 6 end on EOS; q3 reaches the 2048-token budget as in stage 7), autoregressive degenerate check exit 0;
  warm rebuild 108 s, token-out 53.23 ms; `GLM53_DSA_SHARD=1` falls back to a bf16 latent with a warning (readiness
  92 / 100 / 100 %). Host test `tests/test_precision_host.py`: 5 passed.

## 13:45 - 14:20 capacity (try 2) and long-context result

- Capacity, `--max-context 311296 --workspace-mib 1024`: pool 2,433 blocks, 311,235-token copy prompt in 408.5 s,
  copy accuracy 1.0, 648 MB per die free after the prefill: max_model_len 311,296 at 1 slot
  (`capacity/combo_a_kv_s1/capacity.json`). Derived 4 / 8 / 16 / 32-slot values in `doc/context_contract.json`
  (`datatype_sweep`).
- Long-context check: true-next-token top-1 shipped vs selected 0.840 / 0.830 (2K), 0.555 / 0.540 (32K),
  0.700 / 0.695 (100K); TT-vs-TT top-5 overlap 0.84 / 0.85 / 0.83. Pass (no degradation beyond 2 points).
- README, selected_precision_config.json, sweep_results.{json,csv}, Pareto charts written. The default flip is staged
  in `~/glm53-flash-bringup/s8_snapshot_flip` for the coordinator's stage-8 commit (not applied to the live tree).

## Review #1 rework (commit b8198430147 reviewed: MORE-WORK-NEEDED)

- P1: the commit took HEAD's `tt/multichip_decoder.py` (stage-5 defaults: sinkhorn_faces, dsa_q_mm, dsa_tail_shared,
  kda_conv_max_batch=16) plus the `root_cache` hook; that combination was not measured before. New snapshot
  `~/glm53-flash-bringup/s8r1_snapshot` = `git archive b8198430147 models/` + the review-#1 `tt/precision.py`,
  `tools/dtype_sweep.py`, `tools/long_ctx_check.py`, `tests/test_precision_host.py`, `configs/combo_a_kv_dense4.json`
  (+ `tools/memory_probe.py` loading the weight cache). Only `tt/precision.py` differs from the commit (md5 check in
  the records). One device ticket (`scripts/s8r1_inner.sh`, 15:57 - 16:29 UTC):
  - `selected_default_r1` (GLM53_PRECISION unset, cold precision cache: 180 tensors stored): TF top-1 p0 / p1 / p2
    91 / 96 / 91, PC 91 / 95 / 91, top-5 / top-100 100 %, 198 trace replays, non-aligned check equal, TTFT
    226 / 377 / 1026 ms, token-out 51.14 ms, qualitative 5 / 6 EOS (q3 budget), autoregressive degenerate check exit 0,
    one eager step's fidelity counts kda_in / kda_out LoFi x 34, DSA q_a|kv_a / q_b x 33, DSA o x 11, dense MLP x 3.
  - `selected_default_r1_warm`: load 92.1 s (cold 147.8 s), token-out 51.13 ms.
  - `shipped_r1` (GLM53_PRECISION=none, same code): 92 / 98 / 91, PC 91 / 97 / 91, token-out 52.20 ms. Selected:
    -1.06 ms (-2.0 %).
  - `combo_a_kv_dense4_r1` (P2-5, dense MLP bfp4 LoFi on layers 1-2): 92 / 95 / 92, PC 92 / 95 / 91, min 91,
    token-out 51.13 ms, 37 MB per die freed (~+5.8K tokens, estimate). Coordinator: keep combo_a_kv (rule 3: an
    estimated context gain below ~5 % does not override a config with fewer bfp4 groups at equal speed).
  - Capacity probe at `--max-context 337920 --workspace-mib 1024`: free DRAM again sized the pool to 2,433 blocks;
    311,235-token copy prompt, copy accuracy 1.0, 836 MB per die free after the prefill. 311,296 stays the measured
    value ("next step not reached").
- P2-1: combo_a_all reworded to "decode collapse, cause unclassified" (README, selected_precision_config.json):
  `</think>` (154842) loops, deterministic; prefill check 90 / 100 / 100 %; the TF 14 / 32 matches non-precision
  decode-collapse signatures (stage-7 ag_persistent, the open prefill->decode race). Classification runs wait for the
  race fix.
- P2-2: the re-quantized tensor cache name now has a digest of the owning component's source digest and the source
  tensor's memory config / padded shape / dtype; a loaded file is checked against them and rebuilt on mismatch;
  exception layers cache nothing. Host tests: 10 passed.
- P2-3: `doc/context_contract.json` top level = 311,296 (stage-7 values under `stage7_policy_GLM53_PRECISION_none`).
- P2-4: `GLM53_PRECISION` defaults to `none` (one line each, coordinator-approved) in tests/test_multichip_decoder.py,
  test_dsa_shard.py, test_perf_decoder.py and tools/mc_decode_ab.py, dsa_shard_probe.py, s5_sparse_probe.py,
  baseline_dump.py.
- P2-6: records now hold argv, GLM53_* / MESH environment, calling script, base commit and the md5-map digest;
  `sweep_results.*` gained code / requant / fidelity / kv / ccl / precision_env / script columns; older commands are
  marked `[reconstructed]`.
- P3: stale comments fixed (tt/model.py, tt/multichip_decoder.py, tools/dtype_sweep.py, tools/long_ctx_check.py);
  `evidence/` (kv_fp8 probe json, OOM and crash excerpts); `min_top1_perf.png` with the 91 % line; report gate checks
  prefill-check top-1 too; trace replays must equal 2 x decode steps.

## Review #2 rework (914af5b9594 reviewed: MORE-WORK-NEEDED, 2 P2 + 12 P3)

Snapshots `~/glm53-flash-bringup/s8r2_snapshot` (`git archive 914af5b9594 models/` + the review-2 edits) and
`s8r2b_snapshot` (+ the LM-head program hook, the `*_g4*` configs). Three device tickets, QPRIO unset:
- Ticket A (17:30 - 18:05): `tools/mc_matmul_sweep.py --wdtype bfp4 --fidelity LoFi --alt-fidelity HiFi2
  --no-prefill` on all bfp4 groups and the LM head (`geometry/`). At bfp4, ttnn's default kda_in program takes 151 µs,
  `1d:c29:ib4` takes 74 µs. Every family's fastest buildable program had an output hash identical to the current one.
  Tracy windows combo_a vs combo_a_kv (`profile/`). Capacity step 337,920 / 832 MiB: out of DRAM in the warm-up prefill.
- Ticket B (19:20 - 19:50): the HEAD gate. All bfp4 programs (`selected_default_r2`): p2 TF 89 % (fails), 50.01 ms.
  No programs (`combo_a_kv_nospecs_r2`): 91 / 96 / 91, PC 91 / 95 / 91, 51.06 ms. dense4 with all programs: decode
  collapse (TF 14 / 8 / 13 %). Stage-7 policy (`shipped_r2`): 92 / 98 / 91, 51.96 ms. Capacity 337,920 / 832 MiB with
  the MoE row split: out of DRAM in the warm-up prefill.
- Ticket C (to 21:12): kda_in program alone (`combo_a_kv_g4_kdain_r2`): 91 / 96 / 91, PC 91 / 95 / 91, 50.38 ms
  (-3.0 % vs 51.96); all programs but kda_in (`combo_a_kv_g4_rest_r2`): p2 89 %. Coordinator rule: ship combo_a_kv +
  the kda_in program; the rest are "op-level hash identical, full-model numerics shift; not adopted".
- `selected_precision_config.json` now carries `decode_specs: {kda_in: 1d:c29:ib4, layers inner}`; its policy fields
  equal `configs/combo_a_kv_g4_kdain.json`.
- CPU: HF score-trajectory of the r1 autoregressive trajectory: HF top-1 93.75 %, top-5 100 % over 256 tokens. The
  first try was OOM-killed at a 32 GB expert cache; the retry used 16 GB with MemoryMax=80G.
- Code: `tt/precision.py` (wrappers keep packer_l1_acc / fp32_dest_acc_en / approx mode; the pack digest covers the pack
  build code; the ImportError-only fallback is logged; lm_ds skip; `decode_specs` incl. the LM head and a pack rebuild),
  `tools/mc_matmul_sweep.py` (stage-8 options), `tools/dtype_geometry_summary.py`, `tools/dtype_sweep.py`, `tools/
  long_ctx_check.py`, `tools/dtype_sweep_report.py` (one marker and frontier per code generation, measured capacity
  per policy), the three test files (module-scoped autouse fixture instead of the collection-time pin).
- `doc/context_contract.json`: capacity steps, the 1,024 MiB serving workspace, the corrected
  `capability_reduction.reason` (6.64 GB at 6,336 B per token; not a hard device limit).

## Review #3 rework (aa104284595 reviewed: MORE-WORK-NEEDED, 4 P2 + 12 P3)

Code: HEAD 9fcc17c0dc4 (it holds the logits-pad fix 46b2a5a601d). Snapshots `~/glm53-flash-bringup/s8r3_snapshot`
(`git archive 9fcc17c0dc4 models/` + `configs/combo_a_kv_dense4_kdain.json` + the review-3 `tt/precision.py`) and
`s8r3b_snapshot` (+ `tt/precision.py` md5 6f46fec82f0f + `selected_precision_config.json` = combo_a_kv_dense4_g4).
Three device tickets, `scripts/s8r3{a,b,c}_inner.sh`:
- Ticket A (18:05 - 18:32 UTC, P2-1): the reselection on the fixed code, 3 references each, max_seq_len 16,384.
  - `combo_a_kv_g4_h9` (all programs): TF 91 / 96 / 89, PC 91 / 95 / 91, 50.05 ms. Fail. The p2 shift stays on the
    fixed code, so it is real numerics of those programs on the combo_a_kv weights, not the logits-pad bug.
  - `combo_a_kv_g4_rest_h9`: TF 91 / 96 / 89, PC 91 / 95 / 91, 50.77 ms. Fail.
  - `combo_a_kv_dense4_g4_h9`: TF 92 / 95 / 92, PC 92 / 95 / 91, 49.95 ms. Pass, fastest: selected.
  - `combo_a_kv_dense4_kdain_h9`: TF 92 / 95 / 92, PC 92 / 95 / 91, 50.32 ms. Pass, slower.
  - `combo_a_all_h9`: TF 91 / 95 / 90, PC 90 / 96 / 89, 51.04 ms. Fail (PC p2 89 < 90).
  - `kdain_default_h9` (the review-2 default, same code): TF 91 / 96 / 91, PC 91 / 95 / 91, 50.40 ms.
  - Both earlier decode collapses (dense4_g4 14 / 8 / 13, combo_a_all 14 / 32 / 95) were the logits-pad lifetime bug.
    No bisect was necessary: no collapse remains.
- Ticket B (18:34 - 18:56, P2-2): the final gate on HEAD, `GLM53_PRECISION` unset (`final_default_h9`):
  - TF 92 / 95 / 92, PC 92 / 95 / 91, top-5 / top-100 100 %, 198 trace replays, non-aligned check equal;
  - TTFT 226 / 378 / 1024 ms (128 / 161 / 2048 tokens);
  - token-out 49.94 ms = 20.02 t/s/u (warm rebuild 49.96 ms), -3.9 % vs the stage-7 policy (51.96 ms, `shipped_r2`);
  - qualitative suite: 6 prompts, coherent; q3 reaches the budget as in stage 7; q5 is the HF behaviour;
  - autoregressive check: two runs bit-identical, degenerate check exit 0;
  - HF score trajectory (CPU, `hfscore_h9`): top-1 94.9 %, top-5 99.2 %; two picks outside the top 5 (position 22 ".",
    position 24 " sat" rank 21) at flat story-branch points. Stage 6: 97.7 / 99.6 %, worst rank 5. Recorded as a known
    observation.
  - The record md5 map equals HEAD for every `tt/*.py` (`tools/dtype_sweep_report.py --code-ref HEAD`: no warning).
  - Capacity, final weights (free after load 3,139,928,064 B per die), 512-row MoE split:
    - 337,920 at 832 MiB: out of DRAM in the warm-up prefill, DSA indexer `_select_pools` (43 MB)
      (`evidence/h9_cap_337920_moe512.excerpt.txt`).
    - 324,608 at 944 MiB: not a valid step (max_context must be a multiple of 2,048); no device result.
- Ticket C (from 18:57, P2-3): capacity with the 512-row MoE split and 32 MB indexer score blocks
  (`GLM53_DSA_INDEX_BLOCK_MB=32`, bit-identical, stage 6), stop at the first pass:
  - 337,920 at 832 MiB: warm-up and capture pass (peak op `moe_gather_add_rows`); the 337,859-token prefill runs out of
    DRAM in the MoE row split (82.6 MB buffer, 10.3 MB per bank, largest free block 8.6 MB)
    (`evidence/h9c_cap_337920_ix32.excerpt.txt`).
  - 325,632 at 1,000 MiB: pass. Pool 2,545 blocks, 325,571-token copy prompt, prefill 520.4 s, 24 traced decode steps,
    copy accuracy 1.0, 224 MB per die free at the prefill peak. Ticket end 19:27 UTC.
  - Bracket: 325,632 passes, 337,920 fails; the steps between were not probed. The rule of `tt/serving_profiles.py`
    at the 1,024 MiB serving reserve gives the same 325,632 on these weights. The stage-9 serving probe (A2, 1,024 MiB,
    code-default split options) governs the served max_model_len.
- Docs (P2-1 to P2-4 and the P3s):
  - README: Result and selection on `_h9`; capacity tables for the final and the combo_a_kv weights (steps 4 and 5
    of review #2 are one configuration); "One 1-slot number and one sizing rule"; the sweep baseline vs the in-model
    program (kda_in 90.2 µs in model, LM head with `packer_l1_acc=True`); the `precision.applied` layer-kind filter;
    199 decode steps + the prefill logits in the long-context check; 183 MB is the prefill-peak value; the fp8
    sharded latent is another agent's work; `tools/serving_profiles.py` is committed (de9852c604d); the raw Tracy
    op CSVs stay in `logs/s8/`.
  - `doc/context_contract.json`: top-level `supported_context` 325,632, `capacity_evidence`, `limiting_reason_detail`
    and `datatype_sweep` (with `context_sizing`). `capability_reduction` is not changed (DSA-1M work).
  - `tools/dtype_sweep_report.py`: `--code-ref` (md5 check against a git ref), default `--selected final_default_h9`,
    the precision_env label names the policy the default resolved to at run time, no stage-6 capacity for a run
    without a same-code stage-7 build, the final-weights capacity probes. Run:
    `python -m models.autoports.zai_org_glm_5_3_flash.tools.dtype_sweep_report --selected final_default_h9
    --code-ref HEAD` (no md5 warning).

## Review #4 rework (19a4dda4f47 + eb92932986b reviewed: MORE-WORK-NEEDED, 1 P1 + 14 P3)

Host and docs only; no device job.
- P1 (the default id switch made `tt/generator._profiles_budget` fall back to the stage-7 budget: 112,640 tokens at
  1 slot): the coordinator committed `serving_profiles.json` `precision.id` = `combo_a_kv_dense4_g4` (d16e6b9c2f0).
  New host test `tests/test_generator_vllm_host.py::test_default_precision_uses_serving_budget`: the default policy
  id equals the data file's id, the budget source is `serving_profiles.json`, `plan_context(1)` >= 280,576, and at the
  serving reserve `plan_context(1, supported_context(data), serving=True)` does not raise.
  `pytest tests/test_generator_vllm_host.py tests/test_precision_host.py`: 29 passed, 1 failed. The failure is
  `test_dsa_shard_keeps_bf16_latent`, against the DSA-1M agent's uncommitted `tt/precision.py` (fp8 sharded latent);
  that agent owns the test.
- The standalone generator default is 280,576 (the code, combo_a_kv budget, 1,200 MiB reserve), not 294,912; 294,912
  applies after stage 9 ingests the final-weights budget (README, `context_contract.json`).
- P3: the stage-6/7 citation of `final_default_h9` (their owner updates them); the free-run HF control (first
  40 tokens 67.5 % vs 80-85 %, loop entry through rank-5 / rank-21 picks, chat-prompt control 153 vs 156); the
  `measured_on` text and both config md5s (run d96fae9836a8, committed fe43daaeb567: newline + `measured_on`);
  `sweep_results` `pass` = the gate on the minimum over all references (`pass_p0` kept) and one context value per
  weight set (dense4_r1 now 325,632 measured); the p2 = 89 mechanism stated as probable, not proven; "hash identical"
  qualified to the random input; the LM-head program bound (+/- 0.05 ms, sign unknown); the -3.9 % across code
  generations (linked by kdain 50.38 / 50.40 ms); single-point `_h9` frontier.
- Not done (recorded): the 256-row split and the intermediate capacity steps, the long-context rerun, the LM-head
  split, combo_a_all with the fp8 latent, a chat-domain HF score trajectory.
- `full_model.supported_context` (311,296) in `doc/context_contract.json` is stage 6's; not changed here.

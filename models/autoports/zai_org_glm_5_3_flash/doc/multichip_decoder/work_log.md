# Stage 4 work log (multichip decoder)

Times are UTC. Logs are under `~/glm53-flash-bringup/logs/` (review #1 rework: `logs/s4r/`), workspace scripts under
`~/glm53-flash-bringup/scripts/`. Every device job ran through `scripts/run-locked.sh` (one device job at a time on the
host lock, `tt-smi -r` before and after).

## Commands

Environment: `source ~/glm53-flash-bringup/env.sh` (`GLM53_CKPT`, `TT_METAL_HOME`, python env), then from
`$TT_METAL_HOME`, with `T=models/autoports/zai_org_glm_5_3_flash/tests` and `M=models.autoports.zai_org_glm_5_3_flash`:

| What | Command |
|---|---|
| Single-die baseline (one die, policy `opt`) | `GLM53_TP=1 python -m $M.tools.baseline_dump --out ~/glm53-flash-bringup/refs/baseline_opt.pt` |
| Multichip suite, stage-4 options | `GLM53_TP=4 GLM53_MC="topology=linear,ccl_l1=0,norm_sharded=0,dsa_pack=0,pack_a_1d=0,pack_b_1d=0,dsa_1d=0,router_1d=0,ds_shared=0,prefill_2d=0" pytest -sv -p no:cacheprovider $T/test_multichip_decoder.py` |
| Multichip suite, code default (`DEFAULT_MC`) | `GLM53_TP=4 pytest -sv -p no:cacheprovider $T/test_multichip_decoder.py` |
| Tracy (per case) | `GLM53_TP=4 GLM53_DECODER=multichip TRACY_OUT=$MODEL_DIR/doc/multichip_decoder/tracy CASES='test_perf_traced_decode[dsa_moe-1] ...' bash scripts/01-tracy.sh` (`python -m tracy -r -p -v -m pytest $T/test_perf_decoder.py::<case>`, then `tt-perf-report ops.csv --start-signpost PERF_DECODE --end-signpost PERF_DECODE_END --csv perf_report.csv`; prefill: `PERF_PREFILL`) |
| Watcher (per case) | `GLM53_TP=4 GLM53_MC=<stage-4 options> GLM53_WORKER_L1_SIZE=1444864 TT_METAL_WATCHER=10 TT_METAL_WATCHER_NOINLINE=1 TT_METAL_LOGS_PATH=doc/multichip_decoder/watcher/<case> pytest -sv $T/test_multichip_decoder.py::<case>` |
| Block-sharded DSA cache vs replicated | `GLM53_TP=4 pytest -sv -p no:cacheprovider $T/test_dsa_shard.py` |
| Multichip suite DSA cases, sharded cache | `GLM53_TP=4 GLM53_DSA_SHARD=1 pytest -sv $T/test_multichip_decoder.py -k "dsa_moe or paged_permuted or single_die_baseline"` |
| Full-model capacity probe, sharded cache | `GLM53_TP=4 GLM53_DSA_SHARD=1 python -m $M.tools.memory_probe capacity --out DIR --trace-region 67108864 --workspace-mib W --max-context N` |

Workspace scripts: `s4_smoke.sh`, `s4_main.sh` (baseline dump, suite, Tracy), `postfix_all.sh` (post-fp32-fix rerun
chain: `s4_mc_suite2`, `s4_tracy2`), `s4_baseline2.sh` (baseline regeneration + compare), review #1 rework:
`s4r_p12.sh`, `s4r_shard1.sh`, `s4r_cap.sh`.

## 2026-10-16

- 17:11 `s4_smoke.log` (`GLM53_OPT=fused`, `-k "real_weights or runs_multichip"`): **1 failed, 3 passed**.
  `test_real_weights[dsa_moe]` -> `TT_FATAL sparse_sdpa_device_operation.cpp:153: H % tile_h == 0 && H >= tile_h`:
  TP4 leaves 16 DSA query heads per die and `sparse_sdpa` takes whole 32-head tiles. Fix: `tt/dsa.py::_sparse_attend`
  zero-pads the heads to 32 and slices the 16 real heads off the output (cost in the README sharding table).
- 18:50 `s4_baseline_dump.log`: `refs/baseline_opt.pt` (pre-fp32-fix kernels).
- 19:23 `s4_suite.log`: **44 passed** in 33 min (stage-4 code, pre-fix kernels; committed as
  `doc/multichip_decoder/s4_suite.log`). Single-die compare 0.999782 .. 0.999992, rms 0.99651 .. 1.00353.
- 20:11 .. 20:23 `s4_tracy.log`: 9 Tracy windows. The two 1-user KDA decode cases (`kda_dense-1`, `kda_moe-1`)
  printed no PERF line: they had failed with `TT_FATAL ... B * H must be a multiple of 32` in
  `kda.recurrent_decode` (16 local heads x 1 user). `01-tracy.sh` reported `exit=0` from the tracy wrapper and
  `s3_sweep.status` recorded "s4 tracy done 9/9": a false pass, classified in stage 5 (02:31 below). `01-tracy.sh` now
  fails a case whose pytest summary is not a pass.

## 2026-10-17

- 02:31 (stage 5) the 1-user KDA crash root-caused: `kda.recurrent_decode` needs B x H % 32 == 0. Fixed in
  `tt/multichip_decoder.py::MultichipKDA` (decode batch padded with one inactive slot), which every `MCOpts` installs,
  so the stage-4 options carry the fix.
- 04:35 `s4_mc_suite2.log` (post-fix chain, `postfix_all.sh`; `DEFAULT_MC` still equalled the stage-4 options until
  07:23): **43 passed, 1 failed**: `test_matches_single_die_baseline` -> `('kda_moe', 'decode', 0.99873)`, rms ratio
  0.99395. The baseline `refs/baseline_opt.pt` was the 2026-10-16 18:50 file, made with the pre-fix kernels (the fp32
  precision fix changed the single-die decoder), so the compare mixed kernel generations. Not recorded in the stage-4
  docs at the time (review #1 P1-2).
- 05:29 `s4_tracy2.log`: the two 1-user KDA decode windows rerun on the fixed kernels and the stage-4 options
  (overwrote `tracy/test_perf_traced_decode_kda_{dense,moe}-1_/`): 1152 / 1369 µs per pass (merged report / 3 passes).
- 05:33 `s4_baseline_dump2.log`: `refs/baseline_opt.pt` regenerated on the post-fix kernels.
- 08:26 `s4_baseline_cmp2.log` (`s4_baseline2.sh`): passed, but it set no `GLM53_MC`, so it ran the round-3
  `DEFAULT_MC`, not the stage-4 options. Its lines equal `s5_suite.log`. Relabelled in the repo as
  `doc/multichip_decoder/default_mc_baseline_cmp2.log` (review #1 P1-2).
- a0be1c3edd3: stages 4 + 5 committed. The commit message says "perf_report.csv per case is committed": the CSVs
  were gitignored (`.gitignore: *.csv`) and not in the commit; they are force-added with the review #1 rework.

## Review #1 rework (2026-10-17)

See the README sections "Review #1 corrections", "Block-sharded DSA cache" and "Watcher". Logs: `logs/s4r/`.

- 14:20 Code: `tt/dsa_shard.py` (block-sharded DSA latent cache, `MCOpts.dsa_shard` / `GLM53_DSA_SHARD=1`), hooks in
  `tt/dsa.py` (`_write_latent`, `_sparse_attend(latent_rows=)`), `tt/decode_inputs.py` (per-die decode row index),
  `tools/memory_probe.py` (sharded bytes per block), `tests/test_dsa_shard.py`. `tests/test_multichip_decoder.py`:
  quantization-aware reference (as the stage-3 rework suite), rms gate 0.005 on the single-die compare, missing
  baseline fails. Queued `s4r_p12.sh` (baseline dump, stage-4-option suite, DEFAULT_MC suite, watcher) and
  `s4r_shard1.sh`.
- 15:29 `s4r_baseline_dump.log`: `refs/baseline_opt.pt` regenerated (policy `opt`, one die).
- 15:33 `s4r_shard_ab1_layer_gate.log` (first A/B, gated on the layer update): attention-path outputs matched except
  two decode tokens (PCC 0.9927 / 0.9836) and two long synthetic prefills (0.9983 / 0.9985). Diagnosis
  (`tools/dsa_shard_probe.py`, `s4r_probe1.log`, eager steps): on the failing token every head's attention output
  o_lat agrees with the replicated layout to PCC >= 0.99994, no per-die sink weight underflows (w >= 2.8e-4), so the
  layer-level difference is a MoE router near-tie flip downstream of a ~1e-4 attention difference (the synthetic
  streams route in the near-tie regime, `tests/test_functional_decoder.py` NEAR_TIE). The A/B now gates the attention
  sub-block output and counts layer-level near-tie outliers. Side finding: the kernel's u + w (= 1 exactly in real
  arithmetic) deviates by up to 2.8 % per (row, head): `sparse_sdpa`'s P @ V sums and its softmax row sum accumulate
  differently (bf16 L1 accumulation); the replicated path normalizes by the row sum, and the sharded merge uses the
  same row sum (m = 1 - w) so the two layouts normalize alike.
- 17:53 `s4r_suite_s4opts.log` (stage-4 options, policy `opt`): **45 passed** (single-die compare 0.999847 ..
  0.999996, rms 0.99971 .. 1.00196): the 04:35 failure was the stale baseline. Copied to `doc/multichip_decoder/`.
- 19:30 `s4r_suite_default.log` (DEFAULT_MC): 44 passed, 1 failed: `test_matches_single_die_baseline` -> policy
  mismatch ('opt' baseline vs SELECTED 'opt_s5'): the stage-3 rework moved SELECTED between the two suites. HF checks
  passed (508 checks, decode means 0.9931 .. 1.0003). The compares are rerun after a baseline dump under the current
  SELECTED (`s4r_main2.sh`, baseline dumped last).
- 19:45 second A/B attempt (`s4r_shard_ab2_tap_trace_fail.log`): the test's attention tap compiled its copy program
  inside the trace capture (`!is_capturing_trace`); fixed (one eager copy first). Re-queued as `s4r_main2.sh`.
- 20:24 `s4r_shard_ab.log`: sharded vs replicated A/B **2 passed** (attention output worst PCC 0.999908, rms 0.0034;
  4 layer-level near-tie outliers, as diagnosed). 21:21 `s4r_shard_suite.log`: suite DSA cases with the sharded cache,
  16 passed. 21:5x watcher runs on the stage-4 options: 6 / 6 passed, 0 suspicious watcher lines.
- 22:2x Baseline dumped under `opt_s5` (`s4r_baseline_dump_b.log`); compares DEFAULT_MC / stage-4 options / sharded
  passed. The kda_moe decode compare needed the new per-token router near-tie allowance in
  `test_matches_single_die_baseline` (one token, HF margin 2.9e-4, row PCC 0.99645; the stage-5 rework found the same
  token flipping on the single-die side); rms gate kept.
- 23:32 .. 00:08 `s4r_capacity.log`: full-model capacity probe with the sharded cache, **281,283-token prompt, copy
  24/24**, 2199 blocks (281,344 tokens), 335 MB free at the prefill peak, prefill 426.9 s.
- 00:2x (2026-10-18) final: baseline dumped last (`SELECTED` = `opt_s5_pf`), full suite on the final `DEFAULT_MC`
  (`ag_persistent=False`): **45 passed** (`s4r_suite_default_final.log`); stage-4-option compare passed
  (`s4r_cmp_s4opts_final.log`); sharded A/B 2 passed and sharded DSA cases + compare 17 passed (`dsa_shard/*_final.log`).

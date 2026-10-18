# Stage 7 work log (optimized full model)

All device jobs: `~/glm53-flash-bringup/scripts/run-locked.sh` (shared host lock), launched as `systemd-run --user`
units. Logs in `~/glm53-flash-bringup/logs/`, probe records in `~/glm53-flash-bringup/artifacts/s7/`.

## 2026-10-17

- Read the stage-7 goal (`prompts/model_bringup_multigoal/07-optimized-full-model.txt`), skills full-model, optimize,
  tt-enable-tracing, tt-device-usage; stage-5 / stage-6 READMEs.
- New driver `tools/s7_probe.py` (one build per job): TTFT (dispatch vs total, cold / warm, traced vs eager, MoE split
  A/B), decode (model trace, sampler trace, token-out, generator loop), traced-vs-eager prefill check with repeats,
  Tracy windows, readiness runners and the qualitative / autoregressive suite on the same generator.
- `tt/model.py`: options `GLM53_HEAD` (norm_sharded, lm_ds), `GLM53_PREFILL_LENGTHS`, `GLM53_PREFILL_PLAN` (cost plan),
  `GLM53_PREFILL_TRACE` (traced first chunk), `GLM53_MOE_SPLIT_AUTO`, `GLM53_SKIP_INACTIVE`; `_row_to_host`.
  `tt/dsa.py`: `prefill_inputs(host=True)`, `defer_tail` + `write_prefill_tail` (trace-safe open-pool hand-off).
  `tt/generator.py`: `_warm_plans` (every (length, DSA path)), prefill-trace prepare / capture order, `choose_moe_split`
  before the warm prefills. `tools/full_model_checks.py`: `run_with` (shared generator, output dir).
  `tests/test_generator_vllm_host.py`: warm-plan coverage test (11 passed).
- 12:47 the coordinator's readiness run imported a work-in-progress `lm_ds` default and failed (L1 3.05 MB CBs for the
  single 38,912-column DRAM-sharded LM head). Fix: column splits; and every unvalidated option default-off from then on.
- 12:53 Tracy on the 3-layer model: report failed (device-profiler buffers overflowed). Added profiler flushes and
  `TT_TRACY_SKIP_MISSING_DEVICE_DATA=1`, moved to a 1-layer model (`*_l0`).
- 13:24 reduced (0/3/4) check, S=1: traced prefill bit-identical to eager (logits max |diff| 0, greedy tokens equal) for
  100 / 128 / 161 / 300 tokens. `tests/test_model_probe.py` 3 passed with the stage-7 options on.
- 14:08 reduced bucket costs (S=1, eager, ms): 128: 19, 256: 21, 512: 21, 1024: 36, 2048: 66.
- 14:42 full baseline S=1 (`s7_base_s1.log`): TTFT 128 232 ms, 2048 1182 ms (split) / 1060 (no split), 8192 5134 /
  4657; decode model 56.86, token-out 57.24, loop 57.36 ms.
- 14:04 Tracy l0 before/after: LM head 471 µs interleaved; lm_ds splits 10 x 66 µs -> rejected; final norm 65 -> 7 µs.
- 15:06 candidate run failed at warm-up: `NameError: fold` in `tt/fused_decoder.py` (another agent's edit in
  progress). Rerun with a reduced smoke gate first.
- 17:21 full baseline S=8 (`s7_base_s8.log`): 112.8 ms per step with 1, 4 and 8 active slots.
- 18:34 full candidate S=1 (`s7_cand_s1.log`): traced 128 222 vs eager 232 ms; 512 bucket 375 ms; 1000 tokens 752 ms;
  2048 1033 (no split) vs 1153 (split); 8192 4548 vs 5033; decode model 53.23 ms (stage-5 round-6 defaults), loop
  53.74 ms = 18.6 t/s/u. Traced prefill exact (100 / 500 tokens). Decision: 512 bucket on, prefill traces off, split
  auto on, norm_sharded on.
- 19:48 / 20:04 / 20:31 reduced S=8 repeatability: not reproducible with `ag_persistent=1` (stage-5 default), with or
  without `GLM53_SKIP_INACTIVE`; reproducible with `GLM53_MC=ag_persistent=0`. Skip timing (reduced, S=8): 1 active
  5.74 vs 7.13 ms, 4 active 6.31, 8 active 7.19.
- MTP: checkpoint layer 45 counted (7.25 B expert + 185 M other parameters): ~1.07 GB per die, rejected (README).
- 21:46 final S=1 with stage-5 `DEFAULT_MC` (ag_persistent on): teacher forcing top-1 14 % / top-5 32 % / top-100 90 %
  (prefill check 91 / 100 / 100). 22:03 same with `GLM53_MC=ag_persistent=0`: 92 / 100 / 100 (`s7_tfdiag_agp0.log`).
- 22:51 final S=1 (`s7_final_s1.log`, ag_persistent=0, weight cache): TF 92 / 100 / 100, prefill check 91 / 100 / 100,
  qualitative 6 prompts coherent, autoregressive repeat-identical, degenerate check exit 0; TTFT 128 / 161 / 512 /
  2048 / 8192 = 230 / 376 / 379 / 1035 / 4556 ms; decode 53.69 / 54.08 / 54.18 ms (18.46 t/s/u).
- 23:31 final S=8 (`s7_final_s8.log`, skip on): 83.9 / 95.7 / 114.2 ms at 1 / 4 / 8 active; repeats identical.
- Defaults: GLM53_HEAD=norm_sharded, GLM53_PREFILL_LENGTHS=128,512,2048, GLM53_PREFILL_PLAN=cost,
  GLM53_MOE_SPLIT_AUTO=1, GLM53_SKIP_INACTIVE=1 (>= 2 slots), GLM53_PREFILL_TRACE=none. Not committed (instructions).

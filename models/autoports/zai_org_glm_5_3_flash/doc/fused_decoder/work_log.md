# Fused decoder work log (stage 2)

- 2026-10-14 Op survey (`~/glm53-flash-bringup/artifacts/stage2_op_survey.md`, Explore subagent): no per-key-decay fused
  recurrence anywhere; main has the GDN (per-head decay) `fused_recurrent_gated_delta_rule` -> forked.
- 2026-10-14 `mhc_pre` op (tests pass). `rms_norm_pre_all_gather` for the mHC sum of squares rejected (CB size 8.5 MB).
- 2026-10-14 `kda.recurrent_decode` op (port of main's GDN kernel + per-key decay + in-kernel L2 norm + no-relayout
  row gathers + in-place state): 5/5 unit tests on the first build. Fused real-weight / decode-only cases pass.
- 2026-10-15 `mhc_mix` split-K op (projection + Gram partials). Bug found and fixed: non-tile-aligned row counts
  (decode batch 1) -> validation on the padded shape, per-row-count diagonal masks created at setup.
- 2026-10-15 Tracy before / after (`scripts/s2_tracy.sh`), table in README. Decode is now weight-bound.
- 2026-10-15 Bug found by the fused suite: mhc_mix finalize multiplied the Gram tile by the diagonal mask; the Gram tile also mixes the activation's padding rows (non-finite leftovers after mhc_post in the DSA layer) -> 0 * inf = NaN in the row sum (dsa_moe decode PCC 0.44). Bisected with tools/fused_bisect.py (mhc_mix off: pass). Fix: ttnn.where(mask, gram, 0) select; dsa_moe decode >= 0.99992.

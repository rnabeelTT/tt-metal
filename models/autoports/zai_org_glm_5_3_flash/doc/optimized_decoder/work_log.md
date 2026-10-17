# Stage 3 work log (optimized decoder)

## 2026-10-15 .. 2026-10-16: precision policy and program configurations

- Candidates are `OptConfig` entries in `tt/optimized_decoder.py`, each run through Tracy
  (`doc/optimized_decoder/candidates/<name>/`) and `tools/moe_scale_probe.py --policy`.
- bfp4 routed experts are required by memory (E192 at bfp8 does not fit 4 dies). LoFi and HiFi2 give bit-identical
  results on bfp4 weights, so LoFi is taken. HiFi2 left the shared expert 0.8 % low, so it stays HiFi4.
- KDA projections bfp8; DSA W_UK / W_UV bfp8 with a full-grid reuse config at decode rows (`tools/dsa_bmm_probe.py`:
  222 -> 48 µs and 381 -> 45 µs). The reuse config hung the device at 2048 prefill rows, so it applies to decode only.
- 1D program config for the KDA / DSA output projections at decode (`tools/dense_decode_probe.py`: 147 -> 100 µs).
- An `OptConfig` field inserted before `moe` once made the positional candidates run bfp8 experts. The fields now
  follow `moe`, and the Tracy runs were redone.
- The optimized suite holds the bfp4 MoE layers (3, 4) to the measured quantization floor: every PCC bar of the layer
  moves down by 0.015 (`BAR_OVERRIDES`), sequence scale 1 +- 0.03, token scale 1 +- 0.06.

## 2026-10-17: final kernels

- Suite on the fp32-exact kernels, the final stage-2 rewrites and the final stage-1 harness (y_swap near-tie check):
  43 / 43 (`s3_opt_suite3.log`).
- Tracy of the final default refreshed on the same code (`tracy/`, README "Final default").

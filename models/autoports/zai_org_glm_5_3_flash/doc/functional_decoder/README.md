# GLM-5.3-Flash (E192) functional decoder

Stage 1 of the tt-model-bringup flow for `zai-org/GLM-5.3-Flash` on Blackhole. One decoder layer on one die
(`ttnn.open_mesh_device(MeshShape(1, 1))`), real E192 weights (192 of the 288 routed experts per MoE layer, mask of
`autotrust/GLM5.3-Flash-E192-DGX-Spark` applied to the FP8 parent; see `tt/ckpt.py`).

## Layer kinds and implementation

| Kind | Layers | Attention | FFN | Test layer |
|---|---|---|---|---|
| kda_dense | 0-2 | KDA linear attention (64 heads x 128) | dense SwiGLU 12288 (clamp 10) | 0 |
| dsa_moe | 3, 7, ..., 43 | DSA: NoPE MLA (latent 512) + lightning indexer (top-512 pools of 4 + tail) | 192-expert MoE + shared expert | 3 |
| kda_moe | the other 31 | KDA | MoE | 4 |

Every sub-block is wrapped in mHC (4 residual streams, Sinkhorn 20 iterations). The residual is the packed stream
`X [1, 1, T, 4 * 4096]` fp32.

| Module | File | Device path |
|---|---|---|
| mHC | `tt/mhc.py` | `ttnn.experimental.deepseek_prefill.mhc_split_sinkhorn` (parametrization) + `mhc_post` (residual mix) |
| KDA | `tt/kda.py` | prefill: `ttKDA` (`models/demos/deepseek_v3_d_p/tt/kda`, `ttnn.experimental.kda` kernels) per constructed length, runtime `actual_start` / `actual_end`, carried `KdaState`; decode: recurrence on ttnn ops, per-slot fp32 state `[B, 64, 128, 128]` + 3 conv history rows |
| DSA | `tt/dsa.py` | PAGED caches (128-token blocks, block table `[S, T/128 + 1]`): latent block pool `[1, 1, NB*128, 512]` ROW_MAJOR (`ttnn.scatter_rows`), pooled indexer keys (TILE, 32 keys per block); `indexer_score_dsa(page_table=)` (prefill) / `ttnn.experimental.dsa_decode_scores` (decode), `ttnn.topk` 512, `ttnn.transformer.sparse_sdpa(page_table=, page_block_size=128)` for every query row on logical indices (dense rows: `[0..t]`); first dense prefill chunk: `flash_mla_prefill` |
| MoE | `tt/moe_opt.py` (base `tt/moe.py`) | sigmoid router + correction bias (fp32), gather / pairs decode, expert-sorted grouped prefill, routed experts with fp32 DST, swiglu clamp at every activation |
| dense MLP | `tt/mlp.py` | gate / up / down with the swiglu clamp |

## Contract (signatures)

```python
FunctionalDecoder.from_state_dict(state_dict, *, hf_config, layer_idx, mesh_device,
                                  max_batch=32, prefill_lengths=(128, 512, 2048), dsa_kwargs={"max_context": T})
dec.allocate_decode_state(batch)                       # before the first prefill of a DSA layer; before any capture
inputs = dec.prefill_inputs(n, start=s, length=L, slot=k)   # host -> device input construction
X1, state = dec.prefill_forward(X, inputs=inputs, attn_state=state)   # X [1,1,n,16384] fp32, rows >= L padding
dec.commit_prefill(slot, state)                        # KDA carry -> decode slot
dec.set_page_table(pt)                                 # DSA: [S, <= T/128] block ids (default: contiguous per slot)
bufs = dec.decode_inputs(positions)                    # fixed-address inputs; positions[s] = -1: slot s inactive
dec.update_decode_inputs(bufs, positions)              # between steps / trace replays
Y = dec.decode_forward(X_dec, inputs=bufs)             # X_dec [1,1,B,16384]
```

Inactive slots (position -1) keep their state: KDA beta = 0 / decay = 1 and the conv-history shift is masked; DSA
latent / pool-key writes are skipped and the open-pool online-softmax state is kept (`test_inactive_slot_state`).
Paged caches: every index list is logical; the page table maps it in the ops (new kernel support written for this port:
`sparse_sdpa` / `indexer_score_dsa` page_table, `ttnn.experimental.dsa_decode_scores`; unit tests
`tests/test_sparse_sdpa_paged.py`, `tests/test_indexer_paged.py`, `tests/test_dsa_decode_scores.py` are bit-exact /
PCC >= 0.9999995 against the contiguous cache; `test_paged_permuted_blocks` runs a shuffled block table end to end).

Prefill chunks start at multiples of 128; logical lengths are free (padding rows are masked: KDA `actual_end`, DSA
sentinel rows, MoE rows computed and dropped).

## Acceptance

PCC of the layer update `X' - X` >= 0.995 (prefill rows and decode steps) and of the full stream `X'` >= 0.99. The
HF reference rounds the 4-stream residual to bf16 at every mix while the port keeps fp32, so the full-stream PCC has a
floor below 1 that is not a port error; the update PCC is the stricter measure. The full-stream bar (0.99) is a coarse
sanity bar; the measured minimum is 0.99788 (`test_synthetic_weights[dsa_moe]` prefill, `s1_functional7.log`; real
weights: 0.99876).

The tests also bound the update scale |X' - X| TT / HF, because PCC is scale-blind (review #1):

| Measured over | Bound | Measured (`s1_functional7.log`, 2026-10-17) |
|---|---|---|
| One prefill sequence (more than one row) | 1 +- 0.01 | 0.9955 .. 1.0059 (42 sequences; L0 0.9955 .. 0.9968, L3 0.9997 .. 1.0059, L4 0.9995 .. 1.0023) |
| One decode token (a 1-row prefill counts as a token) | 1 +- 0.05 | 0.9803 .. 1.0196 (464 tokens; the 1-row L0 prefill is 0.9934) |
| Mean of a test's decode tokens (8 or more) | 1 +- 0.01 | 0.9959 .. 1.0047 (13 tests) |
| y_swap matches (router near-tie, below) per test / per suite | max(1, 5 %) / max(2, 3 %) of the MoE decode tokens | 1 per test at most; 4 of 298 in the suite |

**The gate catches the review #1 bug (fault injection).** `GLM53_FAULT_ROUTED_SCALE=0.9722` multiplies the routed
output by the bug's factor (test-only hook in `tt/moe.py`, prefill and decode). With it, 17 tests fail: 15 fail and
2 fail at teardown on the decode mean (`s1_fault_routed2.log`, 2026-10-17, final code):

| Layer kind | Check that fails | Tests | Measured with the fault |
|---|---|---|---|
| KDA-MoE (L4), real weights | prefill sequence 1 +- 0.01 | 4: `test_real_weights`, `test_lengths[kda_moe-33]`, `test_batched_users`, `test_determinism` | 0.9846 .. 0.9900 |
| KDA-MoE (L4), synthetic weights | prefill sequence | 1: `test_synthetic_weights[kda_moe]` | 0.9769 |
| DSA-MoE (L3), real weights | prefill sequence | 10: `test_real_weights`, six `test_lengths` (33, 129, 2049, 2051, 2177, 4000), `test_batched_users`, `test_determinism`, `test_paged_permuted_blocks` | 0.9789 .. 0.9839 |
| DSA-MoE (L3), 32 slots / inactive slot | decode mean 1 +- 0.01 | 2 | 0.9823, 0.9828 |

The bug's layer-level factor (faulted / clean update scale, same test) is 0.977 .. 0.979 on L3 and 0.984 .. 0.991 on
L4 (real weights), 0.975 on synthetic L4 and 0.994 on synthetic L3. These checks do not fail with the fault:
`test_lengths[kda_moe-2049]` (prefill 0.9908), `test_synthetic_weights[dsa_moe]` (0.9940), and the L4 decode means
(0.9933, 0.9912). `test_batched_users[kda_moe]` fails with almost no margin (0.9900). So on KDA-MoE the gate catches
the bug through the prefill sequences only. The MoE code is shared with L3, where prefill and decode catch it with
0.6 .. 1.1 % margin.

**Why the per-token bound is wider.** HF runs the layer in bf16, and its mHC post / comb mixing rounds per token. An
fp32 host computation of HF's own mHC on HF's own inputs misses HF's update by 0.978 .. 1.019 per token
(`tools/decode_stage_probe.py`, layer 4). At sequence level the two agree to 0.1 %.

**Port-side mHC offset (stage-1 review #4): fixed for mHC (L4).** The mHC kernels read fp32 operands at TF32
precision, and the Sinkhorn reciprocal (`recip_tile` legacy mode) was 1.4e-3 low. TT's comb column sums were about
1e-3 below 1. That moved the residual update by +0.3 % (post / comb) and +0.1 % (`mhc_post`). After the kernel fix
(fp32 through DST, hi / lo operand split, exact reciprocal; see the fused-decoder README), `decode_stage_l4c.log`
gives:

| Part (layer 4, 8 decode steps, identical inputs) | Before | After |
|---|---|---|
| TT post / comb vs exact fp32 | +0.30 % mean | 0.00 % (-0.01 .. +0.01) |
| device `mhc_post` vs host on the same post / comb | +0.10 % mean | 0.000 % |
| L4 prefill sequences | 1.0035 .. 1.0058 | 0.9995 .. 1.0023 |

The DSA-MoE layer (L3) keeps an update excess of its own. It is not mHC; see "L3 update excess" below.

**L3 update excess (stage-1 review #5): located, not fixed.** On the DSA-MoE layer the update scale is above HF on
short prompts: L3 sequences 0.9997 .. 1.0059, decode means 1.0010 .. 1.0047. Against the HF layer run in fp32 on the
same input (the reference rounding is not the cause) TT is +0.50 % on the 100-row prefill and +0.24 % mean over 8 decode
steps (review #5: +0.34 % / +0.39 % / +0.05 % at 33 / 129 / 2049 rows). The probes on the 100-row prefill (seed 1)
split it as follows (`s1r5_layer_stage_l3.log`, `s1r5_decode_stage_l3.log`, `s1r5_dsa_stage_l3.log`, host checks
in the work log):

| Part | Ratio TT / fp32 | Evidence |
|---|---|---|
| DSA attention output on HF's own attention input | 0.9941 (decode 0.9949) | layer / decode stage probes, `--hf32` |
| of which: bfp8 q_a / q_b / kv_a / o weights | 0.9991 | HF fp32 layer with the port's bfp8 weights (host) |
| of which: `ttnn.rms_norm` of q_a and of the latent | 0.9991 / 0.9993 | `tools/dsa_stage_probe.py` (fp32 torch on the device weights) |
| of which: `flash_mla_prefill` on exact inputs | 0.9983 | same; W_UK, W_UV and o_proj matmuls are exact (1.0001) |
| FFN (exact fp32) on TT's FFN input vs on HF's | 1.0034 | TT FFN input dumped; 4 of 100 rows reroute, the others move as much (1.0026 per row) |
| FFN device compute on the same input | 1.0020 (routed 1.0028, shared 0.9982) | `ffn_out(hf in)`; bfp8 expert weights alone 0.9992 (host) |
| Layer update | 1.0050 | |

The attention deficit drives the FFN term: the HF fp32 layer with its attention output scaled by 0.9942 gives an
update of 1.0035 and an FFN output of 1.0021 (host). So about +0.35 % of the +0.50 % comes from the attention output
being 0.6 % low, and about +0.2 % from the MoE device compute. The attention deficit is in shared op kernels
(`ttnn.rms_norm`, the SDPA softmax of `flash_mla_prefill`, decode `sparse_sdpa`), not in a precision setting of this
port; the SDPA compute kernels use the approximate exp (`exp_packthread_tile<true, ...>`), a likely cause of the
softmax part that is not verified here. Recorded in Limitations; input to the datatype stage.

**Router near-ties: the y_swap reference (stage-1 review #5).** A decode token whose HF 8th-vs-9th expert selection
margin is small can route one expert differently. The port's FFN input differs from HF's bf16-rounded input by about
1 %, which moves the selection scores sigmoid(l) + bias by up to about 2.5e-3. A flipped expert changes the token's
function, not only its rounding. Over 4606 L3 and 2812 L4 HF tokens (review #5, CPU), the 8th -> 9th swap moves the
update scale by up to 17.7 % (L3; 99th percentile 7.9 %) and the delta PCC down to 0.870. So no fixed widened bound fits
a flip. The HF reference therefore also computes, for every decode row with margin < 5e-3, `y_swap`: the unmodified
HF layer rerun with the router's 8th expert replaced by its 9th, weights renormalized as HF does
(`tests/hf_ref/hf_flip_ref.py`; every other row reproduces the cached `y` bit-exactly, which the script checks). Such a
token passes when it meets the normal bars (update PCC 0.995, full PCC 0.99, scale 1 +- 0.05) against `y` or against
`y_swap`. A `y_swap` match is logged (`SWAPS`) and counted. A test may have at most max(1, 5 %) of its MoE decode
tokens matched to `y_swap`, the suite at most max(2, 3 %): a router defect that flips near-tie tokens systematically
fails. The earlier relaxations (near-tie PCC bar 0.98, near-tie scale bound 1 +- 0.08) are removed.

In the final run 194 of the 298 MoE decode tokens have margin < 5e-3 and get a `y_swap`. Four match `y_swap`, and
against it they meet the normal bars with margin:

| Token | HF margin | vs y: update PCC / scale | vs y_swap: update PCC / full PCC / scale |
|---|---|---|---|
| `test_lengths[dsa_moe-2051]` pos 2052 | 1.7e-3 | 0.98972 / 1.0060 | 0.99989 / 0.99997 / 0.9931 |
| `test_paged_permuted_blocks` slot 0 pos 4002 | 1.4e-3 | 0.99191 / 0.9913 | 0.99991 / 0.99997 / 0.9998 |
| `test_batch32_all_active[dsa_moe]` slot 2 pos 39 | 1.9e-4 | 0.99873 / 1.0507 | 0.99993 / 0.99997 / 1.0004 |
| `test_batch32_all_active[kda_moe]` slot 15 pos 130 | 2.0e-4 | 0.99428 / 0.9911 | 0.99963 / 1.00000 / 0.9909 |

For pos 2052 and pos 4002 the TT expert set was also read back earlier (`flip_tt.log`: HF's 9th expert 95 for 162,
and 110 for 178), and for pos 39 (`flip_tt_s1302.log`: expert 31 for 149, the other 7 equal). Before the kernel fix
(review #4), pos 38 and pos 39 both flipped (review #5 CPU counterfactual); now only pos 39 does.

## Results (tests/test_functional_decoder.py, real weights unless noted)

Command: `pytest -sv models/autoports/zai_org_glm_5_3_flash/tests/test_functional_decoder.py` (one die; log
`s1_functional7.log`, 2026-10-17, final code, final kernels, ttnn rebuilt at 01:15 UTC the same day). 42 tests, 42
passed. Four decode tokens match `y_swap` (router near-ties, see Acceptance); the table gives their values against
`y_swap`. Minimum update / full-stream PCC per case: one PCC over all prefill rows of a case (flattened) and one per
decode step (single token):

| case | checks | min update PCC | min full PCC |
|---|---|---|---|
| real weights kda_dense / dsa_moe / kda_moe (prefill 100 at slot 5 of 8, 3 decode steps: eager, then traced) | 4 each | 0.99997 / 0.99984 / 0.99945 | 0.99994 / 0.99992 / 0.99998 |
| lengths kda_dense 1, 31, 33, 128, 129, 2047, 2049, 4000 | 4 each | >= 0.99996 | >= 0.99993 |
| lengths dsa_moe 1, 33, 129, 2049, 2177, 4000 | 4 each | >= 0.99962 | >= 0.99977 |
| lengths dsa_moe 2051 (dense -> sparse switch) | 4 | 0.99981 (decode pos 2052 against y_swap: 0.99989) | 0.99991 |
| lengths kda_moe 33, 2049 | 4 each | 0.99977 / 0.99874 | >= 0.99997 |
| 4 users in slots 1, 3, 4, 6 (lengths 37, 100, 300 or 2100 for DSA, 64), prefilled one by one, decoded together | 16 per kind | 0.99997 / 0.99879 / 0.99951 | >= 0.99935 |
| 32 users, all slots active, mixed positions (DSA: two lengths past 2051), eager then traced | 96 per kind | 0.99992 / 0.99568 / 0.99621 | >= 0.99876 |
| paged caches through a shuffled, interleaved block table (users 4000 / 2100 / 129) | 12 | 0.99910 | 0.99960 |
| inactive slots keep their state (B decodes after idling 4 steps) | 8 per kind | 0.99997 / 0.99989 / 0.99957 | >= 0.99994 |
| determinism (same sequence twice: prefill and traced decode bit-identical) | 10 per kind | 0.99997 / 0.99942 / 0.99929 | >= 0.99971 |
| decode from an empty state (positions 0-3) | 4 per kind | 0.99998 / 0.99993 / 0.99952 | >= 0.99996 |
| synthetic weights kda_dense / dsa_moe / kda_moe | 4 each | 0.99984 / 0.99785 / 0.98976 (prefill, see below) | >= 0.99788 |

Real-weight cases use real token streams (embedding + the real layers below the tested one, computed by the HF
reference); prompt lengths cover the tile (32), page (128), chunk (2048), indexer pool (4) and dense/sparse (2051)
boundaries and non-divisible tails.

## Synthetic MoE layers

Synthetic weights are drawn from the real tensor statistics (`tests/weight_stats.json`: shape / mean / std; small
tensors such as norms, A_log, dt_bias, router bias and mHC base / scale verbatim) and the input stream from the
per-channel statistics of the real stream of the layer kind. On these streams 12-14 % of the tokens have a top-8 vs 9th
expert-selection margin below 1e-3 (real stream: 8.7 %; real router on the synthetic stream: 12 %), so the ~6e-5
relative difference between the TT and HF FFN inputs (after mHC and RMSNorm) flips experts of a few prefill tokens.
Measured on layer 4 (`tools/layer_stage_probe.py`): every stage up to the FFN input at PCC >= 0.99994; FFN output
on the TT input 0.987, on the HF input 0.9998. The synthetic MoE tests therefore assert (a) the MoE block on the HF
FFN input >= 0.995 (0.9984 / 0.9999) and (b) the layer update >= 0.98 (prefill 0.99797 / 0.99206, decode >= 0.9996).
Real-weight cases keep the 0.995 bar throughout.

## Context contract

Advertised context: 1,048,576 tokens (`text_config.max_position_embeddings`; no RoPE, so no scaling limit applies).
Functional decoder: the full advertised context, no capability reduction (`doc/context_contract.json`).

`tests/test_long_context.py`: a 1,048,000-token prompt (not a multiple of the chunk, the page or the pool) is prefilled
in 2048-row chunks into decode slot 0 of a one-slot layer with `max_context` 1,048,576, then 3 decode steps run at
positions 1,048,000-1,048,002. The input rows come from the per-channel statistics of the real stream of the layer
kind; the HF reference (`tests/hf_ref` long mode; vs the plain full HF forward at 4K: bit-identical for the KDA layer, max |diff| 4.9e-4 = bf16 rounding of the segment-wise cache fill for the DSA layer, `scripts/long_mode_check.py`, log `long_mode_check.log` here) fills its layer cache
segment by segment. Logs `~/glm53-flash-bringup/logs/s1_long3_{kda_dense,kda_moe}.log` (2026-10-15, final code) and `s1_long_dsa_moe.log` (2026-10-15, paged caches; the later KDA / dense-MLP fidelity changes do not touch this layer).

| kind | prefill 1,048,000 tokens (one die) | update PCC, last 64 prompt rows | update PCC, decode at 1,048,000..002 |
|---|---|---|---|
| kda_dense (layer 0) | 76 s | 0.99997 | >= 0.99997 |
| dsa_moe (layer 3, paged caches) | 210 s | 0.99733 | >= 0.99975 |
| kda_moe (layer 4) | 98 s | 0.99923 | >= 0.99936 |

Device state per layer and slot at 1M: DSA latent 1 GiB (512 x bf16 per token), pool keys 64 MiB; KDA state 4 MiB
(context-independent). With the paged caches the DSA memory is a shared block pool: the 1M sequence occupies 8192
blocks of 128 tokens, and the same pool serves any mix of shorter sequences.

DSA long-context precision (CPU, `~/glm53-flash-bringup/scripts/selection_{precision,floor}.py`): at 1M the indexer
ranks 262,000 pools and 120-210 of them lie within bf16 epsilon of the top-512 cutoff. In the HF layer itself,
bf16-rounded selection scores change 10-32 of the 512 selected pools per row and move the update of the last 64 rows to
PCC 0.99876 vs the fp32 selection (relative score noise 2**-9: 0.99939). This is the floor of any bf16 implementation
of the selection at this length; the port's error at 1M (1 - 0.99733) is about 2.2x that floor. The decode selection scores are carried in bf16
(`_select_pools`); fp32 scores with an fp32 `ttnn.topk` are a datatype-stage candidate.

Bug found by the 1M test (fixed): the pool-id -> token-id expansion `tok = 4 p + j` was a matmul against a [512, 2048]
expansion matrix. The matmul unit reads fp32 operands at TF32 precision, so pool ids >= 2**11 were off by up to 508
tokens (HiFi4; 8188 at HiFi2; `tools/index_expand_probe.py`) and attention read wrong tokens past ~8K context (update
PCC 0.9877 at 1M). It is now `repeat_interleave` + an elementwise multiply-add (exact).

## Performance (one die, warmed; Tracy + tt-perf-report)

`scripts/01-tracy.sh` (`python -m tracy -r -p -v -m pytest tests/test_perf_decoder.py::<case>`, then
`tt-perf-report --start-signpost PERF_* --end-signpost PERF_*_END`); tables, CSVs and ops CSVs in
`doc/functional_decoder/tracy/<case>/`. Device time = sum of `Device Time` (us) over the ops of one pass in the
signposted window (decode: one of the 3 traced replays).

Recaptured 2026-10-17 12:08 .. 12:24 UTC on the final stage-1 code and kernels, all 12 cases
(`scripts/s1r5_suite.sh`: `GLM53_DECODER=functional scripts/01-tracy.sh`; table by `tools/perf_tables.py
doc/functional_decoder`, log `s1r5_perf_tables.log`):

| case | ops / pass | device time (us) | op-to-op gaps (us) | host wall |
|---|---|---|---|---|
| test_perf_prefill_dsa_moe-128 | 148 | 22366 | 341 | not recorded (console cut) |
| test_perf_prefill_dsa_moe-2048 | 179 | 143527 | 109 | 144.43 ms |
| test_perf_prefill_kda_dense-128 | 80 | 4105 | 1852 | 6.20 ms |
| test_perf_prefill_kda_dense-2048 | 80 | 43737 | 605 | 45.27 ms |
| test_perf_prefill_kda_moe-128 | 118 | 21280 | 1649 | 23.24 ms |
| test_perf_prefill_kda_moe-2048 | 149 | 96660 | 650 | 98.39 ms |
| test_perf_traced_decode_dsa_moe-1 | 181 | 3949 | 153 | 4.747 ms/step |
| test_perf_traced_decode_dsa_moe-32 | 187 | 17246 | 128 | 18.208 ms/step |
| test_perf_traced_decode_kda_dense-1 | 130 | 3539 | 91 | 4.324 ms/step |
| test_perf_traced_decode_kda_dense-32 | 128 | 16542 | 85 | 17.296 ms/step |
| test_perf_traced_decode_kda_moe-1 | 166 | 4082 | 121 | 4.840 ms/step |
| test_perf_traced_decode_kda_moe-32 | 172 | 29316 | 117 | 30.303 ms/step |

Changes since the 2026-10-15 table:
- The fp32-exact mHC kernels (review #4) cost more per layer pass, in every layer kind (two calls each; b916fa98021
  CSVs -> these, `scripts/s1r5_host/mhc_cost.py`): `mhc_split_sinkhorn` 164 -> 179 us at decode, 166 -> 181 us at 128
  rows, 180 -> 194 us at 2048 rows; `mhc_post` 95 .. 97 -> 115 .. 117 us at decode, 145 .. 151 -> 187 .. 188 us at 128
  rows, 1527 .. 1570 -> 1988 .. 2014 us at 2048 rows. That is +35 .. +36 us per decode step and +458 .. +483 us per
  2048-row prefill, per layer.
- The 2048-row MoE prefills (dsa_moe, kda_moe) have 31 more ops and take +12 .. +13 ms: the working tree carries a
  stage-6 memory change in `tt/moe_opt.py` (uncommitted at capture time, not part of stage 1) that runs the routed
  experts of a chunk wider than 1024 rows in 1024-row blocks (`GLM53_MOE_SORTED_ROWS`, default 1024; bit-identical by
  its author's check). With `GLM53_MOE_SORTED_ROWS=0` the routed experts run as in the 2026-10-15 table (by code
  reading; not captured).
- The other cases move by -1.5 .. +1.7 % (device time), most of it the mHC kernels above.

Conclusions (inputs to stages 2-5): one die holds every weight of the layer (bf16 KDA projections: the 32 x 4096 x
32960 input projection alone is 717 us at decode); mHC costs ~25 % of a decode step (the 16384-wide projection runs
on one core: 130 us x 2; the mean of squares 115 us x 2; Sinkhorn, post-mix, composite collapse); the KDA decode
recurrence at 32 users is two fp32 batched matmuls of 3.8 ms (one-row operands padded to tiles); MoE prefill is bound
by reading every expert's weights (all 192 on one die); DSA prefill spends 22.8 ms in flash_mla_prefill and 17 ms in
reshape / transpose at 2048 rows.

## Host fallback audit, determinism, watcher

- Host fallback: `test_no_host_fallback[*]` runs one prefill pass and one decode pass per kind with `ttnn.from_torch`,
  `ttnn.to_torch` and `ttnn.copy_host_to_device_tensor` patched to raise: all pass. Per-call inputs are built by
  `prefill_inputs` / `decode_inputs` before the pass; constants at setup (`GlmDSA.prepare_constants`).
- Determinism: `test_determinism[*]` (bit-identical prefill and traced decode over a repeated sequence) passes.
- Watcher (`TT_METAL_WATCHER=10`, `TT_METAL_WATCHER_NOINLINE=1`; logs `doc/functional_decoder/watcher/<case>/`):
  `test_real_weights[dsa_moe]`, `test_no_host_fallback[dsa_moe]`, `test_decode_only[kda_dense]`,
  `test_decode_only[kda_moe]` pass with no watcher findings.
- KDA prefill under the full watcher: the watcher-instrumented `kda.prepare_chunk_recurrence` program needs a 75.6 KB
  kernel-config ring buffer (default 70,656 B). The ring is the L1 below the worker allocator, so a 16 KiB smaller
  `worker_l1_size` (1,444,864 B; harness `GLM53_WORKER_L1_SIZE`) grows it to 87,040 B. With it, the full watcher (every
  feature on) runs `test_perf_prefill[kda_dense-2048]`, `[kda_moe-2048]`, `[kda_moe-128]` and
  `test_perf_traced_decode[kda_moe-32]` clean (`scripts/01-watcher-kda-l1.sh`, logs `watcher_kda_l1/`). The earlier
  attempts with reduced watcher features (`watcher_reduced/` etc.) are kept for reference.
- Watcher on the fp32-exact mHC kernels (2026-10-17, ttnn rebuilt at 01:15 UTC, `scripts/s1r5_suite.sh`, logs
  `watcher_r5/<case>/`, `TT_METAL_WATCHER=10`, worker_l1_size 1,444,864): `test_real_weights[dsa_moe]`,
  `test_real_weights[kda_moe]` and `test_decode_only[kda_moe]` pass, 0 error / assert / sanitize / overflow lines. The
  watched kernels include `mhc_split_sinkhorn` and `mhc_post` (`kernel_names.txt`).
- Watcher on the stage-1 code (2026-10-15, paged caches, inactive-slot masking, HiFi4 fidelities; logs `watcher_final/<case>/`,
  worker_l1_size 1,444,864): `test_real_weights[dsa_moe]`, `test_lengths[dsa_moe-4000]`, `test_batch32_all_active[dsa_moe]`,
  `test_real_weights[kda_moe]`, `test_lengths[kda_dense-4000]` pass, 0 error / assert / sanitize lines. The `kda_moe` run
  predates the HiFi4 KDA-recurrence change (a compute-fidelity argument only: same kernels, same buffers).

## Limitations

- One die: the layer's full weights (bf16 KDA projections, bfp8 MLP / experts) in one DRAM; tensor parallelism is stage 4.
- Decode performance is far from the target (~4 ms per layer at one user); stage 2 (fusing) and stage 3 address mHC,
  the KDA decode recurrence and op count.
- Routed-expert scale (fixed): the routed output was 0.9756 of the reference at PCC 0.99993. `tools/moe_scale_probe.py`
  showed the bfp8 weight conversion is not the cause (emulated bfp8 weights: 0.9995) and routing / dense projections /
  SwiGLU are exact; the sparse_matmul factories (grouped, 1d) reloaded the fp32 partial sums through SrcA (TF32) at every
  K block. With the partials CB marked UnpackToDestFp32 (as the dense 1d matmul factory does): 1.0036 / PCC 0.99998.
  bfp4 expert weights (datatype stage) cost 0.9826 PCC / 1.0275 scale on their own (same probe).
- KDA attention output scale: 0.9954 of HF on real weights (layer 0, after the HiFi2 -> HiFi4 recurrence fix; PCC 0.99998);
  layer 4 decode: 0.995 .. 0.999 per token. The remaining -0.3 .. -0.5 % is not root-caused (projections, norm epsilon
  and fused sigmoid ruled out). At layer level it is inside the measured update scale (sequence bound 1 +- 0.01, see
  Acceptance); a -0.5 % attention output alone moves the layer update by less than that.
- DSA-MoE update excess (not fixed, see Acceptance "L3 update excess"): the L3 update is +0.50 % above the HF fp32
  layer on a 100-row prompt (+0.34 .. +0.39 % at 33 / 129 rows, +0.05 % at 2049 rows) and +0.24 % mean over 8 decode
  steps. In the suite: L3 prefill sequences up to 1.0059 against the HF bf16 reference, decode means 1.0010 .. 1.0047.
  About +0.35 % comes from the DSA attention output being 0.6 % low (device `ttnn.rms_norm` -0.07 .. -0.09 % on q_a and
  the latent, `flash_mla_prefill` -0.17 %, bfp8 projection weights -0.09 %), which shifts the FFN input; about +0.2 %
  from the MoE device compute (routed +0.28 %, shared -0.18 %). On the 11 DSA layers this uses up to 60 % of the
  sequence bound of 1 +- 0.01. The mHC offset of review #4 (fixed) is a different cause; it only applied to L4.
- mHC kernel cost: the fp32-exact `mhc_split_sinkhorn` / `mhc_post` (review #4) are slower than before (Tracy, same
  shapes in every layer): Sinkhorn 2 calls 164 -> 179 us per decode step; `mhc_post` 97 -> 118 us per decode step and
  1531 -> 2019 us at a 2048-row prefill. `mhc_post` takes the slower SFPU path only for fp32 inputs (this port);
  bf16 / bfp8_b callers keep the FPU path (2026-10-17).
- The MTP layer and the vision tower are not part of the decoder stages (full-model / vLLM stages).

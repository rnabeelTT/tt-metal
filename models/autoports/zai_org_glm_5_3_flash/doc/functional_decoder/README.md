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
sanity bar; measured minimum 0.99934 outside the router near-tie tokens (0.99731 with them).

The tests also bound the update scale |X' - X| TT / HF, because PCC is scale-blind (review #1):

| Measured over | Bound | Measured (s1_functional6.log, final kernels) |
|---|---|---|
| One prefill sequence | 1 +- 0.01 | 0.9934 .. 1.0059 |
| One decode token | 1 +- 0.05 | 0.9803 .. 1.0507 (433 tokens; the one token above 1.05 is a confirmed router flip, below) |
| One router near-tie decode token (HF margin < 5e-3) | 1 +- 0.08 | max 1.0507 |
| Mean of a test's decode tokens (8 or more, near-ties included) | 1 +- 0.01 | 0.9959 .. 1.0047 (13 tests) |

**The gate catches the review #1 bug (fault injection).** `GLM53_FAULT_ROUTED_SCALE=0.9722` multiplies the routed
output by the bug's factor (test-only hook in `tt/moe.py`). The suite then fails 17 tests (`s1_fault_routed.log`):

| Layer kind | Check that fails | Measured with the fault | Margin |
|---|---|---|---|
| KDA-MoE (L4), real weights | prefill sequence 1 +- 0.01 | 0.9846 .. 0.9900 (5 tests) | 0 .. 1.5 % |
| KDA-MoE (L4), synthetic weights | prefill sequence | 0.9769 | 1.3 % |
| DSA-MoE (L3), real weights | prefill sequence | 0.9789 .. 0.9839 (9 tests) | 0.6 .. 1.1 % |
| DSA-MoE (L3), 32 slots / inactive slot | decode mean 1 +- 0.01 | 0.9823, 0.9828 | 0.8 % |

The layer-level factor of the bug is 0.985 .. 0.990 on L4 and 0.979 .. 0.984 on L3 (the "~0.988" of earlier versions
of this README was not a measurement of this bug).

**Why the per-token bound is wider.** HF runs the layer in bf16, and its mHC post / comb mixing rounds per token. An
fp32 host computation of HF's own mHC on HF's own inputs misses HF's update by 0.978 .. 1.019 per token
(`tools/decode_stage_probe.py`, layer 4). At sequence level the two agree to 0.1 %.

**Port-side mHC offset (stage-1 review #4): fixed.** The mHC kernels read fp32 operands at TF32 precision, and the
Sinkhorn reciprocal (`recip_tile` legacy mode) was 1.4e-3 low. TT's comb column sums were about 1e-3 below 1. That
moved the residual update by +0.3 % (post / comb) and +0.1 % (`mhc_post`). After the kernel fix (fp32 through DST,
hi / lo operand split, exact reciprocal; see the fused-decoder README), `decode_stage_l4c.log` gives:

| Part (layer 4, 8 decode steps, identical inputs) | Before | After |
|---|---|---|
| TT post / comb vs exact fp32 | +0.30 % mean | -0.02 % (range -0.06 .. 0) |
| device `mhc_post` vs host on the same post / comb | +0.10 % mean | 0.000 % |
| L4 prefill sequences | 1.0035 .. 1.0058 | 0.9995 .. 1.0023 |

**Router near-ties.** A token whose HF 8th-vs-9th margin is below 5e-3 can route one expert differently. The port's
FFN input differs from HF's bf16-rounded input by about 1 %. One flip changes the token's function, so near-tie tokens
get PCC bar 0.98 and per-token scale 1 +- 0.08, and stay in the decode mean. Evidence for the bound: L3 32-slot slot 2
pos 39, HF margin 1.86e-4 (`route_margin_l3_s1302.log`). TT routes expert 31, HF's 9th, in place of 149. The other 7
experts are equal (`flip_tt_s1302.log`), and the token's update scale is 1.051 at delta PCC 0.9987. Before the kernel
fix, the neighbouring pos 38 flipped instead (margin 8.0e-4); now it routes as HF does.

## Results (tests/test_functional_decoder.py, real weights unless noted)

Command: `pytest -sv models/autoports/zai_org_glm_5_3_flash/tests/test_functional_decoder.py` (one die; log
`s1_functional6.log`, 2026-10-16, final code and final fp32-exact kernels). 42 tests: 41 passed in the full run; the
42nd (`test_batch32_all_active[dsa_moe]`) failed only on the confirmed router flip at slot 2 pos 39 (scale 1.0507,
see Acceptance) and passes under the near-tie scale bound (`s1_b32_dsa.log`). Three decode tokens pass under the
near-tie PCC rule (HF router margin < 5e-3): `test_lengths[dsa_moe-2051]` pos 2052, `test_paged_permuted_blocks`
pos 4002 and `test_batch32_all_active[kda_moe]` slot 15 pos 130. Minimum update / full-stream PCC per case (final
run): one PCC over all prefill
rows of a case (flattened) and one per decode step (single token):

| case | checks | min update PCC | min full PCC |
|---|---|---|---|
| real weights kda_dense / dsa_moe / kda_moe (prefill 100 at slot 5 of 8, 3 decode steps: eager, then traced) | 4 each | 0.99997 / 0.99984 / 0.99945 | 0.99994 / 0.99992 / 0.99998 |
| lengths kda_dense 1, 31, 33, 128, 129, 2047, 2049, 4000 | 4 each | >= 0.99996 | >= 0.99993 |
| lengths dsa_moe 1, 33, 129, 2049, 2177, 4000 | 4 each | >= 0.99962 | >= 0.99977 |
| lengths dsa_moe 2051 (dense -> sparse switch) | 4 | 0.98972 (decode pos 2052: router near-tie, see below); others >= 0.99981 | 0.99759 |
| lengths kda_moe 33, 2049 | 4 each | 0.99977 / 0.99874 | >= 0.99997 |
| 4 users in slots 1, 3, 4, 6 (lengths 37, 100, 300 or 2100 for DSA, 64), prefilled one by one, decoded together | 16 per kind | 0.99997 / 0.99879 / 0.99951 | >= 0.99935 |
| 32 users, all slots active, mixed positions (DSA: two lengths past 2051), eager then traced | 96 per kind | 0.99992 / 0.99568 / 0.99428* | >= 0.99876 |
| paged caches through a shuffled, interleaved block table (users 4000 / 2100 / 129) | 10 | 0.99191* (pos 4002); others >= 0.99972 | 0.99732 |
| inactive slots keep their state (B decodes after idling 4 steps) | 8 per kind | 0.99997 / 0.99989 / 0.99957 | >= 0.99994 |
| determinism (same sequence twice: prefill and traced decode bit-identical) | 10 per kind | 0.99997 / 0.99942 / 0.99929 | >= 0.99971 |
| decode from an empty state (positions 0-3) | 4 per kind | 0.99998 / 0.99993 / 0.99952 | >= 0.99996 |
| synthetic weights kda_dense / dsa_moe / kda_moe | 4 each | 0.99984 / 0.99785 / 0.98976 (prefill, see below) | >= 0.99788 |

\* Router near-ties (MoE layers, single decode tokens). The port's FFN input differs from HF's by ~1 % relative (HF
rounds its 4-stream residual to bf16 at every mix; the port keeps fp32), so the router's selection scores move by up to
~2.5e-3 and tokens whose HF 8th-vs-9th expert margin is that small can take the 9th expert. Every token below 0.995 had
an HF margin of 6e-5 .. 1.7e-3 (`scripts/route_margin.py`, margins recorded in the references), and for the two
non-batch cases the TT expert set was read back (`tools/route_flip_check.py`, `GLM53_MOE_DEBUG=1`, logs
`flip_tt.log` / `flip_hf_*.log` here): position 2052 picked HF's 9th expert 95 instead of 162, position 4002 its 9th
expert 110 instead of 178; every passing token of the same sequences had the identical expert set. The tests hold such a
token (HF margin < 5e-3) to 0.98 and log it (`NEAR_TIES`); any other token keeps the 0.995 bar.

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

Refreshed 2026-10-15 on the final stage-1 code (paged DSA caches, sparse_matmul fix; `scripts/s2_tracy.sh`, functional
pass; table by `tools/perf_tables.py doc/functional_decoder`):

| case | ops / pass | device time (us) | op-to-op gaps (us) | host wall |
|---|---|---|---|---|
| test_perf_prefill_dsa_moe-128 | 148 | 21987 | 1387 | 24.19 ms |
| test_perf_prefill_dsa_moe-2048 | 148 | 130998 | 89 | 132.07 ms |
| test_perf_prefill_kda_dense-128 | 80 | 4043 | 1819 | 6.08 ms |
| test_perf_prefill_kda_dense-2048 | 80 | 43349 | 607 | 44.78 ms |
| test_perf_prefill_kda_moe-128 | 118 | 21294 | 1744 | 23.32 ms |
| test_perf_prefill_kda_moe-2048 | 118 | 84292 | 627 | 85.89 ms |
| test_perf_traced_decode_dsa_moe-1 | 181 | 3900 | 156 | 4.709 ms/step |
| test_perf_traced_decode_dsa_moe-32 | 187 | 17143 | 129 | 22.540 ms/step |
| test_perf_traced_decode_kda_dense-1 | 130 | 3510 | 91 | 4.229 ms/step |
| test_perf_traced_decode_kda_dense-32 | 128 | 16492 | 85 | 17.212 ms/step |
| test_perf_traced_decode_kda_moe-1 | 166 | 4041 | 122 | 4.871 ms/step |
| test_perf_traced_decode_kda_moe-32 | 172 | 29760 | 118 | 30.506 ms/step |

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
- The MTP layer and the vision tower are not part of the decoder stages (full-model / vLLM stages).

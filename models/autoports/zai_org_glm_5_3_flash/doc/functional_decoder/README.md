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
| DSA | `tt/dsa.py` | per-slot ROW_MAJOR latent cache `[S, 1, T, 512]` (`ttnn.scatter_rows`), pooled indexer keys (TILE, paged view), `indexer_score_dsa` (prefill) / batched matmul (decode), `ttnn.topk` 512, `ttnn.transformer.sparse_sdpa` for every query row (dense rows: indices `[0..t]`); first dense prefill chunk: `flash_mla_prefill` |
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
Y = dec.decode_forward(X_dec, inputs=dsa_inputs)       # X_dec [1,1,B,16384]; DSA: dec.attn.decode_inputs / update_decode_inputs
```

Prefill chunks start at multiples of 128; logical lengths are free (padding rows are masked: KDA `actual_end`, DSA
sentinel rows, MoE rows computed and dropped).

## Acceptance

PCC of the layer update `X' - X` >= 0.995 (prefill rows and decode steps) and of the full stream `X'` >= 0.99. The
HF reference rounds the 4-stream residual to bf16 at every mix while the port keeps fp32, so the full-stream PCC has a
floor below 1 that is not a port error; the update PCC is the stricter measure.

## Results (tests/test_functional_decoder.py, real weights unless noted)

Command: `pytest -sv models/autoports/zai_org_glm_5_3_flash/tests/test_functional_decoder.py` (one die; log
`~/glm53-flash-bringup/logs/t1_full_a.log`, later cases `t1_synth3.log`, `t1_audit_decodeonly.log`). 35 tests, all pass.
Minimum update PCC per case (every prefill row and every decode step of the case):

| case | checks | min update PCC | min full PCC |
|---|---|---|---|
| real weights kda_dense / dsa_moe / kda_moe (prefill 100 at slot 5 of 8, 3 decode steps: eager, then traced) | 4 each | 0.99996 / 0.99975 / 0.99908 | 0.99987 / 0.99986 / 0.99997 |
| lengths kda_dense 1, 31, 33, 128, 129, 2047, 2049, 4000 | 4 each | >= 0.99993 | >= 0.99984 |
| lengths dsa_moe 1, 33, 129, 2049, 2051, 2177, 4000 (dense -> sparse at position 2051; 2049 decodes across it) | 4 each | >= 0.99957 | >= 0.99966 |
| lengths kda_moe 33, 2049 | 4 each | 0.99953 / 0.99910 | 0.99998 |
| 4 users in slots 1, 3, 4, 6 (lengths 37, 100, 300 or 2100 for DSA, 64), prefilled one by one, decoded together | 16 per kind | 0.99995 / 0.99882 / 0.99899 | >= 0.99939 |
| determinism (same sequence twice: prefill and traced decode bit-identical) | 10 per kind | 0.99996 / 0.99978 / 0.99864 | >= 0.99987 |
| decode from an empty state (positions 0-3) | 4 per kind | 0.99997 / 0.99981 / 0.99958 | >= 0.99863 |
| synthetic weights kda_dense / dsa_moe / kda_moe | 4 each | 0.99955 / 0.99796 / 0.98772 (prefill, see below) | >= 0.99798 |

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
FFN input >= 0.995 (0.9983 / 0.9998) and (b) the layer update >= 0.98 (prefill 0.99796 / 0.98772, decode >= 0.9995).
Real-weight cases keep the 0.995 bar throughout.

## Context contract

Advertised context: 1,048,576 tokens (`text_config.max_position_embeddings`; no RoPE, so no scaling limit applies).
Functional decoder: the full advertised context, no capability reduction (`doc/context_contract.json`).

`tests/test_long_context.py`: a 1,048,000-token prompt (not a multiple of the chunk, the page or the pool) is prefilled
in 2048-row chunks into decode slot 0 of a one-slot layer with `max_context` 1,048,576, then 3 decode steps run at
positions 1,048,000-1,048,002. The input rows come from the per-channel statistics of the real stream of the layer
kind; the HF reference (`tests/hf_ref` long mode, bit-exact against the full HF forward at 4K) fills its layer cache
segment by segment. Logs `~/glm53-flash-bringup/logs/t1_long_{c,kda_moe,dsa_fix}.log`.

| kind | prefill 1,048,000 tokens (one die) | update PCC, last 64 prompt rows | update PCC, decode at 1,048,000..002 |
|---|---|---|---|
| kda_dense (layer 0) | 140 s | 0.99994 | >= 0.99995 |
| dsa_moe (layer 3) | 209 s | 0.99768 | >= 0.99969 |
| kda_moe (layer 4) | 173 s | 0.99855 | >= 0.99936 |

Device state per layer and slot at 1M: DSA latent 1 GiB (512 x bf16 per token), pool keys 64 MiB; KDA state 4 MiB
(context-independent). Decode token indices are flat cache rows `slot * T + t` in fp32, so `GlmDSA` asserts
`max_batch * max_context <= 2**24` (16 slots at 1M; more than the DRAM holds).

DSA long-context precision (CPU, `~/glm53-flash-bringup/scripts/selection_{precision,floor}.py`): at 1M the indexer
ranks 262,000 pools and 120-210 of them lie within bf16 epsilon of the top-512 cutoff. In the HF layer itself,
bf16-rounded selection scores change 10-32 of the 512 selected pools per row and move the update of the last 64 rows to
PCC 0.99876 vs the fp32 selection (relative score noise 2**-9: 0.99939). This is the floor of any bf16 implementation
of the selection at this length; the port's 0.99768 is near it. The decode selection scores are carried in bf16
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

| case | ops / pass | device time (us) | op-to-op gaps (us) | host wall (Tracy run) |
|---|---|---|---|---|
| prefill kda_dense 128 rows | 80 | 3874 | 3916 | 8.28 ms |
| prefill kda_dense 2048 rows | 80 | 41009 | 2504 | 44.85 ms |
| prefill dsa_moe 128 rows | 137 | 20696 | 918 | 22.10 ms |
| prefill dsa_moe 2048 rows | 137 | 128848 | 77 | 129.92 ms |
| prefill kda_moe 128 rows | 118 | 19997 | 1824 | 22.23 ms |
| prefill kda_moe 2048 rows | 118 | 81801 | 628 | 83.38 ms |
| traced decode kda_dense, 1 user | 124 | 3404 | 88 | 4.10 ms |
| traced decode kda_dense, 32 users | 122 | 16391 | 81 | 19.81 ms |
| traced decode dsa_moe, 1 user | 178 | 3882 | 148 | 4.70 ms |
| traced decode dsa_moe, 32 users | 185 | 16856 | 132 | 20.67 ms |
| traced decode kda_moe, 1 user | 160 | 3948 | 120 | 4.71 ms |
| traced decode kda_moe, 32 users | 166 | 29270 | 115 | 29.98 ms |

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
  `test_decode_only[kda_moe]` pass with no watcher findings. Environment limit: with the watcher enabled in any
  feature combination (full; ring buffer / pause / waypoint / stack usage off; asserts off; NoC sanitize off) the
  `ttnn.experimental.kda.prepare_chunk_recurrence` program of the KDA prefill exceeds the Blackhole kernel config ring
  buffer (75,568 / 72,208 / 72,144 / 72,448 B > 70,656 B, `program.cpp:3197`, fixed by the memory map), so KDA prefill
  cannot run under the watcher (logs `watcher_reduced/`, `watcher_sanitize_only/`, `watcher_no_noc_sanitize/`). The
  KDA layers' decode path, mHC, dense MLP and MoE ran watcher-clean (`test_decode_only`), and the KDA prefill kernel
  is the unmodified origin/main op apart from the base-branch API translation.

## Limitations

- One die: the layer's full weights (bf16 KDA projections, bfp8 MLP / experts) in one DRAM; tensor parallelism is stage 4.
- Decode performance is far from the target (~4 ms per layer at one user); stage 2 (fusing) and stage 3 address mHC,
  the KDA decode recurrence and op count.
- Expert weights: bfp8 conversion leaves the routed output -2.5 % in scale vs bf16 weights (`tools/moe_route_probe.py`);
  revisited in the datatype stage (round-to-nearest pre-quantization).
- The MTP layer and the vision tower are not part of the decoder stages (full-model / vLLM stages).

# GLM-5.3-Flash (E192) multichip decoder

Stage 4 of the tt-model-bringup flow. `tt/multichip_decoder.py` (`MultichipDecoder`) runs the optimized decoder
(`tt/optimized_decoder.py`) on a (1, 4) mesh: p300x2, 4 Blackhole dies, FABRIC_1D. Contract, state, paging and
semantics are those of `tt/functional_decoder.py`.

## Strategy

1D tensor parallelism over the 4 dies (multichip skill default for 1D meshes up to 8 chips):

- Token mixers: TP over heads. KDA has 16 of 64 heads per die. DSA has 16 of 64 query heads per die. The DSA
  latent cache, the pooled index keys and the lightning indexer are replicated, so top-k pool selection needs no
  collective.
- FFN: TP over the intermediate dimension. The 192 routed experts are 512 of 2048 columns per die, and so are the
  shared expert and the dense MLP. All routed experts stay on every die (no expert parallelism), so decode reads a
  quarter of every active expert's weights per die and needs no token dispatch.
- Residual: the 4-stream mHC state stays hidden-sharded through the whole stack. Die d holds columns
  [1024 d, 1024 (d + 1)) of every stream (stream-interleaved). `mhc_mix`, `mhc_pre` and `mhc_post` are column-local
  and run unchanged on that slice.

Rejected alternatives:

| Option | Why not |
|---|---|
| Replicated residual (all-reduce after each sub-block) | Each all-reduce moves 4x the bytes of a reduce-scatter, and every die then repeats the mHC work on the full 16384-wide stream. |
| Expert parallelism (48 experts per die) | One token selects 8 experts. At batch 1 at most 8 of 192 experts are active, so EP puts the load on 1 to 4 dies and needs dispatch / combine collectives. TP spreads every active expert over all 4 dies. GLM-4.5-Air and gpt-oss-120b on the same host use TP for the same reason. |
| Sequence-sharded DSA latent cache | Needed for long context (see Memory). It is a full-model change (stage 6). The decoder keeps the replicated latent cache of the single-die path. |

## Collectives per layer

Per sub-block (attention, then FFN):

| Step | Op | Shape per die | Bytes per die at decode (32 rows) |
|---|---|---|---|
| mHC mix partials | all-gather, dim 1, fp32 | [1, 1, T, 64] -> [1, 4, T, 64] | 8 KB |
| mixer / FFN input | all-gather, dim 3, bf16 | [1, 1, T, 1024] -> [1, 1, T, 4096] | 64 KB |
| mixer / FFN output | reduce-scatter, dim 3, bf16 | [1, 1, T, 4096] -> [1, 1, T, 1024] | 64 KB |

That is 6 collectives per layer. KDA prefill and the MoE do their reduce-scatter inside the module.

## Tensor sharding

| Tensor | Single die | Per die (tp = 4) | Mesh placement |
|---|---|---|---|
| Residual stream X | [1, 1, T, 16384] fp32 | [1, 1, T, 4096] | stream-interleaved, dim -1 |
| mHC fn_T | [16384, 32] fp32 | [4096, 32] | rows permuted to (die, stream, column), dim -2 |
| mHC consts, bcast | — | same | replicated |
| RMSNorm weights | [4096] | same | replicated (the norm runs on the gathered input) |
| KDA input projection | [4096, 32960] | [4096, 8336] | per-rank packed q, k, v, decay rank (repeated per rank), gate, beta; dim -1 |
| KDA conv taps, decay f_b, scale, bias | per channel / head | / 4 | dim -1 |
| KDA output projection | [8192, 4096] | [2048, 4096] | dim -2 |
| KDA decode state | [B, 64, 128, 128] fp32 | [B, 16, 128, 128] | local heads |
| DSA q_a, q_a norm, kv_a, kv_a norm | [4096, 1536], [4096, 512] | same | replicated |
| DSA q_b | [1536, 16384] | [1536, 4096] | dim -1 (head-major) |
| DSA W_UK, W_UV | [1, 64, 256, 512] | [1, 16, 256, 512] | dim 1 (heads) |
| DSA o_proj | [16384, 4096] | [4096, 4096] | dim -2 (head-major rows) |
| DSA indexer (wq_b, wk, k_norm, weights_proj, pool gate) | — | same | replicated |
| DSA latent cache, pooled keys, page table | [NB x 128, 512], ... | same | replicated |
| Router gate, correction bias | [4096, 192] | same | replicated |
| Routed experts gate / up / down | [192, 4096, 2048] x 3 | [192, 4096, 512] x 3 | intermediate dim |
| Shared expert, dense MLP | [4096, 2048 / 12288] | / 4 | intermediate dim |

Padding: none. All sharded dimensions are multiples of 4 x 32. The decode program configs of stage 3 take the
per-die shapes: the DSA per-head matmuls get a full-grid config with 16 heads per die, and the 1D output-projection
configs get K = 2048 (KDA) or 4096 (DSA).

## Memory (per die, full model)

`tools/memory_budget.py --tp 4` (`memory_budget_tp4.json`), with bfp4 routed experts and bfp8 for the other
matmul weights:

| Group | GiB per die |
|---|---|
| Routed experts | 26.58 |
| KDA projections | 1.42 |
| DSA projections (sharded and replicated) | 0.39 |
| Shared experts, dense MLP | 0.37 |
| Indexer, router gates | 0.22 |
| Embedding, LM head | 0.45 |
| **Total** | **29.42 of 31.88** |

With 1 GiB reserved for trace region, activations and CCL buffers, 1.45 GiB remain per die. One user slot of KDA
state is 35.2 MiB per die. The replicated DSA cache is 11,968 B per token per die. That gives about 127K tokens for 1
slot and 31K in total for 32 slots. The advertised context (1,048,576) needs the DSA cache sharded over the dies by
sequence and a bfp8 latent. The full-model stage records the design and the capacity probe in
`doc/context_contract.json`.

## Results

Runs of 2026-10-16. They used the kernels from before the fp32 precision fix; the post-fix rerun is in the
stage-1 to stage-4 rerun chain.

**Correctness.** `tests/test_multichip_decoder.py` (functional suite on the (1, 4) mesh, policy `opt`, bfp4 bars on
the MoE layers): **44 passed** (`logs/s4_suite.log`, 33 min).

| Check | Result |
|---|---|
| Update PCC vs single die (`test_matches_single_die_baseline`, `refs/baseline_opt.pt`) | 0.99978 to 0.99999 (bar 0.999) |
| Update RMS ratio vs single die | 0.9965 to 1.0035 |
| Decode update scale vs HF, per-test mean | 1.0007 to 1.0061 |
| Lowest update PCC vs HF (synthetic kda_moe prefill, bfp4 experts) | 0.9728 (bfp4 bar after the uniform shift) |

**Performance.** Tracy + tt-perf-report, device time of one pass in µs (`doc/multichip_decoder/tracy/`). Decode
reports are normalized by the number of passes: two `mhc_split_sinkhorn` ops per pass.

| Case | One die (`opt`) | TP4 | Speed-up | CCL time in TP4 |
|---|---|---|---|---|
| decode kda_dense, 32 users | 2997 | 1412 | 2.1x | 117 |
| decode kda_moe, 32 users | 12292 | 4756 | 2.6x | 810 |
| decode dsa_moe, 1 user | 2399 | 1718 | 1.4x | 131 |
| decode dsa_moe, 32 users | 12195 | 5274 | 2.3x | 864 |
| prefill kda_dense, 2048 rows | 38487 | 12438 | 3.1x | 2322 |
| prefill kda_moe, 2048 rows | 62114 | 23401 | 2.7x | 4249 |
| prefill dsa_moe, 2048 rows | 108883 | 35763 | 3.0x | 3891 |

The 1-user KDA decode reports (kda_dense-1, kda_moe-1) hold an incomplete pass (one sinkhorn op). They are rerun in
the post-fix chain. At 32 users the collectives take 15 to 17 % of a MoE layer; stage 5 tunes them (topology,
fused all-gather + matmul).

# GLM-5.3-Flash (E192) multichip decoder

Stage 4 of the tt-model-bringup flow. `tt/multichip_decoder.py` (`MultichipDecoder`) runs the optimized decoder
(`tt/optimized_decoder.py`) on a (1, 4) mesh: p300x2, 4 Blackhole dies, FABRIC_1D. Contract, state, paging and
semantics are those of `tt/functional_decoder.py`.

The stage-4 path is `MCOpts()` (every stage-5 option off, linear topology), selected with
`GLM53_MC="topology=linear,ccl_l1=0,norm_sharded=0,dsa_pack=0,pack_a_1d=0,pack_b_1d=0,dsa_1d=0,router_1d=0,ds_shared=0,prefill_2d=0"`.
The code default `DEFAULT_MC` is the stage-5 selection (`doc/optimized_multichip_decoder/README.md`). Exact commands,
dated events and the review #1 rework are in `work_log.md`.

## Review #1 corrections (2026-10-17)

This README was corrected in place after stage-4 review #1 (`~/glm53-flash-bringup/artifacts/stage4_review_1.md`):

- The "CCL time in TP4" column was wrong (it counted the local MoE row gathers as collectives). The CCL numbers
  and their share are recomputed from the committed `tracy/*/perf_report.csv` (see Performance).
- The 1-user KDA Tracy windows of 2026-10-16 were not an "incomplete pass": `kda.recurrent_decode` raised
  `TT_FATAL ... B * H must be a multiple of 32` (16 local heads x 1 user), and the Tracy wrapper reported `exit=0`.
  Stage 5 fixed it (`MultichipKDA` pads the decode batch with one inactive slot; installed for every `MCOpts`, so the
  stage-4 options carry it), see `doc/optimized_multichip_decoder/README.md` and `work_log.md` 02:31. The two windows
  were rerun on the fixed kernels (`logs/s4_tracy2.log`, 05:29) and are in the tables.
- The speed-up table is rebuilt like for like on the final kernels (see Speed-up and efficiency).
- `test_matches_single_die_baseline` failed on the stage-4 path on 2026-10-17 04:35 against a stale baseline; the
  rerun on a regenerated baseline is in Results. `s4_baseline_cmp2.log` was a `DEFAULT_MC` run and is relabelled
  `default_mc_baseline_cmp2.log`.
- "Padding: none" was wrong: see the padding rows of the sharding table.
- The DSA cache is no longer only replicated: a block-sharded cache (`MCOpts.dsa_shard`) is implemented and
  measured (see Block-sharded DSA cache), and the context contract has a `multichip_decoder` entry.

## Strategy

1D tensor parallelism over the 4 dies (multichip skill default for 1D meshes up to 8 chips):

- Token mixers: TP over heads. KDA has 16 of 64 heads per die. DSA has 16 of 64 query heads per die. With the
  default (replicated) DSA cache, the latent cache, the pooled index keys and the lightning indexer are replicated,
  so top-k pool selection needs no collective. With `dsa_shard` the latent is block-sharded (below).
- FFN: TP over the intermediate dimension. The 192 routed experts are 512 of 2048 columns per die, and so are the
  shared expert and the dense MLP. All routed experts stay on every die (no expert parallelism), so decode reads a
  quarter of every active expert's weights per die and needs no token dispatch.
- Residual: the 4-stream mHC state stays hidden-sharded through the whole stack. Die d holds columns
  [1024 d, 1024 (d + 1)) of every stream (stream-interleaved). `mhc_mix`, `mhc_pre` and `mhc_post` are column-local
  and run unchanged on that slice.

Row-parallel output topologies (multichip skill table):

| Topology | Collectives per sub-block | Used |
|---|---|---|
| Replicated residual: matmul partials -> all-reduce | all-reduce (ring: ~2x the bytes of a reduce-scatter) | no |
| Fused all-gather + matmul (input side) | fused AG + matmul, output RS | no: stage 5 measured the fused ops (`doc/optimized_multichip_decoder/README.md`) |
| Fused matmul + reduce-scatter | fused | no: as above |
| Reduce-scatter + delayed all-gather, sharded residual | RS of the output, AG of the next normed input | **yes**: the residual stays sharded (the low-movement family) |

Rejected alternatives:

| Option | Why not |
|---|---|
| Replicated residual (all-reduce after each sub-block) | A ring all-reduce moves about 2x the bytes of a reduce-scatter, so AG(h) + RS(y) moves about the same as AR(y). The win of the sharded residual is the mHC work and the residual memory at a quarter per die: every die would otherwise repeat `mhc_mix` / `mhc_pre` / `mhc_post` on the full 16384-wide fp32 stream. |
| Expert parallelism (48 experts per die) | At 1 user a token selects 8 of 192 experts: EP puts the load on 1 to 4 dies and needs dispatch / combine collectives; TP spreads every active expert over all 4 dies. At 32 users (up to 256 active expert slots) EP would cut the per-die `moe_gather_rows` / untilize volume about 4x (stage-5 audit: 620 µs gather + 261 µs untilize per MoE layer per die) but keep the SwiGLU element-wise volume, and add a dispatch + combine all-to-all per MoE layer. A hybrid (EP at 32 users, TP at 1 user) is a stage-7 / stage-10 candidate; TP is the skill default and 1-user latency is the priority. GLM-4.5-Air and gpt-oss-120b on the same host use TP. |

## Collectives per layer

Per sub-block (attention, then FFN):

| Step | Op | Shape per die | Bytes per die at decode (32 rows) |
|---|---|---|---|
| mHC mix partials | all-gather, dim 1, fp32 | [1, 1, T, 64] -> [1, 4, T, 64] | 8 KB |
| mixer / FFN input | all-gather, dim 3, bf16 | [1, 1, T, 1024] -> [1, 1, T, 4096] | 64 KB |
| mixer / FFN output | reduce-scatter, dim 3, bf16 | [1, 1, T, 4096] -> [1, 1, T, 1024] | 64 KB |

That is 6 collectives per layer. KDA prefill and the MoE do their reduce-scatter inside the module. The block-sharded
DSA cache adds 3 per DSA layer (below).

## Tensor sharding

| Tensor | Single die | Per die (tp = 4) | Mesh placement |
|---|---|---|---|
| Residual stream X | [1, 1, T, 16384] fp32 | [1, 1, T, 4096] | stream-interleaved, dim -1 |
| mHC fn_T | [16384, 32] fp32 | [4096, 32] | rows permuted to (die, stream, column), dim -2 |
| mHC consts, bcast | — | same | replicated |
| RMSNorm weights | [4096] | same | replicated (the norm runs on the gathered input) |
| KDA input projection | [4096, 32960] | [4096, 8336] (tile-padded to 8352) | per-rank packed q, k, v, decay rank (repeated per rank), gate, beta; dim -1 |
| KDA conv taps, decay f_b, scale, bias | per channel / head | / 4 | dim -1 |
| KDA output projection | [8192, 4096] | [2048, 4096] | dim -2 |
| KDA decode state | [B, 64, 128, 128] fp32 | [B', 16, 128, 128], B' = B padded to B x 16 % 32 == 0 | local heads |
| DSA q_a, q_a norm, kv_a, kv_a norm | [4096, 1536], [4096, 512] | same | replicated |
| DSA q_b | [1536, 16384] | [1536, 4096] | dim -1 (head-major) |
| DSA W_UK, W_UV | [1, 64, 256, 512] | [1, 16, 256, 512] | dim 1 (heads) |
| DSA o_proj | [16384, 4096] | [4096, 4096] | dim -2 (head-major rows) |
| DSA indexer (wq_b, wk, k_norm, weights_proj, pool gate) | — | same | replicated |
| DSA pooled keys, page table, open-pool state | [NB' x 32, 128], [S, MB], 3 x [S, 128] | same | replicated |
| DSA latent cache (default) | [NB x 128, 512] bf16 | same | replicated |
| DSA latent cache (`dsa_shard`) | [NB x 128, 512] bf16 | [(ceil((NB - 1) / 4) + 2) x 128, 544] bf16 | global block b on die b % 4 (local block b // 4); + NULL / SINK blocks, 32 control columns |
| Router gate, correction bias | [4096, 192] | same | replicated |
| Routed experts gate / up / down | [192, 4096, 2048] x 3 | [192, 4096, 512] x 3 | intermediate dim |
| Shared expert, dense MLP | [4096, 2048 / 12288] | / 4 | intermediate dim |

Padding (implicit and explicit). All sharded weight dimensions are multiples of 4 x 32, but the per-die compute pads:

| Where | Padding | Cost |
|---|---|---|
| DSA `sparse_sdpa` (`tt/dsa.py::_sparse_attend`) | 16 local query heads zero-padded to 32 (`sparse_sdpa` takes whole 32-head tiles; the stage-4 smoke failed on `H % tile_h == 0` without it), the pad heads sliced off | each die runs half the single-die head work, not a quarter: 33.4 µs per 1-user pass at TP4 vs 50.8 µs on one die (64 heads); 32 users 42.3 vs 60.6 µs (merged reports / 3) |
| KDA fused input projection | per-rank width 6144 + 128 + 2048 + 16 = 8336 columns, TILE-padded to 8352 | +0.2 % columns |
| KDA decay-rank block | the 128-column rank block is repeated on every rank: 4 x 8336 = 33,344 columns vs 32,960 on one die | +1.2 % input-projection weight / FLOPs |
| KDA 1-slot decode (stage 5, `MultichipKDA`) | batch 1 padded to 2 (B x 16 heads % 32 == 0) | one inactive slot of recurrent state per KDA layer (+1 MiB per layer per die) |
| Block-sharded DSA cache (`dsa_shard`) | latent rows 512 -> 544 (control columns), q heads gathered to 64 | +6.25 % latent bytes; per die 64 heads over the die's share of keys |

Program configs and memory of the sharded matmuls, per die (stage-4 options on the final kernels,
`doc/optimized_multichip_decoder/tracy/before/*/perf_report.csv`; config = ttnn's auto config unless named; all
inputs DRAM interleaved; device µs per call):

| Matmul (M x K x N per die) | Decode, 32 rows | Prefill, 2048 rows |
|---|---|---|
| KDA in-proj 4096 x 8352, bfp8, HiFi4 | 87 cores, in0_block_w 2, subblock 1x3, 99.8 | 110 cores, in0_block_w 1, subblock 1x4, 1304 |
| KDA out-proj 2048 x 4096, bfp8 | 64 cores, in0_block_w 8, 1x2 (1D decode config), 29.2 | 110 cores, in0_block_w 1, 1x4, fp32 in / out, 434 |
| DSA q_b 1536 x 4096, bfp8 | 64 cores, in0_block_w 2, 1x2, 30.8 | 110 cores, in0_block_w 1, 1x4, 750 |
| DSA W_UK / W_UV per head, 16 heads, bfp8 | 16 cores (one head per core, full-grid reuse config), in0_block_w 8, 1x4, 17.8 / 14.8 | 80 cores, in0_block_w 1, 473 / 524 |
| DSA o_proj 4096 x 4096, bfp8 | 64 cores, in0_block_w 8, 1x2 (1D), 53.2 | 110 cores, in0_block_w 1, 1x4, 1921 |
| DSA q_a 4096 x 1536 / kv_a 4096 x 512 (replicated), bfp8 | 48 / 16 cores, in0_block_w 2, 66.4 / 55.6 | 100 / 80 cores, in0_block_w 1, 1155 / 372 |
| Indexer wq_b 1536 x 4096 / wk, gate 4096 x 128 / weights 4096 x 32 (replicated), bf16 | 64 / 4 / 1 cores, 35.4 / 49.5 / 30.6 | wk, gate: 64 cores, in0_block_w 2, 118 / 127 (the flash-path window computes no indexer queries) |
| Router 4096 x 192 (replicated), bf16 -> fp32 | 6 cores, 49.4 | 64 cores, in0_block_w 2, 143 |
| Routed experts gate/up 4096 x 1024, down 512 x 4096, bfp4, LoFi | `sparse_matmul` 110 cores, in0_block_w 4, 1x4: 1281 / 755 (32 users) | 110 cores: 3207 / 1914 per call (2 calls each, MoE rows split at 1024) |
| Shared expert 4096 x 1024 / 512 x 4096, bfp8 | 32 / 43 cores, in0_block_w 8, 21.7 / 9.3 (down input in L1) | 110 cores, 499 / 279 |
| Dense MLP gate/up 4096 x 6144, down 3072 x 4096, bfp8 | — (kda_dense decode: 1D configs) | 110 cores, 2999 / 1451 |

Activations between ops are DRAM interleaved on the stage-4 path (the decode all-gather / reduce-scatter outputs
too; `DEFAULT_MC` moves them to L1 and width-shards the decode norms). Op-internal shardings remain, e.g. the
height-sharded `paged_update_cache` input of the pool keys (one tile row per user).

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

KV / state contract per die (the measured stage-6 numbers are in `doc/full_model/memory_budget.json`):

| Item | Replicated cache (default) | Block-sharded cache (`dsa_shard`) |
|---|---|---|
| DSA latent per token, all 11 DSA layers | 11 x 1024 = 11,264 B | 11 x 544 x 2 / 4 = 2,992 B |
| DSA pool keys per token (replicated) | 11 x 64 = 704 B | 704 B |
| **DSA cache per token per die** | **11,968 B** | **3,696 B** |
| KDA state per slot (34 layers, 16 heads, fp32) | 35.65 MB (+ 1 padding slot at max_batch 1) | same |
| Weights + model-level tensors after load (measured, 1 slot) | 32.12 GB of 34.18 GB | same |
| Largest 1-slot context (full-model capacity probe) | 104,448 (stage 6) | **281,344** (2199 blocks; 281,283-token prompt + 24 decode steps, copy 24/24) |

## Block-sharded DSA cache (`MCOpts.dsa_shard`, `GLM53_DSA_SHARD=1`)

Review #1 P1-1: the replicated latent caps the context at ~10 % of the advertised 1,048,576 tokens, and that is a
layout choice, not a device limit. `tt/dsa_shard.py` implements the block-sharded alternative as an option of the TP
decoder (default off until it is selected as a default).

Design (the paged-pool API is unchanged: GlmDSA page tables / vLLM block tables hold GLOBAL block ids, so
`tt/model.py` / `tt/generator.py` keep calling `set_page_table`, `allocate_dsa_caches(num_blocks)`, `bind_kv_cache`):

- Global block b lives on die b % 4 at local block b // 4; a token's latent row is written only on its owner die
  (per-die scatter row index, host and on-device `DeviceDecodeInputs`). The per-die pool has two more blocks: NULL
  and SINK.
- The pooled index keys (64 B of the 1088 B per token per layer), the indexer, the top-512 pool selection and the
  open-pool state stay replicated: the index list of every query row is the replicated layout's list on every die,
  and needs no collective (no distributed top-k). Sharding them too would give 2,992 B per token per die (+24 %
  capacity) at the cost of a candidate all-gather + top-k per DSA layer.
- Each die maps the logical index list through its own page table: owned blocks -> local block, other blocks -> NULL;
  the indexer-overrun column -> SINK. Every list gets a 32-entry prefix [SINK, 31 x NULL].
- Rows are 544 wide: the latent and 32 control columns. Real rows carry 1 in column 512 (ONE), the SINK row 1 in
  column 513, NULL rows -1e4 in column 514. The query gets [0, q . c_self, 1] there (c_self = the query token's own
  latent row). A real key scores q . k as before; the SINK scores the query's own-key logit (identical on every die);
  a NULL key scores -1e4 (weight exactly 0). So the kernel's output carries, per die, u = sum_local p / D (column 512)
  and w = p_SINK / D (column 513) besides o = sum_local p v / D: a flash-decode LSE merge without an LSE output.
- Per DSA layer: all-gather of the 16 local heads' extended queries (every die attends all 64 heads over its own
  keys, `sparse_sdpa` H = 64, v_dim 544) -> T = o / w and M = m / w (m = 1 - w for w < 0.5, u otherwise: no bf16
  cancellation) -> reduce-scatter (sum, fp32) of T and M over the head groups -> o_lat = sum T / sum M on the die's
  16 heads -> W_UV and o_proj as before. Prefill rows run in blocks of `GLM53_DSA_SHARD_ROWS` (256).
- Torch model of the merge (bf16 kernel outputs, 2051 keys spread over 4 dies): relative error 1.8e-3, the bf16
  rounding of the kernel output itself (PCC 0.9999985).

Validation (`dsa_shard/`; layer 3 = DSA + MoE, real weights, (1, 4) mesh, `DEFAULT_MC` + `dsa_shard`):

| Check | Result |
|---|---|
| `tests/test_dsa_shard.py` (`dsa_shard/s4r_shard_ab.log`): sharded vs replicated layout, same weights / inputs, randomly permuted GLOBAL block table; 8 slots (9001-token synthetic user: 4 sparse chunks; 4000-token real-token user; 2049-token user whose decode crosses the dense -> sparse boundary; 100-token user) and 1 slot (6001 tokens); eager + traced decode | **2 passed**. Attention sub-block output: worst PCC 0.999908, worst \|rms ratio - 1\| 0.0034 over every prefill sequence and decode token |
| Same, layer update | all keys >= 0.9999 except 4 (2 per case): two long synthetic prefills (0.99831 / 0.99854) and one decode token per case (0.99267 / 0.98363), each with a matching attention output |
| Diagnosis of those 4 (`tools/dsa_shard_probe.py`, `dsa_shard/s4r_probe1*.log`) | on the failing token every head's attention output agrees to PCC >= 0.99994 (eager and traced give the same 0.983629), no per-die sink weight underflows (w >= 2.8e-4): a MoE router near-tie flip downstream (the synthetic streams route in the near-tie regime) |
| Suite DSA cases vs HF with the sharded cache (`dsa_shard/s4r_shard_suite.log`, `GLM53_DSA_SHARD=1 ... -k "(dsa_moe or paged_permuted) and not single_die_baseline"`) | **16 passed** (real / synthetic weights, lengths 1 .. 4000 incl. 2049 / 2051 / 2177, batched users, determinism, no host fallback, decode-only, permuted blocks, 32 active users, inactive slots); 184 checks, lowest update PCC 0.98213 (real weights, the L3 32-slot decode near-tie token slot 2 pos 38, margin 8.0e-4, which matches the swapped-expert reference at 0.999885; bar 0.98), decode-mean scale 0.9919 .. 0.9958, y_swap 2 of 161 |
| Single-die compare with the sharded cache (`dsa_shard/s4r_cmp_shard.log`, `opt_s5` baseline) | passed: dsa_moe prefill 0.999954 / 0.999937, decode 0.999977 (rms 1.00052 .. 1.00058); the KDA layers are unchanged by the option |

Capacity (`dsa_shard/capacity/`, `GLM53_TP=4 GLM53_DSA_SHARD=1 tools/memory_probe.py capacity --trace-region 67108864
--workspace-mib 900 --max-context 284672`, 2026-10-17 23:32): full 45-layer model on the serving path, 1 slot. Free
after load 1,984,045,568 B per die -> 2199 blocks at 473,088 B per block (measured pool 1,230,837,248 B) -> warm-up and
traced decode capture -> one **281,283-token** prompt (non-aligned) prefilled in 426.9 s -> 24 traced greedy decode
steps at positions 281,283 .. 281,306 copied the repeated natural-text passage 24/24. Peak 33,776,889,856 of
34,111,622,144 B allocatable (335 MB free; 2.69x the replicated 104,448). The 335 MB of headroom suggests ~+40K more
tokens with a smaller workspace (not probed). The advertised 1,048,576 tokens need 3.9 GB of DSA cache per die even
sharded, against ~1.25 GB available after weights: a hard DRAM limit at this weight precision; the next steps
(pool-key sharding: 2,992 B per token; a bfp8 / fp8 latent; host embedding lookup) are in
`doc/full_model/memory_budget.json` `path_to_1M`. The bfp8 / fp8 latent was not attempted: `sparse_sdpa`'s FP8 formats
would also carry the merge's control columns in fp8 (SCALED_FP8 puts only v_dim latent columns in the output), and the
accuracy of an e4m3 latent is unmeasured.

Cost (A/B wall time, one layer, host-synchronized traced replay): decode 1 slot 1730 -> 2152 µs (+422 µs, +24 %), 8
slots 3154 -> 3563 µs (+409 µs) per DSA layer per step: about 4.6 ms per token over 11 DSA layers (~+8 % of a 1-user
step of ~57 ms). The added work per DSA layer: an all-gather of the 16 local heads' extended queries, `sparse_sdpa` on
64 heads instead of 32 (padded), two reduce-scatters and ~25 small element-wise / layout ops. Prefill (wall incl. host
I/O, first-use compiles on both sides): 9001 tokens 3.25 -> 2.93 s, 6001 tokens 1.72 -> 1.75 s: within noise at these
lengths (not separately profiled with Tracy). Memory: the latent pool of the 8-slot test layer
134.3 MB -> 35.9 MB per die (3.74x, the 544 / 512 row widening included).

### fp8 rows (stage-6 review #2 P1-1, 2026-10-20)

The block-sharded layout now takes the stage-8 fp8 e4m3 latent (`GLM53_PRECISION` `kv_latent_dtype fp8_e4m3`, the
default policy). Set `GLM53_DSA_SHARD=1` to select it. Evidence is in `dsa_shard/fp8/` and
`dsa_shard/capacity/fp8_s1_1048576/`.

Row format:

- The rows are 576 columns wide: 512 latent + 64 control columns, 576 B. `sparse_sdpa` needs K rows of a multiple of
  64 B on Blackhole (`sparse_sdpa_device_operation.cpp:208-214`). A 544-column fp8 row (544 B) is not aligned. Thus
  (b) costs 11 x (576 / 4 + 64) = 2,288 B per token per die, not the 2,200 B of the review.
- ONE and SINK are 1.0. This value is exact in e4m3 and in the kernel's bfp8_b tile format.
- KILL is split between the key and the query. The key holds -256 (exact in e4m3). The bf16 query column holds 40.
  The null logit is -10,240 (bf16 rows: -9,984). Each factor has 3 or fewer significant bits. Thus the product is
  exact at every math fidelity, and the null key weight is exactly 0, as before.
- The query stays bf16, and the attention output (`v_dim`) stays 544 columns. The merge is unchanged.
- The pool-key iota constant (`[1, 1, 1, P_alloc]` fp32, 33.6 MB per layer at 1M tokens) is one tensor for all
  block-sharded layers.
- At the context cap, a vLLM pool has `T/128 + 1` blocks plus the dummy block. The pool-key view
  `[KB, 1, P_alloc, 128]` then rounds to `KB = 2`, which adds 67 MB per layer at 1M tokens. In that case the dummy
  block's pool-key tile aliases vLLM's null block 0. The latent tables keep the dummy block. `tt/dsa.py` (the
  replicated layout) has the same rounding.

| Check | Result |
|---|---|
| `tests/test_dsa_shard.py`, fp8 in both layouts (`GLM53_PRECISION` = selected config) | **2 passed**. Attention output worst PCC 0.999909 (8 slots) / 0.999913 (1 slot), worst \|rms ratio - 1\| 0.0042. 3 layer-update keys are outside the bar with a matching attention output (MoE near-ties, as in the bf16 run). Latent pool of the test layer: 67.2 MB -> 19.0 MB per die |
| Full model, teacher forcing on 3 references (`tools/dtype_sweep.py`, `GLM53_DSA_SHARD=1`, combo_a_kv_dense4_g4, `max_seq_len` 16,384) | top-1 91 / 95 / 91, top-5 100 / 100 / 100, prefill check 92. Replicated control on HEAD with the same command (`doc/datatype_sweep/runs/final_default_h9`): 92 / 95 / 92, top-5 100, prefill check 92 |
| Capacity, 1 slot (`tools/memory_probe.py capacity --workspace-mib 600 --max-context 1048576`, `GLM53_MOE_SORTED_ROWS=512`, `GLM53_DSA_INDEX_BLOCK_MB=32`) | **1,048,576 tokens**. Pool 8,193 blocks (2,399,434,752 B + 180,549,632 B of DSA constants). 3,139,928,064 B free after load, 194 MB free at the warm-up peak. A 1,048,515-token non-aligned copy prompt: prefill 2,665.7 s, then 24 traced decode steps copied the passage 24 / 24. Sink headroom >= 73.6 nats in all 11 layers (bound 88.7), no underflow |
| First capacity attempt | out of DRAM in `bind_kv_cache` because of the doubled pool-key view (above). Fixed in `tt/dsa_shard.py` |

Cost against the replicated fp8 layout, same harness:

| Measure | Replicated | Block-sharded |
|---|---|---|
| Token-out decode, 1 slot, `max_seq_len` 16,384 | 49.94 ms | 52.82 ms (+2.88 ms) |
| TTFT at 128 / 161 / 2,048 tokens | 226 / 378 / 1,024 ms | 226 / 374 / 1,024 ms |
| One DSA layer, traced decode replay (1 / 8 slots) | 1,673 / 3,119 µs | 2,125 / 3,532 µs |

TTFT and token-out at depths above 16K tokens were not measured on the same build. The shared device queue was kept
to the decisive jobs.

Decision: the default stays replicated because it is faster. The block-sharded fp8 layout is the long-context
layout. Recommended long serving profile: 1 slot, `max_model_len` 1,048,576, `GLM53_DSA_SHARD=1`, 600 MiB workspace
(`GLM53_DRAM_RESERVE_MIB=600` for the standalone generator), `GLM53_MOE_SORTED_ROWS=512`,
`GLM53_DSA_INDEX_BLOCK_MB=32`. `serving_profiles.json` and the vLLM adapter do not yet select a DSA layout per profile
(stage 9 / 10).

Pool-key sharding (option (a), 11 x (576 / 4 + 16) = 1,760 B per token) is not implemented. `indexer_score_dsa` scores
the full key length of its view (`indexer_score_device_operation.cpp:95-99`). Thus every die must hold a
`P_alloc`-key view, whichever die owns the keys. To shard the pool keys, the model needs a compact per-die pool
numbering, with per-die limits and id maps in the prefill inputs and in the decode trace, and a merge top-k. This is
not necessary for 1,048,576 tokens at 1 slot. It would increase the multi-slot capacity.

## Results

**Correctness, stage-4 options on the final kernels** (`s4r_suite_s4opts.log`, 2026-10-17 17:53, review #1 rework;
`GLM53_MC` = the stage-4 string above, quantization-aware reference, functional bars, rms gate, against
`refs/baseline_opt.pt` regenerated at 15:29 by `tools/baseline_dump.py` at `GLM53_TP=1`): **45 passed** in 9.5 min.

| Check | Result |
|---|---|
| Update PCC vs single die (bar 0.999) | kda_dense 0.999992 .. 0.999996, dsa_moe 0.999847 .. 0.999980, kda_moe 0.999962 .. 0.999993 |
| Update rms ratio vs single die (bar 1 +- 0.005) | 0.99971 .. 1.00196 |
| HF checks (510; quantization-aware reference on the MoE layers) | lowest update PCC 0.98224 (real weights, L3 32-slot decode slot 2 pos 38: an HF router near-tie token, margin 8.0e-4, that matches the swapped-expert reference y_swap at 0.999885; lowest non-near-tie check 0.98968, L4 slot 3 prefill 200; bar 0.98), update rms ratio 0.9719 .. 1.0352 per check |
| Decode update scale vs HF, per-test mean (bar 1 +- 0.01) | 0.9920 .. 1.0010 |
| Router near-tie y_swap matches (module) | 4 of 298 MoE decode tokens (cap 8) |
| 1-slot vs 2-slot KDA decode (`test_single_slot_kda_decode`) | PCC 1.0000000, max abs 0 |

So the 04:35 failure was the stale baseline: on matched kernels the stage-4 path agrees with the single die to
0.99985+ and 0.2 % scale.

The stage-3 rework then moved `SELECTED` from `opt` to `opt_s5` (`opt` + the stage-5 single-die decode options, which
act at tp 1 only), so the `DEFAULT_MC` suite of 19:30 (`s4r_suite_default.log`: 44 passed; HF checks 508, decode-mean
scale 0.9931 .. 1.0003, y_swap 2 of 298) failed its compare on the policy check. The baseline was dumped again under
`opt_s5` and the compares rerun (2026-10-17 22:2x):

| Single-die compare (`opt_s5` baseline) | kda_dense | dsa_moe | kda_moe | rms ratio |
|---|---|---|---|---|
| stage-4 options (`s4r_cmp_s4opts.log`) | 0.999993 .. 0.999996 | 0.999847 .. 0.999977 | 0.999962 .. 0.999975; decode 0.998789 with one near-tie row set aside, other rows 0.999997 | 0.99838 .. 1.00141 |
| `DEFAULT_MC` (`s4r_cmp_default.log`) | 0.999994 .. 0.999996 | 0.999911 .. 0.999954 | decode 0.998790, one near-tie row set aside, other rows 0.999998 | 0.99821 .. 1.00098 |

The kda_moe decode outlier is one token (user A, position 101, row PCC 0.99645) whose HF router margin is 2.9e-4:
its 8th expert flips on one side (the stage-5 rework found the same token on the single-die side, host comparison in
`doc/optimized_multichip_decoder/records/baseline8_rows.txt`). `test_matches_single_die_baseline` now compares per
token when the aggregate misses a bar: a row may fall below 0.999 only if its HF router margin (the cached reference
behind `tools/baseline_dump.py`'s inputs) is below NEAR_TIE (5e-3), at most max(1, 5 %) of the rows, and the other rows
must meet PCC 0.999 and |rms ratio - 1| <= 0.005. Final runs (2026-10-18 00:2x .. 01:1x, baseline dumped last under the then-current `SELECTED` = `opt_s5_pf`, final
`DEFAULT_MC` with `ag_persistent=False`):

| Run | Result |
|---|---|
| Full multichip suite, `DEFAULT_MC` (`s4r_suite_default_final.log`) | **45 passed**; single-die compare kda_dense 0.999994 .. 0.999996, dsa_moe 0.999910 .. 0.999950, kda_moe 0.999963 .. 0.999976 (decode 0.998790 with the near-tie row set aside, other rows 0.999998), rms 0.99821 .. 1.00098; HF: 508 checks, lowest update PCC 0.98227 (real weights, L3 32-slot decode slot 2 pos 38: an HF router near-tie token, margin 8.0e-4, that matches the swapped-expert reference y_swap at 0.999885; lowest non-near-tie check 0.98968, L4 slot 3 prefill 200; bar 0.98), decode-mean scale 0.9931 .. 1.0003, y_swap 2 of 298 |
| Single-die compare, stage-4 options (`s4r_cmp_s4opts_final.log`) | passed: 0.999803 .. 0.999996 (kda_moe decode near-tie row set aside), rms 0.99838 .. 1.00141 |
| Sharded DSA cache A/B (`dsa_shard/s4r_shard_ab_final.log`) | 2 passed: attention output worst PCC 0.999908, worst rms deviation 0.0034 |
| Suite DSA cases + single-die compare, sharded cache (`dsa_shard/s4r_shard_suite_final.log`) | **17 passed**; dsa_moe vs single die 0.999910 .. 0.999977 |

**History.** `doc/multichip_decoder/s4_suite.log` (2026-10-16, stage-4 code, pre-fp32-fix kernels, bfp4 bar shift on
the MoE layers): 44 passed; update PCC vs single die 0.999782 .. 0.999992, rms ratio 0.99651 .. 1.00353; decode-mean
scale vs HF 1.0007 .. 1.0061; lowest update PCC vs HF 0.9728 (synthetic kda_moe prefill). `refs/baseline_opt.pt`
behind those PCCs was regenerated on 2026-10-17 05:33 (post-fix kernels) and again by the review #1 rework run.
`tools/baseline_dump.py` now builds the one-die baseline with the stage-3 single-die-only knobs off (`GLM53_BASELINE_FROZEN=1`, default: no `single_die_r2` / `kda_in_split_ds`, which TP4 does not run; W_UK prefill reuse on as at the 2026-10-18 00:21 dump), so a re-dump reproduces the frozen `refs/baseline_opt.pt` policy (host-side config diff against 08bac32f4c1: identical).
2026-10-17 04:35 (`logs/s4_mc_suite2.log`): 43 passed, 1 failed (the single-die compare of the stage-4 path against
the stale pre-fix baseline, kda_moe decode PCC 0.998733, rms 0.99395; see `work_log.md`).

The suite now gates scale: `test_matches_single_die_baseline` asserts |rms ratio - 1| <= 0.005 besides PCC >= 0.999
and fails (no longer skips) when the baseline file is missing, and the HF checks use the quantization-aware reference
of the stage-3 rework (`QUANT_REF = opt.quant_spec()`, `tests/hf_ref/hf_quant_ref.py`) with the functional bars
instead of the bfp4 bar shift (`GLM53_REF_QUANT=0` restores the shift for comparison).

## Performance

Tracy + tt-perf-report. Two readings of the same windows:

- merged report (`tracy/*/perf_report.csv`, committed): tt-perf-report merges the 4 devices into one row per op (the
  slowest device's row); total / passes (decode: 3 traced passes per window; prefill: one signposted pass);
- device 0 (`ops.csv`, one traced replay session; `scripts/s5_ops.py`): the stage-5 README's convention; 1 to 2 %
  below the merged sum.

Stage-4 windows (`doc/multichip_decoder/tracy/`), µs per pass, merged report. Provenance: decode 32-user and 1-user
dsa_moe windows and the prefill windows ran 2026-10-16 20:11 .. 20:23 (`logs/s4_tracy.log`) on the pre-fp32-fix
kernels; the two 1-user KDA windows are the 2026-10-17 05:29 rerun (`logs/s4_tracy2.log`) on the fixed kernels with
the stage-4 options (the 2026-10-16 runs crashed, above). CCL = all-gather + reduce-scatter rows:

| Case | TP4 total | CCL | CCL share | Kernels / date |
|---|---|---|---|---|
| decode kda_dense, 1 user | 1152 | 98 | 8.5 % | post-fix, 10-17 05:29 |
| decode kda_dense, 32 users | 1412 | 100 | 7.1 % | pre-fix, 10-16 |
| decode kda_moe, 1 user | 1369 | 107 | 7.8 % | post-fix, 10-17 05:31 |
| decode kda_moe, 32 users | 4756 | 128 | 2.7 % | pre-fix, 10-16 |
| decode dsa_moe, 1 user | 1718 | 108 | 6.3 % | pre-fix, 10-16 |
| decode dsa_moe, 32 users | 5274 | 118 | 2.2 % | pre-fix, 10-16 |
| prefill kda_dense, 2048 rows | 12438 | 1975 | 15.9 % | pre-fix, 10-16 |
| prefill kda_moe, 2048 rows | 23401 | 2009 | 8.6 % | pre-fix, 10-16 |
| prefill dsa_moe, 2048 rows | 35763 | 1735 | 4.9 % | pre-fix, 10-16 |

The collectives are 2 to 3 % of a 32-user MoE layer and 6 to 9 % of a 1-user layer; at decode the MoE (sparse
matmuls, row gathers, untilize) and the element-wise / data-movement ops dominate. The earlier "15 to 17 %" counted the
local `MoeGatherRows` / `MoeGatherAddRows` (658 µs at kda_moe-32) as collectives.

### Speed-up and efficiency (like for like, final kernels)

Single die: the stage-3 final `opt` windows (`doc/optimized_decoder/tracy/`, 2026-10-17 13:07 .. 13:10). TP4: the
stage-5 `tracy/before` windows (stage-4 options, same kernels, `doc/optimized_multichip_decoder/tracy/before/`,
2026-10-17 07:1x), device 0. Both prefill sides run the MoE rows split at 1024 (`GLM53_MOE_SORTED_ROWS` default,
4 routed sparse matmuls per pass). Efficiency = speed-up / 4.

| Case | Single die | TP4 (stage-4 options) | Speed-up | Efficiency |
|---|---|---|---|---|
| decode dsa_moe, 1 user | 2421 | 1745 | 1.39x | 35 % |
| decode kda_moe, 1 user | 2287 | 1338 | 1.71x | 43 % |
| decode kda_dense, 1 user | 2082 | 1134 | 1.84x | 46 % |
| decode kda_moe, 32 users | 11530 | 4599 | 2.51x | 63 % |
| decode dsa_moe, 32 users | 11779 | 4975 | 2.37x | 59 % |
| decode kda_dense, 32 users | 2901 | not re-measured on the final kernels (stage-4 run, pre-fix kernels: 1398 device 0) | (2.08x) | (52 %) |
| prefill kda_dense, 2048 rows | 29966 | 12179 | 2.46x | 62 % |
| prefill kda_moe, 2048 rows | 66865 | 25601 | 2.61x | 65 % |
| prefill dsa_moe, 2048 rows | 97938 | 33523 | 2.92x | 73 % |

The 1-user decode efficiency is low because the replicated work (the five DSA / indexer projections of the normed
input, the norms on the gathered input, the mHC Sinkhorn: 179 µs per pass) and the 6 collectives do not shrink with
TP; stage 5 cuts the replicated projections and norms (`DEFAULT_MC`: dsa_moe 1 user 1745 -> 1348 µs).

## Watcher

- Stage-5 runs (`doc/optimized_multichip_decoder/watcher/`, `DEFAULT_MC`, `test_real_weights[*]`: 100-row prefill and
  B = 8 decode per kind) passed with no watcher assert. Each process ends with `Timed out while waiting for active
  ethernet core 29-25` at mesh close: the fabric-router teardown under the watcher on this board (the non-watcher
  suites close cleanly; the 122B port shows the same).
- Stage-4-option runs (review #1 P2-4; `doc/multichip_decoder/watcher/<case>/`, `GLM53_MC` = the stage-4 string,
  `GLM53_WORKER_L1_SIZE=1444864`, `TT_METAL_WATCHER=10`): `test_real_weights[kda_dense|dsa_moe|kda_moe]` (linear
  collectives with DRAM outputs, interleaved norms on the gathered h, the separate DSA projections at 16 heads),
  `test_batch32_all_active[kda_moe]` (32-user MoE), `test_lengths[dsa_moe-2051]` (multi-chunk prefill, dense -> sparse
  decode) and `test_single_slot_kda_decode` (the 1-slot padded KDA path). Results per case in
  `watcher/s4r_watcher.status`: **all 6 passed**, every `watcher/<case>/generated/watcher/watcher.log` has 0 lines
  matching error / assert / sanitize / overflow / exception / bad; the processes end with the same eth-core 29-25
  teardown timeout as stage 5 (watcher-only fabric teardown; the non-watcher suites close cleanly).

## Limitations

- The 1-user decode efficiency is 35 to 46 % (above). The stage-4 path is superseded by `DEFAULT_MC`.
- The single-die compare covers single-chunk prefill (100 / 64 rows) and 3 eager B = 8 decode steps; traced decode,
  32-user decode, multi-chunk prefill and B = 1 are compared against HF only.

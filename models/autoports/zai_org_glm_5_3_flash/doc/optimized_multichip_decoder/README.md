# GLM-5.3-Flash (E192) optimized multichip decoder

Stage 5 of the tt-model-bringup flow. The pass optimizes `tt/multichip_decoder.py` (`MultichipDecoder`, TP4 on the
(1, 4) mesh of a p300x2: 4 Blackhole dies, FABRIC_1D) in place. Contract, state, paging and semantics stay those of
`tt/functional_decoder.py`. Decode is the priority (tokens/s/user, then 32-user throughput), then prefill.

The stage-5 changes are options of one dataclass, `MCOpts` (`tt/multichip_decoder.py`). `DEFAULT_MC` holds the
selected values. `GLM53_MC="key=value,..."` overrides fields at construction (A/B runs). `MCOpts()` (all options off, linear
topology) is the stage-4 path; `GLM53_MC="topology=linear,ccl_l1=0,norm_sharded=0,dsa_pack=0,pack_a_1d=0,pack_b_1d=0,
dsa_1d=0,router_1d=0,ds_shared=0,prefill_2d=0"` selects it.

## Result

`DEFAULT_MC` (selected): ring topology when the fabric routes it (`topology="auto"`), 2 links, decode collective
outputs in L1, width-sharded decode RMSNorms, the five replicated DSA projections packed into two matmuls
(q_a | kv_a bfp8 and indexer wk | pool gate | head weights bf16, both interleaved on 1D decode configs), the DSA q_b /
wq_b and the MoE router on 1D decode configs, the packed weights shared with prefill (`ds_shared`: the separate weights
are freed, no second copy), and the DSA prefill matmuls on an explicit 2D config (`prefill_2d`).

Decode, device time of one traced replay (Tracy, device 0, µs; `tracy/before` = stage-4 options, `tracy/after` =
`DEFAULT_MC`, same code and run day; the 1-user KDA "before" runs already carry the 1-slot fix, without which they
fail):

| Layer kind | Users | Before | After | Change |
|---|---|---|---|---|
| kda_dense (L0) | 1 | 1134 | 979 | -13.7 % |
| kda_moe (L4) | 1 | 1338 | 1157 | -13.5 % |
| dsa_moe (L3) | 1 | 1745 | 1348 | -22.8 % |
| kda_moe | 32 | 4599 | 4380 | -4.8 % |
| dsa_moe | 32 | 4975 | 4536 | -8.8 % |

Decode, traced replay wall (`tools/mc_decode_ab.py`, 30 back-to-back replays, one build per row, µs; rounds 5 and 3):

| Layer kind | Users | Stage-4 options (base / base2) | DEFAULT_MC | Change |
|---|---|---|---|---|
| kda_dense | 1 | 1167 / 1166 | 1035 | -11.3 % |
| kda_moe | 1 | 1371 / 1371 | 1213 | -11.5 % |
| dsa_moe | 1 | 1805 / 1804 | 1432 | -20.6 % |
| kda_moe | 32 | 4610 / 4547 | 4361 | -4.8 % |
| dsa_moe | 32 | 5122 / 5121 | 4753 | -7.2 % |

For the 45-layer stack (3 kda_dense, 31 kda_moe, 11 dsa_moe layers) the layer replays sum to 65.9 ms per token
before and 56.5 ms after at 1 user (decoder layers only, without embedding, head and sampling): 15.2 -> 17.7
tokens/s/user from the layers (-14.3 %). At 32 users the MoE dominates (see Open items).

Prefill, 2048 rows, device time of the signposted pass (Tracy, device 0, µs):

| Layer kind | Before | After | Change | CCL before -> after |
|---|---|---|---|---|
| kda_dense | 12179 | 11670 | -4.2 % | 1889 -> 1365 |
| kda_moe | 25601 | 24815 | -3.1 % | 2150 -> 1527 |
| dsa_moe | 33523 | 31488 | -6.1 % | 1617 -> 1032 |

Correctness on the final `DEFAULT_MC` (`tests/test_multichip_decoder.py`, `logs/s5_suite2.log`): **45 passed** in 9 min
(the stage-4 functional suite on the (1, 4) mesh with the bfp4 bars on the MoE layers, the single-die baseline compare
and the new 1-slot KDA test; the intermediate default of round 3 also passed 45 / 45, `logs/s5_suite.log`).

| Check | Stage 4 (`logs/s4_suite.log`) | Stage 5 |
|---|---|---|
| Update PCC vs the single-die TTNN decoder (`refs/baseline_opt.pt`, bar 0.999) | 0.99978 to 0.99999 | kda_dense 0.999993 to 0.999996, dsa_moe 0.999909 to 0.999954, kda_moe 0.999963 to 0.999993 |
| Lowest update (delta) PCC vs HF over the suite (synthetic kda_moe prefill, bfp4 experts) | 0.972796 | 0.972638 |
| 1-slot vs 2-slot KDA decode (3 eager steps + 1 traced replay) | failed (`TT_FATAL`) | bit-identical |
| Full model on the round-3 `DEFAULT_MC` (DRAM-sharded q_a / kv_a pack; main agent's `tools/readiness prefill-check`, 45 layers, `logs/s6_prefill_check3.log`) | - | top-1 0.92, top-5 1.00 over 100 positions |

Watcher (`TT_METAL_WATCHER=10`, `scripts/s5_watcher.sh`, worker L1 1,444,864 B so that the watcher-instrumented KDA
prefill programs fit the kernel-config ring as in stage 1): `test_real_weights` of every layer kind (prefill, eager and
traced decode) **passed** with no watcher assert (`watcher/<case>/`). Each process then aborts at mesh close with
`Timed out while waiting for active ethernet core 29-25 to become active again`: the fabric-router teardown under the
watcher on this board (seen on every watcher process of the 122B port too); a board reset clears it.

## Method

- A/B probe: `tools/mc_decode_ab.py` builds one real layer per (kind, batch), then for every variant sets the
  options, runs one eager step, captures the decode trace and times 30 back-to-back replays (`replay`, the
  device-bound step) and 30 replays with the host input copies (`step`, as `tests/test_perf_decoder.py`). One build
  serves every variant. `base` runs first and `base2` last: their difference is the run noise (0 to 50 µs). Unit
  checks (`--units`) compare each option with the stage-4 path on one input and with an fp64 product of the same
  device weights.
- Prefill probe: `tools/mc_prefill_ab.py` (warm 2048-row pass, best of 3, host wall with a sync; `--save` / `--ref`
  compare outputs across builds).
- Tracy: `~/glm53-flash-bringup/scripts/s5_tracy.sh before|after` (`tests/test_perf_decoder.py` windows,
  `tt-perf-report`). Device times are device 0, one traced replay (decode) or the signposted pass (prefill), sum of
  `DEVICE KERNEL DURATION` (`~/glm53-flash-bringup/scripts/s5_ops.py`). `tracy/before` = stage-4 options on the
  same code, `tracy/after` = `DEFAULT_MC`.

Correction to the stage-4 README: its column "CCL time in TP4" (117 to 864 µs) summed the merged 4-device report.
Per device, one decode pass holds 6 collectives at 95 to 100 µs in total (2 % to 7 % of the layer). The stage-4
"incomplete 1-user KDA pass" was a failure: see "1-slot KDA decode" below.

## Operation-topology audit (decode, per layer, per device)

Stage-4 measurements (`doc/multichip_decoder/tracy`, dsa_moe 1 user = 1680 µs device, kda_moe 32 users = 4620 µs).

| Op sequence | Material cost | Candidate | Constraint | Action |
|---|---|---|---|---|
| mHC mix: `mhc_mix` split-K partials -> `ttnn.sum` dim 1 (transpose, reduce, transpose) -> all-gather fp32 [T, 64] -> sum -> 9 small ops (Gram diagonal, rsqrt, scale) | 42 µs + 13 µs CCL per sub-block | `fast_reduce_nc` for the two sums | fp32-exact sums (the mHC precision fix) | rejected: 4.3e-4 relative error (TF32 inputs of the FPU add) vs 7e-8 for `ttnn.sum` |
| `mhc_split_sinkhorn` | 60 µs per call, 2 per layer, 1 core | multi-core over users | one tile holds all 32 users; 40 sequential normalizations of that tile | no gain possible by row split (see Sinkhorn) |
| all-gather h [T, 1024] bf16 -> RMSNorm [T, 4096] | 17 µs + **65 µs** (ttnn splits an interleaved norm by tile rows: 1 core) | width-sharded L1 norm on 32 cores | fp32 accumulation kept | **taken** (`norm_sharded`) |
| DSA: five projections of the same normed x (q_a 1536, kv_a 512 bfp8; indexer wk 128, pool gate 128, head weights 32 bf16) on 1 to 48 cores | **250 µs** | two packed matmuls on 1D configs (DRAM-sharded measured) | mixed dtypes: one bfp8 pack, one bf16 pack | **taken** (`dsa_pack`, `pack_a_1d`, `pack_b_1d`, `ds_shared`) |
| DSA q_a / kv_a norms [T, 1536] / [T, 512] | 25 + 11 µs on 1 core | sharded norm (24 / 16 cores) | - | **taken** (`norm_sharded`) |
| MoE router [T, 4096] x [4096, 192] bf16, fp32 out | 49 µs on 6 cores | DRAM-sharded or 1D config | fp32 routing precision (stage-1 route flips) | **taken**: 1D config, 6 cores, in0 blocks of 8 (`router_1d`) |
| mixer / FFN output: row-parallel matmul -> reduce-scatter bf16 | 17 to 32 µs per RS | fused matmul + RS; bfp8 payload; ring | - | ring taken; fused / bfp8 rejected (CCL table) |
| MoE 32 users: sorted dispatch -> `moe_gather_rows` [G x 32, 4096] -> sparse gate/up -> SwiGLU -> sparse down -> untilize -> gather-add | gather 620, untilize 261, SwiGLU element-wise 362, sparse matmuls 1866 µs | fewer padded rows per expert group | tile = 32 rows per expert group; ~1.8 real rows per group at 32 users | open item (below) |
| inter-layer boundary | none: the residual stays stream-interleaved, hidden-sharded | - | - | contract below |

## Candidates and decisions

Replay wall of one traced decode step in µs (`tools/mc_decode_ab.py`); each row is measured against `base` of the
same build. Records: `~/glm53-flash-bringup/artifacts/s5/ab{1,2,3,5}.jsonl`, `pf{2,5}.jsonl`, logs `logs/s5_ab1.log`,
`logs/s5_ab2.log`, `logs/s5_round3.log`, `logs/s5_round5.log`, per-run `logs/s5_ab_<kind>_<batch>_<time>.log`.

| Option (`MCOpts`) | Measurement | Precision (unit check, `--units`) | Decision |
|---|---|---|---|
| `norm_sharded` (decode RMSNorms width-sharded on 32 / 24 / 16 cores) | dsa_moe 1 user 1806 -> 1684; kda_moe 32 users 4587 -> 4417 | PCC 0.999999 vs the 1-core norm, max 1 bf16 ulp | taken |
| `dsa_pack` (5 projections -> 2 packed matmuls) | dsa_moe 1 user 1806 -> 1686; with `norm_sharded` 1548 | q_a / kv_a / wk / gate / weights within the bf16 rounding of the separate matmuls (fp64 reference) | taken |
| pack A (q_a, kv_a) DRAM-sharded vs 1D (`pack_a_1d`) | dsa_moe 1 user: DS 1487, 1D 1436; 32 users: DS 4800, 1D 4758 (Tracy: the DS kernel ran 62.6 µs on 80 cores, 142 GB/s) | DS with packer L1 accumulation: q_a 3.0e-3 vs 2.9e-3 separate | 1D taken (faster; interleaved weight, exact accumulation) |
| pack B (wk, gate, weights) DRAM-sharded vs 1D (`pack_b_1d`) | dsa_moe 1 user: DS 1548, 1D 1519 | gate (fp32 out): DS 7.0e-4, 1D 3.1e-4 = separate | 1D taken (faster, exact, no bank padding) |
| router DRAM-sharded (`router_ds`) vs 1D (`router_1d`) | kda_moe 1 user: DS 1222, 1D 1224 (base 1398 with norm/L1/ring 1255) | fp32 logits: interleaved 3.0e-4, DS 3.9e-3 (5.2e-4 to 7.1e-4 with L1 accumulation), 1D 3.0e-4 | 1D taken (same speed, routing-exact, no padding) |
| `topology` ring vs linear | decode dsa_moe 1 user 1806 -> 1783; prefill kda_moe 26.9 / 25.9 -> 25.5 ms, dsa_moe 33.8 -> 33.3, kda_dense 12.8 -> 12.2 ms | all-gather bit-identical; reduce-scatter sums in another order (prefill dsa_moe PCC vs linear 0.9995, suite passes) | taken as `auto` (`ttnn.get_usable_topology`: Ring on this box; Linear on a line-cabled box) |
| `num_links` 1 vs 2 | dsa_moe 1 user 1821 (1 link) vs 1806; prefill kda_moe 27.1 vs 26.9 ms | identical | 2 (all available) |
| `ccl_l1` (decode collective outputs in L1) | dsa_moe 1 user 1806 -> 1797; kda_moe 32 users 4587 -> 4501 | identical | taken |
| `ag_persistent` (preallocated all-gather outputs) | dsa_moe 1 user 1525 -> 1516; kda_moe 1 user 1255 -> 1249 | identical values | not taken (see Rejected) |
| `mix_nc` (fast_reduce_nc for the mix sums) | dsa_moe 1 user 1806 -> 1778 | 4.3e-4 relative vs 7e-8 | rejected (mHC fp32 exactness) |
| `ds_shared` (packed weights serve prefill, separate weights freed) | prefill dsa_moe 33.8 -> 32.8 ms | prefill output bit-identical (max abs 0) to the separate weights | taken |
| `dsa_1d` (q_b, wq_b decode on 1D configs) | dsa_moe 1 user 1487 -> 1481 (with 1D pack A: 1436 -> 1432); 32 users 4758 -> 4753 | - | taken (same weights, no memory) |
| `prefill_2d` (DSA prefill matmuls on an explicit 2D config) | prefill dsa_moe 2048 rows: 33.35 / 33.32 ms -> 31.39 ms (`logs/s5_round5.log`, `pf5.jsonl`) | - | taken (output PCC 1.0 vs the auto config) |
| 1-slot KDA: per-slot inputs allocated with the padded slot count | kda_moe 1 user: stage-4 options 1398 -> 1371, DEFAULT_MC 1224 -> 1213; kda_dense 1044 -> 1035 (the in-tile pads of the [B, 1] inputs ran on 1 core at 10 to 11 µs each) | identical (same rows) | taken |

## Collectives (OPT-009 fields)

Six collectives per layer, all `ttnn.experimental.*_async` through `models/tt_transformers/tt/ccl.py` (`MeshCCL` in
`tt/multichip_decoder.py`), `chunks_per_sync` 10, `num_workers_per_link` 2, `num_buffers_per_channel` 2, 2 links,
cluster axis none (1D mesh).

Decode (Tracy, device 0, µs per call, before -> after; 2 calls per kind per layer):

| Collective | Payload per device | Dtype | Output memory (after) | Persistent buffers | kda_moe 1 user | dsa_moe 1 user | kda_moe 32 users | Decision |
|---|---|---|---|---|---|---|---|---|
| all-gather of the mix partials, dim 1 | [T, 64] -> [4, T, 64] | fp32 | L1 | none (`ag_persistent` measured, not adopted) | 12.0 -> 9.9 | 12.7 -> 9.4 | 12.8 -> 10.2 | ring, L1 |
| all-gather of h, dim 3 | [T, 1024] -> [T, 4096] | bf16 | L1 | none (same) | 17.8 -> 12.6 | 17.0 -> 11.6 | 18.6 -> 11.7 | ring, L1 |
| reduce-scatter of the mixer / FFN partials, dim 3 | [T, 4096] -> [T, 1024] | bf16 (intermediate DRAM) | L1 | none (122B: no gain) | 16.2 -> 14.2 | 19.3 -> 15.4 | 18.6 -> 11.9 | ring, L1 |

Per layer: 6 collectives, 92 to 100 µs before, 68 to 73 µs after. Prefill: 1.6 to 2.2 ms -> 1.0 to 1.5 ms per layer
(ring).

## Inter-layer residual contract (for the full model)

- Input and output of every layer: the 4-stream mHC state X [1, 1, T, 4 x 4096] fp32, stream-interleaved and
  hidden-sharded. Device d holds columns [1024 d, 1024 (d + 1)) of every stream: [1, 1, T, 4096] per device, DRAM
  interleaved, TILE.
- No collective sits between layers. Inside a layer, each sub-block gathers the 64 fp32 mix partials and the 1024-wide
  h of every device, and reduce-scatters the 4096-wide output back to the 1024-wide slice before `mhc_post`.
- The embedding writes X in this layout (`tt/model.py`), and the head gathers only at the end (hc_head, final norm).
- Decode rows: one tile of users (B <= 32). With B = 1 the KDA layers pad their batch to 2 internally.

## 1-slot KDA decode

`kda.recurrent_decode` needs batch x local heads % 32 == 0. TP4 leaves 16 KDA heads per die, so a decoder built for
one slot (B = 1, a single-user serving profile) failed in every KDA layer (`TT_FATAL ... B * H must be a multiple of
32`, `logs/s5_ab1.log`). `MultichipKDA` allocates the state for 2 slots, pads x and the per-slot inputs with one
inactive row (state unchanged, the row is dropped), and leaves B >= 2 untouched.
`tests/test_multichip_decoder.py::test_single_slot_kda_decode` checks slot 0 of a 1-slot decoder against slot 0 of a
2-slot decoder (3 eager steps and one traced replay).

## Sinkhorn

`mhc_split_sinkhorn` already splits token tiles over cores (prefill: 64 cores at 2048 rows). Decode has one tile (all
32 users), and the tile is the unit of FPU / SFPU work: a core with 8 valid rows takes as long as a core with 32. Its
60 µs (stage 4; 90 µs with the fp32-exact kernel committed on 2026-10-17) are 40 sequential row / column normalizations of that tile (8 permutation matmuls, a reciprocal and a multiply
each, hi / lo operand splits for fp32 exactness). Splitting users over cores gives no gain. A kernel-level speed-up
(e.g. HiFi3 for the permutation matmuls, exact because the permutation operand is 0 / 1) needs a program-factory
change and a rebuild; it is an open item.

## Memory and context contract

`DEFAULT_MC` adds no persistent device memory: both DSA packs are interleaved concatenations of the separate weights
(no padding), `ds_shared` frees the separate weights after packing (the build peaks at one extra packed copy of one
layer, 11 MB, for a moment), and the router keeps its weight. Decode collective outputs in L1 and
the sharded norms are activations of a step (L1, < 300 KB). The context contract (`doc/context_contract.json`,
`optimized_multichip_decoder`) is unchanged. Rejected variants would have cost memory: a DRAM-sharded indexer pack
+1.8 MB per DSA layer, a DRAM-sharded router +0.5 MB per MoE layer (bank padding), separate decode copies of the
packed weights +12.9 MB per DSA layer.

## Rejected and not adopted

| Candidate | Evidence | Reason |
|---|---|---|
| Multi-core sinkhorn (split users over cores) | 60 µs per call at 1 and at 32 users (stage-4 Tracy), 90 µs after the fp32-exact kernel | one tile holds every user; see Sinkhorn |
| `fast_reduce_nc` mix sums (`mix_nc`) | -28 µs per layer | 4.3e-4 relative error (TF32 inputs) vs 7e-8: breaks the fp32-exact mHC |
| DRAM-sharded router / indexer pack | same speed as 1D or slower; 2x to 10x the error on fp32 outputs | 1D configs are exact, as fast, and need no bank padding |
| Persistent all-gather outputs (`ag_persistent`) | -6 to -9 µs per layer (0.5 %) | the 122B port on this board saw traced replays lose bit-reproducibility with persistent all-gather outputs; the gain is at the noise level here; option kept |
| Fused all-gather + matmul (`all_gather_minimal_matmul_async`) | not built here: an RMSNorm over the gathered row sits between the all-gather and every consumer matmul | 122B on this board (decode M = 32): 0.140 to 0.598 ms vs 0.067 to 0.070 ms for all-gather + 1D matmul (`qwen_qwen3_5_122b_a10b/doc/optimized_multichip_decoder/README.md`) |
| Fused matmul + reduce-scatter (`matmul_reduce_scatter_async`) | output projections 28 µs (KDA), 52 µs (DSA) + RS 13 to 15 µs | 122B on this board: 2D configs only, at most 8 cores compute for one tile of rows: 0.063 to 0.087 ms vs 0.047 ms separate |
| bfp8 collective payloads | RS / AG are 9 to 15 µs, latency bound | 122B on this board: -1.4 µs RS, -2.6 µs AG but an extra typecast; within noise |
| `all_reduce_async` instead of reduce-scatter (replicated residual) | - | 4x the bytes and the mHC work on every die (stage 4); 122B: 38.4 vs 35.9 µs |
| Fused MoE gate op (`generalized_moe_gate`, sigmoid + bias + top-k) | would replace ~15 router ops (~90 µs at 1 user) | bf16 scores in the op; the GLM router runs fp32 scores and fp32 top-k (stage-1 route-flip fix) |
| tt-perf-report: "Output subblock 1x1 is small" (1D decode matmuls) | one output tile per core is the widest grid for N <= 64 tiles | a 1x2 subblock halves the cores of a bandwidth-bound matmul |
| tt-perf-report: "HiFi2 may also work" (decode and prefill matmuls at HiFi4) | stage 3: HiFi2 left the shared expert 0.8 % low; HiFi4 norms / router / dense MLP / KDA projections are stage-1 accuracy fixes | precision policy belongs to the datatype sweep (stage 8) |
| tt-perf-report: "place input 0 in L1" (decode activations in DRAM) | collective outputs moved to L1 (`ccl_l1`): -9 µs per layer; the other decode activations are 64 to 256 KB | further moves cost a reshard each; the activations are not bandwidth relevant |
| tt-perf-report: "Try a DRAM-sharded program config" (indexer pack, router, shared expert) | measured for the indexer pack and the router (rows above) | slower or less exact; the shared expert (22 µs) would need a DRAM-sharded copy of 4.4 MB per MoE layer or a prefill path on the sharded weight |

## Open items

- Sinkhorn kernel: 90 µs per call (fp32-exact kernel), 180 µs per layer = 15 % of a 1-user layer. The permutation
  matmuls could run at HiFi3 (exact for a 0 / 1 operand); needs a program-factory change and a rebuild.
- 32-user MoE (about 3.3 ms of a 4.5 ms kda_moe layer): `moe_gather_rows` 620 µs and `untilize` 261 µs move 195 expert
  groups x 32 rows x 4096 per die (about 1.8 real rows per group), and the SwiGLU element-wise ops run on the same
  6240 rows (362 µs). Smaller row groups (tiny tiles) or a fused gather + SwiGLU would cut this. The down projection
  cannot write row-major output (its resident block, 1.7 MB, exceeds the grouped factory's L1 budget at TP4).
- KDA input projection (100 µs per KDA layer, 36 MB bfp8 at 365 GB/s): a DRAM-sharded weight (8352 -> 8448 columns)
  shared with the KDA prefill kernels.
- Prefill matmuls still on `in0_block_w = 1` (tt-perf-report advice, `tracy/after/*/perf_report_advice.txt`): the
  dense MLP (2048 x 4096 x 6144: 3.0 ms, 2048 x 3072 x 4096: 1.5 ms per dense layer), the shared expert (0.5 + 0.3 ms
  per MoE layer), the KDA output projection (0.4 ms) and the DSA packed projections (1.0 + 0.2 ms). `prefill_2d`
  (explicit 2D config) took the DSA q_b / wq_b / o_proj from 33.3 to 31.4 ms per dsa_moe layer with a bit-identical
  output; the same config in `tt/mlp.py`, `tt/moe.py` and the KDA prefill is a stage-7 (prefill) item.
- Single-user serving runs the decoder with `max_batch` slots (`tt/model.py` default 32): the 1-user numbers above
  need `GLM53_MAX_BATCH=1` in the full model.

## Artifacts

- Code: `tt/multichip_decoder.py` (`MCOpts`, `DEFAULT_MC`, `MeshCCL`, `DSMatmul`, `IL1DMatmul`,
  `sharded_rms_norm`, `MultichipDSA`, `MultichipKDA`, router hook), `tt/moe.py` (collective links / output memory
  hooks), `tests/test_multichip_decoder.py::test_single_slot_kda_decode`, `tools/mc_decode_ab.py`,
  `tools/mc_prefill_ab.py`.
- Tracy + tt-perf-report: `tracy/before/<case>/`, `tracy/after/<case>/` (`ops.csv.gz`, `perf_report.{txt,csv}`,
  `perf_report_advice.txt` for the advice runs, `console.log`).
- Watcher: `watcher/<case>/` (`console.log`, `generated/watcher/watcher.log`).
- Logs (`~/glm53-flash-bringup/logs/`): `s5_ab1.log`, `s5_ab2.log`, `s5_round3.log`, `s5_round5.log`,
  `s5_tracy.log`, `s5_suite.log` (round-3 default), `s5_suite2.log` (final), `s5_watcher.log` (first attempt: KDA
  prefill programs over the kernel-config ring under the watcher), `s5_watcher2.log` (final); probe records
  `artifacts/s5/*.jsonl`.
- Commands:
  ```
  GLM53_TP=4 pytest models/autoports/zai_org_glm_5_3_flash/tests/test_multichip_decoder.py
  GLM53_TP=4 python -m models.autoports.zai_org_glm_5_3_flash.tools.mc_decode_ab --kind dsa_moe --batch 1 --units \
      --variants "base:;fin:topology=auto,ccl_l1=1,norm_sharded=1,dsa_pack=1,pack_a_1d=1,pack_b_1d=1,dsa_1d=1,router_1d=1" \
      --out ab.jsonl
  GLM53_TP=4 python -m models.autoports.zai_org_glm_5_3_flash.tools.mc_prefill_ab --kind dsa_moe --variants "base:;ring:topology=ring" --out pf.jsonl
  ~/glm53-flash-bringup/scripts/s5_tracy.sh after        # GLM53_MC unset = DEFAULT_MC
  ```

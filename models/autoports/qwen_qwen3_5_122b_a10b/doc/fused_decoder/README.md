# Fused decoder — Qwen/Qwen3.5-122B-A10B (stage 2)

Status: evidence complete (equivalence, before/after perf with complete profiler windows, watcher, stress); commands in work_log.md.

`tt/fused_decoder.py`: `FusedDecoder(FunctionalDecoder)` — identical public contract (prefill/decode, paged KV,
GDN state, `start_pos` chunking, batch), with a rewritten op graph selected by `FusionFlags`. Layer kinds, context
contract (262144) and non-aligned lengths are unchanged (`doc/context_contract.json` not modified).

## Op-sequence audit (from the stage-1 Tracy tables, one die, real weights)

Decode, one step (per replay, complete windows): GDN layer 115 ops / 1.608 ms; attention layer 84 ops / 1.439 ms.
Prefill T=128: GDN 306 ops / 27.6 ms; attention 132 ops / 28.7 ms. Dominant costs: MoE sparse matmuls (prefill), dense
in-projections (decode). Full per-op tables: `../functional_decoder/tracy/<kind>/*_perf_report.txt`.

| # | subgraph (functional) | ops | device us (decode) | class | rewrite |
|---|---|---|---|---|---|
| 1 | router: linear, softmax, fill_pad, topk, slice x2, fill_pad, reduce, div, typecast, untilize | 11 | ~130 | dedicated op | `TTMoEGate` -> router matmul + `generalized_moe_gate` (top-k + softmax over selected == softmax + renormalize) |
| 2 | shared expert: gate matmul, up matmul (silu fused), mul, down matmul, sigmoid-gate matmul [3072x32] on 1 core, sigmoid, mul | 7 | ~160 | shared-LHS rewrite | one packed [3072 x (1024\|1024\|1)] matmul + 3 slices, SiLU folded into the multiply |
| 3 | attention q(+gate)/k/v projections | 3 | ~220 | shared-LHS rewrite | one [3072 x 17408] matmul + 3 slices (`AttentionConfig.fuse_qkv`, opt-in) |
| 4 | masked expert groups (prefill / batched decode): swiglu -> down -> multiply(routing) [E,C,3072] -> reduce | – | prefill | algebraic rewrite | multiply the [E,C,1024] intermediate before the linear down projection (exact, 3x fewer elements) |
| 5 | routing weights x expert outputs + sum over k (gather decode) | 2 | ~15 | op merge candidate | kept: folding into one matmul needs a K=8 (sub-tile) contraction (segfaults, see Ornith notes) |
| 6 | SDPA, RoPE, concat-heads, RMSNorm, TopK, paged cache ops | – | – | already dedicated ops (qwen36) | none |
| 7 | GDN single-device decode conv/recurrent chain (untilize/tilize/slice/concat/reshape, ~40 ops) | ~40 | ~250 | structural | not rewritten here: this is the qwen36 single-device module; stage 4 replaces it with `TPGatedDeltaNet` on the (1,4) mesh, whose decode path is the one to optimise |

## Equivalence (fused vs HF, fused vs functional) — `tests/test_fused_decoder.py`, logs `02-fused-full.log`, `02-fix2-fused-attn.log`

| case | linear_attention | full_attention |
|---|---|---|
| fused vs HF prefill/decode T=5 | 0.99990 / 0.99981 | 0.99960 / 0.99961 |
| T=129 | 0.99886 / 0.99847 | 0.99979 / 0.99986 |
| T=333 | 0.99854 / 0.99818 | 0.99986 / 0.99993 |
| fused vs functional, identical weights+inputs, T=333 prefill / decode | 0.99999998 / 0.99999999 | 0.9999994 / 0.9999989 |
| batch 32 per-user prefill + batched decode vs HF | 0.99941 / 0.99911 | pass (>= 0.995, see log) |
| traced decode 3 replays vs HF; repeated replay bit-identical | 0.99881 / 0.99884 / 0.99857 | pass |
| real weights T=37 / T=300 (prefill, decode) | 0.99824 / 0.99947 ; 0.99882 / 0.99961 | pass (>= 0.995, see log) |
| chunked prefill 2048+2048+333 (`start_pos` continuation) vs HF, then decode at 4429 (`02-fused-chunked.log`) | pass (>= 0.995, see log) | 0.99997 / 0.99999; K/V cache content per 512-position window: K 0.99986, V 0.99987 (all 9 windows) |

Every fused value equals the functional value to within 1e-5 (same numerics, fewer dispatches). Determinism: repeated
prefill bit-identical (GDN), repeated traced decode replay bit-identical (attention).

## Performance before/after — `tracy/<kind>_<functional|fused>/*_perf_report.{txt,csv}` (real weights, one die)

Windows are complete: decode = 3 traced replays, every replay session has the same op count (checked in `decode_ops.csv`,
`METAL TRACE REPLAY SESSION ID`); values per replay / per prefill. Host wall from the test output.

| kind | measurement | functional ops / device | fused ops / device | delta | host wall functional -> fused |
|---|---|---|---|---|---|
| linear_attention | prefill T=128 | 306 / 27.64 ms | 321 / 26.72 ms | -3.3 % | 30.3 -> 30.1 ms |
| full_attention | prefill T=128 | 132 / 28.69 ms | 148 / 27.73 ms | -3.4 % | 29.7 -> 29.1 ms |
| linear_attention | traced decode B=1 | 115 / 1.610 ms | 118 / 1.464 ms | -9.1 % | 1.697 -> 1.553 ms/step |
| full_attention | traced decode B=1 | 85 / 1.340 ms | 89 / 1.128 ms | -15.8 % | 1.430 -> 1.228 ms/step |

Where the decode time went (per replay, us): GDN Matmul 649 -> 550 (packed shared expert: three matmuls incl. a one-core
[3072x32] scalar matmul -> one), router chain -> gate op; attention Matmul 654 -> 486 (fused QKV + packed shared expert).
The "functional" attention baseline here (1.340 ms) is already below the stage-1 table (1.439 ms) because the T==1 head-concat
rewrite (NLPConcatHeads on one core, 105 us -> transpose+reshape) lives in the shared kernel and applies to both graphs.
Prefill: -3 % device, host wall flat — the MoE sparse matmuls (65-78 %) are untouched by fusing and belong to the optimize
stage (program configs, chunk width, dtype). Op counts rise slightly in prefill because the gate op processes tokens in
<=110-token groups (slices/reshards) and the packed shared expert adds slices; both are cheaper than what they replace.

Model-level estimate (36 GDN + 12 attention layers, one die, device time): 36 x 1.464 + 12 x 1.128 = 66.2 ms/token
(stage 1: 75.2 ms), i.e. ~15 tok/s single-die-equivalent before optimization.

Pointers left for the optimize stage (from these tables): the router gate matmul `32 x 3072 x 256` runs 70 us on 8 cores
(SLOW, 4.8% of the GDN layer / 6.2% of the attention layer; the gate op removed the ~63 us softmax/top-k tail, not the
matmul), both RMSNorms run ~49 us each on one core, and the packed shared-expert matmul is 56 us (SLOW) on 65 cores.
Window completeness was checked on the committed `*_perf_report.csv` files (3 replays each with identical op sequences).

## Fallback / structural-op audit of the fused runtime path
* `test_fused_runtime_fallback_audit`: 0 `from_torch`/`to_torch` in aligned prefill, chunk continuation and decode for both
  kinds (same accepted boundary as stage 1 for non-aligned GDN prefill: the host-built valid_len selector).
* Remaining layout ops in the fused decode graph (per replay, counted from `tracy/*_fused/decode_perf_report.csv`):
  GDN 7 UntilizeWithUnpadding + 5 TilizeWithValPadding + 2 Tilize + 1 InterleavedToSharded + 2 ShardedToInterleaved;
  attention 1 TilizeWithValPadding + 3 InterleavedToSharded + 2 ShardedToInterleaved (no untilize). All but one live in
  the qwen36 token mixers (GDN conv-state / recurrent path, KV update) and the gate op's height-shard contract; the one
  TilizeWithValPadding common to both kinds (~3 us, 8 cores) is the autoport's tilize of the gate op's row-major routing
  scores for the routing-weight multiply in `_routed_gather` (`moe.py`). The single-device GDN chain is replaced by
  `TPGatedDeltaNet` in stage 4; the stage-3 decoder still carries it (see doc/optimized_decoder).

## Router compute policy (changed by the gate op)
`TTMoEGate` runs the router matmul at HiFi2 with fp32 accumulation and the kernel's exp over the raw selected logits
(`softmax_position="post"`), versus the functional HiFi4 matmul + numerically-stable softmax. On the tested inputs the selected
expert ids and weights match (fused-vs-functional PCC 0.99999998); `softmax_position="pre"` is the fallback if flips or
overflow appear on long real prompts. The shared-expert sigmoid gate is an input activation of the final multiply.

Stage-3 amendment: the gate op is now used for the single-tile decode rows only. With more than 32 rows (prefill) its output
was measured bit-nondeterministic (5 of 29 identical T=37 layer prefills differed at bf16-ulp level, 0 of 29 with the functional
softmax/top-k router), so `Qwen35MoEFused._route` sends prefill rows through the functional router (details in
doc/optimized_decoder/README.md section 3 and work_log.md, 2026-09-24 06:35).

## Rejected / not applicable
* Folding routing-weight scaling + expert sum into one matmul (`[1,1,1,k] @ [1,1,k,H]`): not attempted here — K=8 is below one
  tile and the [1,k,1,H] -> [1,1,k,H] move is not a view (a padded-shape change), so it would add a real data-movement op for a
  2-op saving; no in-repo repro exists. Kept multiply + sum.
* `ttnn.swiglu` at decode (M=1): pads rows to a full tile and returns [1,k,32,N]; slice+slice+multiply(SiLU activation) kept.
* TTMoEGate's derived 2D program config: sized for one M tile; ttnn auto program config used (prefill routes hundreds of rows).
* Single-device GDN decode conv/recurrent chain rewrites (Ornith's head-major/fused-taps): not on the (1,4) TP path (stage 4).

## Watcher / stress
* Watcher (`TT_METAL_WATCHER=10`): fused real-weight T=37 prefill+decode both kinds and fused batched decode B=32 (attention):
  pass, `watcher.log` 10,886 lines, 0 exception/fatal/sanitizer/overflow (`watcher/`).
* Stress: `test_fused_stress_repeated_runs` — 20 prefill+decode cycles over lengths 5..1500 per kind, all finite, identical
  cycles bit-identical.

## Dense Qwen3.5-9B regression for the shared-kernel changes (`02-9b-decode-check.log`, `02-9b-units.log`)
The T==1 head-concat rewrite and the fused-QKV hook live in the shared gated-attention kernel, so the dense 9B path was
re-checked on one die with the amended tree:
* `tools/check_qwen36_9b_decode.py`: 9B layer 3 (16 q heads / 4 KV heads / head_dim 256 / hidden 4096), paged prefill T=37
  then paged decode vs the HF `Qwen3_5DecoderLayer`: B=1 pos 37 PCC 0.99676; B=4 distinct positions 5/37/129/300 PCC
  0.99739 / 0.99661 / 0.99697 / 0.99680 (threshold 0.99). RESULT PASS.
* 9B unit suite (`qwen36/tests/unit/test_gdn.py`, `test_attention.py`, `test_layer.py`, `MESH_DEVICE=P150`): 5/5 PASS.
* The 9B full-model test remains OPEN from stage 1 (hangs in `Embedding.__init__ -> ttnn.as_tensor` on this host before any
  device work; not caused by these changes).

## Shared-code changes in this stage
`AttentionConfig.fuse_qkv` (default False), `load_attention_weights(..., fuse_qkv)`, `gated_attention_forward_ttnn(qkv_proj_weight=)`
(opt-in fused projection), and the T==1 head-concat path (transpose+reshape instead of one-core NLPConcatHeads; applies to
every caller at decode, numerically the same permutation).

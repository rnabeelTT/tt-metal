# Functional decoder — Qwen/Qwen3.5-122B-A10B (stage 1)

Status: complete. Correctness (HF parity at 262107 tokens GDN / 16421 attention), capacity (262107), batching, traced decode,
determinism, runtime fallback audit, watcher, and perf evidence are recorded below; every command is in work_log.md.

## What this stage delivers

`models/autoports/qwen_qwen3_5_122b_a10b/tt/functional_decoder.py` — `FunctionalDecoder(LightweightModule)` for one HF
`Qwen3_5MoeDecoderLayer` of either kind:

| layer kind | count | token mixer | feed-forward |
|---|---|---|---|
| `linear_attention` | 36 | Gated DeltaNet (qwen36 kernels: chunk prefill, recurrent decode, conv window) | 256-expert MoE top-8 + gated shared expert |
| `full_attention` | 12 | gated GQA attention, 32 q / 2 kv heads, head_dim 256, partial RoPE 0.25, paged KV | same |

Supporting modules: `tt/model_config.py` (`Qwen35MoeModelArgs`), `tt/weights.py` (streaming safetensors mapping,
HF->internal key remap), `tt/moe.py` (`Qwen35MoE`: router, `ttnn.sparse_matmul` routed experts, shared expert),
`tt/layer.py` (`Qwen35MoeDecoderLayer`, feed-forward hook on `Qwen36DecoderLayer`).

## Public contract

```python
layer = FunctionalDecoder.from_state_dict(state_dict, hf_config=cfg, layer_idx=i, mesh_device=dev,
                                          max_batch_size=B, max_seq_len=262144)
layer.allocate_kv_cache(max_batch_size=B, max_seq_len=..., block_size=64)   # full_attention layers
layer.allocate_gdn_state(batch_size=B)                                       # linear_attention layers
out = layer.prefill_forward(x, start_pos=0, page_table=pt, user_id=u)        # x: [1, T, 3072] (any T)
out = layer.decode_forward(x, current_pos=pos, page_table=pt, cos=cos, sin=sin)  # x: [B, 1, 3072]
```
Details in the module docstring. Prefill pads internally to the 128-token GDN chunk and masks/slices; decode is a
single traced-safe step (all inputs are device tensors; GDN state and paged KV are updated in place).

## Evidence

Device: one Blackhole p300c die (1x1 mesh, `l1_small_size=24576`). HF reference: `Qwen3_5MoeDecoderLayer` bf16 eager on CPU.
Precision in this stage: routed experts bfp8_b (HiFi2, fp32 dest acc OFF), shared expert / GDN / attention projections
bfp8_b, router and norms bf16, activations bf16, KV cache bf16, GDN state bf16 buffers (fp32 inside the recurrent step).
Acceptance bar PCC >= 0.995 (default). Log files: `~/qwen35-122b-bringup/logs/01-*.log`.

### HF-vs-TTNN PCC, real weights (layers 0 and 3 of the checkpoint)

| layer kind | T | prefill PCC | decode PCC (at pos T) |
|---|---|---|---|
| linear_attention (L0) | 37 | 0.99824 | 0.99947 |
| linear_attention (L0) | 300 | 0.99882 | 0.99961 |
| full_attention (L3) | 37 | 0.99957 | 0.99936 |
| full_attention (L3) | 300 | 0.99968 | 0.99988 |

### Synthetic weights (from `tests/weight_stats.json`, real shapes), all PCC >= 0.995

| case | linear_attention | full_attention |
|---|---|---|
| prefill/decode T=5 | 0.99990 / 0.99981 | 0.99960 / 0.99961 |
| T=128 (GDN chunk boundary) | 0.99897 / 0.99820 | 0.99978 / 0.99992 |
| T=129 (just across) | 0.99886 / 0.99847 | 0.99979 / 0.99986 |
| T=333 (non-divisible) | 0.99854 / 0.99818 | 0.99986 / 0.99993 |
| batch 2: per-user prefill, batched decode | 0.99933 / 0.99902 | 0.99967 / 0.99980 |
| batch 32: per-user prefill, batched decode | 0.99941 / 0.99911 | 0.99969 / 0.99978 |
| batch 4, distinct prompt lengths 5/37/129/300, decode at each user's own position (per-user HF refs) | pass (all users >= 0.995) | pass (all users >= 0.995) |
| traced decode, 4 replays after prefill T=40 | 0.99857 .. 0.99905 | 0.99975 .. 0.99982 |
| chunked prefill 2048+2048+333 (start_pos), decode at 4429 | 0.99758 / 0.99744 | 0.99997 / 0.99999 (after the page-table offset fix; 0.99888 / 0.99911 with the bug) |
| long HF parity, chunked 2048, state carried: GDN 65573 / 262107 tokens; attention 16421 (last-2048 / all / decode) | 0.99732 / 0.99741 / 0.99790 ; 0.99744 / 0.99737 / 0.99745 | 0.99999 / 0.99998 / 0.99999 (0.99931 / 0.99894 / 0.99943 with the bug) |

### Paged KV cache, page table, current position
Chunked prefill K/V ownership is checked position by position: after a 2048+2048+333 chunked prefill through a
random-permutation page table, K and V are read back through the page table and compared with HF's cache per 512-position
window (`test_chunked_prefill_kv_cache_matches_hf`; also with real weights for 2048+333 in
`test_real_weight_chunked_attention_prefill`). This check was added after the stage review found that later chunks had
been written to the first chunk's blocks (fixed: the chunk page table now starts at `start_pos // block_size`).
Results: synthetic 2048+2048+333, every 512-position window K and V PCC 0.99986 (10 windows), decode at 4429 PCC 0.99999;
real weights 2048+333: layer prefill PCC 0.99979, all K/V windows 0.99986, decode at 2381 PCC 0.99988 (`logs/01-fix2-attn.log`).
Full-attention tests use a random-permutation page table (`_page_table`), block size 64, paged prefill through
`paged_fill_cache` (chunk_page_table/chunk_start_idx) and paged decode through `paged_update_cache` +
`paged_scaled_dot_product_attention_decode` with a device `current_pos` tensor. Batch-32 decode exercises one KV slot per
user. The GDN state is a batched external buffer updated in place; per-user prefill writes its row.

### Capability contract

| claim | evidence | remaining risk |
|---|---|---|
| context 262144 (HF `max_position_embeddings`) supported | `test_full_context_capacity` (262107 tokens, both kinds); `test_long_hf_parity`: GDN HF parity at 262107 tokens (PCC 0.9974), attention HF parity at 16421 tokens (PCC 0.99998) — HF eager attention scores are 34 GB at 16K and 137 GB at 32K on this host | attention HF parity beyond 16K is not CPU-computable here; attention eager decode at 262K measured 59.9 ms on one die after the fix (1.05 s in the pre-fix run, which included first-call compile) |
| any logical length T (non-aligned) | T=5/37/129/300/333/4429/262107 | non-aligned GDN prefill uses the host-built valid_len selector (see audit) |
| chunked prefill continuation (start_pos) | 2048+2048+333 vs HF; 262107 in 2048 chunks | start_pos must be a multiple of 128 and of the KV block size (asserted); GDN continuation is single-user |
| both layer kinds, real config shapes | tests above; `test_target_decoder_contract` | none |
| batch up to 32 | batch 2 and 32 decode vs HF | batched *prefill* is per user (vLLM semantics) |
| paged KV cache, page table, current position | random page tables, random slots via batch tests | none |
| traced decode | `test_synthetic_traced_decode` (capture + 4 replays with refreshed inputs) | none |

### Determinism, fallback audit, watcher
* `test_synthetic_repeated_input_determinism`: bit-identical prefill and decode outputs for repeated inputs (both kinds).
* `test_runtime_fallback_audit_device_passes`: `ttnn.from_torch`/`ttnn.to_torch` are wrapped with a counter during one
  aligned prefill (T=256), one chunk-continuation prefill (start_pos=256, T=128) and one decode step, both layer kinds:
  0 host transfers in every measured pass (RoPE cos/sin are sliced from the resident device tables; a fresh GDN state
  is zeroed with device ops). Accepted, explicit boundary: a NON-aligned GDN prefill (valid_len) builds the fixed-bucket
  one-hot selector on the host inside the qwen36 kernel (2 small `from_torch`, `ttnn_gated_deltanet.py:207`,
  `ttnn_delta_rule_seq.py:455`); attention non-aligned prefill: 0. `test_runtime_fallback_audit_source` additionally
  AST-scans the autoport forward paths.
* Watcher (`TT_METAL_WATCHER=10`; real-weight T=37 eager prefill+decode both kinds AND the chunked 2048+2048+333 attention
  prefill with the K/V cache check; run on the final code): tests pass, `watcher.log` (10,894 lines) has no
  exception/fatal/sanitizer/overflow lines (`watcher/pytest.log`, `watcher/watcher.tail.log`; full log in
  `~/qwen35-122b-bringup/artifacts/01-watcher/`). Coverage note: traced decode and batch-32 paths were not run under watcher.

### Performance (warmed, single layer, one die) — `tracy/<kind>/*_perf_report.{txt,csv}`

Evidence files: `*_perf_report.{txt,csv}`, `*_stacked.csv`, `*_tracy.console.log` are committed (force-added past .gitignore); the raw
`*_ops.csv` (0.4-6.4 MB) are kept untracked next to them and in `~/qwen35-122b-bringup/`.
Real weights, layer 0 (linear_attention) and layer 3 (full_attention). Tracy ops CSV -> `tt-perf-report` with the `PERF_*`
signposts (`Device Time` column, microseconds; `python -m tracy -r -p -v -m pytest <node id>`). The device profiler buffers hold
only a few hundred ops per dump, so: the decode window is 3 traced replays (every replay session complete: 115 / 84 ops each,
checked in `decode_ops.csv` column `METAL TRACE REPLAY SESSION ID`), and the 2048-token prefill is run as 4 start_pos chunks of
512 with a `ReadDeviceProfiler` flush between chunks (all ops present). Values are per replay / per full prefill.

| kind | measurement | ops | device time | top op classes (device time) | host wall |
|---|---|---|---|---|---|
| linear_attention | prefill T=128 | 306 | 27.59 ms | SparseMatmul 17.65 ms, BinaryNg 2.54, Unary 2.23, ReshapeView 1.21 | 30.4 ms |
| full_attention | prefill T=128 | 132 | 28.72 ms | SparseMatmul 21.55 ms, Unary 2.13, BinaryNg 1.65, Matmul 0.79 | 30.1 ms |
| linear_attention | prefill T=2048 (4 x 512 chunks) | 1940 | 411.5 ms | SparseMatmul 271.1 ms, BinaryNg 38.8, Unary 35.4, ReshapeView 16.2 | – |
| full_attention | prefill T=2048 (4 x 512 chunks) | 1248 | 483.7 ms | SparseMatmul 377.6 ms, Unary 33.5, BinaryNg 25.9, Slice 12.1 | – |
| linear_attention | traced decode B=1 (per replay) | 115 | 1.608 ms | Matmul 0.649 ms, SparseMatmul 0.302, ReshapeView 0.150, BinaryNg 0.141 | 1.689 ms/step |
| full_attention | traced decode B=1 (per replay) | 84 | 1.439 ms | Matmul 0.654 ms, SparseMatmul 0.305, LayerNorm 0.109, NLPConcatHeads 0.105 | 1.531 ms/step |

Reading: the MoE prefill (masked 32-token expert groups over the selected-expert union) is 65-78% of prefill; the dense
in-projections dominate decode. Model-level estimate from the single-layer numbers: 36 x 1.608 + 12 x 1.439 = 75.2 ms of
device time per decode token on one die (~13 tok/s before any optimization; the model itself needs the 4-die mesh).
The stage-1 first-pass numbers (10-replay windows, 26.9/28.6 ms prefill, 1.26/1.37 ms decode) were computed over windows in
which the profiler had dropped ops in later replays and are superseded by this table.

### Shared-code changes made in this stage (all in this branch, see work_log.md)
`qwen36/tt/layer.py` feed-forward hook, `qwen36/tt/mlp.py` `forward(x, mode=None)`, `qwen36/tt/model.py` layer-class and
per-layer state hooks, `qwen36/tt/gdn/gated_deltanet.py` batched external state, `experimental/.../ttnn_gated_attention.py`
batched paged decode (Q permute, shard grid), `experimental/.../ttnn_delta_rule_seq.py` L1 relayout budget,
`experimental/.../ttnn_delta_rule_ops.py` decode-state L1 budget.

### Shared-code regression evidence (Qwen3.5-9B, this branch)
`unit/test_gdn.py`, `unit/test_attention.py`, `unit/test_layer.py` pass with real 9B weights (GDN decode/prefill kernels and the
non-paged attention prefill path; NO executed 9B test reaches the paged-decode hunks, which are argued below: `permute(2,0,1,3)`
equals the old `transpose(1,2)` at B=1, sdpa_decode requires `q_shape[0]==1` so the old layout never worked at B>1, and
`num_cores_to_corerangeset(B, grid, row_wise=True)` equals the old single-row range for B <= grid width). The full-model 9B tests (`unit/test_model.py`, `test_prefill.py`) could not be
completed: the 9B model load stalled >60 min in the embedding weight upload (`Embedding.__init__ -> ttnn.as_tensor`, no I/O,
not a path this port touches). The new thresholds keep existing configurations on their previous paths: decode-state L1
budget 96 MiB (9B B=32 = 64 MiB), seq-adapter L1 budget 32 MiB (= 9B 2048-token chunk), width rule only for GDN width > 4096,
attention permute/shard grid identical at B=1. See work_log.md 02:20.

### Known limitations carried to later stages
* Single-device only; the model does not fit one die (experts 116B params). Stage 4 shards experts (TP on the intermediate
  dim) and uses the qwen36 TP token mixers.
* Decode MoE path: per-token gather for T <= 4 users, masked tile-group path above; prefill chunk 32. Not yet tuned.

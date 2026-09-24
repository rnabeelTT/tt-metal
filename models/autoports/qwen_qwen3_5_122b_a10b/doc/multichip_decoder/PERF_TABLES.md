# Stage-4 mesh perf tables (Tracy, tt-perf-report signposted windows)

Single-chip = stage-3 `OptimizedDecoder`/`SELECTED` on one die (committed tables, `doc/optimized_decoder/tracy/*_optimized`).
Mesh = `MultichipDecoder`/`MULTICHIP_SELECTED` on the (1,4) P300x2, device time summed over the window on each device
(decode = one traced replay, 3 collected). Speedup = single-chip / max over devices; efficiency = speedup / 4.

| layer | case | single-chip ms (ops) | mesh ms per device 0/1/2/3 (ops/device) | speedup | efficiency | CCL ms/device |
|---|---|---|---|---|---|---|
| GDN (L0) | prefill_128 | 16.197 (285) | 8.198/8.188/8.213/8.206 (138) | 1.97x | 49% | 0.164/0.141/0.188/0.157 |
| GDN (L0) | prefill_2048 | 124.241 (1550) | 49.944/49.883/49.957/49.963 (948) | 2.49x | 62% | 1.289/1.257/1.365/1.343 |
| GDN (L0) | decode | 0.827 (95) | 0.474/0.471/0.471/0.473 (103) | 1.75x | 44% | 0.048/0.045/0.054/0.048 |
| attention (L3) | prefill_128 | 15.138 (104) | 9.026/9.017/9.035/9.029 (93) | 1.68x | 42% | 0.084/0.068/0.089/0.076 |
| attention (L3) | prefill_2048 | 88.948 (812) | 47.147/47.117/47.167/47.198 (768) | 1.88x | 47% | 0.716/0.691/0.774/0.781 |
| attention (L3) | decode | 0.656 (92) | 0.416/0.419/0.419/0.416 (84) | 1.57x | 39% | 0.047/0.050/0.061/0.046 |

### GDN (L0) prefill_128 — device 0 top ops (8.198 ms, 138 ops per window)

| op (shape) | fidelity | cores | count | ms | share |
|---|---|---|---|---|---|
| SparseMatmul active=?/256 x 64 x 3072 x 512 | LoFi BF16 x BFP4 => BF16 | 16 | 2 | 3.756 | 46% |
| Unary | BF16, BF16 => BF16 | 110 | 4 | 1.187 | 14% |
| SparseMatmul active=?/256 x 64 x 256 x 3072 | LoFi BF16 x BFP4 => BF16 | 48 | 2 | 1.071 | 13% |
| FastReduceNC | BF16 => BF16 | 110 | 2 | 0.559 | 7% |
| BinaryNg | BF16, BF16 => BF16 | 110 | 11 | 0.243 | 3% |
| Slice | BF16 => BF16 | 110 | 13 | 0.201 | 2% |
| Matmul 128 x 3072 x 5152 | HiFi2 BF16 x BFP8 => BF16 | 32 | 1 | 0.107 | 1% |
| LayerNorm | BF16, BF16 => BF16 | 4 | 2 | 0.098 | 1% |
| MatmulReduceScatterAsync 128 x 2048 x 3072 | HiFi4 FP32 x BFP8 => FP32 | 44 | 1 | 0.094 | 1% |
| Unary | BF16 => BF16 | 110 | 5 | 0.091 | 1% |

Op-to-op gaps on device 0: 89.500 ms per window.

### GDN (L0) prefill_2048 — device 0 top ops (49.944 ms, 948 ops per window)

| op (shape) | fidelity | cores | count | ms | share |
|---|---|---|---|---|---|
| SparseMatmul active=377/96512 x 32 x 3072 x 512 | LoFi BF16 x BFP4 => BF16 | 16 | 4 | 16.837 | 34% |
| SparseMatmul active=377/96512 x 32 x 256 x 3072 | LoFi BF16 x BFP4 => BF16 | 48 | 4 | 7.071 | 14% |
| Embeddings | UINT32, BF16 => BF16 | 110 | 4 | 4.366 | 9% |
| Scatter | UINT32, UINT32 => UINT32 | 1 | 12 | 2.398 | 5% |
| Sort | FP32 => FP32 | 64 | 4 | 1.533 | 3% |
| UntilizeCodegen | BF16 => BF16 | 110 | 4 | 1.470 | 3% |
| BinaryNg | BF16, BF16 => BF16 | 110 | 64 | 1.251 | 3% |
| Embeddings | UINT32, BF16 => BF16 | 16 | 32 | 1.209 | 2% |
| Accumulation | FP32 => FP32 | 1 | 8 | 1.104 | 2% |
| Scatter | BF16, UINT32 => BF16 | 1 | 4 | 1.074 | 2% |

Op-to-op gaps on device 0: 563.266 ms per window.

### GDN (L0) decode — device 0 top ops (0.474 ms, 103 ops per window)

| op (shape) | fidelity | cores | count | ms | share |
|---|---|---|---|---|---|
| SparseMatmul active=?/256 x 32 x 3072 x 512 | LoFi BF16 x BFP4 => BF16 | 16 | 1 | 0.062 | 13% |
| BinaryNg | BF16, BF16 => BF16 | 110 | 15 | 0.058 | 12% |
| Matmul 32 x 3072 x 5152 | HiFi2 BF16 x BFP8 => BF16 | 41 | 1 | 0.051 | 11% |
| AllGatherAsync | BF16 => BF16 | 12 | 2 | 0.024 | 5% |
| ReduceScatterMinimalAsync | BF16 => UINT8 | 12 | 2 | 0.024 | 5% |
| ReshapeView | BF16 => BF16 | 4 | 4 | 0.023 | 5% |
| Matmul 32 x 2048 x 3072 | HiFi2 BF16 x BFP8 => BF16 | 32 | 1 | 0.022 | 5% |
| SparseMatmul active=?/256 x 32 x 256 x 3072 | LoFi BF16 x BFP4 => BF16 | 48 | 1 | 0.018 | 4% |
| Ternary | BF16, BF16 => BF16 | 110 | 4 | 0.016 | 3% |
| Matmul 32 x 3072 x 544 | HiFi2 BF16 x BFP8 => BF16 | 17 | 1 | 0.015 | 3% |

Op-to-op gaps on device 0: 1.936 ms per window.

### attention (L3) prefill_128 — device 0 top ops (9.026 ms, 93 ops per window)

| op (shape) | fidelity | cores | count | ms | share |
|---|---|---|---|---|---|
| SparseMatmul active=?/256 x 64 x 3072 x 512 | LoFi BF16 x BFP4 => BF16 | 16 | 2 | 4.709 | 52% |
| SparseMatmul active=?/256 x 64 x 256 x 3072 | LoFi BF16 x BFP4 => BF16 | 48 | 2 | 1.297 | 14% |
| Unary | BF16, BF16 => BF16 | 110 | 4 | 1.206 | 13% |
| FastReduceNC | BF16 => BF16 | 110 | 2 | 0.561 | 6% |
| BinaryNg | BF16, BF16 => BF16 | 110 | 11 | 0.235 | 3% |
| Slice | BF16 => BF16 | 110 | 10 | 0.192 | 2% |
| LayerNorm | BF16, BF16 => BF16 | 4 | 2 | 0.099 | 1% |
| Matmul 128 x 3072 x 4608 | HiFi2 BF16 x BFP8 => BF16 | 32 | 1 | 0.095 | 1% |
| Unary | BF16 => BF16 | 110 | 3 | 0.083 | 1% |
| Matmul 128 x 3072 x 256 | HiFi4 BF16 x BF16 => BF16 | 32 | 1 | 0.066 | 1% |

Op-to-op gaps on device 0: 70.954 ms per window.

### attention (L3) prefill_2048 — device 0 top ops (47.147 ms, 768 ops per window)

| op (shape) | fidelity | cores | count | ms | share |
|---|---|---|---|---|---|
| SparseMatmul active=377/96512 x 32 x 3072 x 512 | LoFi BF16 x BFP4 => BF16 | 16 | 4 | 16.831 | 36% |
| SparseMatmul active=377/96512 x 32 x 256 x 3072 | LoFi BF16 x BFP4 => BF16 | 48 | 4 | 7.086 | 15% |
| Embeddings | UINT32, BF16 => BF16 | 110 | 4 | 4.320 | 9% |
| Scatter | UINT32, UINT32 => UINT32 | 1 | 12 | 2.398 | 5% |
| Sort | FP32 => FP32 | 64 | 4 | 1.532 | 3% |
| UntilizeCodegen | BF16 => BF16 | 110 | 4 | 1.467 | 3% |
| BinaryNg | BF16, BF16 => BF16 | 110 | 64 | 1.267 | 3% |
| Embeddings | UINT32, BF16 => BF16 | 16 | 32 | 1.199 | 3% |
| Accumulation | FP32 => FP32 | 1 | 8 | 1.104 | 2% |
| Scatter | BF16, UINT32 => BF16 | 1 | 4 | 1.074 | 2% |

Op-to-op gaps on device 0: 608.876 ms per window.

### attention (L3) decode — device 0 top ops (0.416 ms, 84 ops per window)

| op (shape) | fidelity | cores | count | ms | share |
|---|---|---|---|---|---|
| SparseMatmul active=?/256 x 32 x 3072 x 512 | LoFi BF16 x BFP4 => BF16 | 16 | 1 | 0.062 | 15% |
| Matmul 32 x 3072 x 4608 | HiFi2 BF16 x BFP8 => BF16 | 48 | 1 | 0.054 | 13% |
| BinaryNg | BF16, BF16 => BF16 | 110 | 10 | 0.036 | 9% |
| AllGatherAsync | BF16 => BF16 | 12 | 2 | 0.024 | 6% |
| ReduceScatterMinimalAsync | BF16 => UINT8 | 12 | 2 | 0.023 | 5% |
| Matmul 32 x 2048 x 3072 | HiFi2 BF16 x BFP8 => BF16 | 32 | 1 | 0.021 | 5% |
| SparseMatmul active=?/256 x 32 x 256 x 3072 | LoFi BF16 x BFP4 => BF16 | 48 | 1 | 0.018 | 4% |
| Matmul 32 x 3072 x 544 | HiFi2 BF16 x BFP8 => BF16 | 17 | 1 | 0.015 | 4% |
| SdpaDecode | BF16, BFP8 => BF16 | 110 | 1 | 0.013 | 3% |
| LayerNorm | BF16, BF16 => BF16 | 32 | 2 | 0.012 | 3% |

Op-to-op gaps on device 0: 1.864 ms per window.

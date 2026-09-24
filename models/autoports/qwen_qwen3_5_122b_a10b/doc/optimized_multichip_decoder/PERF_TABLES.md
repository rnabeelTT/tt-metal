# Stage-5 mesh perf tables (Tracy, tt-perf-report signposted windows)

Single-chip = stage-3 `OptimizedDecoder`/`SELECTED` on one die (committed tables, `doc/optimized_decoder/tracy/*_optimized`).
Mesh = `MultichipDecoder`/`MULTICHIP_SELECTED` on the (1,4) P300x2, device time summed over the window on each device
(decode = one traced replay, 3 collected). Speedup = single-chip / max over devices; efficiency = speedup / 4.

| layer | case | single-chip ms (ops) | mesh ms per device 0/1/2/3 (ops/device) | speedup | efficiency | CCL ms/device |
|---|---|---|---|---|---|---|
| GDN (L0) | prefill_128 | 16.197 (285) | 8.199/8.187/8.205/8.202 (139) | 1.97x | 49% | 0.165/0.141/0.182/0.154 |
| GDN (L0) | prefill_2048 | 124.241 (1550) | 60.971/60.929/61.014/60.987 (924) | 2.04x | 51% | 1.369/1.339/1.345/1.385 |
| GDN (L0) | decode | 0.827 (95) | 0.472/0.474/0.473/0.473 (103) | 1.74x | 44% | 0.046/0.049/0.055/0.048 |
| attention (L3) | prefill_128 | 15.138 (104) | 9.035/9.022/9.039/9.031 (94) | 1.67x | 42% | 0.099/0.088/0.094/0.076 |
| attention (L3) | prefill_2048 | 88.948 (812) | 58.231/58.194/58.302/58.293 (744) | 1.53x | 38% | 0.781/0.709/0.775/0.816 |
| attention (L3) | decode | 0.656 (92) | 0.406/0.407/0.409/0.408 (84) | 1.61x | 40% | 0.046/0.046/0.054/0.048 |

### GDN (L0) prefill_128 — device 0 top ops (8.199 ms, 139 ops per window)

| op (shape) | fidelity | cores | count | ms | share |
|---|---|---|---|---|---|
| SparseMatmul active=?/256 x 64 x 3072 x 512 | LoFi BF16 x BFP4 => BF16 | 16 | 2 | 3.757 | 46% |
| Unary | BF16 => BF16 | 110 | 10 | 1.279 | 16% |
| SparseMatmul active=?/256 x 64 x 256 x 3072 | LoFi BF16 x BFP4 => BF16 | 48 | 2 | 1.065 | 13% |
| FastReduceNC | BF16 => BF16 | 110 | 2 | 0.559 | 7% |
| BinaryNg | BF16, BF16 => BF16 | 110 | 11 | 0.243 | 3% |
| Slice | BF16 => BF16 | 110 | 13 | 0.202 | 2% |
| Matmul 128 x 3072 x 5152 | HiFi2 BF16 x BFP8 => BF16 | 32 | 1 | 0.107 | 1% |
| LayerNorm | BF16, BF16 => BF16 | 4 | 2 | 0.099 | 1% |
| MatmulReduceScatterAsync 128 x 2048 x 3072 | HiFi4 FP32 x BFP8 => FP32 | 44 | 1 | 0.094 | 1% |
| ChunkGdnPrepOperation | BF16, BF16 => FP32 | 64 | 1 | 0.068 | 1% |

Op-to-op gaps on device 0: 90.581 ms per window.

### GDN (L0) prefill_2048 — device 0 top ops (60.971 ms, 924 ops per window)

| op (shape) | fidelity | cores | count | ms | share |
|---|---|---|---|---|---|
| SparseMatmul active=377/96512 x 32 x 3072 x 512 | LoFi BF16 x BFP4 => BF16 | 16 | 4 | 16.843 | 28% |
| Accumulation | FP32 => FP32 | 8 | 4 | 8.776 | 14% |
| SparseMatmul active=377/96512 x 32 x 256 x 3072 | LoFi BF16 x BFP4 => BF16 | 48 | 4 | 7.074 | 12% |
| Permute | FP32 => FP32 | 110 | 12 | 5.324 | 9% |
| Embeddings | UINT32, BF16 => BF16 | 110 | 4 | 4.346 | 7% |
| Scatter | UINT32, UINT32 => UINT32 | 1 | 8 | 1.676 | 3% |
| UntilizeCodegen | BF16 => BF16 | 110 | 4 | 1.475 | 2% |
| BinaryNg | BF16, BF16 => BF16 | 110 | 64 | 1.246 | 2% |
| Embeddings | UINT32, BF16 => BF16 | 16 | 32 | 1.212 | 2% |
| Scatter | BF16, UINT32 => BF16 | 1 | 4 | 1.097 | 2% |

Op-to-op gaps on device 0: 412.812 ms per window.

### GDN (L0) decode — device 0 top ops (0.472 ms, 103 ops per window)

| op (shape) | fidelity | cores | count | ms | share |
|---|---|---|---|---|---|
| SparseMatmul active=?/256 x 32 x 3072 x 512 | LoFi BF16 x BFP4 => BF16 | 16 | 1 | 0.062 | 13% |
| BinaryNg | BF16, BF16 => BF16 | 110 | 15 | 0.058 | 12% |
| Matmul 32 x 3072 x 5152 | HiFi2 BF16 x BFP8 => BF16 | 41 | 1 | 0.051 | 11% |
| AllGatherAsync | BF16 => BF16 | 12 | 2 | 0.024 | 5% |
| ReshapeView | BF16 => BF16 | 4 | 4 | 0.023 | 5% |
| ReduceScatterMinimalAsync | BF16 => UINT8 | 12 | 2 | 0.023 | 5% |
| Matmul 32 x 2048 x 3072 | HiFi2 BF16 x BFP8 => BF16 | 32 | 1 | 0.022 | 5% |
| SparseMatmul active=?/256 x 32 x 256 x 3072 | LoFi BF16 x BFP4 => BF16 | 48 | 1 | 0.018 | 4% |
| Ternary | BF16, BF16 => BF16 | 110 | 4 | 0.016 | 3% |
| Matmul 32 x 3072 x 544 | HiFi2 BF16 x BFP8 => BF16 | 17 | 1 | 0.015 | 3% |

Op-to-op gaps on device 0: 1.933 ms per window.

### attention (L3) prefill_128 — device 0 top ops (9.035 ms, 94 ops per window)

| op (shape) | fidelity | cores | count | ms | share |
|---|---|---|---|---|---|
| SparseMatmul active=?/256 x 64 x 3072 x 512 | LoFi BF16 x BFP4 => BF16 | 16 | 2 | 4.710 | 52% |
| SparseMatmul active=?/256 x 64 x 256 x 3072 | LoFi BF16 x BFP4 => BF16 | 48 | 2 | 1.294 | 14% |
| Unary | BF16 => BF16 | 110 | 8 | 1.288 | 14% |
| FastReduceNC | BF16 => BF16 | 110 | 2 | 0.559 | 6% |
| BinaryNg | BF16, BF16 => BF16 | 110 | 11 | 0.236 | 3% |
| Slice | BF16 => BF16 | 110 | 10 | 0.192 | 2% |
| LayerNorm | BF16, BF16 => BF16 | 4 | 2 | 0.099 | 1% |
| Matmul 128 x 3072 x 4608 | HiFi2 BF16 x BFP8 => BF16 | 32 | 1 | 0.095 | 1% |
| Matmul 128 x 3072 x 544 | HiFi2 BF16 x BFP8 => BF16 | 36 | 1 | 0.066 | 1% |
| Matmul 128 x 3072 x 256 | HiFi4 BF16 x BF16 => BF16 | 32 | 1 | 0.066 | 1% |

Op-to-op gaps on device 0: 73.625 ms per window.

### attention (L3) prefill_2048 — device 0 top ops (58.231 ms, 744 ops per window)

| op (shape) | fidelity | cores | count | ms | share |
|---|---|---|---|---|---|
| SparseMatmul active=377/96512 x 32 x 3072 x 512 | LoFi BF16 x BFP4 => BF16 | 16 | 4 | 16.831 | 29% |
| Accumulation | FP32 => FP32 | 8 | 4 | 8.778 | 15% |
| SparseMatmul active=377/96512 x 32 x 256 x 3072 | LoFi BF16 x BFP4 => BF16 | 48 | 4 | 7.079 | 12% |
| Permute | FP32 => FP32 | 110 | 12 | 5.320 | 9% |
| Embeddings | UINT32, BF16 => BF16 | 110 | 4 | 4.334 | 7% |
| Scatter | UINT32, UINT32 => UINT32 | 1 | 8 | 1.676 | 3% |
| UntilizeCodegen | BF16 => BF16 | 110 | 4 | 1.469 | 3% |
| BinaryNg | BF16, BF16 => BF16 | 110 | 64 | 1.282 | 2% |
| Embeddings | UINT32, BF16 => BF16 | 16 | 32 | 1.193 | 2% |
| Scatter | BF16, UINT32 => BF16 | 1 | 4 | 1.097 | 2% |

Op-to-op gaps on device 0: 468.376 ms per window.

### attention (L3) decode — device 0 top ops (0.406 ms, 84 ops per window)

| op (shape) | fidelity | cores | count | ms | share |
|---|---|---|---|---|---|
| SparseMatmul active=?/256 x 32 x 3072 x 512 | LoFi BF16 x BFP4 => BF16 | 16 | 1 | 0.062 | 15% |
| Matmul 32 x 3072 x 4608 | HiFi2 BF16 x BFP8 => BF16 | 72 | 1 | 0.045 | 11% |
| BinaryNg | BF16, BF16 => BF16 | 110 | 10 | 0.036 | 9% |
| AllGatherAsync | BF16 => BF16 | 12 | 2 | 0.024 | 6% |
| ReduceScatterMinimalAsync | BF16 => UINT8 | 12 | 2 | 0.022 | 5% |
| Matmul 32 x 2048 x 3072 | HiFi2 BF16 x BFP8 => BF16 | 32 | 1 | 0.021 | 5% |
| SparseMatmul active=?/256 x 32 x 256 x 3072 | LoFi BF16 x BFP4 => BF16 | 48 | 1 | 0.018 | 4% |
| Matmul 32 x 3072 x 544 | HiFi2 BF16 x BFP8 => BF16 | 17 | 1 | 0.015 | 4% |
| SdpaDecode | BF16, BFP8 => BF16 | 110 | 1 | 0.013 | 3% |
| LayerNorm | BF16, BF16 => BF16 | 32 | 2 | 0.012 | 3% |

Op-to-op gaps on device 0: 1.892 ms per window.

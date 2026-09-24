### Device time (tt-perf-report signposted windows; decode = one traced replay, 3 collected)

| layer | case | fused ms | fused ops | optimized ms | optimized ops | change |
|---|---|---|---|---|---|---|
| GDN (L0) | prefill_128 | 26.744 | 315 | 16.197 | 285 | -39.4% |
| GDN (L0) | prefill_2048 | 398.894 | 1994 | 124.241 | 1550 | -68.9% |
| GDN (L0) | decode | 1.288 | 91 | 0.827 | 95 | -35.8% |
| attention (L3) | prefill_128 | 27.713 | 132 | 15.138 | 104 | -45.4% |
| attention (L3) | prefill_2048 | 471.375 | 1248 | 88.948 | 812 | -81.1% |
| attention (L3) | decode | 1.125 | 88 | 0.656 | 92 | -41.6% |

### Roofline accounting (decode, one die, bytes per token at the stored dtypes / 512 GB/s)

| layer | bytes/token (MB) | roofline ms | device ms (optimized replay) | fraction | host wall ms/step (3 replays) |
|---|---|---|---|---|---|
| GDN (L0) | 152.3 | 0.297 | 0.827 | 36% | 0.912 |
| attention (L3) | 137.9 | 0.269 | 0.656 | 41% | 0.772 |

### Dominant decode rows (optimized, one replay; dtype / fidelity as measured)

**GDN (L0)**

| op | us | cores | bound | DRAM % | fidelity | in0 dtype | weight dtype | in0_block_w |
|---|---|---|---|---|---|---|---|---|
| MatmulDeviceOperation 32 x 3072 x 20608 | 185.6 | 65 | DRAM | 66.60 | LoFi  | bf16 | bf8 | 8 |
| SparseMatmulDeviceOperation active=?/256 x 32 x 3072 x 2 | 119.2 | 64 |  |  | LoFi  | bf16 | bf4 | 32 |
| SparseMatmulDeviceOperation active=?/256 x 32 x 1024 x 3 | 59.3 | 48 |  |  | LoFi  | bf16 | bf4 | 32 |
| MatmulDeviceOperation 32 x 8192 x 3072 | 57.8 | 8 | DRAM | 85.10 | LoFi  | bf16 | bf8 | 8 |
| MatmulDeviceOperation 32 x 3072 x 2080 | 20.7 | 33 | SLOW | 60.17 | HiFi2 | bf16 | bf8 | 8 |
| BinaryNgDeviceOperation | 19.7 | 110 |  |  | BF16, | bf16 | bf16 |  |
| ReshapeViewDeviceOperation | 19.4 | 8 |  |  | BF16  | bf16 | nan |  |
| TernaryDeviceOperation | 14.3 | 110 |  |  | BF16, | bf16 | bf16 |  |
| TernaryDeviceOperation | 12.5 | 110 |  |  | BF16, | bf16 | bf16 |  |
| TernaryDeviceOperation | 12.2 | 110 |  |  | BF16, | bf16 | bf16 |  |
| MatmulDeviceOperation 32 x 1024 x 3072 | 11.9 | 32 | SLOW | 54.98 | HiFi2 | bf16 | bf8 | 8 |
| MatmulDeviceOperation b={64} x 32 x 128 x 128 | 11.8 | 64 | SLOW | 34.65 | HiFi2 | bf16 | bf16 | 4 |

**attention (L3)**

| op | us | cores | bound | DRAM % | fidelity | in0 dtype | weight dtype | in0_block_w |
|---|---|---|---|---|---|---|---|---|
| MatmulDeviceOperation 32 x 3072 x 17408 | 129.3 | 8 | DRAM | 80.80 | LoFi  | bf16 | bf8 | 3 |
| SparseMatmulDeviceOperation active=?/256 x 32 x 3072 x 2 | 118.2 | 64 |  |  | LoFi  | bf16 | bf4 | 32 |
| SparseMatmulDeviceOperation active=?/256 x 32 x 1024 x 3 | 61.1 | 48 |  |  | LoFi  | bf16 | bf4 | 32 |
| MatmulDeviceOperation 32 x 8192 x 3072 | 58.0 | 8 | DRAM | 84.68 | LoFi  | bf16 | bf8 | 8 |
| SdpaDecodeDeviceOperation | 24.3 | 64 |  |  | BF16, | bf16 | bf8 |  |
| MatmulDeviceOperation 32 x 3072 x 2080 | 20.8 | 33 | SLOW | 60.08 | HiFi2 | bf16 | bf8 | 8 |
| ReshapeViewDeviceOperation | 15.5 | 16 |  |  | BF16  | bf16 | nan |  |
| MatmulDeviceOperation 32 x 1024 x 3072 | 11.9 | 32 | SLOW | 54.76 | HiFi2 | bf16 | bf8 | 8 |
| PermuteDeviceOperation | 10.6 | 110 |  |  | BF16  | bf16 | nan |  |
| MatmulDeviceOperation 32 x 3072 x 256 | 9.1 | 8 | SLOW | 34.15 | HiFi2 | bf16 | bf16 | 32 |
| FillPadDeviceOperation | 7.6 | 110 |  |  | BF16  | bf16 | nan |  |
| BinaryNgDeviceOperation | 7.5 | 110 |  |  | BF16, | bf16 | bf16 |  |

### Op-class totals per decode replay (fused -> optimized, us)

**GDN (L0)**

| op class | fused us | optimized us |
|---|---|---|
| MatmulDeviceOperation | 550.6 | 312.3 |
| SparseMatmulDeviceOperation | 303.4 | 177.9 |
| LayerNormDeviceOperation | 111.0 | 24.0 |
| BinaryNgDeviceOperation | 93.0 | 89.0 |
| ReshapeViewDeviceOperation | 76.7 | 75.7 |
| TernaryDeviceOperation | 51.3 | 48.2 |
| UnaryDeviceOperation | 32.5 | 31.5 |
| SliceDeviceOperation | 30.0 | 27.0 |
| CopyDeviceOperation | 19.0 | 13.7 |
| ShardedToInterleavedDeviceOperation | 3.4 | 6.9 |
| FillPadDeviceOperation | 6.8 | 6.8 |
| InterleavedToShardedDeviceOperation | 0.5 | 4.7 |
| FastReduceNCDeviceOperation | 3.5 | 3.2 |
| TilizeWithValPaddingDeviceOperation | 2.7 | 2.7 |

**attention (L3)**

| op class | fused us | optimized us |
|---|---|---|
| MatmulDeviceOperation | 486.0 | 229.1 |
| SparseMatmulDeviceOperation | 302.2 | 177.7 |
| LayerNormDeviceOperation | 108.8 | 21.5 |
| BinaryNgDeviceOperation | 50.5 | 49.2 |
| ReshapeViewDeviceOperation | 42.5 | 41.3 |
| SliceDeviceOperation | 32.4 | 31.1 |
| SdpaDecodeDeviceOperation | 24.9 | 24.4 |
| UnaryDeviceOperation | 14.4 | 14.5 |
| PermuteDeviceOperation | 13.3 | 13.5 |
| TransposeDeviceOperation | 10.9 | 10.8 |
| PagedUpdateCacheDeviceOperation | 8.2 | 8.2 |
| ShardedToInterleavedDeviceOperation | 3.4 | 7.9 |
| FillPadDeviceOperation | 7.4 | 7.1 |
| InterleavedToShardedDeviceOperation | 3.3 | 6.7 |

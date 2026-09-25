# Stage-7 perf tables (Tracy, tt-perf-report signposted windows, reduced one-layer-per-kind model)

Before = stage-5 `MULTICHIP_SELECTED` (committed tables, `doc/optimized_multichip_decoder/tracy`); after = the stage-7 selected
policy (seven decode op-count knobs + sorted_min_tokens=256; prefill_256 has no stage-5 row). Device time = sum over the window per device (decode: per traced replay); ops = per replay.

| layer | case | before ms/device 0/1/2/3 (ops) | after ms/device 0/1/2/3 (ops) | after gaps ms (dev 0) | delta (max device) |
|---|---|---|---|---|---|
| GDN (L0) | prefill_128 | 8.199/8.187/8.205/8.202 (139/139/139/139) | 8.200/8.192/8.208/8.206 (136/136/136/136) | 2.940 | +0.002 ms (+0.0 %) |
| GDN (L0) | prefill_256 | n/a | 11.174/11.152/11.190/11.187 (228/228/228/228) | 3.992 | n/a |
| GDN (L0) | prefill_2048 | 60.971/60.929/61.014/60.987 (924/924/924/924) | 61.061/61.012/61.127/61.102 (912/912/912/912) | 10.368 | +0.114 ms (+0.2 %) |
| GDN (L0) | decode | 0.472/0.474/0.473/0.473 (103/103/103/103) | 0.452/0.454/0.453/0.453 (82/82/82/82) | 0.091 | -0.020 ms (-4.3 %) |
| attention (L3) | prefill_128 | 9.035/9.022/9.039/9.031 (94/94/94/94) | 9.021/9.012/9.027/9.018 (94/94/94/94) | 2.077 | -0.012 ms (-0.1 %) |
| attention (L3) | prefill_256 | n/a | 10.645/10.625/10.667/10.661 (186/186/186/186) | 2.514 | n/a |
| attention (L3) | prefill_2048 | 58.231/58.194/58.302/58.293 (744/744/744/744) | 58.263/58.218/58.249/58.209 (744/744/744/744) | 7.895 | -0.039 ms (-0.1 %) |
| attention (L3) | decode | 0.406/0.407/0.409/0.408 (84/84/84/84) | 0.395/0.398/0.399/0.397 (77/77/77/77) | 0.077 | -0.010 ms (-2.3 %) |

48-layer decode layer-stack device lower bound (36 x GDN + 12 x attention, max over devices): before 21.98 ms, after 21.14 ms.

## GDN (L0) traced decode, device 0, per replay (82 ops)

| op | count | device us | gaps us |
|---|---|---|---|
| Slice | 15 | 17.1 | 20.9 |
| BinaryNg | 15 | 60.3 | 7.8 |
| ReshapeView | 11 | 45.3 | 8.6 |
| Unary | 6 | 14.6 | 5.9 |
| LayerNorm | 5 | 23.3 | 4.9 |
| Copy | 4 | 6.6 | 2.2 |
| Ternary | 4 | 15.4 | 6.3 |
| AllGatherAsync | 2 | 24.3 | 12.4 |
| ShardedToInterleaved | 2 | 2.1 | 2.5 |
| Matmul 32 x 512 x 2048 | 2 | 20.4 | 7.8 |
| Matmul b={16} x 32 x 128 x 128 | 2 | 8.2 | 1.3 |
| ReduceScatterMinimalAsync | 2 | 21.8 | 1.4 |
| Matmul 32 x 3072 x 5184 | 1 | 50.8 | 0.5 |
| Transpose | 1 | 1.8 | 0.5 |
| Matmul 32 x 2048 x 3072 | 1 | 22.4 | 0.5 |
| Matmul 32 x 3072 x 256 | 1 | 9.2 | 1.0 |
| InterleavedToSharded | 1 | 0.5 | 1.5 |
| GeneralizedMoeGate | 1 | 1.4 | 1.5 |
| SparseMatmul active=?/256 x 32 x 3072 x 512 | 1 | 61.9 | 0.6 |
| SparseMatmul active=?/256 x 32 x 256 x 3072 | 1 | 17.6 | 0.6 |
| TilizeWithValPadding | 1 | 2.7 | 0.6 |
| FastReduceNC | 1 | 3.3 | 0.5 |
| Matmul 32 x 3072 x 544 | 1 | 15.5 | 0.5 |
| Matmul 32 x 256 x 3072 | 1 | 5.1 | 0.5 |

## attention (L3) traced decode, device 0, per replay (77 ops)

| op | count | device us | gaps us |
|---|---|---|---|
| Slice | 15 | 20.1 | 11.6 |
| BinaryNg | 10 | 36.1 | 5.3 |
| ShardedToInterleaved | 6 | 5.4 | 7.7 |
| LayerNorm | 4 | 21.2 | 4.0 |
| Transpose | 4 | 7.7 | 2.9 |
| InterleavedToSharded | 4 | 2.5 | 6.0 |
| Copy | 3 | 4.5 | 1.9 |
| ReshapeView | 3 | 13.7 | 2.4 |
| Unary | 3 | 8.5 | 1.5 |
| AllGatherAsync | 2 | 23.9 | 15.9 |
| RotaryEmbeddingHf | 2 | 7.9 | 2.0 |
| Concat | 2 | 3.3 | 1.3 |
| FillPad | 2 | 9.4 | 1.5 |
| PagedUpdateCache | 2 | 6.4 | 2.9 |
| ReduceScatterMinimalAsync | 2 | 21.7 | 1.5 |
| Matmul 32 x 3072 x 4608 | 1 | 45.7 | 0.5 |
| NLPCreateQKVHeadsDecode | 1 | 4.7 | 1.0 |
| SdpaDecode | 1 | 13.3 | 0.5 |
| NLPConcatHeadsDecode | 1 | 1.3 | 0.6 |
| Matmul 32 x 2048 x 3072 | 1 | 21.3 | 0.5 |
| Matmul 32 x 3072 x 256 | 1 | 9.2 | 1.0 |
| GeneralizedMoeGate | 1 | 1.4 | 1.5 |
| SparseMatmul active=?/256 x 32 x 3072 x 512 | 1 | 61.9 | 0.6 |
| SparseMatmul active=?/256 x 32 x 256 x 3072 | 1 | 17.7 | 0.5 |
| TilizeWithValPadding | 1 | 2.7 | 0.6 |
| FastReduceNC | 1 | 3.5 | 0.5 |
| Matmul 32 x 3072 x 544 | 1 | 15.3 | 0.5 |
| Matmul 32 x 256 x 3072 | 1 | 5.1 | 0.5 |

| layer | case | single-chip ms | mesh (1x4) ms | speedup | efficiency |
|---|---|---|---|---|---|
| full_attention | prefill_128 | 16.097 | 15.356 | 1.05x | 26% |
| full_attention | prefill_2048 | 229.755 | 238.668 | 0.96x | 24% |
| full_attention | traced_decode | 0.727 | 0.458 | 1.59x | 40% |
| linear_attention | prefill_128 | 18.511 | 15.867 | 1.17x | 29% |
| linear_attention | prefill_2048 | 250.441 | 238.718 | 1.05x | 26% |
| linear_attention | traced_decode | 0.867 | 0.584 | 1.48x | 37% |

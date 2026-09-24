| layer | case | single-chip ms | mesh (1x4) ms | speedup | efficiency |
|---|---|---|---|---|---|
| full_attention | prefill_128 | 16.097 | 15.133 | 1.06x | 27% |
| full_attention | prefill_2048 | 229.755 | 128.171 | 1.79x | 45% |
| full_attention | traced_decode | 0.727 | 0.449 | 1.62x | 40% |
| linear_attention | prefill_128 | 18.511 | 15.530 | 1.19x | 30% |
| linear_attention | prefill_2048 | 250.441 | 132.990 | 1.88x | 47% |
| linear_attention | traced_decode | 0.867 | 0.583 | 1.49x | 37% |

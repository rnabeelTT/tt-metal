| op | before: n / ms (avg cores) | after: n / ms (avg cores) |
|---|---|---|
| SparseMatmulDeviceOperation | 2 / 12.16 (32) | 2 / 4.68 (110) |
| PermuteDeviceOperation | 4 / 14.21 (84) | 2 / 0.13 (59) |
| AccumulationDeviceOperation | 2 / 10.77 (4) | 1 / 0.14 (1) |
| BinaryNgDeviceOperation | 40 / 4.82 (110) | 41 / 4.52 (110) |
| ScatterDeviceOperation | 3 / 2.40 (1) | 3 / 2.40 (1) |
| EmbeddingsDeviceOperation | 10 / 2.11 (69) | 10 / 2.12 (69) |
| MatmulDeviceOperation | 8 / 1.38 (87) | 11 / 2.69 (65) |
| UntilizeWithUnpaddingDeviceOperation | 23 / 1.48 (74) | 24 / 1.47 (73) |
| SliceDeviceOperation | 20 / 1.13 (93) | 21 / 1.66 (94) |
| UntilizeCodegenDeviceOperation | 2 / 0.81 (87) | 2 / 0.80 (87) |
| UnaryDeviceOperation | 6 / 0.29 (110) | 8 / 1.17 (110) |
| TypecastDeviceOperation | 9 / 0.36 (90) | 10 / 0.36 (81) |
| ReduceScatterMinimalAsyncDeviceOperation | 2 / 0.34 (12) | 2 / 0.34 (12) |
| ReshapeViewDeviceOperation | 17 / 0.21 (77) | 17 / 0.21 (77) |
| ReduceDeviceOperation | 3 / 0.20 (95) | 3 / 0.20 (95) |
| TilizeWithValPaddingDeviceOperation | 10 / 0.14 (62) | 11 / 0.17 (59) |
| SoftmaxDeviceOperation | 1 / 0.11 (64) | 1 / 0.12 (64) |
| TopKDeviceOperation | 1 / 0.05 (64) | 1 / 0.05 (64) |
| FillPadDeviceOperation | 1 / 0.01 (64) | 1 / 0.01 (64) |
| UntilizeDeviceOperation | - | 1 / 0.01 (1) |
| TilizeDeviceOperation | - | 1 / 0.00 (1) |
| **total** | **164 ops / 52.98 ms** | **173 ops / 23.24 ms** |

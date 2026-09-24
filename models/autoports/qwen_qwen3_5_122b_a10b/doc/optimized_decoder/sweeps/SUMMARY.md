# Stage-3 sweep summary (real weights, one Blackhole die)

PCC = prefill T=128 / decode vs the HF layer; prefill ms = warmed wall T=128; decode ms = traced, 3 replays (wall/3).

## family `baseline_stage2`

| candidate | layer | PCC prefill | PCC decode | prefill ms | decode ms | materialized (routed gu/down, shared, mixer, kv) | note |
|---|---|---|---|---|---|---|---|
| stage2_bfp8_hifi2 | GDN (L0) | 0.99882 | 0.99958 | 28.66 | 1.488 |  |  |
| stage2_bfp8_hifi2 | attention (L3) | 0.99960 | 0.99983 | 28.53 | 1.19 |  |  |

## family `dense_1d`

| candidate | layer | PCC prefill | PCC decode | prefill ms | decode ms | materialized (routed gu/down, shared, mixer, kv) | note |
|---|---|---|---|---|---|---|---|
| dense_1d_in64_out32 | GDN (L0) | 0.99882 | 0.99958 | 18.1 | 1.14 | bf4 / bf4 / bf8 / bf8 |  |
| dense_1d_in88_out32 | GDN (L0) | 0.99882 | 0.99958 | 18.22 | 1.143 | bf4 / bf4 / bf8 / bf8 |  |
| dense_1d_in44_out32 | GDN (L0) | 0.99882 | 0.99958 | 18.06 | 1.147 | bf4 / bf4 / bf8 / bf8 |  |
| dense_1d_in64_out48 | GDN (L0) | 0.99882 | 0.99958 | 18.09 | 1.135 | bf4 / bf4 / bf8 / bf8 |  |
| dense_1d_in64_out96 | GDN (L0) | 0.99882 | 0.99958 | 18.05 | 1.137 | bf4 / bf4 / bf8 / bf8 |  |
| dense_1d_in88_out88 | GDN (L0) | 0.99882 | 0.99958 | 18.88 | 1.141 | bf4 / bf4 / bf8 / bf8 |  |
| dense_1d_in64_out32_bw16 | GDN (L0) | 0.99882 | 0.99958 | 18.24 | 1.147 | bf4 / bf4 / bf8 / bf8 |  |
| dense_1d_in64_out32_bw32 | GDN (L0) | 0.99882 | 0.99958 | 17.94 | 1.149 | bf4 / bf4 / bf8 / bf8 |  |
| dense_1d_in64_out32 | attention (L3) | 0.99959 | 0.99983 | 15.78 | 0.888 | bf4 / bf4 / bf8 / bf8 / bf16 |  |
| dense_1d_in88_out32 | attention (L3) | 0.99959 | 0.99983 | 15.8 | 0.889 | bf4 / bf4 / bf8 / bf8 / bf16 |  |
| dense_1d_in44_out32 | attention (L3) | 0.99959 | 0.99983 | 16.12 | 0.897 | bf4 / bf4 / bf8 / bf8 / bf16 |  |
| dense_1d_in64_out48 | attention (L3) | 0.99959 | 0.99983 | 15.87 | 0.886 | bf4 / bf4 / bf8 / bf8 / bf16 |  |
| dense_1d_in64_out96 | attention (L3) | 0.99959 | 0.99983 | 15.87 | 0.882 | bf4 / bf4 / bf8 / bf8 / bf16 |  |
| dense_1d_in88_out88 | attention (L3) | 0.99959 | 0.99983 | 15.8 | 0.882 | bf4 / bf4 / bf8 / bf8 / bf16 |  |
| dense_1d_in64_out32_bw16 | attention (L3) | 0.99959 | 0.99983 | 15.91 | 0.892 | bf4 / bf4 / bf8 / bf8 / bf16 |  |
| dense_1d_in64_out32_bw32 | attention (L3) | 0.99959 | 0.99983 | 16.12 | 0.897 | bf4 / bf4 / bf8 / bf8 / bf16 |  |

## family `dense_ds`

| candidate | layer | PCC prefill | PCC decode | prefill ms | decode ms | materialized (routed gu/down, shared, mixer, kv) | note |
|---|---|---|---|---|---|---|---|
| dense_ds_r1 | GDN (L0) | - | - | - | - | - | FAIL: TT_THROW @ /home/raahem/tt-metal/tt_metal/impl/program/program.cpp:2065: tt::exception |
| dense_ds_r2 | GDN (L0) | - | - | - | - | - | FAIL: TT_THROW @ /home/raahem/tt-metal/tt_metal/impl/program/program.cpp:2065: tt::exception |
| dense_ds_r3 | GDN (L0) | 0.99882 | 0.99958 | 18.07 | 0.944 | bf4 / bf4 / bf8 / bf8 |  |
| dense_ds_r1 | attention (L3) | 0.99959 | 0.99983 | 15.88 | 0.731 | bf4 / bf4 / bf8 / bf8 / bf16 |  |
| dense_ds_r2 | attention (L3) | 0.99959 | 0.99983 | 15.93 | 0.834 | bf4 / bf4 / bf8 / bf8 / bf16 |  |
| dense_ds_r3 | attention (L3) | 0.99959 | 0.99983 | 15.91 | 0.769 | bf4 / bf4 / bf8 / bf8 / bf16 |  |

## family `dense_ds2`

| candidate | layer | PCC prefill | PCC decode | prefill ms | decode ms | materialized (routed gu/down, shared, mixer, kv) | note |
|---|---|---|---|---|---|---|---|
| ds2_attn_all_r1 | GDN (L0) | 0.99882 | 0.99960 | 18.14 | 0.883 | bf4 / bf4 / bf8 / bf8 |  |
| ds2_attn_o_r1 | GDN (L0) | 0.99882 | 0.99960 | 17.91 | 0.885 | bf4 / bf4 / bf8 / bf8 |  |
| ds2_gdn_o_r1 | GDN (L0) | 0.99882 | 0.99960 | 17.93 | 0.869 | bf4 / bf4 / bf8 / bf8 |  |
| ds2_gdn_mega_r3 | GDN (L0) | 0.99882 | 0.99958 | 17.9 | 0.928 | bf4 / bf4 / bf8 / bf8 |  |
| ds2_attn_all_r1 | attention (L3) | 0.99959 | 0.99983 | 15.79 | 0.735 | bf4 / bf4 / bf8 / bf8 / bf16 |  |
| ds2_attn_o_r1 | attention (L3) | 0.99959 | 0.99983 | 15.99 | 0.76 | bf4 / bf4 / bf8 / bf8 / bf16 |  |
| ds2_gdn_o_r1 | attention (L3) | 0.99959 | 0.99983 | 15.72 | 0.776 | bf4 / bf4 / bf8 / bf8 / bf16 |  |
| ds2_gdn_mega_r3 | attention (L3) | 0.99959 | 0.99983 | 15.77 | 0.774 | bf4 / bf4 / bf8 / bf8 / bf16 |  |

## family `ds_cores`

| candidate | layer | PCC prefill | PCC decode | prefill ms | decode ms | materialized (routed gu/down, shared, mixer, kv) | note |
|---|---|---|---|---|---|---|---|
| ds_cores_32 | attention (L3) | 0.99959 | 0.99983 | 15.84 | 0.732 | bf4 / bf4 / bf8 / bf8 / bf16 |  |
| ds_cores_16 | attention (L3) | - | - | - | - | - | FAIL: TT_THROW @ /home/raahem/tt-metal/tt_metal/impl/program/program.cpp:2065: tt::exception |
| ds_cores_8 | attention (L3) | - | - | - | - | - | FAIL: TT_THROW @ /home/raahem/tt-metal/tt_metal/impl/program/program.cpp:2065: tt::exception |

## family `gather`

| candidate | layer | PCC prefill | PCC decode | prefill ms | decode ms | materialized (routed gu/down, shared, mixer, kv) | note |
|---|---|---|---|---|---|---|---|
| gather_gu64_in96_dn64_in32 | GDN (L0) | 0.99882 | 0.99958 | 21.34 | 1.375 | bf4 / bf4 / bf8 / bf8 |  |
| gather_gu64_in32_dn64_in32 | GDN (L0) | 0.99882 | 0.99958 | 21.07 | 1.358 | bf4 / bf4 / bf8 / bf8 |  |
| gather_gu64_in8_dn64_in8 | GDN (L0) | 0.99882 | 0.99958 | 21.16 | 1.373 | bf4 / bf4 / bf8 / bf8 |  |
| gather_gu32_in96_dn32_in32 | GDN (L0) | 0.99882 | 0.99958 | 21.1 | 1.363 | bf4 / bf4 / bf8 / bf8 |  |
| gather_gu110_in96_dn96_in32 | GDN (L0) | - | - | - | - | - | FAIL: TT_FATAL @ /home/raahem/tt-metal/tt_metal/impl/program/program.cpp:153: not on_dispatch_core |
| gather_gu64_in48_dn64_in16 | GDN (L0) | 0.99882 | 0.99958 | 21.16 | 1.362 | bf4 / bf4 / bf8 / bf8 |  |
| gather_gu64_in96_dn64_in32 | attention (L3) | 0.99952 | 0.99983 | 19.46 | 1.078 | bf4 / bf4 / bf8 / bf8 / bf16 |  |
| gather_gu64_in32_dn64_in32 | attention (L3) | 0.99952 | 0.99983 | 19.24 | 1.063 | bf4 / bf4 / bf8 / bf8 / bf16 |  |
| gather_gu64_in8_dn64_in8 | attention (L3) | 0.99952 | 0.99983 | 19.2 | 1.08 | bf4 / bf4 / bf8 / bf8 / bf16 |  |
| gather_gu32_in96_dn32_in32 | attention (L3) | 0.99952 | 0.99983 | 19.44 | 1.064 | bf4 / bf4 / bf8 / bf8 / bf16 |  |
| gather_gu110_in96_dn96_in32 | attention (L3) | - | - | - | - | - | FAIL: TT_FATAL @ /home/raahem/tt-metal/tt_metal/impl/program/program.cpp:153: not on_dispatch_core |
| gather_gu64_in48_dn64_in16 | attention (L3) | 0.99952 | 0.99983 | 19.48 | 1.072 | bf4 / bf4 / bf8 / bf8 / bf16 |  |

## family `gdn_dtype2`

| candidate | layer | PCC prefill | PCC decode | prefill ms | decode ms | materialized (routed gu/down, shared, mixer, kv) | note |
|---|---|---|---|---|---|---|---|
| gdn2_bfp8 | GDN (L0) | 0.99882 | 0.99958 | 18.29 | 1.284 | bf4 / bf4 / bf8 / bf8 |  |
| gdn2_bfp4 | GDN (L0) | 0.99527 | 0.99773 | 17.97 | 1.225 | bf4 / bf4 / bf8 / bf4 |  |

## family `gdn_state`

| candidate | layer | PCC prefill | PCC decode | prefill ms | decode ms | materialized (routed gu/down, shared, mixer, kv) | note |
|---|---|---|---|---|---|---|---|
| gdn_state_bf16 | GDN (L0) | 0.99882 | 0.99960 | 18.15 | 0.884 | bf4 / bf4 / bf8 / bf8 |  |
| gdn_state_fp32 | GDN (L0) | 0.99882 | 0.99957 | 18.63 | 0.941 | bf4 / bf4 / bf8 / bf8 |  |

## family `kv`

| candidate | layer | PCC prefill | PCC decode | prefill ms | decode ms | materialized (routed gu/down, shared, mixer, kv) | note |
|---|---|---|---|---|---|---|---|
| kv_bf16 | GDN (L0) | 0.99882 | 0.99958 | 21.15 | 1.374 | bf4 / bf4 / bf8 / bf8 |  |
| kv_bfp8 | GDN (L0) | 0.99882 | 0.99958 | 20.93 | 1.379 | bf4 / bf4 / bf8 / bf8 |  |
| kv_bf16 | attention (L3) | 0.99952 | 0.99983 | 19.16 | 1.074 | bf4 / bf4 / bf8 / bf8 / bf16 |  |
| kv_bfp8 | attention (L3) | 0.99954 | 0.99983 | 19.2 | 1.077 | bf4 / bf4 / bf8 / bf8 / bf8 |  |

## family `kv_final`

| candidate | layer | PCC prefill | PCC decode | prefill ms | decode ms | materialized (routed gu/down, shared, mixer, kv) | note |
|---|---|---|---|---|---|---|---|
| kvf_bf16 | attention (L3) | 0.99959 | 0.99983 | 15.75 | 0.846 | bf4 / bf4 / bf8 / bf8 / bf16 |  |
| kvf_bfp8_k64 | attention (L3) | 0.99962 | 0.99986 | 15.83 | 0.824 | bf4 / bf4 / bf8 / bf8 / bf8 |  |
| kvf_bfp8_k256 | attention (L3) | 0.99962 | 0.99985 | 15.81 | 0.808 | bf4 / bf4 / bf8 / bf8 / bf8 |  |

## family `masked`

| candidate | layer | PCC prefill | PCC decode | prefill ms | decode ms | materialized (routed gu/down, shared, mixer, kv) | note |
|---|---|---|---|---|---|---|---|
| masked_c32_gu32_in64_dn32 | GDN (L0) | 0.99882 | 0.99958 | 21.22 | 1.373 | bf4 / bf4 / bf8 / bf8 |  |
| masked_c32_gu64_in64_dn32 | GDN (L0) | 0.99882 | 0.99958 | 21.74 | 1.374 | bf4 / bf4 / bf8 / bf8 |  |
| masked_c32_gu16_in96_dn16 | GDN (L0) | 0.99882 | 0.99958 | 25.08 | 1.367 | bf4 / bf4 / bf8 / bf8 |  |
| masked_c64_gu32_in64_dn32 | GDN (L0) | 0.99882 | 0.99958 | 17.88 | 1.372 | bf4 / bf4 / bf8 / bf8 |  |
| masked_c64_gu64_in96_dn64 | GDN (L0) | 0.99882 | 0.99958 | 19.23 | 1.368 | bf4 / bf4 / bf8 / bf8 |  |
| masked_c128_gu64_in96_dn64 | GDN (L0) | - | - | - | - | - | FAIL: TT_THROW @ /home/raahem/tt-metal/tt_metal/impl/program/program.cpp:2065: tt::exception |
| masked_c32_gu32_in64_dn32 | attention (L3) | 0.99952 | 0.99983 | 19.17 | 1.082 | bf4 / bf4 / bf8 / bf8 / bf16 |  |
| masked_c32_gu64_in64_dn32 | attention (L3) | 0.99952 | 0.99983 | 20.15 | 1.075 | bf4 / bf4 / bf8 / bf8 / bf16 |  |
| masked_c32_gu16_in96_dn16 | attention (L3) | 0.99952 | 0.99983 | 23.46 | 1.079 | bf4 / bf4 / bf8 / bf8 / bf16 |  |
| masked_c64_gu32_in64_dn32 | attention (L3) | 0.99952 | 0.99983 | 15.77 | 1.075 | bf4 / bf4 / bf8 / bf8 / bf16 |  |
| masked_c64_gu64_in96_dn64 | attention (L3) | 0.99952 | 0.99983 | 17.41 | 1.079 | bf4 / bf4 / bf8 / bf8 / bf16 |  |
| masked_c128_gu64_in96_dn64 | attention (L3) | - | - | - | - | - | FAIL: TT_THROW @ /home/raahem/tt-metal/tt_metal/impl/program/program.cpp:2065: tt::exception |

## family `masked2`

| candidate | layer | PCC prefill | PCC decode | prefill ms | decode ms | materialized (routed gu/down, shared, mixer, kv) | note |
|---|---|---|---|---|---|---|---|
| masked2_c64_gu32_in96_dn32_in32 | GDN (L0) | 0.99882 | 0.99958 | 18.32 | 1.286 | bf4 / bf4 / bf8 / bf8 |  |
| masked2_c64_gu32_in32_dn32_in16 | GDN (L0) | 0.99882 | 0.99958 | 18.05 | 1.286 | bf4 / bf4 / bf8 / bf8 |  |
| masked2_c64_gu16_in96_dn16_in32 | GDN (L0) | 0.99882 | 0.99958 | 21.61 | 1.284 | bf4 / bf4 / bf8 / bf8 |  |
| masked2_c128_gu32_in32_dn32_in32 | GDN (L0) | - | - | - | - | - | FAIL: TT_FATAL @ /home/raahem/tt-metal/ttnn/cpp/ttnn/operations/matmul/device/matmul_device_operation.cpp:2802: devi |
| masked2_c128_gu16_in64_dn16_in32 | GDN (L0) | - | - | - | - | - | FAIL: TT_FATAL @ /home/raahem/tt-metal/ttnn/cpp/ttnn/operations/matmul/device/matmul_device_operation.cpp:2802: devi |
| masked2_c128_gu32_in16_dn32_in16 | GDN (L0) | - | - | - | - | - | FAIL: TT_FATAL @ /home/raahem/tt-metal/ttnn/cpp/ttnn/operations/matmul/device/matmul_device_operation.cpp:2802: devi |
| masked2_c64_gu32_in96_dn32_in32 | attention (L3) | 0.99959 | 0.99983 | 15.92 | 0.985 | bf4 / bf4 / bf8 / bf8 / bf16 |  |
| masked2_c64_gu32_in32_dn32_in16 | attention (L3) | 0.99959 | 0.99983 | 16.26 | 0.985 | bf4 / bf4 / bf8 / bf8 / bf16 |  |
| masked2_c64_gu16_in96_dn16_in32 | attention (L3) | 0.99959 | 0.99983 | 21.17 | 0.986 | bf4 / bf4 / bf8 / bf8 / bf16 |  |
| masked2_c128_gu32_in32_dn32_in32 | attention (L3) | - | - | - | - | - | FAIL: TT_FATAL @ /home/raahem/tt-metal/ttnn/cpp/ttnn/operations/matmul/device/matmul_device_operation.cpp:2802: devi |
| masked2_c128_gu16_in64_dn16_in32 | attention (L3) | - | - | - | - | - | FAIL: TT_FATAL @ /home/raahem/tt-metal/ttnn/cpp/ttnn/operations/matmul/device/matmul_device_operation.cpp:2802: devi |
| masked2_c128_gu32_in16_dn32_in16 | attention (L3) | - | - | - | - | - | FAIL: TT_FATAL @ /home/raahem/tt-metal/ttnn/cpp/ttnn/operations/matmul/device/matmul_device_operation.cpp:2802: devi |

## family `mega_split`

| candidate | layer | PCC prefill | PCC decode | prefill ms | decode ms | materialized (routed gu/down, shared, mixer, kv) | note |
|---|---|---|---|---|---|---|---|
| mega_split_r1 | GDN (L0) | 0.99882 | 0.99961 | 18.47 | 0.88 | bf4 / bf4 / bf8 / bf8 |  |
| mega_split_r2 | GDN (L0) | 0.99882 | 0.99961 | 18.01 | 0.903 | bf4 / bf4 / bf8 / bf8 |  |
| mega_split_r3 | GDN (L0) | 0.99882 | 0.99961 | 18.55 | 0.895 | bf4 / bf4 / bf8 / bf8 |  |

## family `mixer_weights`

| candidate | layer | PCC prefill | PCC decode | prefill ms | decode ms | materialized (routed gu/down, shared, mixer, kv) | note |
|---|---|---|---|---|---|---|---|
| attn_bfp8 | GDN (L0) | 0.99882 | 0.99958 | 21.05 | 1.37 | bf4 / bf4 / bf8 / bf8 |  |
| attn_bfp4 | GDN (L0) | 0.99882 | 0.99958 | 21.03 | 1.374 | bf4 / bf4 / bf8 / bf8 |  |
| gdn_bfp8 | GDN (L0) | 0.99882 | 0.99958 | 20.91 | 1.366 | bf4 / bf4 / bf8 / bf8 |  |
| gdn_bfp4 | GDN (L0) | 0.99528 | 0.99774 | 21.09 | 1.371 | bf4 / bf4 / bf8 / bf4 |  |
| attn_bfp8 | attention (L3) | 0.99952 | 0.99983 | 19.17 | 1.074 | bf4 / bf4 / bf8 / bf8 / bf16 |  |
| attn_bfp4 | attention (L3) | 0.99466 | 0.99819 | 19.19 | 1.024 | bf4 / bf4 / bf8 / bf4 / bf16 |  |
| gdn_bfp8 | attention (L3) | 0.99952 | 0.99983 | 19.19 | 1.075 | bf4 / bf4 / bf8 / bf8 / bf16 |  |
| gdn_bfp4 | attention (L3) | 0.99952 | 0.99983 | 19.55 | 1.081 | bf4 / bf4 / bf8 / bf8 / bf16 |  |

## family `norm`

| candidate | layer | PCC prefill | PCC decode | prefill ms | decode ms | materialized (routed gu/down, shared, mixer, kv) | note |
|---|---|---|---|---|---|---|---|
| norm_interleaved | GDN (L0) | 0.99882 | 0.99958 | 21.03 | 1.369 | bf4 / bf4 / bf8 / bf8 |  |
| norm_sharded | GDN (L0) | 0.99882 | 0.99958 | 20.98 | 1.306 | bf4 / bf4 / bf8 / bf8 |  |
| norm_interleaved | attention (L3) | 0.99952 | 0.99983 | 19.26 | 1.074 | bf4 / bf4 / bf8 / bf8 / bf16 |  |
| norm_sharded | attention (L3) | 0.99952 | 0.99983 | 19.16 | 1.002 | bf4 / bf4 / bf8 / bf8 / bf16 |  |

## family `norm_ds`

| candidate | layer | PCC prefill | PCC decode | prefill ms | decode ms | materialized (routed gu/down, shared, mixer, kv) | note |
|---|---|---|---|---|---|---|---|
| norm_ds_off | attention (L3) | 0.99962 | 0.99985 | 15.85 | 0.731 | bf4 / bf4 / bf8 / bf8 / bf8 |  |
| norm_ds_on | attention (L3) | 0.99962 | 0.99985 | 15.77 | 0.725 | bf4 / bf4 / bf8 / bf8 / bf8 |  |

## family `precision`

| candidate | layer | PCC prefill | PCC decode | prefill ms | decode ms | materialized (routed gu/down, shared, mixer, kv) | note |
|---|---|---|---|---|---|---|---|
| routed_bfp8_hifi2 | GDN (L0) | 0.99882 | 0.99958 | 28.77 | 1.486 |  |  |
| routed_bfp8_lofi | GDN (L0) | 0.99882 | 0.99958 | 28.4 | 1.482 |  |  |
| routed_bfp4gu_bfp8down_lofi | GDN (L0) | 0.99882 | 0.99958 | 23.26 | 1.414 |  |  |
| routed_bfp4_lofi | GDN (L0) | 0.99882 | 0.99958 | 21.15 | 1.37 |  |  |
| routed_bfp4_hifi2 | GDN (L0) | 0.99882 | 0.99958 | 21.69 | 1.378 |  |  |
| routed_bfp8_hifi2 | attention (L3) | 0.99952 | 0.99983 | 28.6 | 1.189 |  |  |
| routed_bfp8_lofi | attention (L3) | 0.99952 | 0.99983 | 28.05 | 1.181 |  |  |
| routed_bfp4gu_bfp8down_lofi | attention (L3) | 0.99952 | 0.99983 | 21.85 | 1.117 |  |  |
| routed_bfp4_lofi | attention (L3) | 0.99952 | 0.99983 | 19.36 | 1.076 |  |  |
| routed_bfp4_hifi2 | attention (L3) | 0.99952 | 0.99983 | 19.64 | 1.084 |  |  |

## family `prefill_mode_1024`

| candidate | layer | PCC prefill | PCC decode | prefill ms | decode ms | materialized (routed gu/down, shared, mixer, kv) | note |
|---|---|---|---|---|---|---|---|
| prefill_masked | attention (L3) | nan | nan | 126.36 | 0.773 | bf4 / bf4 / bf8 / bf8 / bf16 |  |
| prefill_sorted | attention (L3) | nan | nan | 80.62 | 0.775 | bf4 / bf4 / bf8 / bf8 / bf16 |  |

## family `prefill_mode_128`

| candidate | layer | PCC prefill | PCC decode | prefill ms | decode ms | materialized (routed gu/down, shared, mixer, kv) | note |
|---|---|---|---|---|---|---|---|
| prefill_masked | GDN (L0) | 0.99882 | 0.99960 | 18.15 | 0.884 | bf4 / bf4 / bf8 / bf8 |  |
| prefill_sorted | GDN (L0) | - | - | - | - | - | FAIL: TT_FATAL @ /home/raahem/tt-metal/ttnn/cpp/ttnn/operations/data_movement/scatter/scatter.cpp:109: !(input_dtype |
| prefill_masked | attention (L3) | 0.99959 | 0.99983 | 15.77 | 0.777 | bf4 / bf4 / bf8 / bf8 / bf16 |  |
| prefill_sorted | attention (L3) | - | - | - | - | - | FAIL: TT_FATAL @ /home/raahem/tt-metal/ttnn/cpp/ttnn/operations/data_movement/scatter/scatter.cpp:109: !(input_dtype |
| prefill_masked | GDN (L0) | 0.99882 | 0.99960 | 18.17 | 0.882 | bf4 / bf4 / bf8 / bf8 |  |
| prefill_sorted | GDN (L0) | - | - | - | - | - | FAIL: TT_FATAL @ /home/raahem/tt-metal/ttnn/cpp/ttnn/operations/matmul/device/utilities/matmul_utilities.cpp:216: !( |
| prefill_masked | attention (L3) | 0.99959 | 0.99983 | 15.89 | 0.776 | bf4 / bf4 / bf8 / bf8 / bf16 |  |
| prefill_sorted | attention (L3) | - | - | - | - | - | FAIL: TT_FATAL @ /home/raahem/tt-metal/ttnn/cpp/ttnn/operations/matmul/device/utilities/matmul_utilities.cpp:216: !( |
| prefill_masked | GDN (L0) | 0.99882 | 0.99960 | 18.13 | 0.883 | bf4 / bf4 / bf8 / bf8 |  |
| prefill_sorted | GDN (L0) | 0.99882 | 0.99960 | 46.83 | 0.886 | bf4 / bf4 / bf8 / bf8 |  |
| prefill_masked | attention (L3) | 0.99959 | 0.99983 | 15.79 | 0.774 | bf4 / bf4 / bf8 / bf8 / bf16 |  |
| prefill_sorted | attention (L3) | 0.99959 | 0.99983 | 38.49 | 0.774 | bf4 / bf4 / bf8 / bf8 / bf16 |  |
| prefill_masked | GDN (L0) | 0.99882 | 0.99960 | 18.11 | 0.881 | bf4 / bf4 / bf8 / bf8 |  |
| prefill_sorted | GDN (L0) | 0.99882 | 0.99960 | 41.48 | 0.883 | bf4 / bf4 / bf8 / bf8 |  |
| prefill_masked | attention (L3) | 0.99959 | 0.99983 | 15.83 | 0.774 | bf4 / bf4 / bf8 / bf8 / bf16 |  |
| prefill_sorted | attention (L3) | 0.99959 | 0.99983 | 38.5 | 0.773 | bf4 / bf4 / bf8 / bf8 / bf16 |  |

## family `prefill_mode_2048`

| candidate | layer | PCC prefill | PCC decode | prefill ms | decode ms | materialized (routed gu/down, shared, mixer, kv) | note |
|---|---|---|---|---|---|---|---|
| prefill_masked | GDN (L0) | 0.99916 | 0.99993 | 244.69 | 0.882 | bf4 / bf4 / bf8 / bf8 |  |
| prefill_sorted | GDN (L0) | - | - | - | - | - | FAIL: TT_FATAL @ /home/raahem/tt-metal/ttnn/cpp/ttnn/operations/data_movement/scatter/scatter.cpp:109: !(input_dtype |
| prefill_masked | attention (L3) | 0.99978 | 0.99990 | 259.45 | 0.777 | bf4 / bf4 / bf8 / bf8 / bf16 |  |
| prefill_sorted | attention (L3) | - | - | - | - | - | FAIL: TT_FATAL @ /home/raahem/tt-metal/ttnn/cpp/ttnn/operations/data_movement/scatter/scatter.cpp:109: !(input_dtype |
| prefill_masked | GDN (L0) | 0.99916 | 0.99993 | 244.6 | 0.883 | bf4 / bf4 / bf8 / bf8 |  |
| prefill_sorted | GDN (L0) | - | - | - | - | - | FAIL: TT_FATAL @ /home/raahem/tt-metal/ttnn/cpp/ttnn/operations/matmul/device/utilities/matmul_utilities.cpp:216: !( |
| prefill_masked | attention (L3) | 0.99978 | 0.99990 | 259.2 | 0.775 | bf4 / bf4 / bf8 / bf8 / bf16 |  |
| prefill_sorted | attention (L3) | - | - | - | - | - | FAIL: TT_FATAL @ /home/raahem/tt-metal/ttnn/cpp/ttnn/operations/matmul/device/utilities/matmul_utilities.cpp:216: !( |
| prefill_masked | GDN (L0) | 0.99916 | 0.99993 | 244.63 | 0.881 | bf4 / bf4 / bf8 / bf8 |  |
| prefill_sorted | GDN (L0) | 0.99916 | 0.99993 | 164.49 | 0.883 | bf4 / bf4 / bf8 / bf8 |  |
| prefill_masked | attention (L3) | 0.99978 | 0.99990 | 259.02 | 0.772 | bf4 / bf4 / bf8 / bf8 / bf16 |  |
| prefill_sorted | attention (L3) | 0.99978 | 0.99990 | 135.37 | 0.774 | bf4 / bf4 / bf8 / bf8 / bf16 |  |

## family `prefill_mode_256`

| candidate | layer | PCC prefill | PCC decode | prefill ms | decode ms | materialized (routed gu/down, shared, mixer, kv) | note |
|---|---|---|---|---|---|---|---|
| prefill_masked | attention (L3) | nan | nan | 30.7 | 0.777 | bf4 / bf4 / bf8 / bf8 / bf16 |  |
| prefill_sorted | attention (L3) | nan | nan | 43.66 | 0.774 | bf4 / bf4 / bf8 / bf8 / bf16 |  |

## family `prefill_mode_4096`

| candidate | layer | PCC prefill | PCC decode | prefill ms | decode ms | materialized (routed gu/down, shared, mixer, kv) | note |
|---|---|---|---|---|---|---|---|
| prefill_masked | attention (L3) | nan | nan | 517.61 | 0.733 | bf4 / bf4 / bf8 / bf8 / bf16 |  |
| prefill_sorted | attention (L3) | nan | nan | 244.71 | 0.735 | bf4 / bf4 / bf8 / bf8 / bf16 |  |

## family `prefill_mode_512`

| candidate | layer | PCC prefill | PCC decode | prefill ms | decode ms | materialized (routed gu/down, shared, mixer, kv) | note |
|---|---|---|---|---|---|---|---|
| prefill_masked | attention (L3) | nan | nan | 62.19 | 0.78 | bf4 / bf4 / bf8 / bf8 / bf16 |  |
| prefill_sorted | attention (L3) | nan | nan | 54.43 | 0.774 | bf4 / bf4 / bf8 / bf8 / bf16 |  |

## family `router`

| candidate | layer | PCC prefill | PCC decode | prefill ms | decode ms | materialized (routed gu/down, shared, mixer, kv) | note |
|---|---|---|---|---|---|---|---|
| router_ttnn | attention (L3) | 0.99959 | 0.99983 | 15.97 | 0.987 | bf4 / bf4 / bf8 / bf8 / bf16 |  |
| router_derived | attention (L3) | 0.99959 | 0.99983 | 15.73 | 0.924 | bf4 / bf4 / bf8 / bf8 / bf16 |  |
| router_1d8 | attention (L3) | 0.99959 | 0.99983 | 16.05 | 0.93 | bf4 / bf4 / bf8 / bf8 / bf16 |  |

## family `sdpa`

| candidate | layer | PCC prefill | PCC decode | prefill ms | decode ms | materialized (routed gu/down, shared, mixer, kv) | note |
|---|---|---|---|---|---|---|---|
| sdpa_default_32_64 | GDN (L0) | 0.99882 | 0.99958 | 21.17 | 1.371 | bf4 / bf4 / bf8 / bf8 |  |
| sdpa_32_128 | GDN (L0) | 0.99882 | 0.99958 | 20.89 | 1.367 | bf4 / bf4 / bf8 / bf8 |  |
| sdpa_32_256 | GDN (L0) | 0.99882 | 0.99958 | 21.52 | 1.375 | bf4 / bf4 / bf8 / bf8 |  |
| sdpa_0_0 | GDN (L0) | 0.99882 | 0.99958 | 21.94 | 1.367 | bf4 / bf4 / bf8 / bf8 |  |
| sdpa_default_32_64 | attention (L3) | 0.99952 | 0.99983 | 19.2 | 1.074 | bf4 / bf4 / bf8 / bf8 / bf16 |  |
| sdpa_32_128 | attention (L3) | 0.99952 | 0.99982 | 19.35 | 1.075 | bf4 / bf4 / bf8 / bf8 / bf16 |  |
| sdpa_32_256 | attention (L3) | 0.99952 | 0.99985 | 19.26 | 1.078 | bf4 / bf4 / bf8 / bf8 / bf16 |  |
| sdpa_0_0 | attention (L3) | 0.99952 | 0.99983 | 19.29 | 1.077 | bf4 / bf4 / bf8 / bf8 / bf16 |  |

## family `sdpa_long`

| candidate | layer | PCC prefill | PCC decode | prefill ms | decode ms | materialized (routed gu/down, shared, mixer, kv) | note |
|---|---|---|---|---|---|---|---|
| sdpa_long_default_32_64 | attention (L3) | 0.99959 | 0.99983 | 15.89 | 1.1 | bf4 / bf4 / bf8 / bf8 / bf16 |  |
| sdpa_long_32_128 | attention (L3) | 0.99959 | 0.99983 | 15.82 | 1.099 | bf4 / bf4 / bf8 / bf8 / bf16 |  |
| sdpa_long_32_256 | attention (L3) | 0.99959 | 0.99986 | 16.06 | 1.111 | bf4 / bf4 / bf8 / bf8 / bf16 |  |
| sdpa_long_32_512 | attention (L3) | - | - | - | - | - | FAIL: TT_THROW @ /home/raahem/tt-metal/tt_metal/impl/program/program.cpp:2065: tt::exception |
| sdpa_long_kv_bfp8_32_64 | attention (L3) | 0.99962 | 0.99984 | 16.06 | 1.08 | bf4 / bf4 / bf8 / bf8 / bf8 |  |
| sdpa_long_kv_bfp8_32_256 | attention (L3) | 0.99962 | 0.99984 | 16.09 | 1.063 | bf4 / bf4 / bf8 / bf8 / bf8 |  |

## family `shared`

| candidate | layer | PCC prefill | PCC decode | prefill ms | decode ms | materialized (routed gu/down, shared, mixer, kv) | note |
|---|---|---|---|---|---|---|---|
| shared_bfp8_hifi2 | GDN (L0) | 0.99882 | 0.99958 | 21.29 | 1.372 |  |  |
| shared_bfp8_lofi | GDN (L0) | 0.99882 | 0.99958 | 20.89 | 1.37 |  |  |
| shared_bfp4_lofi | GDN (L0) | 0.99879 | 0.99958 | 21.84 | 1.369 |  |  |
| shared_bfp8_hifi2 | attention (L3) | 0.99959 | 0.99983 | 19.16 | 1.078 |  |  |
| shared_bfp8_lofi | attention (L3) | 0.99952 | 0.99983 | 19.18 | 1.08 |  |  |
| shared_bfp4_lofi | attention (L3) | 0.99932 | 0.99979 | 19.29 | 1.077 |  |  |

## family `shared_pc`

| candidate | layer | PCC prefill | PCC decode | prefill ms | decode ms | materialized (routed gu/down, shared, mixer, kv) | note |
|---|---|---|---|---|---|---|---|
| shared_pc_None | attention (L3) | 0.99959 | 0.99983 | 16.04 | 0.987 | bf4 / bf4 / bf8 / bf8 / bf16 |  |
| shared_pc_16 | attention (L3) | 0.99959 | 0.99983 | 16.08 | 0.942 | bf4 / bf4 / bf8 / bf8 / bf16 |  |
| shared_pc_32 | attention (L3) | 0.99959 | 0.99983 | 16.04 | 0.938 | bf4 / bf4 / bf8 / bf8 / bf16 |  |
| shared_pc_44 | attention (L3) | 0.99959 | 0.99983 | 15.86 | 0.937 | bf4 / bf4 / bf8 / bf8 / bf16 |  |
| shared_pc_64 | attention (L3) | 0.99959 | 0.99983 | 16.07 | 0.937 | bf4 / bf4 / bf8 / bf8 / bf16 |  |
| shared_pc_88 | attention (L3) | 0.99959 | 0.99983 | 15.83 | 0.938 | bf4 / bf4 / bf8 / bf8 / bf16 |  |

## family `sorted_geom_2048`

| candidate | layer | PCC prefill | PCC decode | prefill ms | decode ms | materialized (routed gu/down, shared, mixer, kv) | note |
|---|---|---|---|---|---|---|---|
| sorted_gu64_in32_dn64_in32 | attention (L3) | nan | nan | 136.55 | 0.773 | bf4 / bf4 / bf8 / bf8 / bf16 |  |
| sorted_gu64_in96_dn64_in32 | attention (L3) | nan | nan | 134.92 | 0.773 | bf4 / bf4 / bf8 / bf8 / bf16 |  |
| sorted_gu32_in64_dn32_in32 | attention (L3) | nan | nan | 133.61 | 0.775 | bf4 / bf4 / bf8 / bf8 / bf16 |  |
| sorted_gu64_in8_dn64_in8 | attention (L3) | nan | nan | 135.91 | 0.775 | bf4 / bf4 / bf8 / bf8 / bf16 |  |
| sorted_gu96_in32_dn96_in32 | attention (L3) | - | - | - | - | - | FAIL: TT_FATAL @ /home/raahem/tt-metal/tt_metal/impl/program/program.cpp:153: not on_dispatch_core |

## family `sorted_group_2048`

| candidate | layer | PCC prefill | PCC decode | prefill ms | decode ms | materialized (routed gu/down, shared, mixer, kv) | note |
|---|---|---|---|---|---|---|---|
| sorted_g32 | attention (L3) | nan | nan | 134.58 | 0.774 | bf4 / bf4 / bf8 / bf8 / bf16 |  |
| sorted_g64 | attention (L3) | nan | nan | 161.92 | 0.775 | bf4 / bf4 / bf8 / bf8 / bf16 |  |
| sorted_g128 | attention (L3) | nan | nan | 220.21 | 0.777 | bf4 / bf4 / bf8 / bf8 / bf16 |  |

## family `sorted_group_512`

| candidate | layer | PCC prefill | PCC decode | prefill ms | decode ms | materialized (routed gu/down, shared, mixer, kv) | note |
|---|---|---|---|---|---|---|---|
| sorted_g32 | attention (L3) | nan | nan | 59.27 | 0.777 | bf4 / bf4 / bf8 / bf8 / bf16 |  |
| sorted_g64 | attention (L3) | nan | nan | 93.46 | 0.775 | bf4 / bf4 / bf8 / bf8 / bf16 |  |
| sorted_g128 | attention (L3) | nan | nan | 160.28 | 0.775 | bf4 / bf4 / bf8 / bf8 / bf16 |  |

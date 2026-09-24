# Stage-4 mesh sweep summary (real weights, (1,4) P300x2, `tools/sweep_multichip.py`)

PCC = prefill T=128 / decode vs the HF layer; prefill ms = warmed wall T=128 (host bound, see README section 4); decode ms = traced, 3 replays (wall/3); delta vs the `mc_selected` row of the same layer (`baseline.jsonl`).

## family `baseline`

| candidate | layer | PCC prefill | PCC decode | prefill ms | decode ms | decode delta | note |
|---|---|---|---|---|---|---|---|
| `mc_selected` | GDN (L0) | 0.99992 | 0.99996 | 14.06 | 0.608 | +0 us | |
| `mc_selected` | attention (L3) | 0.99985 | 0.99993 | 14.29 | 0.482 | +0 us | |

## family `conv1d`

| candidate | layer | PCC prefill | PCC decode | prefill ms | decode ms | decode delta | note |
|---|---|---|---|---|---|---|---|
| `conv_blk1024` | GDN (L0) | 0.99992 | 0.99996 | 14.19 | 0.608 | +0 us | |
| `conv_blk1536` | GDN (L0) | 0.99992 | 0.99996 | 14.34 | 0.609 | +1 us | |
| `conv_blk3072` | GDN (L0) | 0.99992 | 0.99996 | 13.97 | 0.608 | +0 us | |

## family `dense_1d`

| candidate | layer | PCC prefill | PCC decode | prefill ms | decode ms | decode delta | note |
|---|---|---|---|---|---|---|---|
| `attn_qkvNone_woNone` | attention (L3) | 0.99985 | 0.99993 | 14.71 | 0.481 | -1 us | |
| `attn_qkv32_wo32` | attention (L3) | 0.99985 | 0.99993 | 14.41 | 0.474 | -8 us | |
| `attn_qkv64_wo64` | attention (L3) | 0.99985 | 0.99993 | 14.65 | 0.476 | -6 us | |
| `attn_qkv88_wo33` | attention (L3) | 0.99985 | 0.99993 | 14.23 | 0.473 | -9 us | |
| `attn_qkv44_wo44` | attention (L3) | 0.99985 | 0.99993 | 14.27 | 0.473 | -9 us | |

## family `down`

| candidate | layer | PCC prefill | PCC decode | prefill ms | decode ms | decode delta | note |
|---|---|---|---|---|---|---|---|
| `dn48_in8` | GDN (L0) | 0.99992 | 0.99996 | 13.95 | 0.608 | +0 us | |
| `dn96_in8` | GDN (L0) | – | – | – | – | – | FAILED: TT_FATAL @ /home/raahem/tt-metal/tt_metal/impl/program/program.cpp:153: not on_dispatch_core info: Illegal kernel placement for bmm_large_bl |
| `dn32_in8` | GDN (L0) | 0.99992 | 0.99996 | 14.00 | 0.613 | +5 us | |
| `dn24_in8` | GDN (L0) | 0.99992 | 0.99996 | 14.08 | 0.613 | +5 us | |
| `dn48_in8` | attention (L3) | 0.99985 | 0.99993 | 14.29 | 0.481 | -1 us | |
| `dn96_in8` | attention (L3) | – | – | – | – | – | FAILED: TT_FATAL @ /home/raahem/tt-metal/tt_metal/impl/program/program.cpp:153: not on_dispatch_core info: Illegal kernel placement for bmm_large_bl |
| `dn32_in8` | attention (L3) | 0.99985 | 0.99993 | 14.57 | 0.485 | +3 us | |
| `dn24_in8` | attention (L3) | 0.99985 | 0.99993 | 14.47 | 0.486 | +4 us | |

## family `gate_up`

| candidate | layer | PCC prefill | PCC decode | prefill ms | decode ms | decode delta | note |
|---|---|---|---|---|---|---|---|
| `gu16_in32` | GDN (L0) | 0.99992 | 0.99996 | 14.31 | 0.609 | +1 us | |
| `gu16_in96` | GDN (L0) | 0.99992 | 0.99996 | 14.58 | 0.603 | -5 us | |
| `gu8_in32` | GDN (L0) | 0.99992 | 0.99996 | 15.30 | 0.627 | +19 us | |
| `gu16_in8` | GDN (L0) | 0.99992 | 0.99996 | 13.92 | 0.653 | +45 us | |
| `gu16_in32` | attention (L3) | 0.99985 | 0.99993 | 14.48 | 0.482 | +0 us | |
| `gu16_in96` | attention (L3) | 0.99985 | 0.99993 | 14.31 | 0.476 | -6 us | |
| `gu8_in32` | attention (L3) | 0.99985 | 0.99993 | 14.49 | 0.501 | +19 us | |
| `gu16_in8` | attention (L3) | 0.99985 | 0.99993 | 14.61 | 0.526 | +44 us | |

## family `gdn_1d`

| candidate | layer | PCC prefill | PCC decode | prefill ms | decode ms | decode delta | note |
|---|---|---|---|---|---|---|---|
| `gdn_qkvzNone_outNone` | GDN (L0) | 0.99992 | 0.99996 | 14.43 | 0.607 | -1 us | |
| `gdn_qkvz32_out32` | GDN (L0) | 0.99992 | 0.99996 | 13.99 | 0.608 | +0 us | |
| `gdn_qkvz64_out64` | GDN (L0) | 0.99992 | 0.99996 | 13.83 | 0.607 | -1 us | |
| `gdn_qkvz88_out33` | GDN (L0) | 0.99992 | 0.99996 | 14.13 | 0.606 | -2 us | |
| `gdn_qkvz44_out44` | GDN (L0) | 0.99992 | 0.99996 | 13.86 | 0.608 | +0 us | |

## family `gdn_state`

| candidate | layer | PCC prefill | PCC decode | prefill ms | decode ms | decode delta | note |
|---|---|---|---|---|---|---|---|
| `state_bf16` | GDN (L0) | 0.99992 | 0.99996 | 13.98 | 0.614 | +6 us | |
| `state_fp32` | GDN (L0) | 0.99992 | 0.99996 | 13.82 | 0.623 | +15 us | |

## family `norm_mode`

| candidate | layer | PCC prefill | PCC decode | prefill ms | decode ms | decode delta | note |
|---|---|---|---|---|---|---|---|
| `norm_A` | GDN (L0) | 0.99992 | 0.99996 | 14.06 | 0.609 | +1 us | |
| `norm_B` | GDN (L0) | – | – | – | – | – | FAILED: TT_FATAL @ /home/raahem/tt-metal/ttnn/cpp/ttnn/operations/matmul/device/matmul_device_operation.cpp:76: a_shape[-1] == b_shape[-2] info: The |
| `norm_A` | attention (L3) | 0.99985 | 0.99993 | 14.33 | 0.482 | +0 us | |
| `norm_B` | attention (L3) | – | – | – | – | – | FAILED: TT_FATAL @ /home/raahem/tt-metal/ttnn/cpp/ttnn/operations/matmul/device/matmul_device_operation.cpp:76: a_shape[-1] == b_shape[-2] info: The |

## family `sdpa`

| candidate | layer | PCC prefill | PCC decode | prefill ms | decode ms | decode delta | note |
|---|---|---|---|---|---|---|---|
| `sdpa_0_0` | attention (L3) | 0.99985 | 0.99993 | 14.38 | 0.527 | +45 us | |
| `sdpa_32_256` | attention (L3) | 0.99985 | 0.99993 | 14.51 | 0.520 | +38 us | |

## family `shared`

| candidate | layer | PCC prefill | PCC decode | prefill ms | decode ms | decode delta | note |
|---|---|---|---|---|---|---|---|
| `sharedNone` | GDN (L0) | 0.99992 | 0.99996 | 14.52 | 0.609 | +1 us | |
| `shared8` | GDN (L0) | 0.99992 | 0.99996 | 14.02 | 0.589 | -19 us | |
| `shared16` | GDN (L0) | 0.99992 | 0.99996 | 14.08 | 0.582 | -26 us | |
| `shared17` | GDN (L0) | 0.99992 | 0.99996 | 14.40 | 0.585 | -23 us | |
| `sharedNone` | attention (L3) | 0.99985 | 0.99993 | 14.37 | 0.481 | -1 us | |
| `shared8` | attention (L3) | 0.99985 | 0.99993 | 14.26 | 0.461 | -21 us | |
| `shared16` | attention (L3) | 0.99985 | 0.99993 | 14.27 | 0.456 | -26 us | |
| `shared17` | attention (L3) | 0.99985 | 0.99993 | 14.23 | 0.458 | -24 us | |

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// GDN decode head step, reader. Unit n = b * Nv + h (user b, value head h; q / k head h / (Nv / Nk)):
//   * q, k, v: row b of the FP32 conv output ([1, 1, 32, qkv] TILE, L1) gathered into row 0 of zeroed tiles (two
//     64-byte face-row reads per tile);
//   * z: row b of the head's z columns of the in-proj output ([1, B, P] bf16 TILE, L1: 32-byte face rows);
//   * the gate inputs as column 0 of four FP32 tiles (every row): a, dt_bias, -exp(A_log), b (a / b are the in-proj's
//     columns a_tile * 32 + h / b_tile * 32 + h; dt_bias / -exp(A_log) from the per-die params tensor);
//   * the head's Kt x Vt state tiles ([Bmax, Nv, Dk, Dv] TILE, DRAM).
// Once per launch (after the first unit's state): the gated-norm weight row (params row 2) into Vt FP32 tiles, persistent
// (the compute pops it after the last unit).
// VSPLIT = P (one work item per core): item n = p * U + u (U = B * Nv units, runtime arg 6): unit u's value columns
// [p Vt, (p + 1) Vt) of its VT_FULL (Vt here is the slice): v / z / state / norm-weight slices; q, k and the gates whole.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "tools/profiler/kernel_profiler.hpp"

namespace {
constexpr uint32_t F32_FACE = 1024, F32_ROW = 64;  // fp32 face / face row bytes
constexpr uint32_t BF_FACE = 512, BF_ROW = 32;     // bf16
constexpr uint32_t F32_TILE = 4096, BF_TILE = 2048;

// byte offset of row r, column half c (0: cols 0-15, 1: 16-31) in a tile with the given face / face-row sizes
inline uint32_t row_off(uint32_t r, uint32_t c, uint32_t face, uint32_t row) {
    return ((r >> 4) * 2 + c) * face + (r & 15) * row;
}
// element (r, col) of an FP32 tile, as a word index
inline uint32_t f32_idx(uint32_t r, uint32_t col) {
    return (((r >> 4) * 2 + (col >> 4)) * 256) + (r & 15) * 16 + (col & 15);
}

template <typename Acc>
inline void gather_row(const Acc& src, uint32_t l1, uint32_t page0, uint32_t ntiles, uint32_t r, uint32_t face, uint32_t row,
                       uint32_t tile) {
    for (uint32_t i = 0; i < ntiles; ++i) {
        for (uint32_t c = 0; c < 2; ++c) {
            noc_async_read(src.get_noc_addr(page0 + i, row_off(r, c, face, row)), l1 + i * tile + c * face, row);
        }
    }
}

inline void zero(uint32_t l1, uint32_t bytes) {
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
    for (uint32_t o = 0; o < bytes; o += MEM_ZEROS_SIZE) {
        noc_async_read(zeros, l1 + o, MEM_ZEROS_SIZE);
    }
}

// column 0 of an FP32 tile (all 32 rows) <- v (other elements stay zero)
inline void fill_col0(uint32_t l1, uint32_t v) {
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1);
    for (uint32_t r = 0; r < 32; ++r) {
        p[f32_idx(r, 0)] = v;
    }
}
}  // namespace

void kernel_main() {
    const uint32_t conv_addr = get_arg_val<uint32_t>(0);
    const uint32_t proj_addr = get_arg_val<uint32_t>(1);
    const uint32_t params_addr = get_arg_val<uint32_t>(2);
    const uint32_t state_addr = get_arg_val<uint32_t>(3);
    const uint32_t start = get_arg_val<uint32_t>(4);
    const uint32_t count = get_arg_val<uint32_t>(5);

    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Vt = get_compile_time_arg_val(1);
    constexpr uint32_t Nv = get_compile_time_arg_val(2);
    constexpr uint32_t Nk = get_compile_time_arg_val(3);
    constexpr uint32_t conv_ct = get_compile_time_arg_val(4);  // conv_out tiles per row (qkv / 32)
    constexpr uint32_t proj_ct = get_compile_time_arg_val(5);  // in-proj output tiles per row
    constexpr uint32_t z_tile = get_compile_time_arg_val(6);   // first z tile of the in-proj output
    constexpr uint32_t a_tile = get_compile_time_arg_val(7);
    constexpr uint32_t b_tile = get_compile_time_arg_val(8);
    constexpr uint32_t cb_q = get_compile_time_arg_val(9);
    constexpr uint32_t cb_k = get_compile_time_arg_val(10);
    constexpr uint32_t cb_v = get_compile_time_arg_val(11);
    constexpr uint32_t cb_gin = get_compile_time_arg_val(12);
    constexpr uint32_t cb_S = get_compile_time_arg_val(13);
    constexpr uint32_t cb_z = get_compile_time_arg_val(14);
    constexpr uint32_t cb_w = get_compile_time_arg_val(15);
    constexpr uint32_t cb_scr = get_compile_time_arg_val(16);
    constexpr auto conv_args = TensorAccessorArgs<17>();
    constexpr auto proj_args = TensorAccessorArgs<conv_args.next_compile_time_args_offset()>();
    constexpr auto params_args = TensorAccessorArgs<proj_args.next_compile_time_args_offset()>();
    constexpr auto state_args = TensorAccessorArgs<params_args.next_compile_time_args_offset()>();
    constexpr uint32_t kv = Kt * Vt;
    constexpr uint32_t GQA = Nv / Nk;
    if (count == 0) {
        return;
    }
#ifdef VSPLIT
    constexpr uint32_t VTF = VT_FULL;  // the head's value tiles; this core holds Vt of them
    const uint32_t U = get_arg_val<uint32_t>(6);
    auto unit_of = [&](uint32_t n) { return n % U; };
    auto part_of = [&](uint32_t n) { return n / U; };
#else
    constexpr uint32_t VTF = Vt;
    auto unit_of = [&](uint32_t n) { return n; };
    auto part_of = [&](uint32_t n) { return 0 * n; };
#endif

    const auto s_conv = TensorAccessor(conv_args, conv_addr, F32_TILE);
    const auto s_proj = TensorAccessor(proj_args, proj_addr, BF_TILE);
    const auto s_par = TensorAccessor(params_args, params_addr, F32_TILE);
    const auto s_state = TensorAccessor(state_args, state_addr, BF_TILE);

    static_assert(Nv <= 32, "the per-head params sit in params tile 0");
    const uint32_t scr = get_write_ptr(cb_scr);
    volatile tt_l1_ptr uint32_t* par = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr);
    // the gate inputs of a unit as column 0 of four FP32 tiles (a / b: the bf16 face rows holding column h of the a / b
    // tiles, row b, read to scr + 4 tiles; dt_bias / -exp(A_log) from params tile 0 at scr)
    auto push_gates = [&](uint32_t b, uint32_t h) {
        const uint32_t ab_l1 = scr + 4 * F32_TILE;
        noc_async_read(s_proj.get_noc_addr(a_tile, row_off(b, (h & 31) >> 4, BF_FACE, BF_ROW)), ab_l1, BF_ROW);
        noc_async_read(s_proj.get_noc_addr(b_tile, row_off(b, (h & 31) >> 4, BF_FACE, BF_ROW)), ab_l1 + 64, BF_ROW);
        cb_reserve_back(cb_gin, 4);
        noc_async_read_barrier();
        // scr is rewritten by the NOC for every unit and read here by the RISC: drop the cached lines (Blackhole L1
        // cache), or a unit can read an earlier unit's a / b
        invalidate_l1_cache();
        volatile tt_l1_ptr uint16_t* ab = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(ab_l1);
        const uint32_t a_bits = static_cast<uint32_t>(ab[h & 15]) << 16;
        const uint32_t b_bits = static_cast<uint32_t>(ab[32 + (h & 15)]) << 16;
        const uint32_t g = get_write_ptr(cb_gin);
        fill_col0(g, a_bits);
        fill_col0(g + F32_TILE, par[f32_idx(1, h)]);
        fill_col0(g + 2 * F32_TILE, par[f32_idx(0, h)]);
        fill_col0(g + 3 * F32_TILE, b_bits);
        cb_push_back(cb_gin, 4);
    };
    // params: [1, 1, 32, 128] FP32 per die (row 0: -exp(A_log) per head, row 1: dt_bias, row 2: gated-norm weight).
    // Tile 0 and the first unit's gate inputs go first (the compute starts on them), then the zeroing and the weight.
    {
        DeviceZoneScopedN("rd_gates0");
        noc_async_read(s_par.get_noc_addr(0), scr, F32_TILE);
        push_gates(unit_of(start) / Nv, unit_of(start) % Nv);
    }
    // k and v must be zero outside row 0 (the rank-1 update k^T (x) u sums over all rows; 0 * garbage could be NaN). The
    // other gathered tiles only reach rows / elements the compute never reads back. Zero the first slot of the two
    // (double-buffered) here, the second one after the first unit is handed over.
    zero(get_write_ptr(cb_k), Kt * F32_TILE);
    noc_async_read_barrier();
    bool v_zeroed = false;

    for (uint32_t w = 0; w < count; ++w) {
        const uint32_t n = unit_of(start + w);  // the unit (user, value head)
        const uint32_t p = part_of(start + w);  // its value-column slice
        const uint32_t b = n / Nv;
        const uint32_t h = n - b * Nv;
        const uint32_t hk = h / GQA;
        if (w > 0) {
            push_gates(b, h);
        }

        DeviceZoneScopedN("rd_unit");
        // q / k first (the norms start on them), then v / z (v's first slot zeroed on the way)
        cb_reserve_back(cb_q, Kt);
        cb_reserve_back(cb_k, Kt);
        gather_row(s_conv, get_write_ptr(cb_q), hk * Kt, Kt, b, F32_FACE, F32_ROW, F32_TILE);
        gather_row(s_conv, get_write_ptr(cb_k), (Nk + hk) * Kt, Kt, b, F32_FACE, F32_ROW, F32_TILE);
        if (!v_zeroed) {
            zero(get_write_ptr(cb_v), Vt * F32_TILE);
        }
        noc_async_read_barrier();
        cb_push_back(cb_q, Kt);
        cb_push_back(cb_k, Kt);
        cb_reserve_back(cb_v, Vt);
        cb_reserve_back(cb_z, Vt);
        gather_row(s_conv, get_write_ptr(cb_v), 2 * Nk * Kt + h * VTF + p * Vt, Vt, b, F32_FACE, F32_ROW, F32_TILE);
        gather_row(s_proj, get_write_ptr(cb_z), z_tile + h * VTF + p * Vt, Vt, b, BF_FACE, BF_ROW, BF_TILE);
        noc_async_read_barrier();
        cb_push_back(cb_v, Vt);
        cb_push_back(cb_z, Vt);
        if (w == 0) {
            zero(get_write_ptr(cb_k), Kt * F32_TILE);  // the second slots (the S read barrier below covers them)
            zero(get_write_ptr(cb_v), Vt * F32_TILE);
            v_zeroed = true;
        }

        cb_reserve_back(cb_S, kv);
        const uint32_t s = get_write_ptr(cb_S);
        for (uint32_t ki = 0; ki < Kt; ++ki) {
            for (uint32_t j = 0; j < Vt; ++j) {
                const uint32_t src = n * (Kt * VTF) + ki * VTF + p * Vt + j;
                noc_async_read(s_state.get_noc_addr(src), s + (ki * Vt + j) * BF_TILE, BF_TILE);
            }
        }
        if (w == 0) {
            // the gated-norm weight (params row 2), read once: the compute needs it last
            for (uint32_t t = 1; t < 4; ++t) {
                noc_async_read(s_par.get_noc_addr(t), scr + t * F32_TILE, F32_TILE);
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_S, kv);
        if (w == 0) {
            invalidate_l1_cache();
            cb_reserve_back(cb_w, Vt);
            volatile tt_l1_ptr uint32_t* wt = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_w));
            for (uint32_t j = 0; j < Vt; ++j) {
                for (uint32_t col = 0; col < 32; ++col) {
                    wt[j * 1024 + f32_idx(0, col)] = par[(p * Vt + j) * 1024 + f32_idx(2, col)];
                }
            }
            cb_push_back(cb_w, Vt);
        }
    }
}

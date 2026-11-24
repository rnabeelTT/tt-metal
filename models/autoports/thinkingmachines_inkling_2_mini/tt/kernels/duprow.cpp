// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Row 0 -> row 1 of every bf16 tile of an interleaved [1, 1, 1, H] tensor, in place (tt/moe_fused.py: one decode row
// runs the fused MoE's 2-row path with the row twice; the padding row 1 of each tile becomes the copy). A tile row r < 16
// is 16 bf16 in face 0 at byte 32 r and 16 in face 1 at byte 512 + 32 r.
//
// runtime args: addr, first tile, tile count

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t addr = get_arg_val<uint32_t>(0);
    const uint32_t t0 = get_arg_val<uint32_t>(1);
    const uint32_t nt = get_arg_val<uint32_t>(2);
    constexpr uint32_t cb = get_compile_time_arg_val(0);
    constexpr uint32_t TB = 2048;
    constexpr auto args = TensorAccessorArgs<1>();
    const auto acc = TensorAccessor(args, addr, TB);
    const uint32_t l1 = get_write_ptr(cb);
    for (uint32_t t = 0; t < nt; ++t) {
        noc_async_read(acc.get_noc_addr(t0 + t), l1 + t * TB, TB);
    }
    noc_async_read_barrier();
    for (uint32_t t = 0; t < nt; ++t) {
        volatile uint32_t* w = reinterpret_cast<volatile uint32_t*>(l1 + t * TB);
        for (uint32_t f = 0; f < 2; ++f) {  // faces 0, 1: row 0 is 8 words at word 128 f, row 1 the next 8
            for (uint32_t i = 0; i < 8; ++i) {
                w[128 * f + 8 + i] = w[128 * f + i];
            }
        }
        noc_async_write(l1 + t * TB, acc.get_noc_addr(t0 + t), TB);
    }
    noc_async_write_barrier();
}

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// fold reader (RISCV_1): stream i = core's index: the HT fp32 tiles of kq row i (pages i HT + t), in order, BLK tiles
// per CB push. ct: [HT, BLK, acc(kq)]; rt: [kq addr, i]
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t HT = get_compile_time_arg_val(0);
    constexpr uint32_t BLK = get_compile_time_arg_val(1);
    constexpr auto a = TensorAccessorArgs<2>();
    constexpr uint32_t F32 = 4096;
    const uint32_t addr = get_arg_val<uint32_t>(0), i = get_arg_val<uint32_t>(1);
    const auto s = TensorAccessor(a, addr, F32);
    for (uint32_t t0 = 0; t0 < HT; t0 += BLK) {
        cb_reserve_back(0, BLK);
        uint32_t w = get_write_ptr(0);
        for (uint32_t t = t0; t < t0 + BLK; ++t) {
            noc_async_read(s.get_noc_addr(i * HT + t), w, F32);
            w += F32;
        }
        noc_async_read_barrier();
        cb_push_back(0, BLK);
    }
}

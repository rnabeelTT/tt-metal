// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Writes this core's NOC coordinates (my_x, my_y on the kernel's NOC) into a 64-byte slot of a DRAM buffer, slot =
// logical y * grid_x + logical x (runtime arg). moe_decode/experts.py noc_coords reads the table on the host (the
// ttnn build has no logical -> NOC core binding).
//
// runtime args: [out_addr, slot]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t slot = get_arg_val<uint32_t>(1);
    constexpr uint32_t cb = get_compile_time_arg_val(0);
    constexpr auto out_args = TensorAccessorArgs<1>();
    const auto s_out = TensorAccessor(out_args, out_addr, 64);
    cb_reserve_back(cb, 1);
    volatile tt_l1_ptr uint32_t* v = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb));
    v[0] = my_x[noc_index];
    v[1] = my_y[noc_index];
    v[2] = 1;
    noc_async_write(get_write_ptr(cb), s_out.get_noc_addr(slot), 64);
    noc_async_write_barrier();
}

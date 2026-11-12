// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Multi-row paged KV update, compute: for each of the 2 cache tile rows, untilize the Wt cache tiles (the writer
// patches the rows in place), then tilize the patched block back (the stock paged_fused_update_cache pair of steps).

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/tilize.h"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"

void kernel_main() {
    constexpr uint32_t cb_cache = get_compile_time_arg_val(0);
    constexpr uint32_t cb_ucache = get_compile_time_arg_val(1);
    constexpr uint32_t cb_ucache2 = get_compile_time_arg_val(2);
    constexpr uint32_t cb_out = get_compile_time_arg_val(3);
    constexpr uint32_t Wt = get_compile_time_arg_val(4);

    compute_kernel_hw_startup(cb_cache, cb_ucache);
    for (uint32_t g = 0; g < 2; ++g) {
        compute_kernel_lib::untilize<Wt, cb_cache, cb_ucache>(1);
        compute_kernel_lib::tilize<Wt, cb_ucache2, cb_out>(1);
    }
}

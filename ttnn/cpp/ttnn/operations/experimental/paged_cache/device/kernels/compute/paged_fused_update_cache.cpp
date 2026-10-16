// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/tilize.h"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "experimental/kernel_args.h"

void kernel_main() {
    const auto has_work = get_arg(args::has_work);
    if (!has_work) {
        return;
    }

    constexpr auto Wt = get_arg(args::Wt);
    constexpr auto num_heads = get_arg(args::num_heads);

    // Which of the two inputs this instance serves is structural: the factory places one compute
    // kernel per input shard grid, and each binds only its own input buffer.
    DataflowBuffer in_dfb(dfb::in);
    DataflowBuffer untilized_in_dfb(dfb::untilized_in);
    DataflowBuffer cache_dfb(dfb::cache);
    DataflowBuffer untilized_cache_dfb(dfb::untilized_cache);
    DataflowBuffer untilized_cache2_dfb(dfb::untilized_cache2);
    DataflowBuffer out_dfb(dfb::out);

    compute_kernel_hw_startup(dfb::in, dfb::untilized_in);

    // Untilize input (single block, init only - no uninit needed)
    compute_kernel_lib::untilize<
        Wt,
        dfb::in,
        dfb::untilized_in,
        compute_kernel_lib::untilize_config::InitUninitMode::InitOnly,
        compute_kernel_lib::untilize_config::WaitMode::WaitBlock,
        compute_kernel_lib::untilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(
        in_dfb, untilized_in_dfb, 1);

    for (uint32_t cur_head = 0; cur_head < num_heads; ++cur_head) {
        // Untilize a block from the cache with reconfiguration from previous iteration
        compute_kernel_lib::untilize<Wt, dfb::cache, dfb::untilized_cache>(cache_dfb, untilized_cache_dfb, 1);

        // Wait on writer to update block. Tilize with reconfiguration
#ifdef ARCH_QUASAR
        // Quasar: tilize_init programs unpack+math only and pack_reconfig_data_format is gasket-only, so
        // the packer's L1 destination (BFD) still points at dfb::untilized_cache from the untilize above
        // -> the re-tilized block would land in the untilize ring and dfb::out would never be written
        // (all-zero cache block). Retarget the packer BFD before packing. Same fix as update_cache.cpp.
        pack_init(dfb::out);
#endif
        compute_kernel_lib::tilize<Wt, dfb::untilized_cache2, dfb::out>(untilized_cache2_dfb, out_dfb, 1);
    }
}

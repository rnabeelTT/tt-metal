// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Indexer pooling (exact): compute helpers (as tt/engram_exact ex_common.hpp).
#pragma once
#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"

namespace ex {
inline void copy_to(uint32_t cb, uint32_t t, uint32_t d) {
    reconfig_data_format_srca(cb);
    copy_init(cb);
    copy_tile(cb, t, d);
}
inline void bcast_row_to(uint32_t cb, uint32_t t, uint32_t d) {  // row 0 of tile t to every row of DST d
    reconfig_data_format(cb, cb);
    unary_bcast_init<BroadcastType::ROW>(cb);
    unary_bcast<BroadcastType::ROW>(cb, t, d);
}
inline void pack_to(uint32_t d, uint32_t cb) {
    cb_reserve_back(cb, 1);
    pack_reconfig_data_format(cb);
    pack_tile(d, cb);
    cb_push_back(cb, 1);
}
inline void sadd(uint32_t a, uint32_t b, uint32_t o) {
    add_binary_tile_init();
    add_binary_tile(a, b, o);
}
inline void smul(uint32_t a, uint32_t b, uint32_t o) {
    mul_binary_tile_init();
    mul_binary_tile(a, b, o);
}
inline void to_bf16(uint32_t d) {
    typecast_tile_init<(uint32_t)DataFormat::Float32, (uint32_t)DataFormat::Float16_b>();
    typecast_tile<(uint32_t)DataFormat::Float32, (uint32_t)DataFormat::Float16_b>(d);
}
}  // namespace ex

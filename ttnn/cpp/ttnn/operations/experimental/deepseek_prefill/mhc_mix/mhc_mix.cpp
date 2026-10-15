// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "mhc_mix.hpp"

#include "device/mhc_mix_device_operation.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::mhc_mix {

ttnn::Tensor mhc_mix(const ttnn::Tensor& x, const ttnn::Tensor& fn_t, uint32_t num_chunks) {
    return ttnn::prim::mhc_mix(x, fn_t, num_chunks);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::mhc_mix

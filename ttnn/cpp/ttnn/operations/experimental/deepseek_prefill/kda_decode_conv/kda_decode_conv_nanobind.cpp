// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "kda_decode_conv_nanobind.hpp"

#include <nanobind/nanobind.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "kda_decode_conv.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::kda_decode_conv::detail {

void bind_experimental_kda_decode_conv_operation(nb::module_& mod) {
    ttnn::bind_function<"kda_decode_conv", "ttnn.experimental.deepseek_prefill.">(
        mod,
        R"doc(
            One decode step of the KDA short causal convolution (4 taps) with SiLU, for up to 32 slots, in one op:

                acc = cur * tap3 + h0 * tap0 + h1 * tap1 + h2 * tap2      (fp32)
                qkv = active ? silu(acc) : 0                             (select: an idle row cannot leak NaN)
                h0, h1, h2 <- active ? (h1, h2, cur) : (h0, h1, h2)      (in place)

            with cur = proj[:, :width]. Replaces a slice, a multiply, three typecast + addcmul pairs, a SiLU and four
            selects of the unfused step.

            Args:
                * :attr:`proj`: [1,1,B,P] BFLOAT16 TILE (P >= width); its first ``width`` columns are the current row.
                * :attr:`h0`, :attr:`h1`, :attr:`h2`: [1,1,B,width] BFLOAT16 TILE conv history (oldest first),
                  updated in place.
                * :attr:`tap0` .. :attr:`tap3`: [1,1,1,width] FLOAT32 TILE taps (tap3 multiplies the current row).
                * :attr:`active`: [1,1,B,1] FLOAT32 TILE slot mask (1 active, 0 idle).
                * :attr:`width`: conv width (multiple of 32).

            B (the logical rows) must fit one tile row (B <= 32). All tensors DRAM interleaved.

            Returns:
                qkv [1,1,B,width] FLOAT32 TILE, DRAM.
        )doc",
        &kda_decode_conv,
        nb::arg("proj").noconvert(),
        nb::arg("h0").noconvert(),
        nb::arg("h1").noconvert(),
        nb::arg("h2").noconvert(),
        nb::arg("tap0").noconvert(),
        nb::arg("tap1").noconvert(),
        nb::arg("tap2").noconvert(),
        nb::arg("tap3").noconvert(),
        nb::arg("active").noconvert(),
        nb::arg("width"));
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::kda_decode_conv::detail

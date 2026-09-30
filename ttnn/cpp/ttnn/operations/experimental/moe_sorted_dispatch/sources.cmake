# Source files for ttnn_op_experimental_moe_sorted_dispatch.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_EXPERIMENTAL_MOE_SORTED_DISPATCH_API_HEADERS moe_sorted_dispatch.hpp)

set(TTNN_OP_EXPERIMENTAL_MOE_SORTED_DISPATCH_SRCS
    device/moe_sorted_dispatch_device_operation.cpp
    device/moe_sorted_dispatch_program_factory.cpp
    moe_sorted_dispatch.cpp
)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/experimental/moe_sorted_dispatch/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
# Listed here rather than inline in CMakeLists.txt so that
# add/remove/rename doesn't touch a file with metalium-developers-infra
# as a required co-owner.
set(TTNN_OP_EXPERIMENTAL_MOE_SORTED_DISPATCH_NANOBIND_SRCS moe_sorted_dispatch_nanobind.cpp)

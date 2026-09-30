# Source files for ttnn_op_experimental_moe_gather_add_rows.
set(TTNN_OP_EXPERIMENTAL_MOE_GATHER_ADD_ROWS_API_HEADERS moe_gather_add_rows.hpp)

set(TTNN_OP_EXPERIMENTAL_MOE_GATHER_ADD_ROWS_SRCS
    device/moe_gather_add_rows_device_operation.cpp
    device/moe_gather_add_rows_program_factory.cpp
    moe_gather_add_rows.cpp
)

set(TTNN_OP_EXPERIMENTAL_MOE_GATHER_ADD_ROWS_NANOBIND_SRCS moe_gather_add_rows_nanobind.cpp)

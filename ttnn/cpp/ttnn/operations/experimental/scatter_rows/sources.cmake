# Source files for ttnn_op_experimental_scatter_rows.
set(TTNN_OP_EXPERIMENTAL_SCATTER_ROWS_API_HEADERS scatter_rows.hpp)

set(TTNN_OP_EXPERIMENTAL_SCATTER_ROWS_SRCS
    device/scatter_rows_device_operation.cpp
    device/scatter_rows_program_factory.cpp
    scatter_rows.cpp
)

set(TTNN_OP_EXPERIMENTAL_SCATTER_ROWS_NANOBIND_SRCS scatter_rows_nanobind.cpp)

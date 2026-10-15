set(TTNN_OP_EXPERIMENTAL_KDA_RECURRENT_DECODE_API_HEADERS recurrent_decode.hpp)

set(TTNN_OP_EXPERIMENTAL_KDA_RECURRENT_DECODE_SRCS
    recurrent_decode.cpp
    device/recurrent_decode_device_operation.cpp
    device/recurrent_decode_program_factory.cpp
)

set(TTNN_OP_EXPERIMENTAL_KDA_RECURRENT_DECODE_NANOBIND_SRCS recurrent_decode_nanobind.cpp)

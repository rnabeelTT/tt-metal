set(TTNN_OP_EXPERIMENTAL_DSA_DECODE_SCORES_API_HEADERS dsa_decode_scores.hpp)

set(TTNN_OP_EXPERIMENTAL_DSA_DECODE_SCORES_SRCS
    dsa_decode_scores.cpp
    device/dsa_decode_scores_device_operation.cpp
    device/dsa_decode_scores_program_factory.cpp
)

set(TTNN_OP_EXPERIMENTAL_DSA_DECODE_SCORES_NANOBIND_SRCS dsa_decode_scores_nanobind.cpp)

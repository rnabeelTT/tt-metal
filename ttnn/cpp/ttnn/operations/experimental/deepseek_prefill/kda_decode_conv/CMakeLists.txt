include(sources.cmake)

add_library(ttnn_op_experimental_deepseek_prefill_kda_decode_conv ${LIB_TYPE})
add_library(
    TTNN::Ops::Experimental::DeepSeekPrefill::KdaDecodeConv
    ALIAS ttnn_op_experimental_deepseek_prefill_kda_decode_conv
)

target_precompile_headers(ttnn_op_experimental_deepseek_prefill_kda_decode_conv REUSE_FROM TTNN::PCHFull)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_deepseek_prefill_kda_decode_conv)

set_target_properties(
    ttnn_op_experimental_deepseek_prefill_kda_decode_conv
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

file(GLOB_RECURSE kernels device/kernels/*)

target_sources(
    ttnn_op_experimental_deepseek_prefill_kda_decode_conv
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES ${TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_KDA_DECODE_CONV_API_HEADERS}
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
    PRIVATE
        ${TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_KDA_DECODE_CONV_SRCS}
)

target_link_libraries(ttnn_op_experimental_deepseek_prefill_kda_decode_conv PRIVATE TT::Metalium PUBLIC TTNN::Core)

install(
    TARGETS
        ttnn_op_experimental_deepseek_prefill_kda_decode_conv
    FILE_SET
    kernels
        DESTINATION
            ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/kda_decode_conv
        COMPONENT ttnn-runtime
)

install(
    TARGETS
        ttnn_op_experimental_deepseek_prefill_kda_decode_conv
    FILE_SET
    api
        COMPONENT ttnn-dev
    LIBRARY
        COMPONENT tar
)

if(TARGET ttnn)
    target_sources(ttnn PRIVATE ${TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_KDA_DECODE_CONV_NANOBIND_SRCS})
endif()

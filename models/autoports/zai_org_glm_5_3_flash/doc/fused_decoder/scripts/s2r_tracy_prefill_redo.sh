#!/usr/bin/env bash
# stage-2 review #1: re-capture the six fused prefill Tracy cases (their raw ops.csv of the s2r_tracy_dsa.sh run was lost)
source ~/glm53-flash-bringup/env.sh
export CASES='test_perf_prefill[dsa_moe-128] test_perf_prefill[dsa_moe-2048] test_perf_prefill[kda_dense-128] test_perf_prefill[kda_dense-2048] test_perf_prefill[kda_moe-128] test_perf_prefill[kda_moe-2048]'
GLM53_DECODER=fused TRACY_OUT=$TT_METAL_HOME/$MODEL_DIR/doc/fused_decoder/tracy bash ~/glm53-flash-bringup/scripts/01-tracy.sh
echo ALLDONE

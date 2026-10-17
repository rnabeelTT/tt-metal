#!/usr/bin/env bash
# refreshed stage-1 (functional, paged DSA) and stage-2 (fused) Tracy windows, all 12 perf cases each
source ~/glm53-flash-bringup/env.sh
export CASES='test_perf_prefill[kda_dense-128] test_perf_prefill[kda_dense-2048] test_perf_prefill[dsa_moe-128] test_perf_prefill[dsa_moe-2048] test_perf_prefill[kda_moe-128] test_perf_prefill[kda_moe-2048] test_perf_traced_decode[kda_dense-1] test_perf_traced_decode[kda_dense-32] test_perf_traced_decode[dsa_moe-1] test_perf_traced_decode[dsa_moe-32] test_perf_traced_decode[kda_moe-1] test_perf_traced_decode[kda_moe-32]'
D=$MODEL_DIR/doc
GLM53_DECODER=functional TRACY_OUT=$D/functional_decoder/tracy bash ~/glm53-flash-bringup/scripts/01-tracy.sh
GLM53_DECODER=fused TRACY_OUT=$D/fused_decoder/tracy bash ~/glm53-flash-bringup/scripts/01-tracy.sh
echo ALLDONE

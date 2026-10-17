#!/usr/bin/env bash
# stage-2 review #1 (P2): re-capture the fused Tracy cases on the final code (all 12: the new flags touch DSA, the dense
# MLP and the routed experts), and the four functional DSA cases in the same session (the functional KDA cases are
# unchanged by this work)
source ~/glm53-flash-bringup/env.sh
D=$TT_METAL_HOME/$MODEL_DIR/doc
DSA='test_perf_prefill[dsa_moe-128] test_perf_prefill[dsa_moe-2048] test_perf_traced_decode[dsa_moe-1] test_perf_traced_decode[dsa_moe-32]'
KDA='test_perf_prefill[kda_dense-128] test_perf_prefill[kda_dense-2048] test_perf_prefill[kda_moe-128] test_perf_prefill[kda_moe-2048] test_perf_traced_decode[kda_dense-1] test_perf_traced_decode[kda_dense-32] test_perf_traced_decode[kda_moe-1] test_perf_traced_decode[kda_moe-32]'
CASES="$DSA $KDA" GLM53_DECODER=fused TRACY_OUT=$D/fused_decoder/tracy bash ~/glm53-flash-bringup/scripts/01-tracy.sh
CASES="$DSA" GLM53_DECODER=functional TRACY_OUT=$D/functional_decoder/tracy bash ~/glm53-flash-bringup/scripts/01-tracy.sh
echo ALLDONE

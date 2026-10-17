#!/usr/bin/env bash
# stage-2 review #2: fused MoE prefill-2048 Tracy without the uncommitted stage-6 MoE row split (GLM53_MOE_SORTED_ROWS=0
# = the committed stage-2 MoE path); the s2r2_chain.sh captures ran with the split (default 1024 in the worktree)
source ~/glm53-flash-bringup/env.sh
export CASES='test_perf_prefill[kda_moe-2048] test_perf_prefill[dsa_moe-2048]'
GLM53_MOE_SORTED_ROWS=0 GLM53_DECODER=fused TRACY_OUT=$TT_METAL_HOME/$MODEL_DIR/doc/fused_decoder/tracy_nosplit \
  bash ~/glm53-flash-bringup/scripts/01-tracy.sh
echo ALLDONE

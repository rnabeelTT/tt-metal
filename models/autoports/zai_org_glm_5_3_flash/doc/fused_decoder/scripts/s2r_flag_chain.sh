#!/usr/bin/env bash
# stage-2 review #1: layer-level bit-exactness of the new fused flags (tools/fused_flag_check.py; L0 kda_dense, L3
# dsa_moe, L4 kda_moe), then the Tracy re-capture with the flags on (s2r_tracy_dsa.sh)
source ~/glm53-flash-bringup/env.sh; cd $TT_METAL_HOME
F=dsa_heads_op,mlp_act_fold,kda_decode_fold
L=~/glm53-flash-bringup/logs/s2r_flag_check.log
~/glm53-flash-bringup/scripts/run-locked.sh $L timeout 3600 python_env/bin/python -m models.autoports.zai_org_glm_5_3_flash.tools.fused_flag_check $F 0 3 4
n=$(grep -c "bit-exact True" $L)
if [ "$n" != 6 ]; then echo "flag check: $n / 6 bit-exact; no Tracy"; exit 1; fi
GLM53_FUSED_ON=$F bash ~/glm53-flash-bringup/scripts/s2r_tracy_dsa.sh > ~/glm53-flash-bringup/logs/s2r_tracy_dsa.log 2>&1

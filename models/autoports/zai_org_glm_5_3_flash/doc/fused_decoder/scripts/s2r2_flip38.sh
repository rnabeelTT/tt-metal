#!/bin/bash
# stage-2 review #2 P2-c: fused-path router set at L3 slot 2 pos 37-39 (suite5 near-tie token pos 38, delta PCC 0.98296)
source ~/glm53-flash-bringup/env.sh; cd $TT_METAL_HOME
L=~/glm53-flash-bringup/logs; R=~/glm53-flash-bringup/scripts/run-locked.sh
GLM53_MOE_DEBUG=1 $R $L/flip_tt_s1302_fused.log timeout 2400 python_env/bin/python models/autoports/zai_org_glm_5_3_flash/tools/route_flip_check.py --layer 3 --seed 1302 --prefill 37 --steps 3 --slot 2 --fused

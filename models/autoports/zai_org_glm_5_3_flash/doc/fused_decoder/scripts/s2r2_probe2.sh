#!/usr/bin/env bash
# stage-2 review #2 P1-d adapted retry: KDA prefill qkv projection with untilize_out (tools/fusing_probe.py kdaprefill)
source ~/glm53-flash-bringup/env.sh; cd $TT_METAL_HOME
~/glm53-flash-bringup/scripts/run-locked.sh ~/glm53-flash-bringup/logs/s2r2_probe2.log \
  timeout 900 python_env/bin/python -m models.autoports.zai_org_glm_5_3_flash.tools.fusing_probe kdaprefill

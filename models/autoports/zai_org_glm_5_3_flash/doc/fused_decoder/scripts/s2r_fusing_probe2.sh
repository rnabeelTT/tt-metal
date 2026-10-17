#!/usr/bin/env bash
# stage-2 review #1: graph-fusing probe, second pass (grouped concat, prefill 64/128 rows, mlp / moe / kda sections)
source ~/glm53-flash-bringup/env.sh; cd $TT_METAL_HOME
~/glm53-flash-bringup/scripts/run-locked.sh ~/glm53-flash-bringup/logs/s2r_fusing_probe2.log \
  timeout 1800 python_env/bin/python -m models.autoports.zai_org_glm_5_3_flash.tools.fusing_probe heads2 mlp moe kdanorm kdaconv

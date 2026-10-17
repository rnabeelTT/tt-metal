#!/usr/bin/env bash
# stage-2 review #2 P1 a-e: fusing probe (finalize, relayouts, sharded norm, KDA prefill slice + untilize)
source ~/glm53-flash-bringup/env.sh; cd $TT_METAL_HOME
~/glm53-flash-bringup/scripts/run-locked.sh ~/glm53-flash-bringup/logs/s2r2_probe.log \
  timeout 1800 python_env/bin/python -m models.autoports.zai_org_glm_5_3_flash.tools.fusing_probe finalize relayout

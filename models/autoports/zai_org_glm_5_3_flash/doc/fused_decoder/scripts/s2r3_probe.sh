#!/usr/bin/env bash
# stage-2 review #3 P2-2: fast_reduce_nc chunk sum and the narrower KDA prefill matmul at bf16 / HiFi4 (tools/fusing_probe.py)
source ~/glm53-flash-bringup/env.sh; cd $TT_METAL_HOME
~/glm53-flash-bringup/scripts/run-locked.sh ~/glm53-flash-bringup/logs/s2r3_probe2.log \
  timeout -k 60 1800 python_env/bin/python -m models.autoports.zai_org_glm_5_3_flash.tools.fusing_probe chunksum kdaprefill2

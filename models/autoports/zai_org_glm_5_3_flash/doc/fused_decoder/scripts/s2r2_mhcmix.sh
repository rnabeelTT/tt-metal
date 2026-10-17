#!/usr/bin/env bash
# stage-2 review #2 P2-d: mhc_mix unit tests incl. T=1 / T=8 with non-finite padding rows through _mixes_split_k
source ~/glm53-flash-bringup/env.sh; cd $TT_METAL_HOME
~/glm53-flash-bringup/scripts/run-locked.sh ~/glm53-flash-bringup/logs/s2r2_mhc_mix.log \
  timeout 1800 python_env/bin/pytest -sv -p no:cacheprovider $MODEL_DIR/tests/test_mhc_mix.py

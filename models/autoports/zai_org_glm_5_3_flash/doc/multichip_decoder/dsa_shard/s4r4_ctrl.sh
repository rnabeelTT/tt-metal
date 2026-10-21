#!/bin/bash
# Stage-4 review #4 P3-3: replicated-fp8 control of test_batch32_all_active[dsa_moe] (no watcher), pure git archive HEAD
source ~/glm53-flash-bringup/env.sh >/dev/null
SNAP=$HOME/glm53-flash-bringup/s4r4_snap; MD=models/autoports/zai_org_glm_5_3_flash
cd $SNAP; echo "snapshot $(cat BASE_COMMIT) (pure git archive HEAD)"; md5sum -c MD5SUMS | sed 's/^/md5check /'; sed 's/^/md5 /' MD5SUMS
export PYTHONPATH=$SNAP:$PYTHONPATH GLM53_TP=4 GLM53_CACHE_DIR=$HOME/.cache/glm53_flash
unset GLM53_LAYERS GLM53_MC GLM53_DSA_SHARD GLM53_MAX_BATCH GLM53_MAX_SEQ_LEN GLM53_DRAM_RESERVE_MIB GLM53_MOE_SORTED_ROWS
export GLM53_PRECISION=$SNAP/$MD/doc/datatype_sweep/selected_precision_config.json
python -c "import models.autoports.zai_org_glm_5_3_flash.tt.dsa as m; print('imports', m.__file__)"
exec python -m pytest -sv -p no:cacheprovider "$MD/tests/test_multichip_decoder.py::test_batch32_all_active[dsa_moe]"

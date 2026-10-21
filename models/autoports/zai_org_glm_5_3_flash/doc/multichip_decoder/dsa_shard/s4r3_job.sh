#!/bin/bash
# Stage-4 review #3 P2-2: ONE device ticket on a pure `git archive HEAD` snapshot (s4r3_snap, BASE_COMMIT 51997a179ba).
# (a) fp8 watcher test_dsa_shard[b1]; (b) bf16 (GLM53_PRECISION=none, module default) test_dsa_shard; (c) committed
# sharded row check test_model_probe (GLM53_DSA_SHARD=1) at 8 and 1 slots; (d) optional fp8 sharded watcher
# test_batch32_all_active[dsa_moe]. tt-smi -r between steps (watcher: one pytest process per test).
source ~/glm53-flash-bringup/env.sh >/dev/null
SNAP=$HOME/glm53-flash-bringup/s4r3_snap; MD=models/autoports/zai_org_glm_5_3_flash; L=$HOME/glm53-flash-bringup/logs/s4r3
F=$MD/tests; PREC=$SNAP/$MD/doc/datatype_sweep/selected_precision_config.json
RESET() { env -u PYTHONPATH -u VIRTUAL_ENV /home/raahem/.tenstorrent-venv/bin/tt-smi -r > /dev/null 2>&1; echo "reset rc=$? $(date -u)"; }
cd $SNAP; echo "snapshot $(cat BASE_COMMIT) (pure git archive HEAD)"; md5sum -c MD5SUMS | sed 's/^/md5check /'; sed 's/^/md5 /' MD5SUMS
export PYTHONPATH=$SNAP:$PYTHONPATH GLM53_TP=4 GLM53_CACHE_DIR=$HOME/.cache/glm53_flash
unset GLM53_LAYERS GLM53_MC GLM53_PRECISION GLM53_DSA_SHARD GLM53_MAX_BATCH GLM53_MAX_SEQ_LEN GLM53_DRAM_RESERVE_MIB GLM53_MOE_SORTED_ROWS GLM53_PROBE_SLOTS
python -c "import models.autoports.zai_org_glm_5_3_flash.tt.dsa_shard as m; print('imports', m.__file__)"
WENV="GLM53_WORKER_L1_SIZE=1444864 TT_METAL_WATCHER=10 TT_METAL_WATCHER_NOINLINE=1 TT_METAL_WATCHER_DISABLE_ETH=1 TT_METAL_LOGGER_LEVEL=Warning"
step() { name=$1; secs=$2; shift 2; echo "=== step $name start $(date -u): $*" | tee -a $L/$name.log
  timeout -k 60 $secs "$@" >> $L/$name.log 2>&1; rc=$?
  echo "=== step $name exit code: $rc $(date -u)" | tee -a $L/$name.log; echo "$name rc=$rc $(date -u)" >> $L/status; RESET; }
wl() { w=$L/watcher/$1/generated/watcher/watcher.log; echo "watcher $1 suspicious_lines=$( [ -f $w ] && grep -ciE 'error|assert|sanitiz|overflow|exception|bad' $w || echo NA)" | tee -a $L/status; }
mkdir -p $L/watcher/shard_fp8_b1 $L/watcher/shard_fp8_batch32_dsa_moe
step a_watch_fp8_b1 1500 env $WENV GLM53_PRECISION=$PREC TT_METAL_LOGS_PATH=$L/watcher/shard_fp8_b1 \
  python -m pytest -sv -p no:cacheprovider "$F/test_dsa_shard.py::test_shard_matches_replicated[b1]"; wl shard_fp8_b1
step b_dsa_shard_bf16 1800 python -m pytest -sv -p no:cacheprovider $F/test_dsa_shard.py
step c_probe_s8_shard 900 env GLM53_DSA_SHARD=1 GLM53_PROBE_SLOTS=8 GLM53_DEALLOC_CHECK=1 python -m pytest -sv -p no:cacheprovider $F/test_model_probe.py
step c_probe_s1_shard 900 env GLM53_DSA_SHARD=1 GLM53_PROBE_SLOTS=1 GLM53_DEALLOC_CHECK=1 python -m pytest -sv -p no:cacheprovider $F/test_model_probe.py
step d_watch_fp8_batch32 1200 env $WENV GLM53_PRECISION=$PREC GLM53_DSA_SHARD=1 TT_METAL_LOGS_PATH=$L/watcher/shard_fp8_batch32_dsa_moe \
  python -m pytest -sv -p no:cacheprovider "$F/test_multichip_decoder.py::test_batch32_all_active[dsa_moe]"; wl shard_fp8_batch32_dsa_moe
echo "ALLDONE $(date -u)" >> $L/status

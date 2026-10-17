#!/bin/bash
# stage-2 review #2 P2-a: watcher on correctness cases (fused decoder, real weights, HF-compared) + op unit tests, final kernels
source ~/glm53-flash-bringup/env.sh; cd $TT_METAL_HOME
MD=$TT_METAL_HOME/models/autoports/zai_org_glm_5_3_flash; W=$MD/doc/fused_decoder/watcher_correctness; rm -rf $W; mkdir -p $W/suite $W/ops
L=~/glm53-flash-bringup/logs; R=~/glm53-flash-bringup/scripts/run-locked.sh
export GLM53_WORKER_L1_SIZE=1444864 TT_METAL_WATCHER=10 TT_METAL_WATCHER_NOINLINE=1
TT_METAL_LOGS_PATH=$W/suite $R $L/s2_watcher_correct.log timeout 10800 python_env/bin/pytest -sv -p no:cacheprovider $MD/tests/test_fused_decoder.py \
  -k "test_real_weights or test_batched_users or test_batch32_all_active or test_inactive_slot_state or (test_lengths and (kda_dense-2049 or dsa_moe-2051 or kda_moe-2049 or dsa_moe-1))"
cp $L/s2_watcher_correct.log $W/suite/console.log
TT_METAL_LOGS_PATH=$W/ops $R $L/s2_watcher_ops.log timeout 5400 python_env/bin/pytest -sv -p no:cacheprovider $MD/tests/test_kda_recurrent_decode.py $MD/tests/test_mhc_mix.py $MD/tests/test_mhc_pre.py $MD/tests/test_mhc_post.py $MD/tests/test_mhc_split_sinkhorn.py
cp $L/s2_watcher_ops.log $W/ops/console.log
s=$(cat $W/*/generated/watcher/watcher.log 2>/dev/null | grep -ciE 'error|assert|sanitiz|overflow')
echo "s2 watcher correctness: suite $(grep -E '^=+ .*(passed|failed)' $W/suite/console.log | tail -1) ops $(grep -E '^=+ .*(passed|failed)' $W/ops/console.log | tail -1) suspicious=$s" >> $L/s3_sweep.status

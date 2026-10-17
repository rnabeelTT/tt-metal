#!/usr/bin/env bash
# stage-2 review #2: after the P1 rewrites (decode_rm_relayout, mhc_finalize_fold): layer-level bit-exactness (B 32 and
# B 1, 5 decode steps incl. 4 traced replays), the fused suite (suite6), the watcher correctness subset, then Tracy:
# all 12 fused cases and the 8 functional KDA cases, same session
source ~/glm53-flash-bringup/env.sh; cd $TT_METAL_HOME
L=~/glm53-flash-bringup/logs; R=~/glm53-flash-bringup/scripts/run-locked.sh
MD=$TT_METAL_HOME/$MODEL_DIR; F=dsa_heads_op,mlp_act_fold,kda_decode_fold,decode_rm_relayout,mhc_finalize_fold
$R $L/s2r2_flag_check.log timeout 5400 python_env/bin/python -m models.autoports.zai_org_glm_5_3_flash.tools.fused_flag_check $F 0 3 4
n=$(grep -c "bit-exact True" $L/s2r2_flag_check.log); echo "flag check $n / 36" >> $L/s2r2.status
[ "$n" != 36 ] && { echo "not bit-exact: stop" >> $L/s2r2.status; exit 1; }
$R $L/s2_fused_suite6.log timeout 14400 python_env/bin/pytest -sv -p no:cacheprovider $MD/tests/test_fused_decoder.py
echo "suite6: $(grep -E '^=+ .*(passed|failed)' $L/s2_fused_suite6.log | tail -1)" >> $L/s2r2.status
W=$MD/doc/fused_decoder/watcher_correctness; rm -rf $W/suite; mkdir -p $W/suite
GLM53_WORKER_L1_SIZE=1444864 TT_METAL_WATCHER=10 TT_METAL_WATCHER_NOINLINE=1 TT_METAL_LOGS_PATH=$W/suite \
  $R $L/s2r2_watcher_suite.log timeout 10800 python_env/bin/pytest -sv -p no:cacheprovider $MD/tests/test_fused_decoder.py \
  -k "test_real_weights or test_batched_users or test_batch32_all_active or test_inactive_slot_state or (test_lengths and (kda_dense-2049 or dsa_moe-2051 or kda_moe-2049 or dsa_moe-1))"
cp $L/s2r2_watcher_suite.log $W/suite/console.log
echo "watcher suite: $(grep -E '^=+ .*(passed|failed)' $W/suite/console.log | tail -1) suspicious=$(cat $W/suite/generated/watcher/watcher.log 2>/dev/null | grep -ciE 'error|assert|sanitiz|overflow')" >> $L/s2r2.status
D=$MD/doc; ALL='test_perf_prefill[kda_dense-128] test_perf_prefill[kda_dense-2048] test_perf_prefill[kda_moe-128] test_perf_prefill[kda_moe-2048] test_perf_prefill[dsa_moe-128] test_perf_prefill[dsa_moe-2048] test_perf_traced_decode[kda_dense-1] test_perf_traced_decode[kda_dense-32] test_perf_traced_decode[kda_moe-1] test_perf_traced_decode[kda_moe-32] test_perf_traced_decode[dsa_moe-1] test_perf_traced_decode[dsa_moe-32]'
KDA='test_perf_prefill[kda_dense-128] test_perf_prefill[kda_dense-2048] test_perf_prefill[kda_moe-128] test_perf_prefill[kda_moe-2048] test_perf_traced_decode[kda_dense-1] test_perf_traced_decode[kda_dense-32] test_perf_traced_decode[kda_moe-1] test_perf_traced_decode[kda_moe-32]'
CASES="$ALL" GLM53_DECODER=fused TRACY_OUT=$D/fused_decoder/tracy bash ~/glm53-flash-bringup/scripts/01-tracy.sh > $L/s2r2_tracy.log 2>&1
CASES="$KDA" GLM53_DECODER=functional TRACY_OUT=$D/functional_decoder/tracy bash ~/glm53-flash-bringup/scripts/01-tracy.sh >> $L/s2r2_tracy.log 2>&1
echo "tracy: $(grep -c 'exit=0' $L/s2r2_tracy.log) / 20 $(date -Is)" >> $L/s2r2.status

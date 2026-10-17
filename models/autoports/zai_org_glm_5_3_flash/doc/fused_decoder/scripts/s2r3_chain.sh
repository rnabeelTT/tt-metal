#!/usr/bin/env bash
# stage-2 review #3: P2-2 probes (fast_reduce_nc chunk sum; narrower KDA prefill matmul at bf16 / HiFi4), then P2-1 the
# op unit tests under the watcher one module per pytest process (own TT_METAL_LOGS_PATH each: the watcher log restarts
# with every device open; TT_METAL_WATCHER_APPEND=1 keeps every session of a module in one log), and two B=1 traced fused decode cases under the watcher
source ~/glm53-flash-bringup/env.sh; cd $TT_METAL_HOME
L=~/glm53-flash-bringup/logs; R=~/glm53-flash-bringup/scripts/run-locked.sh; S=$L/s2r3.status
MD=$TT_METAL_HOME/$MODEL_DIR; W=$MD/doc/fused_decoder/watcher_correctness
$R $L/s2r3_probe.log timeout -k 60 1800 python_env/bin/python -m models.autoports.zai_org_glm_5_3_flash.tools.fusing_probe chunksum kdaprefill2
echo "probe exit $(grep -o 'exit code: [0-9]*' $L/s2r3_probe.log | tail -1)" >> $S
export GLM53_WORKER_L1_SIZE=1444864 TT_METAL_WATCHER=10 TT_METAL_WATCHER_NOINLINE=1 TT_METAL_WATCHER_APPEND=1  # keep every device session
rm -rf $W/ops; mkdir -p $W/ops
for m in test_kda_recurrent_decode test_mhc_mix test_mhc_pre test_mhc_post test_mhc_split_sinkhorn; do
  mkdir -p $W/ops/$m
  TT_METAL_LOGS_PATH=$W/ops/$m $R $L/s2r3_w_$m.log timeout -k 60 3600 python_env/bin/pytest -sv -p no:cacheprovider $MD/tests/$m.py
  cp $L/s2r3_w_$m.log $W/ops/$m/console.log
  echo "$m: $(grep -E '^=+ .*(passed|failed)' $W/ops/$m/console.log | tail -1) starts=$(grep -c 'starting' $W/ops/$m/generated/watcher/watcher.log 2>/dev/null) suspicious=$(grep -ciE 'error|assert|sanitiz|overflow' $W/ops/$m/generated/watcher/watcher.log 2>/dev/null)" >> $S
done
for c in 'kda_moe-1' 'dsa_moe-1'; do
  d=$W/b1_traced_$c; rm -rf $d; mkdir -p $d
  GLM53_DECODER=fused TT_METAL_LOGS_PATH=$d $R $L/s2r3_w_b1_$c.log timeout -k 60 3600 python_env/bin/pytest -sv -p no:cacheprovider "$MD/tests/test_perf_decoder.py::test_perf_traced_decode[$c]"
  cp $L/s2r3_w_b1_$c.log $d/console.log
  echo "b1 $c: $(grep -E '^=+ .*(passed|failed)' $d/console.log | tail -1) suspicious=$(grep -ciE 'error|assert|sanitiz|overflow' $d/generated/watcher/watcher.log 2>/dev/null)" >> $S
done
echo "DONE $(date -Is)" >> $S

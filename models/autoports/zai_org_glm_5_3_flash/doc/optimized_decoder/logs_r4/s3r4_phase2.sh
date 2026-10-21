#!/bin/bash
# stage-3 review #4 final gate on the review #4 default (one ticket per case)
S=~/glm53-flash-bringup/scripts/s3r4_one.sh
$S suite; $S watcher_suite; $S watcher_decode1
for k in dsa_moe-1 kda_dense-1 kda_moe-1 kda_dense-32 kda_moe-32 dsa_moe-32; do $S tracy - "test_perf_traced_decode[$k]"; done
$S tracy - "test_perf_traced_decode[dsa_moe-1]" 16384; $S tracy opt_s5_pf_r3 "test_perf_traced_decode[dsa_moe-1]" 16384
for k in kda_dense-2048 kda_moe-2048 dsa_moe-2048 kda_dense-128 kda_moe-128 dsa_moe-128; do $S tracy - "test_perf_prefill[$k]"; done
echo "$(date -Is) phase2 done" >> ~/glm53-flash-bringup/logs/s3r4.status

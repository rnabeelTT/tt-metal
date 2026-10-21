#!/bin/bash
# stage-3 review #4: the gate cases that dsa_q_mm (now off in the default) changes, rerun on the final default
S=~/glm53-flash-bringup/scripts/s3r4_one.sh
$S tracy - "test_perf_traced_decode[dsa_moe-1]"; $S suite; $S watcher_suite; $S watcher_decode1
echo "$(date -Is) phase3 done" >> ~/glm53-flash-bringup/logs/s3r4.status

#!/bin/bash
# One thing on the chips at a time: the v2 release chains first (user, 2026-10-23: scorecards and fused-kernel work
# before MTP tuning), then the rest. Logs: runs/<chain>.log. runs/HOLD pauses between chains.
R=$HOME/ornith-fusion/runs
for c in cn_v2_chain ornith_v2_chain q122_v2_chain glm_chain ornith_p150_chain q122_hub_smoke ornith_mtp q122_mtp; do
  while [ -e $R/HOLD ] || $HOME/ornith-fusion/bin/chips_busy; do sleep 60; done
  echo "== $c $(date +%T)"
  bash $R/$c.sh > $R/$c.log 2>&1
  echo "   $c exit $? $(date +%T)"
done
echo "queue done $(date +%T)"

#!/bin/bash
# One thing on the chips at a time: the run chains in order, after the 122B card chain. Logs: runs/<chain>.log
R=$HOME/ornith-fusion/runs
while pgrep -f "bash $R/q122_(card_)?chain.sh|bash $R/q122_longctx.sh|bash $R/q122_gpqa.sh|bash $R/c30_chain.sh|bash $R/gdn_ms.sh|bash $R/q122_gdn.sh" >/dev/null; do sleep 60; done
for c in cn_v2_chain glm_chain ornith_mtp ornith_p150_chain q122_hub_smoke q122_mtp; do
  # runs/HOLD: kernel work has the chips between chains
  while [ -e $R/HOLD ] || $HOME/ornith-fusion/bin/chips_busy; do sleep 60; done
  echo "== $c $(date +%T)"
  bash $R/$c.sh > $R/$c.log 2>&1
  echo "   $c exit $? $(date +%T)"
done
echo "queue done $(date +%T)"

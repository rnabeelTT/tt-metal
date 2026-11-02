#!/bin/bash
# GDN mixer profile for plan step 4 (after the queued chains): per-layer traced decode, p300x2, shipped config (fused
# MoE + replicated boundary), QWEN35_GDN_FUSED_DECODE 0 / 1 (the kda conv + recurrence POC ops), b1 and b8.
export PATH=$HOME/ornith-fusion/bin:$PATH
T=$HOME/tt-metal-bringups/models/autoports/ornith_ai_ornith_1_5_35b_a3b/fusion/tools
R=$HOME/ornith-fusion/runs
while pgrep -fx "bash $R/cn_hub_smoke.sh" >/dev/null || pgrep -fx "bash $R/ornith_p150_chain.sh" >/dev/null || pgrep -fx "bash $R/glm_chain.sh" >/dev/null || pgrep -fx "bash $R/q122_chain.sh" >/dev/null || chips_busy; do sleep 60; done
echo "gdn prof start $(date +%T)"
for F in 0 1; do
  PROF_ENV="--env QWEN35_MOE_FUSED_DECODE=1 --env QWEN35_DECODE_REPLICATED=1 --env QWEN35_GDN_FUSED_DECODE=$F" \
    $T/run_layer_prof.sh "traced_decode_batched and (b1- or b8-)" gdn_f${F}
  f=$(find $R/prof/gdn_f${F} -name "ops_perf_results*.csv" | sort | tail -1)
  [ -n "$f" ] && python3 $T/segments.py "$f" 2>&1 | tail -n 12
done
echo "gdn prof done $(date +%T)"

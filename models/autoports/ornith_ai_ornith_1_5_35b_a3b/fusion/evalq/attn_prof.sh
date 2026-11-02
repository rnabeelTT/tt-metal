#!/bin/bash
# Fused attention decode ops: Ornith per-layer profile b1 / b8, attention ops off / on (GDN ops on, shipped fast config)
export PATH=$HOME/ornith-fusion/bin:$PATH
T=$HOME/tt-metal-gdnwip/models/autoports/ornith_ai_ornith_1_5_35b_a3b/fusion/tools
R=$HOME/ornith-fusion/runs
for F in ${FS:-0 1}; do
  rm -rf $R/prof/attnops_f$F
  PROF_ENV="--env QWEN35_MOE_FUSED_DECODE=1 --env QWEN35_DECODE_REPLICATED=1 --env QWEN35_GDN_DECODE_OPS=1 --env QWEN35_ATTN_DECODE_OPS=$F" \
    timeout 2400 $T/run_layer_prof.sh "attn and traced_decode_batched and (b1- or b8-)" attnops_f$F
  docker rm -f ornith-prof-attnops_f$F >/dev/null 2>&1
  f=$(find $R/prof/attnops_f$F -name "ops_perf_results*.csv" | sort | tail -1)
  echo "== attn prof ops=$F"; [ -n "$f" ] && python3 $T/segments.py "$f" 2>&1 | grep -E "==|mixer|layer|batch|b1|b8"
done
echo "attn prof done $(date +%T)"

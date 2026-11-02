#!/bin/bash
# Step 6 fused decode add + RMSNorm boundary op: Ornith layer tests (GDN + attention layers) with QWEN35_ADDNORM_DECODE=1,
# then the per-layer profile b1 / b8 addnorm off / on (GDN + attention ops on), from the gdn_wip tree
export PATH=$HOME/ornith-fusion/bin:$PATH
T=$HOME/tt-metal-gdnwip/models/autoports/ornith_ai_ornith_1_5_35b_a3b/fusion/tools
R=$HOME/ornith-fusion/runs; mkdir -p $R/addnorm
cd $T
OPS="--env QWEN35_MOE_FUSED_DECODE=1 --env QWEN35_DECODE_REPLICATED=1 --env QWEN35_GDN_DECODE_OPS=1 --env QWEN35_ATTN_DECODE_OPS=1"
K="real_weight or batched_decode or traced_decode or nonaligned or runtime_fallback or prefill_after_traced"
for A in ${AS:-1}; do
  IMAGE=ornith-dev MESH=P300x2 TNAME=addnorm-pcc TENV="$OPS --env QWEN35_LAYER_TESTS_POLICY=shipped --env QWEN35_ADDNORM_DECODE=$A" timeout 3600 ./trun.sh \
    "python -m pytest -q -s models/autoports/ornith_ai_ornith_1_5_35b_a3b/tests/test_multichip_decoder.py -k '$K' 2>&1 | grep -E 'PCC|pcc|passed|failed|FAILED|Error|assert|tests/' | grep -v 'decode user' | tail -n 60" > $R/addnorm/pcc_a$A.log 2>&1
  echo "== layer tests addnorm=$A $(date +%T)"; cat $R/addnorm/pcc_a$A.log
done
[ -n "$NOPROF" ] && exit 0
for A in ${PAS:-0 1}; do
  for Y in gdn attn; do
    rm -rf $R/prof/addnorm_${Y}_a$A
    PROF_ENV="$OPS --env QWEN35_ADDNORM_DECODE=$A" timeout 2400 ./run_layer_prof.sh "$Y and traced_decode_batched and (b1- or b8-)" addnorm_${Y}_a$A
    docker rm -f ornith-prof-addnorm_${Y}_a$A >/dev/null 2>&1
    f=$(find $R/prof/addnorm_${Y}_a$A -name "ops_perf_results*.csv" | sort | tail -1)
    echo "== $Y prof addnorm=$A"; [ -n "$f" ] && python3 ./segments.py "$f" --labels=b1,b8 2>&1 | grep -E "==|mixer|moe|boundary|layer"
  done
done
echo "addnorm layer done $(date +%T)"

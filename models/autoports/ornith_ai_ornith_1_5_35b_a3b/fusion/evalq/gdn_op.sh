#!/bin/bash
# Fused GDN decode mixer (models/demos/blackhole/qwen36/tt/gdn_decode, QWEN35_GDN_DECODE_OPS):
#  1. op test on one die: Ornith TP4 / TP2 and 122B TP4 shapes, B = 1 / 2 / 8 / 32, vs torch FP32;
#  2. Ornith GDN layer tests on 4 dies (shipped policy + fused MoE + replicated boundary), ops off / on: real weights vs
#     HF, shipped-policy PCC, batched b32 / distinct positions, traced decode, prefill after traced decode,
#     non-aligned continuation, runtime fallback audit;
#  3. per-layer device profile, traced batched decode b1 / b8, ops off / on (segments.py).
export PATH=$HOME/ornith-fusion/bin:$PATH
T=$HOME/tt-metal-bringups/models/autoports/ornith_ai_ornith_1_5_35b_a3b/fusion/tools
R=$HOME/ornith-fusion/runs; mkdir -p $R/gdn
cd $T
echo "gdn op start $(date +%T)"
IMAGE=ornith-dev MESH=P300 DEVS=0,1 TNAME=gdn-op timeout 1800 ./trun.sh \
  "python models/autoports/ornith_ai_ornith_1_5_35b_a3b/tests/test_gdn_decode_ops.py tp4 tp2 q122" > $R/gdn/op.log 2>&1
rc=$?
echo "gdn op exit $rc $(date +%T)"
grep -E "pcc|Error|error:|assert|Traceback" $R/gdn/op.log | grep -v riscv-tt-elf | head -40
[ $rc = 0 ] || exit 1
SHIP="--env QWEN35_MOE_FUSED_DECODE=1 --env QWEN35_DECODE_REPLICATED=1 --env QWEN35_LAYER_TESTS_POLICY=shipped"
K="gdn and (real_weight or shipped_policy or batched_decode or traced_decode or nonaligned or fallback_audit or prefill_after_traced)"
for F in 0 1; do
  IMAGE=ornith-dev MESH=P300x2 TNAME=gdn-pcc TENV="$SHIP --env QWEN35_GDN_DECODE_OPS=$F" timeout 3600 ./trun.sh \
    "python -m pytest -q -s models/autoports/ornith_ai_ornith_1_5_35b_a3b/tests/test_multichip_decoder.py -k '$K' 2>&1 | grep -E 'PCC|pcc|passed|failed|FAILED|Error|assert' | grep -v 'shared memory' | tail -n 40" > $R/gdn/pcc_f$F.log 2>&1
  echo "== gdn layer tests ops=$F $(date +%T)"; cat $R/gdn/pcc_f$F.log
done
for F in 0 1; do
  PROF_ENV="--env QWEN35_MOE_FUSED_DECODE=1 --env QWEN35_DECODE_REPLICATED=1 --env QWEN35_GDN_DECODE_OPS=$F" \
    $T/run_layer_prof.sh "traced_decode_batched and (b1- or b8-)" gdnops_f${F}
  f=$(find $R/prof/gdnops_f${F} -name "ops_perf_results*.csv" | sort | tail -1)
  echo "== gdn prof ops=$F"; [ -n "$f" ] && python3 $T/segments.py "$f" 2>&1 | tail -n 14
done
echo "gdn op done $(date +%T)"

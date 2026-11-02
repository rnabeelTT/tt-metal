#!/bin/bash
# Qwen3-Coder-Next blackhole-fast-v2 (= blackhole-fast + the fused GDN decode mixer + attention decode ops): package (coder_next_fusion,
# tt-model-fast-v2.yaml), then v2 vs the published blackhole-fast, both served exactly as shipped: p300x2 card rows +
# real text + greedy + evals (HumanEval chat, MBPP, GSM8K CoT, IFEval), then p300 rows + greedy.
export PATH=$HOME/ornith-fusion/bin:$PATH MODEL_ID=Qwen/Qwen3-Coder-Next
C=$HOME/tt-metal-codernext
T=$C/models/autoports/ornith_ai_ornith_1_5_35b_a3b/fusion/tools
R=$HOME/ornith-fusion/runs
while chips_busy; do sleep 60; done
echo "cn v2 chain start $(date +%T)"
(cd $C && tt-model package --container models/autoports/qwen_qwen3_coder_next/tt-model-fast-v2.yaml > $R/package_cnv2.log 2>&1; echo "package exit $?")
NEW=$C/build/qwen3-coder-next-blackhole-fast-v2/tt_kernel_manifest.json
[ -f $NEW ] || { echo "no package"; tail -n 20 $R/package_cnv2.log; exit 1; }
PUB=raahemnabeel/qwen3-coder-next-blackhole-fast
for P in p300x2 p300; do
  EV=""; [ $P = p300x2 ] && EV="humaneval_chat mbpp_instruct gsm8k_cot_llama_hf ifeval"
  GREEDY=1 EVALS=$EV $T/card_bench.sh card_cnv2_$P $NEW $P
  # the published blackhole-fast was measured the same way at its release: runs/card_cnnew_<profile>
  python3 $T/greedy.py compare card_cnnew_$P card_cnv2_$P 2>&1 | tail -n 3
done
echo "cn v2 chain done $(date +%T)"

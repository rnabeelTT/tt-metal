#!/bin/bash
# Qwen3-Coder-Next blackhole-fast (one fast repo: steps 1-5 + 7): package (coder_next_fusion, tt-model-fast.yaml), served
# exactly as shipped: p300x2 card rows + real text + greedy + evals (HumanEval chat, MBPP, GSM8K CoT, IFEval), p300 rows +
# greedy, then a p300 sampled-output check (garble gate; 2 dies keep the stock top-k).
export PATH=$HOME/ornith-fusion/bin:$HOME/.local/bin:$PATH MODEL_ID=Qwen/Qwen3-Coder-Next CACHE_NAME=qwen3-coder-next-blackhole
C=$HOME/tt-metal-codernext
T=$C/models/autoports/ornith_ai_ornith_1_5_35b_a3b/fusion/tools
R=$HOME/ornith-fusion/runs
while systemctl --user is-active -q p300-fix || systemctl --user is-active -q q122-s7b || systemctl --user is-active -q ornith-publish || systemctl --user is-active -q q122-s7c || chips_busy; do sleep 60; done
echo "cn fast chain start $(date +%T)"
(cd $C && tt-model package --container models/autoports/qwen_qwen3_coder_next/tt-model-fast.yaml > $R/package_cnfast.log 2>&1; echo "package exit $?")
NEW=$C/build/qwen3-coder-next/tt_kernel_manifest.json
IMG=$(python3 -c "import json; print(json.load(open('$NEW'))['container']['built']['image'])"); echo "image $IMG"
for P in p300x2 p300; do
  EV=""; [ $P = p300x2 ] && EV="humaneval_chat mbpp_instruct gsm8k_cot_llama_hf ifeval"
  rm -rf $R/card_cnfast_$P
  GREEDY=1 EVALS=$EV $T/card_bench.sh card_cnfast_$P $NEW $P
  python3 $T/greedy.py compare card_cnnew_$P card_cnfast_$P 2>&1 | tail -n 3
done
E="--env QWEN35_MOE_FUSED_DECODE=1 --env QWEN35_DECODE_REPLICATED=1 --env QWEN35_GDN_DECODE_OPS=1 --env QWEN35_ATTN_DECODE_OPS=1 --env QWEN35_ROWS_TOPK=1"
rm -rf $R/cnfast_p300g
GARBLE=1 ROWS="128:128:1:4" $T/milestone.sh cnfast_p300g cn-g $IMG p300 $E
python3 -c "
import json; d=json.load(open('$R/cnfast_p300g/garble.json')); print('garble', d['summary']); print([(r['n_words'], r['text'][:50]) for r in d['rows']][:3])"
echo "cn fast chain done $(date +%T)"

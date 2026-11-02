#!/bin/bash
# Qwen3-Coder-30B-A3B fused expert-parallel MoE (branch c30_fusion, worktree ~/tt-metal-c30):
#  1. op tests (ornith_fusion tree): ep (one die's window, [gate | up], 1 / 2 / 8 / 32 rows) + the m / rows regressions;
#  2. decode tests vs HF (real weights, 4 dies) with the fused path off / on;
#  3. package the fast bundle, then both bundles served as shipped on p300x2 (card rows, real text, greedy, evals) and
#     p300 (the default profile).
export PATH=$HOME/ornith-fusion/bin:$PATH MODEL_ID=Qwen/Qwen3-Coder-30B-A3B-Instruct CACHE_NAME=qwen3-coder-30b-a3b
C=$HOME/tt-metal-c30
T=$C/models/autoports/ornith_ai_ornith_1_5_35b_a3b/fusion/tools
R=$HOME/ornith-fusion/runs; mkdir -p $R/c30
echo "c30 chain start $(date +%T)"
(cd $HOME/tt-metal-bringups/models/autoports/ornith_ai_ornith_1_5_35b_a3b/fusion/tools
 IMAGE=ornith-dev MESH=P300 DEVS=0,1 TNAME=ep-op timeout 2400 ./trun.sh \
   "python models/autoports/ornith_ai_ornith_1_5_35b_a3b/tests/test_moe_decode_ops.py ep experts rows tp2" > $R/c30/op.log 2>&1
 echo "c30 op test exit $? $(date +%T)"; grep -E "^ep rows|experts trial 0|worst row|Error|error:|assert" $R/c30/op.log | tail -n 30)
cd $T
for F in 0 1; do
  IMAGE=ornith-dev MESH=P300x2 TNAME=c30-pcc TENV="--env QWEN3_MOE_FUSED_DECODE=$F" timeout 7200 ./trun.sh \
    "python -m pytest -q -s models/demos/blackhole/qwen3_coder_30b_a3b/tests/test_multichip_decoder.py -k 'test_multichip_decode_batch or test_multichip_multi_step_decode_vs_hf or test_expert_window_can_be_empty' 2>&1 | grep -iE 'pcc|passed|failed|error|assert' | tail -n 30" > $R/c30/pcc_f$F.log 2>&1
  echo "== c30 pcc fused=$F $(date +%T)"; cat $R/c30/pcc_f$F.log
done
(cd $C && tt-model package --container models/demos/blackhole/qwen3_coder_30b_a3b/tt-model-fast.yaml > $R/package_c30.log 2>&1; echo "package exit $?")
NEW=$C/build/qwen3-coder-30b-a3b-blackhole-fast/tt_kernel_manifest.json
[ -f $NEW ] || { echo "no package"; exit 1; }
# p300x2 (4 dies, 32 experts per die) with the card evals (chat-safe HumanEval + MBPP, as Coder-Next), then the default
# profile p300 (2 dies, 64 experts per die); greedy on every run
for P in p300x2 p300; do
  EV=""; [ $P = p300x2 ] && EV="humaneval_chat mbpp_instruct gsm8k_cot_llama_hf ifeval"
  GREEDY=1 EVALS=$EV $T/card_bench.sh card_c30new_$P $NEW $P
  GREEDY=1 EVALS=$EV $T/card_bench.sh card_c30pub_$P raahemnabeel/qwen3-coder-30b-a3b $P
  python3 $T/greedy.py compare card_c30pub_$P card_c30new_$P 2>&1 | tail -n 3
done
# p150 (1 die: all 128 experts in one window): greedy + the single-user rows, both bundles
GREEDY=1 $T/card_bench.sh card_c30new_p150 $NEW p150 128:128:1:4 1024:256:1:4
GREEDY=1 $T/card_bench.sh card_c30pub_p150 raahemnabeel/qwen3-coder-30b-a3b p150 128:128:1:4 1024:256:1:4
python3 $T/greedy.py compare card_c30pub_p150 card_c30new_p150 2>&1 | tail -n 3
echo "c30 chain done $(date +%T)"

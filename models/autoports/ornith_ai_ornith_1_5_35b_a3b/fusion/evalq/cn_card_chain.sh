#!/bin/bash
# Qwen3-Coder-Next card measurements: both bundles served exactly as shipped, p300x2 then p300 (new first; GREEDY on
# every run, so the p300 bfp4 fused path is checked against the published one), then the evals already ran in cn_chain.
export PATH=$HOME/ornith-fusion/bin:$PATH MODEL_ID=Qwen/Qwen3-Coder-Next
T=$HOME/tt-metal-codernext/models/autoports/ornith_ai_ornith_1_5_35b_a3b/fusion/tools
NEW=$HOME/tt-metal-codernext/build/qwen3-coder-next-blackhole-fast/tt_kernel_manifest.json
PUB=raahemnabeel/qwen3-coder-next
for P in p300x2 p300; do
  EV=""; [ $P = p300x2 ] && EV="humaneval_chat mbpp_instruct"  # chat-safe HumanEval + MBPP rescore
  GREEDY=1 EVALS=$EV $T/card_bench.sh card_cnnew_$P $NEW $P
  GREEDY=1 EVALS=$EV $T/card_bench.sh card_cnpub_$P $PUB $P
  python3 $T/greedy.py compare card_cnpub_$P card_cnnew_$P 2>&1 | tail -n 12
done
echo "cn card chain done $(date +%T)"

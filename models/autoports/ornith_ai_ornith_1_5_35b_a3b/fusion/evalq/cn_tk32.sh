#!/bin/bash
# Coder-Next with the 32-bit rows top-k (package tt-model/qwen3-coder-next:873a5264e629, already built by tk32_ports.sh):
# p300 card rows + greedy as shipped (vs card_cnfast_p300), p300x2 spot rows (vs card_cnfast_p300x2).
export PATH=$HOME/ornith-fusion/bin:$HOME/.local/bin:$PATH MODEL_ID=Qwen/Qwen3-Coder-Next
R=$HOME/ornith-fusion/runs; C=$HOME/tt-metal-codernext
TC=$C/models/autoports/ornith_ai_ornith_1_5_35b_a3b/fusion/tools
NEW=$C/build/qwen3-coder-next/tt_kernel_manifest.json
while systemctl --user is-active -q rtk-time || chips_busy; do sleep 60; done
echo "cn tk32 start $(date +%T)"
rm -rf $R/card_cnfast3_p300 $R/card_cnfast3_p300x2
GREEDY=1 $TC/card_bench.sh card_cnfast3_p300 $NEW p300
python3 $TC/greedy.py compare card_cnfast_p300 card_cnfast3_p300 2>&1 | grep -E "^(sequential|concurrent):"
GREEDY=1 $TC/card_bench.sh card_cnfast3_p300x2 $NEW p300x2 128:128:1:4 128:128:8:32 1024:256:32:128
python3 $TC/greedy.py compare card_cnfast_p300x2 card_cnfast3_p300x2 2>&1 | grep -E "^(sequential|concurrent):"
echo "cn tk32 done $(date +%T)"

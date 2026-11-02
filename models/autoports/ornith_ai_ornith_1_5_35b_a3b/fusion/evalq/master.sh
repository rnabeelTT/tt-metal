#!/bin/bash
# one thing on the chips at a time: Coder-Next chain (running) -> Coder-Next card runs -> Qwen3.5-122B chain
export PATH=$HOME/ornith-fusion/bin:$PATH
R=$HOME/ornith-fusion/runs
while pgrep -fx "bash $R/cn_chain.sh" >/dev/null || pgrep -f "tt-model package" >/dev/null; do sleep 60; done
bash $R/cn_card_chain.sh > $R/cn_card_chain.log 2>&1
bash $R/q122_chain.sh > $R/q122_chain.log 2>&1
echo "master done $(date +%T)"

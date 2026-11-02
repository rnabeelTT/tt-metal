#!/bin/bash
# Publish Ornith-1.5-122B as raahemnabeel/ornith-1.5-122b (new public repo + catalog; user's go-ahead 2026-10-27), then
# a fresh-pull hub smoke on p300x2 (greedy vs the card run, real text).
export PATH=$HOME/ornith-fusion/bin:$HOME/.local/bin:$PATH MODEL_ID=ornith-ai/Ornith-1.5-122B CACHE_NAME=ornith-1.5-122b
set -o pipefail
Q=$HOME/tt-metal-q122; R=$HOME/ornith-fusion/runs; T=$Q/models/autoports/ornith_ai_ornith_1_5_35b_a3b/fusion/tools
REPO=raahemnabeel/ornith-1.5-122b
while chips_busy; do sleep 60; done
echo "o122 publish start $(date +%T)"
(cd $Q && tt-model package --container models/autoports/qwen_qwen3_5_122b_a10b/tt-model-ornith122.yaml > $R/package_o122pub.log 2>&1; echo "package exit $?")
B=$Q/build/ornith-1.5-122b
grep -q "GPQA-Diamond (all 198 questions) | 0.712" $B/README.md && grep -q "tt-model serve $REPO --profile" $B/README.md || { echo "README lacks the card"; exit 1; }
python3 -c "import json; print('image', json.load(open('$B/tt_kernel_manifest.json'))['container']['built']['image'])"
tt-model push $B --publish > $R/push_o122.log 2>&1; echo "push exit $?"; tail -n 4 $R/push_o122.log
L=fin_smoke_o122; rm -rf $R/$L; mkdir -p $R/$L
tt-model pull $REPO > $R/$L/pull.log 2>&1; echo "pull exit $? $(tail -n 3 $R/$L/pull.log | tr -s ' \n' ' ')"
GREEDY=1 $T/card_bench.sh $L $REPO p300x2 128:128:1:4 128:128:32:128
grep -h -m2 "image" $R/$L/serve.log
python3 $T/greedy.py compare card_o122_p300x2 $L 2>&1 | tail -n 3
echo "o122 publish done $(date +%T)"

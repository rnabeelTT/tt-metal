#!/bin/bash
# Coder-Next fast build republished as the model's main repo raahemnabeel/qwen3-coder-next (user renamed the repos:
# original -> -v1, -blackhole-fast -> plain), then a fresh-pull hub smoke on p300x2.
export PATH=$HOME/ornith-fusion/bin:$HOME/.local/bin:$PATH MODEL_ID=Qwen/Qwen3-Coder-Next
C=$HOME/tt-metal-codernext; R=$HOME/ornith-fusion/runs; T=$C/models/autoports/ornith_ai_ornith_1_5_35b_a3b/fusion/tools
while chips_busy; do sleep 60; done
echo "cn rename start $(date +%T)"
(cd $C && tt-model package --container models/autoports/qwen_qwen3_coder_next/tt-model-fast.yaml > $R/package_cnrename.log 2>&1; echo "package exit $?")
B=$C/build/qwen3-coder-next
grep -q "v1 -> current" $B/README.md && grep -q "tt-model serve raahemnabeel/qwen3-coder-next --profile" $B/README.md || { echo "README lacks the renamed card"; exit 1; }
python3 -c "import json; print('image', json.load(open('$B/tt_kernel_manifest.json'))['container']['built']['image'])"
tt-model push $B > $R/push_cnrename.log 2>&1; echo "push exit $?"; tail -n 4 $R/push_cnrename.log
L=fin_smoke_cn2; rm -rf $R/$L; mkdir -p $R/$L
tt-model pull raahemnabeel/qwen3-coder-next > $R/$L/pull.log 2>&1; echo "pull exit $? $(tail -n 3 $R/$L/pull.log | tr -s ' \n' ' ')"
GREEDY=1 $T/card_bench.sh $L raahemnabeel/qwen3-coder-next p300x2 128:128:1:4 128:128:32:128
grep -h -m2 "image" $R/$L/serve.log
python3 $T/greedy.py compare card_cnfast3_p300x2 $L 2>&1 | tail -n 3
echo "cn rename done $(date +%T)"

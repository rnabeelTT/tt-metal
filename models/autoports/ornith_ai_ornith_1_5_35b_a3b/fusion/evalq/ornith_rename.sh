#!/bin/bash
# Ornith fast build republished as the model's main repo raahemnabeel/ornith-1.5-35b-a3b (package / image name
# ornith-1.5-35b-a3b; the original is now raahemnabeel/ornith-1.5-35b-a3b-v1), then a fresh-pull hub smoke on p300x2.
export PATH=$HOME/ornith-fusion/bin:$HOME/.local/bin:$PATH MODEL_ID=ornith-ai/Ornith-1.5-35B-A3B
C=$HOME/tt-metal-bringups; R=$HOME/ornith-fusion/runs; T=$C/models/autoports/ornith_ai_ornith_1_5_35b_a3b/fusion/tools
while chips_busy; do sleep 60; done
echo "ornith rename start $(date +%T)"
(cd $C && tt-model package --container models/autoports/ornith_ai_ornith_1_5_35b_a3b/tt-model-fast.yaml > $R/package_orename.log 2>&1; echo "package exit $?")
B=$C/build/ornith-1.5-35b-a3b
grep -q "v1 -> current" $B/README.md && grep -q "tt-model serve raahemnabeel/ornith-1.5-35b-a3b --profile" $B/README.md || { echo "README lacks the renamed card"; exit 1; }
python3 -c "import json; m=json.load(open('$B/tt_kernel_manifest.json')); print('image', m['container']['built']['image'], 'repo', m.get('repo'), 'name', m.get('name'))"
tt-model push $B > $R/push_orename.log 2>&1; echo "push exit $?"; tail -n 6 $R/push_orename.log
L=fin_smoke_orn2; rm -rf $R/$L; mkdir -p $R/$L
tt-model pull raahemnabeel/ornith-1.5-35b-a3b > $R/$L/pull.log 2>&1; echo "pull exit $? $(tail -n 3 $R/$L/pull.log | tr -s ' \n' ' ')"
GREEDY=1 $T/card_bench.sh $L raahemnabeel/ornith-1.5-35b-a3b p300x2 128:128:1:4 128:128:32:128
grep -h -m2 "image" $R/$L/serve.log
python3 $T/greedy.py compare card_fast3_p300x2 $L 2>&1 | tail -n 3
echo "ornith rename done $(date +%T)"

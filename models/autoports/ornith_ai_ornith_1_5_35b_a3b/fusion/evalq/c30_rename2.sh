#!/bin/bash
# Coder-30B (user, 2026-10-27, after the Coder-Next / 122B work): original repo -> -v1, the fast repo -> the plain name
# (hf repos move, done by hand 2026-10-27 after deleting an older private -v1 at the user's request), repackage as name qwen3-coder-30b-a3b, push, fresh-pull hub smoke on p300x2.
export PATH=$HOME/ornith-fusion/bin:$HOME/.local/bin:$PATH MODEL_ID=Qwen/Qwen3-Coder-30B-A3B-Instruct CACHE_NAME=qwen3-coder-30b-a3b
C=$HOME/tt-metal-c30; R=$HOME/ornith-fusion/runs; T=$C/models/autoports/ornith_ai_ornith_1_5_35b_a3b/fusion/tools
REPO=raahemnabeel/qwen3-coder-30b-a3b
set -o pipefail
while chips_busy; do sleep 60; done
echo "c30 rename start $(date +%T)"
(cd $C && tt-model package --container models/demos/blackhole/qwen3_coder_30b_a3b/tt-model-fast.yaml > $R/package_c30rename.log 2>&1; echo "package exit $?")
B=$C/build/qwen3-coder-30b-a3b
grep -q "v1 -> current" $B/README.md && grep -q "tt-model serve $REPO --profile" $B/README.md || { echo "README lacks the renamed card"; exit 1; }
python3 -c "import json; print('image', json.load(open('$B/tt_kernel_manifest.json'))['container']['built']['image'])"
tt-model push $B > $R/push_c30rename.log 2>&1; echo "push exit $?"; tail -n 4 $R/push_c30rename.log
L=fin_smoke_c30b; rm -rf $R/$L; mkdir -p $R/$L
tt-model pull $REPO > $R/$L/pull.log 2>&1; echo "pull exit $? $(tail -n 3 $R/$L/pull.log | tr -s ' \n' ' ')"
GREEDY=1 $T/card_bench.sh $L $REPO p300x2 128:128:1:4 128:128:32:128
grep -h -m2 "image" $R/$L/serve.log
python3 $T/greedy.py compare card_c30new_p300x2 $L 2>&1 | tail -n 3
echo "c30 rename done $(date +%T)"

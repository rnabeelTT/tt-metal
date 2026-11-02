#!/bin/bash
# Ornith blackhole-fast: repackage with the final card (tt-model-fast.yaml card block) and push into the EXISTING
# raahemnabeel/ornith-1.5-35b-a3b-blackhole-fast (user, 2026-10-26: one fast repo per model, new revision).
export PATH=$HOME/ornith-fusion/bin:$HOME/.local/bin:$PATH
C=$HOME/tt-metal-bringups; R=$HOME/ornith-fusion/runs
while systemctl --user is-active -q q122-s7b; do sleep 60; done
echo "ornith publish start $(date +%T)"
(cd $C && tt-model package --container models/autoports/ornith_ai_ornith_1_5_35b_a3b/tt-model-fast.yaml > $R/package_ofast3.log 2>&1; echo "package exit $?")
B=$C/build/ornith-1.5-35b-a3b-blackhole-fast
grep -q "original -> blackhole-fast" $B/README.md || { echo "README lacks the new card"; exit 1; }
python3 -c "import json; print('image', json.load(open('$B/tt_kernel_manifest.json'))['container']['built']['image'])"
tt-model push $B > $R/push_ofast.log 2>&1; echo "push exit $?"; tail -n 5 $R/push_ofast.log
echo "ornith publish done $(date +%T)"

#!/bin/bash
# Qwen3.5-122B blackhole-fast: repackage with the final card and push into the EXISTING raahemnabeel/qwen3-5-122b-a10b-blackhole-fast
export PATH=$HOME/ornith-fusion/bin:$HOME/.local/bin:$PATH
Q=$HOME/tt-metal-q122; R=$HOME/ornith-fusion/runs
echo "q122 publish start $(date +%T)"
(cd $Q && tt-model package --container models/autoports/qwen_qwen3_5_122b_a10b/tt-model-fast.yaml > $R/package_q122fast2.log 2>&1; echo "package exit $?")
B=$Q/build/qwen3-5-122b-a10b
grep -q "published -> blackhole-fast" $B/README.md && grep -q "138 of 198" $B/README.md || { echo "README lacks the new card"; exit 1; }
python3 -c "import json; print('image', json.load(open('$B/tt_kernel_manifest.json'))['container']['built']['image'])"
tt-model push $B > $R/push_q122fast.log 2>&1; echo "push exit $?"; tail -n 6 $R/push_q122fast.log
echo "q122 publish done $(date +%T)"

#!/bin/bash
# Coder-Next blackhole-fast: repackage with the final card and push into the EXISTING raahemnabeel/qwen3-coder-next-blackhole-fast
export PATH=$HOME/ornith-fusion/bin:$HOME/.local/bin:$PATH
C=$HOME/tt-metal-codernext; R=$HOME/ornith-fusion/runs
while systemctl --user is-active -q cn-fast; do sleep 30; done
echo "cn publish start $(date +%T)"
(cd $C && tt-model package --container models/autoports/qwen_qwen3_coder_next/tt-model-fast.yaml > $R/package_cnfast2.log 2>&1; echo "package exit $?")
B=$C/build/qwen3-coder-next
grep -q "published -> blackhole-fast" $B/README.md && grep -q "0.824" $B/README.md || { echo "README lacks the new card"; exit 1; }
python3 -c "import json; print('image', json.load(open('$B/tt_kernel_manifest.json'))['container']['built']['image'])"
tt-model push $B > $R/push_cnfast.log 2>&1; echo "push exit $?"; tail -n 5 $R/push_cnfast.log
echo "cn publish done $(date +%T)"

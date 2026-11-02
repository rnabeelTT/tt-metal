#!/bin/bash
# Package-only (no serve, no push) of the template-card builds for Ornith-35B and Coder-Next while the eval queue
# holds the chips: the BuildKit cache was evicted, so this is a cold compile; it warms ccache for the later re-push.
export PATH=$HOME/ornith-fusion/bin:$HOME/.local/bin:$PATH
R=$HOME/ornith-fusion/runs
(cd $HOME/tt-metal-bringups && MODEL_ID=ornith-ai/Ornith-1.5-35B-A3B nice -n 15 tt-model package --container models/autoports/ornith_ai_ornith_1_5_35b_a3b/tt-model-fast.yaml > $R/pkg_orn.log 2>&1); echo "orn package exit $? $(date +%T)"
(cd $HOME/tt-metal-codernext && MODEL_ID=Qwen/Qwen3-Coder-Next nice -n 15 tt-model package --container models/autoports/qwen_qwen3_coder_next/tt-model-fast.yaml > $R/pkg_cn.log 2>&1); echo "cn package exit $? $(date +%T)"

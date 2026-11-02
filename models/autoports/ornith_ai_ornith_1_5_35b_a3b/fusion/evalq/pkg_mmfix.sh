#!/bin/bash
# Package the three multimodal bundles with the plugin's encoder-split fix (vllm-tt-plugin-pub 8b1f8b1); no serve/push.
export PATH=$HOME/ornith-fusion/bin:$HOME/.local/bin:$PATH
R=$HOME/ornith-fusion/runs
(cd $HOME/tt-metal-q122 && MODEL_ID=ornith-ai/Ornith-1.5-122B CACHE_NAME=ornith-1.5-122b nice -n 15 tt-model package --container models/autoports/qwen_qwen3_5_122b_a10b/tt-model-ornith122.yaml > $R/pkg_mmfix_o122.log 2>&1); echo "o122 package exit $? $(date +%T)"
(cd $HOME/tt-metal-q122 && MODEL_ID=Qwen/Qwen3.5-122B-A10B CACHE_NAME=qwen3-5-122b-a10b nice -n 15 tt-model package --container models/autoports/qwen_qwen3_5_122b_a10b/tt-model-fast.yaml > $R/pkg_mmfix_q122.log 2>&1); echo "q122 package exit $? $(date +%T)"
(cd $HOME/tt-metal-bringups && MODEL_ID=ornith-ai/Ornith-1.5-35B-A3B nice -n 15 tt-model package --container models/autoports/ornith_ai_ornith_1_5_35b_a3b/tt-model-fast.yaml > $R/pkg_mmfix_orn.log 2>&1); echo "orn package exit $? $(date +%T)"

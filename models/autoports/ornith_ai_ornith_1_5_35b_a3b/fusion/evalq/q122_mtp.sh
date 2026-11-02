#!/bin/bash
# Qwen3.5-122B MTP parameters on p300x2: the packaged blackhole-fast image + q122_fusion models/ with the GDN decode ops.
export PATH=$HOME/ornith-fusion/bin:$PATH MODEL_ID=Qwen/Qwen3.5-122B-A10B CACHE_NAME=qwen3-5-122b-a10b
export T=$HOME/tt-metal-q122/models/autoports/ornith_ai_ornith_1_5_35b_a3b/fusion/tools
NEW=$HOME/tt-metal-q122/build/qwen3-5-122b-a10b-blackhole-fast/tt_kernel_manifest.json
IMG=$(python3 -c "import json; print(json.load(open('$NEW'))['container']['built']['image'])")
bash $HOME/ornith-fusion/runs/mtp_sweep.sh q122_mtp $IMG 1 --env QWEN35_MOE_FUSED_DECODE=1 --env QWEN35_DECODE_REPLICATED=1 --env QWEN35_GDN_DECODE_OPS=1 --env QWEN35_ATTN_DECODE_OPS=1
echo "q122 mtp done $(date +%T)"

#!/bin/bash
# Ornith MTP parameters on p300x2: the blackhole-fast image + ornith_fusion models/ with the GDN decode ops (the next build).
export PATH=$HOME/ornith-fusion/bin:$PATH MODEL_ID=ornith-ai/Ornith-1.5-35B-A3B CACHE_NAME=ornith-1.5-35b-a3b-v2
export T=$HOME/tt-metal-bringups/models/autoports/ornith_ai_ornith_1_5_35b_a3b/fusion/tools
bash $HOME/ornith-fusion/runs/mtp_sweep.sh orn_mtp tt-model/ornith-1.5-35b-a3b-blackhole-fast:e7d45a33dd80 1 \
  --env QWEN35_MOE_FUSED_DECODE=1 --env QWEN35_DECODE_REPLICATED=1 --env QWEN35_GDN_DECODE_OPS=1 --env QWEN35_ATTN_DECODE_OPS=1
echo "ornith mtp done $(date +%T)"

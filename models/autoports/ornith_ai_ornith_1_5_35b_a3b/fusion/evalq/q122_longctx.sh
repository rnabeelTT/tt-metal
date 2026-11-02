#!/bin/bash
# 122B p300x2-longctx card rows, rerun: the shipped profile's 2048-token budget is smaller than one image item (4096
# tokens) and vLLM refuses to boot (both bundles). New bundle re-packaged with a 4096 budget; the published bundle gets
# the same budget through SERVE_ARGS. MTP-off single-user rows + one long prompt that is chunked.
export PATH=$HOME/ornith-fusion/bin:$PATH MODEL_ID=Qwen/Qwen3.5-122B-A10B CACHE_NAME=qwen3-5-122b-a10b
Q=$HOME/tt-metal-q122
T=$Q/models/autoports/ornith_ai_ornith_1_5_35b_a3b/fusion/tools
R=$HOME/ornith-fusion/runs
NEW=$Q/build/qwen3-5-122b-a10b-blackhole-fast/tt_kernel_manifest.json
# packaged in the first run (image af3d5f52a5eb); the bench rows only
$T/card_bench.sh card_q122new_longctx $NEW p300x2-longctx 128:128:1:4 2048:128:1:4 16384:128:1:2
SERVE_ARGS="--max-num-batched-tokens 4096" $T/card_bench.sh card_q122pub_longctx raahemnabeel/qwen3-5-122b-a10b p300x2-longctx 128:128:1:4 2048:128:1:4 16384:128:1:2
echo "q122 longctx done $(date +%T)"

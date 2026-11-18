#!/bin/bash
# Serve thinkingmachines/Inkling-Small (text) with the hybrid-MoE overlay and native MTP speculative decoding (single
# user) on the QB2, through vLLM + the TT plugin (OpenAI-compatible API on PORT, model name inkling-small-hybrid).
#
# usage: serve/serve_inkling.sh ; logs -> serve/server_inkling.log
# env:   the configuration: routed bfp4 with layers 2-17 gate/up bfp4 + down bfp8, 192 device slots per layer, fused
#                 decode kernels, host <-> device sockets, MTP k=2; the int4 host library of the 67 experts per layer
#                 outside the device's 191 built on first use, plans/inkling_plan.npz)
#        CTX (max model len, default 32768), K (MTP drafts per verify pass, 2), PORT (8000),
#        TRACE (1: verify / drafter traces), TRACE_MIB (trace region, default 400; fast 320), HYB_IMAGE, WCACHE, LIB, FP32ACC (1), PREFETCH (1), STREAM (1: prefill
#                 streaming; 0 = the CPU computes every missed expert), NS (staging slots: 32 at CTX > 32768, else 128), PCHUNK / PGROUP (prefill chunk / group: 1536 / 10 at CTX > 32768, else 2048 / 8),
#        EXTRA (more docker -e flags)
# Code: this checkout's Inkling port, the M2.7 port, qwen36 and hybrid_moe are mounted over the image's copies.
P=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)   # models/experimental/hybrid_moe
REPO=$(cd "$P/../../.." && pwd)
PROFILE=${PROFILE:-fast}
CTX=${CTX:-32768}
PLAN=/opt/tt-metal/models/experimental/hybrid_moe/plans/inkling_plan.npz
case $PROFILE in
  # fast: little DRAM beside the experts -> 1024-row prefill launches (half the 2048-row transients)
  fast) K=${K:-2}; POLICY=early_gu4dn8; S=192; LIB=${LIB:-/hyblib/inkling_int4_fb191}
        PENV="-e HYB_FUSED=1 -e HYB_ROWS=$((K + 1)) -e INKLING_PREFILL_CHUNK=${PCHUNK:-$([ "$CTX" -gt 32768 ] && echo 1536 || echo 2048)} -e HYB_SOCKET=1"
        # 4-page host -> device socket FIFOs (L1 on the socket cores): 8 left the FP32-accumulation prefill matmuls' circular
        # buffers 32 KB short (TT_THROW static CB clash at warm-up)
        PENV="$PENV -e HYB_SOCKET_H2D_DEPTH=${H2D_DEPTH:-4}"
        PENV="$PENV -e HYB_PREFETCH=${PREFETCH:-1} -e HYB_PLAN=$PLAN -e HYB_PLAN_WEIGHT=mass"
        # first use: build the int4 library of every expert outside each layer's 191 device slots (hybrid/firstboot.py)
        PENV="$PENV -e HYB_LIB_KEYS=port:thinkingmachines_inkling_small -e HYB_BUILD_STATIC=191"
        PENV="$PENV -e HYB_BUILD_LAYERS=$(seq -s, 2 41) -e QWEN35_MOE_FP32_ACC=${FP32ACC:-1}"
        # STREAM=1 (default): missed experts streamed to the device in prefill (the engine compiles every prefill
        # program before its trace captures: a program first compiled after them hung the device, HANDOFF_INKLING.md)
        if [ "${STREAM:-1}" = 1 ]; then
          # prefill streaming: missed experts copied over PCIe into 128 staging slots and computed on the device
          # (gate/up bfp4, down bfp8 copies of the 67 cold experts per layer, built on first use), the prefill
          # layer-major in groups of 16 chunks so a staged expert serves 16K tokens (tools/prefill_bench.py: 340 ->
          # 1037 tok/s at 16K)
          PENV="$PENV -e HYB_DEV_EXPERTS=${DEVEXP:-/hyblib/inkling_dev_gu4dn8_s191} -e HYB_STREAM_SLOTS=${NS:-$([ "$CTX" -gt 32768 ] && echo 32 || echo 128)}"
          PENV="$PENV -e HYB_DEV_POLICY=/opt/tt-metal/models/autoports/thinkingmachines_inkling_small/doc/precision/all_gu4dn8.json"
          PENV="$PENV -e HYB_STREAM_ROUNDS=${SROUNDS:-3} -e HYB_STREAM_MIN=${SMIN:-4} -e HYB_STREAM_MIN_LONG=1"
          PENV="$PENV -e HYB_STREAM_LONG_FROM=${SLONG:-1024} -e HYB_STREAM_CQ1=${CQ1:-0} -e HYB_PF_PREFETCH_MIN=${PFMIN:-0} -e HYB_LRU=0"
          PENV="$PENV -e INKLING_PREFILL_GROUP=${PGROUP:-$([ "$CTX" -gt 32768 ] && echo 10 || echo 8)}"
          # split: the fewest-token missed experts of a prefill step on the CPU while the rest stream (1-4K prompts:
          # 7% faster); costs for Inkling's experts: an 18.4 MB copy ~0.65 ms, an int4 expert ~0.5 ms + 0.05 per token
          PENV="$PENV -e HYB_PF_SPLIT=${PFSPLIT:-1} -e HYB_PF_COPY_MS=${COPY_MS:-4} -e HYB_PF_CPU_MS=0.5,0.05 -e HYB_HOSTK_THREADS=${HK_THREADS:-8}"
        fi
        # FETCH=1: missed experts copied into 16 LRU device slots per layer during decode (M2.7's PCIe fetch; 175 static
        # slots, so the int4 / device-format libraries cover everything outside the top 175; layers 18-41 fetch their
        # down projections as bfp4 copies). fixed policy: 30% of a layer's misses, at most 2 per layer and step
        if [ "${FETCH:-0}" = 1 ]; then
          LIB=/hyblib/inkling_int4_fb175
          PENV="$PENV -e HYB_LIB=$LIB -e HYB_DEV_EXPERTS=/hyblib/inkling_dev_gu4dn8_s175 -e HYB_BUILD_STATIC=175 -e HYB_LRU=16"
          PENV="$PENV -e HYB_FETCH_DN_EXPERTS=/hyblib/inkling_dev_dn4_s175 -e HYB_FETCH_DN_LAYERS=18-41"
          PENV="$PENV -e HYB_FETCH_POLICY=fixed -e HYB_FETCH_FRAC=${FFRAC:-0.3} -e HYB_FETCH_MAX=${FMAX:-2}"
        fi
        TRACE_MIB=${TRACE_MIB:-320} ;;
  *) echo "PROFILE=$PROFILE: only fast (the one configuration)"; exit 1 ;;
esac
IMG=${HYB_IMAGE:-tt-model/minimax-m2.7-hybrid-qb2:0c9857a46f16}
HF_HOST=${HF_HOST:-${HF_HOME:-/opt/tenstorrent/models}}
CACHE=${INKLING_CACHE:-$HOME/.cache/inkling}
TR_MIB=${TRACE_MIB:-400}
TT_CFG='{"tt": {"trace_mode": "all", "sample_on_device_mode": "all", "trace_region_size": '$((TR_MIB * 1048576))', "l1_small_size": 24576, "fabric_config": "FABRIC_1D"}}'
# image / audio inputs (the towers run on the host, tt/mm.py); MM_LIMIT='{"image": 0, "audio": 0}': text only
MM_LIMIT=${MM_LIMIT:-'{"image": 8, "audio": 4}'}
# the TT plugin with the generic multimodal items (branch inkling-mm of the plugin fork) over the image's copy
PLUGIN_SRC=${PLUGIN_SRC:-$HOME/vllm-tt-plugin-inkling/src/vllm_tt_plugin}
PLUGIN_MOUNT=(); [ -d "$PLUGIN_SRC" ] && PLUGIN_MOUNT=(-v "$PLUGIN_SRC:/opt/tt-venv/lib/python3.12/site-packages/vllm_tt_plugin:ro")
PORT_DIR=$REPO/models/autoports/thinkingmachines_inkling_small
(cd "$P" && python3 -m hybrid.hostk >/dev/null)
mkdir -p "$CACHE/tt" "$CACHE/logs" "$HOME/.cache/hybrid-moe"
docker rm -f inkling-serve >/dev/null 2>&1
docker run --rm --device /dev/tenstorrent --entrypoint tt-smi "$IMG" -r >/dev/null 2>&1
docker run -d --name inkling-serve \
  --device /dev/tenstorrent --ipc host --ulimit memlock=-1:-1 \
  --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G \
  -v "$HF_HOST:/hf" -v "$CACHE:/cache" -v "$HOME/.cache/hybrid-moe:/hyblib" \
  -v "$PORT_DIR:/opt/tt-metal/models/autoports/thinkingmachines_inkling_small" \
  -v "$REPO/models/autoports/minimaxai_minimax_m2_7:/opt/tt-metal/models/autoports/minimaxai_minimax_m2_7" \
  -v "$REPO/models/demos/blackhole/qwen36:/opt/tt-metal/models/demos/blackhole/qwen36" \
  -v "$P:/opt/tt-metal/models/experimental/hybrid_moe" \
  -v "$REPO/tt_metal/impl/dispatch/kernels/cq_prefetch.cpp:/opt/tt-metal/tt_metal/impl/dispatch/kernels/cq_prefetch.cpp:ro" \
  -v "$PORT_DIR/vllm_bundle:/extra_models:ro" "${PLUGIN_MOUNT[@]}" \
  --user "$(id -u):$(id -g)" -p ${PORT:-8000}:8000 \
  -e ARCH_NAME=blackhole -e HF_MODEL=thinkingmachines/Inkling-Small -e MESH_DEVICE=P300x2 -e HF_HUB_OFFLINE=1 \
  -e HOME=/tmp -e HF_HOME=/hf -e HF_HUB_CACHE=/hf/hub -e EXTRA_MODELS_DIR=/extra_models \
  -e TT_METAL_LOGGER_LEVEL=Error -e TT_METAL_CACHE=/cache/tt_fix -e TT_METAL_LOGS_PATH=/cache/logs \
  -e VLLM_ENGINE_READY_TIMEOUT_S=7200 \
  -e QWEN35_PRECISION_CONFIG=/opt/tt-metal/models/autoports/thinkingmachines_inkling_small/doc/precision/$POLICY.json \
  -e HYB_ENABLE=1 -e HYB_PORT=thinkingmachines_inkling_small -e HYB_POC_DIR=/opt/tt-metal/models/experimental/hybrid_moe \
  -e HYB_SLOTS=$S -e HYB_LIB=$LIB \
  -e HYB_STREAM_SLOTS=0 -e HYB_SKIP=0 -e HYB_LOWBIT=0 -e HYB_HOSTK_THREADS=${HK_THREADS:-6} -e HYB_DECODE_ROWS=8 \
  -e HYB_WCACHE=${WCACHE:-/cache/wcache} $PENV \
  -e INKLING_SPEC_K=$K -e INKLING_TRACE=${TRACE:-1} ${EXTRA} \
  -e PYTHONPATH=/opt/tt-metal/models/experimental/hybrid_moe/serve:/opt/tt-metal:/opt/tt-metal/models/experimental/hybrid_moe \
  --entrypoint bash "$IMG" -c "cd /tmp && exec vllm serve thinkingmachines/Inkling-Small --max-model-len $CTX \
    --max-num-seqs 1 --block-size 64 --additional-config '$TT_CFG' --served-model-name inkling-small-hybrid \
    --enable-auto-tool-choice --tool-call-parser inkling --reasoning-parser inkling --port 8000 \
    --limit-mm-per-prompt '$MM_LIMIT'" \
  && (docker logs -f inkling-serve > "$P/serve/server_inkling.log" 2>&1 &)
echo "started (PROFILE $PROFILE, CTX $CTX, K $K); tail -f $P/serve/server_inkling.log"

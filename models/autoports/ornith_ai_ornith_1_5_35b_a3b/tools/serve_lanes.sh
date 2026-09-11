#!/usr/bin/env bash
#
# Run N independent replicas of this model, one per Blackhole die ("lanes").
#
# WHY THIS EXISTS. The model is replicated, not sharded: 18.2 GB against ~30 GB usable
# per die, so one die holds a complete copy and a single stream never leaves it. Decode
# is dispatch-bound (~8.4 us per program regardless of tensor size), so tensor
# parallelism would run the same op count on smaller data and add collectives -- extra
# dies cannot make one stream faster. They serve more streams. Measured on this QB2,
# 64-token completions:
#
#     lanes (dies)      1        2        4
#     aggregate     26.07    54.47    91.24 tok/s
#     per stream    29.05    28.40    25.89 tok/s
#     TTFT (med)      397      404      420 ms
#
# 3.50x at four dies (one stream per lane, 256-token completions, end to end through
# vLLM). That is the ONLY way extra dies pay off for this model, and until now it was
# documented but not scripted: an operator who ran the p300x2 profile got one four-die
# replica at ~29 tok/s instead of four one-die replicas at 91. This script is the
# difference.
#
# Aggregate carries TTFT, so it lands under per-stream x lanes -- the same four lanes
# measure 80.63 at 64-token completions. Per-stream decaying 29.05 -> 25.89 with lane
# count is HOST contention between four vLLM processes, not the dies.
#
# Each lane is a separate vLLM process on its own port. There is no load balancer here
# on purpose -- put your own in front (nginx, HAProxy, or the client-side round-robin
# that tools/serve_bench.py --base does) so the deployment keeps whatever routing,
# health-checking and retry policy it already has.
#
# Usage:
#   serve_lanes.sh start [N]     launch N lanes on dies 0..N-1   (default: all dies)
#   serve_lanes.sh wait  [N]     block until every lane answers /v1/models
#   serve_lanes.sh status        per-lane pid / port / ready
#   serve_lanes.sh stop          stop every lane this script started
#   serve_lanes.sh urls          print lane base URLs, space separated
#
# Lane i serves on port (BASE_PORT + i), default 8100. Override with LANE_BASE_PORT.
# Logs and pidfiles land in LANE_RUN_DIR (default /tmp/ornith-lanes).

set -uo pipefail

BASE_PORT="${LANE_BASE_PORT:-8100}"
RUN_DIR="${LANE_RUN_DIR:-/tmp/ornith-lanes}"
MODEL_NAME="${LANE_MODEL_NAME:-ornith-1.5-35b-a3b}"
MAX_MODEL_LEN="${LANE_MAX_MODEL_LEN:-4096}"
# max_num_seqs is 1 by necessity, not by choice: qwen36 keeps the Gated DeltaNet
# recurrent state in external buffers and set_external_state asserts a batch-1 shape,
# so a wider batch dies at engine init. Concurrency comes from lanes. If that assert is
# ever lifted, raise this and the two multiply.
MAX_NUM_SEQS="${LANE_MAX_NUM_SEQS:-1}"
READY_TIMEOUT="${LANE_READY_TIMEOUT:-1800}"

die() { echo "serve_lanes: $*" >&2; exit 1; }

# One die is a PARTIAL p300 board. TT_VISIBLE_DEVICES alone makes tt-metal call the QB2
# a CUSTOM cluster and refuse to start, so the single-die descriptor is mandatory, not
# an optimization.
: "${TT_METAL_HOME:?set TT_METAL_HOME (source ~/ornith-bringup/env.sh; activate_venv)}"
MESH_DESC="$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto"

WEIGHTS="${ORNITH_WEIGHTS:-${WEIGHTS_VIEW:-}}"

# Without this every lane dies at startup with "No TT model architecture is registered
# for model: <weights path>". The plugin discovers models by listing SUBDIRECTORIES of
# EXTRA_MODELS_DIR for a vllm_metadata.json, so it points at the PARENT of this bundle,
# not at the bundle. tt-model sets it from tt-model.yaml's extra_models_dir; launching
# vllm directly, as here, does not, so set it ourselves.
export EXTRA_MODELS_DIR="${EXTRA_MODELS_DIR:-$TT_METAL_HOME/models/autoports}"

detect_dies() {
    local n=0
    for d in /dev/tenstorrent/[0-9]*; do [ -e "$d" ] && n=$((n + 1)); done
    echo "$n"
}

# Lane indices worth reporting on: every die present, plus every lane we have a
# pidfile for (which covers a lane whose die node is not listable from here).
lane_indices() {
    { for d in /dev/tenstorrent/[0-9]*; do [ -e "$d" ] && basename "$d"; done
      for f in "$RUN_DIR"/lane*.pid; do
          [ -e "$f" ] || continue
          f="${f##*/lane}"; echo "${f%.pid}"
      done
    } 2>/dev/null | sort -n -u
}

lane_port() { echo $((BASE_PORT + $1)); }
lane_pidfile() { echo "$RUN_DIR/lane$1.pid"; }
lane_log() { echo "$RUN_DIR/lane$1.log"; }

lane_ready() {
    curl -fsS --max-time 3 "http://127.0.0.1:$(lane_port "$1")/v1/models" >/dev/null 2>&1
}

lane_pid() {
    local f; f="$(lane_pidfile "$1")"
    [ -f "$f" ] || return 1
    local p; p="$(cat "$f" 2>/dev/null)"
    [ -n "$p" ] && kill -0 "$p" 2>/dev/null && echo "$p"
}

cmd_start() {
    local n="${1:-$(detect_dies)}"
    local total; total="$(detect_dies)"
    [ "$total" -gt 0 ] || die "no /dev/tenstorrent/N devices found"
    [ "$n" -le "$total" ] || die "asked for $n lanes but only $total dies present"
    [ -n "$WEIGHTS" ] || die "set ORNITH_WEIGHTS (or WEIGHTS_VIEW) to the weights directory"
    [ -f "$MESH_DESC" ] || die "single-die mesh descriptor not found: $MESH_DESC"
    mkdir -p "$RUN_DIR"

    for i in $(seq 0 $((n - 1))); do
        if lane_pid "$i" >/dev/null; then
            echo "lane $i: already running (pid $(lane_pid "$i"))"
            continue
        fi
        local port; port="$(lane_port "$i")"
        # Each lane is pinned to exactly one die and is a complete, independent replica;
        # nothing is shared between lanes, so one dying does not disturb the others.
        TT_VISIBLE_DEVICES="$i" \
        TT_MESH_GRAPH_DESC_PATH="$MESH_DESC" \
        nohup vllm serve "$WEIGHTS" \
            --served-model-name "$MODEL_NAME" \
            --max-num-seqs "$MAX_NUM_SEQS" \
            --block-size 64 \
            --max-model-len "$MAX_MODEL_LEN" \
            --additional-config '{"tt": {"trace_region_size": 1073741824}}' \
            --port "$port" \
            > "$(lane_log "$i")" 2>&1 &
        echo $! > "$(lane_pidfile "$i")"
        echo "lane $i: die $i, port $port, pid $!, log $(lane_log "$i")"
    done
    echo
    echo "Lanes are still loading weights and capturing traces; this takes minutes."
    echo "Block until they answer:   $0 wait $n"
}

cmd_wait() {
    local n="${1:-$(detect_dies)}"
    local deadline=$((SECONDS + READY_TIMEOUT))
    echo "waiting up to ${READY_TIMEOUT}s for $n lane(s)..."
    while :; do
        local ready=0 dead=0
        for i in $(seq 0 $((n - 1))); do
            if lane_ready "$i"; then
                ready=$((ready + 1))
            elif ! lane_pid "$i" >/dev/null; then
                dead=$((dead + 1))
            fi
        done
        if [ "$ready" -eq "$n" ]; then
            echo "all $n lane(s) ready"
            return 0
        fi
        if [ "$dead" -gt 0 ]; then
            echo "ERROR: $dead lane(s) exited before becoming ready; see $RUN_DIR/lane*.log" >&2
            cmd_status
            return 1
        fi
        if [ "$SECONDS" -ge "$deadline" ]; then
            echo "ERROR: timed out with $ready/$n ready" >&2
            cmd_status
            return 1
        fi
        sleep 10
    done
}

cmd_status() {
    # Enumerate the union of dies present and pidfiles written, so a lane still shows up
    # if the device nodes cannot be listed from this context (detect_dies would return 0
    # and `seq 0 -1` would silently print nothing but the header).
    printf "%-6s %-8s %-8s %-8s %s\n" lane port pid ready log
    for i in $(lane_indices); do
        local p; p="$(lane_pid "$i" || echo -)"
        local r=no; lane_ready "$i" && r=yes
        [ "$p" = "-" ] && [ "$r" = "no" ] && [ ! -f "$(lane_pidfile "$i")" ] && continue
        printf "%-6s %-8s %-8s %-8s %s\n" "$i" "$(lane_port "$i")" "$p" "$r" "$(lane_log "$i")"
    done
}

cmd_stop() {
    for i in $(lane_indices); do
        local p; p="$(lane_pid "$i" || true)"
        if [ -n "${p:-}" ]; then
            echo "lane $i: stopping pid $p"
            kill "$p" 2>/dev/null
        fi
        rm -f "$(lane_pidfile "$i")"
    done
    # Give vLLM a chance to release the devices cleanly before anyone reopens them; a
    # half-closed process leaves /dev/shm/tt_device_* behind and the next open fails
    # with "Sysmem mapped at unexpected NOC address".
    for _ in $(seq 1 30); do
        local alive=0
        for i in $(lane_indices); do lane_pid "$i" >/dev/null && alive=1; done
        [ "$alive" -eq 0 ] && break
        sleep 1
    done
    echo "stopped"
}

cmd_urls() {
    local n="${1:-$(detect_dies)}" out=""
    for i in $(seq 0 $((n - 1))); do out="$out http://127.0.0.1:$(lane_port "$i")"; done
    echo "${out# }"
}

case "${1:-}" in
    start)  shift; cmd_start "$@" ;;
    wait)   shift; cmd_wait "$@" ;;
    status) shift; cmd_status "$@" ;;
    stop)   shift; cmd_stop "$@" ;;
    urls)   shift; cmd_urls "$@" ;;
    *) sed -n '2,40p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac

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
#     aggregate     31.66    61.55   109.03 tok/s
#     per stream    33.15    32.35    28.52 tok/s
#     TTFT (med)      394      395      410 ms
#
# 3.44x at four dies (one stream per lane, 256-token completions, end to end through
# vLLM). That is the ONLY way extra dies pay off for this model, and until now it was
# documented but not scripted: an operator who ran the p300x2 profile got one four-die
# replica at ~29 tok/s instead of four one-die replicas at 91. This script is the
# difference.
#
# Aggregate carries TTFT, so it lands under per-stream x lanes -- the same four lanes
# measure 102.10 at 128-token completions. Per-stream decaying 33.15 -> 28.52 with lane
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

# Four lanes load 18.2 GB of weights each, concurrently, off the same host. vLLM's
# default VLLM_ENGINE_READY_TIMEOUT_S is 600 s and the API server hard-fails the moment
# it expires -- which is how a slow-but-healthy 4-lane start reads as a crash. Give the
# engine room; `wait` still bounds the whole thing with LANE_READY_TIMEOUT.
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-3600}"

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
        # --reasoning-parser / --tool-parser ARE NOT OPTIONAL, and this script shipped
        # without them. tt-model.yaml declares them as `capabilities` and the launcher
        # turns them into these flags, but a hand-rolled `vllm serve` gets neither -- so
        # every lane returned the model's THINKING as its answer, `</think>` and all.
        # Stage 11 measured what that costs on ifeval, changing no prompt or threshold:
        #   prompt-level strict  32.14 without the parser -> 60.71 with it
        #   inst-level strict    46.51                    -> 69.77
        # i.e. lanes -- the deployment this package RECOMMENDS for two and four dies --
        # were serving at roughly half the instruction-following quality of the container
        # for the same weights. Caught 2026-09-17 when a 4-lane verification printed
        # answers with "</think>" inline.
        #
        # SPELL THE TOOL FLAG EXACTLY LIKE THIS. `--tool-parser hermes` does not exist:
        # vLLM's FlexibleArgumentParser normalises '_'->'-', the real flag is
        # `--tool-call-parser`, and vLLM hard-errors on it unless
        # `--enable-auto-tool-choice` is also passed. Getting it wrong does not fail
        # loudly at the CLI -- the server starts, then dies in abstract_tool_parser.py
        # with "Failed to load module 'hermes' from hermes." because the name is treated
        # as a plugin PATH. First attempt at this fix brought up 1 lane of 4 that way.
        # This is the same spelling tt_kernel/packaging.py emits for the container.
        #
        # KEEP THE ENV ASSIGNMENTS WELDED TO `nohup` -- NO COMMENT MAY SIT BETWEEN THEM.
        # A `#` line after a trailing backslash ENDS the continuation: bash then parses
        # `VAR=x VAR2=y` as a standalone assignment, setting them as plain shell
        # variables in THIS shell and never exporting them to the child. The lane then
        # launches with no TT_VISIBLE_DEVICES and no single-die descriptor, so every
        # lane tries to open the whole 4-die mesh at once and all four hang until
        # vLLM's 600 s VLLM_ENGINE_READY_TIMEOUT_S fires. That is exactly what the
        # comment block above did when it was added: 4/4 lanes had worked repeatedly
        # before it, 0/4 after, and it read as a "startup flake".
        TT_VISIBLE_DEVICES="$i" \
        TT_MESH_GRAPH_DESC_PATH="$MESH_DESC" \
        nohup vllm serve "$WEIGHTS" \
            --served-model-name "$MODEL_NAME" \
            --max-num-seqs "$MAX_NUM_SEQS" \
            --block-size 64 \
            --max-model-len "$MAX_MODEL_LEN" \
            --additional-config '{"tt": {"trace_region_size": 1073741824}}' \
            --reasoning-parser qwen3 \
            --enable-auto-tool-choice --tool-call-parser hermes \
            --port "$port" \
            > "$(lane_log "$i")" 2>&1 &
        local lpid=$!
        echo "$lpid" > "$(lane_pidfile "$i")"
        echo "lane $i: die $i, port $port, pid $lpid, log $(lane_log "$i")"
        # Read the pinning back out of the CHILD's environment rather than trusting that
        # we wrote it. This is the only check that would have caught the comment-in-
        # continuation bug, which lost TT_VISIBLE_DEVICES silently and cost a full
        # 4-lane verification run.
        #
        # POLL, do not sample once. A prefix assignment is applied to the environment at
        # EXEC, not at fork, so for the moment between `&` and vllm actually starting,
        # /proc/<pid>/environ still shows THIS shell's environment with no
        # TT_VISIBLE_DEVICES in it. Checking immediately reports a false failure on a
        # perfectly good launch -- which it did, on the first run of this guard.
        local pinned=no
        for _ in $(seq 1 100); do
            if [ -r "/proc/$lpid/environ" ] \
               && tr '\0' '\n' < "/proc/$lpid/environ" | grep -qx "TT_VISIBLE_DEVICES=$i"; then
                pinned=yes; break
            fi
            kill -0 "$lpid" 2>/dev/null || break
            sleep 0.1
        done
        if [ "$pinned" != yes ]; then
            echo "serve_lanes: FATAL lane $i did not inherit TT_VISIBLE_DEVICES=$i;" >&2
            echo "             it would open every die instead of one. Killing it." >&2
            kill "$lpid" 2>/dev/null; rm -f "$(lane_pidfile "$i")"
            die "die pinning did not reach the lane process"
        fi
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

#!/usr/bin/env bash
# Frontier-Bench v0.1 (= terminal-bench@3.0.0, 74 tasks; the GPU/H100 tasks are excluded by default)
# with Harbor + Terminus-2 (AGENT=claude-code | oracle also work). Thin wrapper over tbench_run.sh
# (see its header for env: sampling, CPUS/MEM_MB, N_ATTEMPTS, TASKS/EXCLUDE_TASKS/LIMIT, GPU_TASKS=1).
#   frontierbench_run.sh v0.1 <label> <base_url> <model_id> <concurrency>
# Results: ~/ornith-fusion/runs/<label>/frontierbench/   Final line: RESULT frontierbench ...
set -euo pipefail
if [ $# -ne 5 ]; then sed -n '2,7p' "$0"; exit 2; fi
case "$1" in v0.1|0.1|3.0.0) ;; *) echo "variant must be v0.1"; exit 2;; esac
shift
exec "$(dirname "$0")/tbench_run.sh" frontier "$@"

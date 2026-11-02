#!/usr/bin/env bash
# SWE Atlas Codebase QnA (124 tasks, scaleapi/SWE-Atlas@49e4af3) with Harbor + mini-swe-agent 2.4.6 and
# SWE Atlas' qa config (step_limit 250); Ornith card: temp 1.0, top_p 0.95, 128K context, 5 runs averaged
# (N_ATTEMPTS=5). The rubric judge (Claude Opus 4.5 in the official verifier) is deferred to files:
# the run ends with JUDGE_PENDING <run>/judge/judge_inputs.jsonl; after judging:
#   python3 ~/evals/sweatlas/sweatlas_judge.py ingest <run_dir> <judgments.jsonl>
# Thin wrapper over tbench_run.sh (see its header for env).
#   sweatlas_run.sh qna <label> <base_url> <model_id> <concurrency>
# Results: ~/ornith-fusion/runs/<label>/sweatlas_qna/
set -euo pipefail
if [ $# -ne 5 ]; then sed -n '2,10p' "$0"; exit 2; fi
case "$1" in qna) ;; *) echo "variant must be qna"; exit 2;; esac
shift
exec "$(dirname "$0")/tbench_run.sh" sweatlas-qna "$@"

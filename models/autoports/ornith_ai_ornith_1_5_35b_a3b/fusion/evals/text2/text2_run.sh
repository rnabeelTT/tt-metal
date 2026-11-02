#!/usr/bin/env bash
# text2_run.sh <bench> <label> <base_url> <model_id> <concurrency>
# Model-card benchmarks (Qwen3.5-122B / Ornith-1.5 cards) against an OpenAI-compatible server.
# Output: ${RUNS_ROOT:-~/ornith-fusion/runs}/<label>/<bench>/ ; last line: RESULT <bench> <metric>=<value> n=<items>
# (preceded by JUDGE_PENDING <judge_inputs.jsonl> when an LLM judge is needed; then run judge_ingest.py).
# Env: MAX_TOKENS (32768) TEMPERATURE TOP_P TOP_K (defaults by model: Ornith-35B 1.0/0.95/20, others 0.6/0.95/20)
#      PRESENCE_PENALTY (unset) LIMIT (first N items, smoke) REQUEST_TIMEOUT (14400 s) MAX_RETRIES (6)
#      SAVE_REASONING (1) RUNS_ROOT
# Benches: see `text2_run.sh list`. Re-running the same command resumes (cached successful requests are reused).
# Search-agent benches (browsecomp browsecomp_zh widesearch seal0 hle_tools): SearXNG + webtools service
#   (~/evals/websearch/run_webtools.sh start; auto-started), agent env in common/search_agent.py (AGENT_MAX_TURNS
#   CTX_MODE CTX_TRIGGER FOLD_TARGET KEEP_REASONING EPISODES ...), SAMPLE_N=<k> fixed random subset (seed 1234).
set -euo pipefail
HERE=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
if [ "${1:-}" = list ]; then
  ls "$HERE/benches" | sed -n 's/\.\(py\|sh\)$//p' | grep -v '^_' | grep -v __init__ | sort -u; exit 0
fi
if [ $# -ne 5 ]; then sed -n '2,10p' "$0"; exit 2; fi
B=$1
if [ -f "$HERE/benches/$B.sh" ]; then exec bash "$HERE/benches/$B.sh" "$@"; fi
if [ -f "$HERE/benches/${B//-/_}.py" ]; then exec "$HERE/.venv/bin/python" -u "$HERE/run.py" "$@"; fi
echo "unknown bench $B; have:"; "$0" list; exit 2

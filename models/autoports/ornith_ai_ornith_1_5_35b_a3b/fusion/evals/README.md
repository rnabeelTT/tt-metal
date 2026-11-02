# Eval harness (snapshot of our own code under ~/evals)

Only the code we wrote: runners (`*_run.sh`), the text benchmark harness (`text2/`: benches, common, exact
continuation `extend_truncated.py`, judge ingest), Qwen text sets (`qwen_text/`), vision runner, web tools and the
SearXNG / sandbox setup (`websearch/`), simulated users (`usersim/`), Claude-agent judging (`judging/`), agent
adapters (`*_support/`, `claweval/`, `agent_proxy/`) and the fixed sample id lists (`samples/`, seed 1234).
Third-party harnesses (lm-evaluation-harness, Harbor, OpenHands benchmarks, SWE-bench, SciCode, lmms-eval,
VLMEvalKit, OmniDocBench, terminal-bench tasks, ...) and all datasets / outputs are not included: clone them at the
versions REF_EVALS.md names. Paths are the box's (`$HOME/evals`, `$HOME/ornith-fusion/runs`).

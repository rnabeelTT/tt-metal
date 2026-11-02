"""summarize.py <out_dir> <domain> [<domain> ...]
Reads <out_dir>/<domain>.json (tau2 Results), computes the official metrics with tau2.metrics.agent_metrics
(pass^k from the per-task success counts, success = reward == 1), prints per-domain lines and the final
  RESULT tau2 pass^1=<mean of per-domain pass^1, x100> n=<tasks>
(the single TAU2-Bench number on the Qwen3.5 card is taken to be the unweighted mean of the airline / retail /
telecom pass^1 -- inferred, the card does not say). Writes summary.json + request_counts.json.
"""
import json
import statistics as st
import sys
from pathlib import Path

from tau2.data_model.simulation import Results
from tau2.metrics.agent_metrics import compute_metrics, is_successful

out = Path(sys.argv[1])
domains = sys.argv[2:]
summary = {"bench": "tau2", "domains": {}}
p1s, n_tasks, harness_fail_total = [], 0, 0
counts = {}
for d in domains:
    p = out / f"{d}.json"
    if not p.exists():
        print(f"tau2 {d}: MISSING {p}")
        continue
    res = Results.load(p)
    sims = res.simulations
    if not sims:
        print(f"tau2 {d}: no simulations")
        continue
    m = compute_metrics(res)
    tasks = {s.task_id for s in sims}
    term = {}
    for s in sims:
        term[s.termination_reason.value] = term.get(s.termination_reason.value, 0) + 1
    harness_fail = sum(1 for s in sims if s.reward_info and "harness_exception" in str((s.reward_info.info or {}).get("note", "")))
    harness_fail_total += harness_fail
    agent_req = [sum(1 for msg in s.messages if msg.role == "assistant") - 1 for s in sims if s.messages]  # minus greeting
    user_req = [sum(1 for msg in s.messages if msg.role == "user") for s in sims if s.messages]
    counts[d] = {"simulations": len(sims), "agent_requests": sum(agent_req), "user_sim_requests": sum(user_req),
                 "agent_requests_per_sim": round(st.mean(agent_req), 2) if agent_req else None,
                 "user_sim_requests_per_sim": round(st.mean(user_req), 2) if user_req else None}
    p1 = m.pass_hat_ks.get(1, float("nan"))
    p1s.append(p1)
    n_tasks += len(tasks)
    trials = res.info.num_trials
    summary["domains"][d] = {
        "tasks": len(tasks), "trials": trials, "simulations": len(sims),
        "pass_hat_k": {f"pass^{k}": round(v * 100, 2) for k, v in m.pass_hat_ks.items()},
        "avg_reward": round(m.avg_reward * 100, 2),
        "success_rate": round(100 * sum(is_successful(s.reward_info.reward) for s in sims if s.reward_info) / len(sims), 2),
        "termination_reasons": term, "harness_exceptions": harness_fail,
        "agent_llm": res.info.agent_info.llm, "agent_llm_args": {k: v for k, v in (res.info.agent_info.llm_args or {}).items() if k != "api_key"},
        "user_llm": res.info.user_info.llm,
    }
    pk = " ".join(f"pass^{k}={v * 100:.2f}" for k, v in m.pass_hat_ks.items())
    print(f"tau2 {d}: tasks={len(tasks)} trials={trials} sims={len(sims)} {pk} terminations={term}"
          + (f" HARNESS_EXCEPTIONS={harness_fail}" if harness_fail else ""))
summary["request_counts"] = counts
value = round(100 * sum(p1s) / len(p1s), 2) if p1s else None
summary.update(metric="pass^1 (mean over domains)", value=value, n=n_tasks, harness_exceptions=harness_fail_total)
(out / "summary.json").write_text(json.dumps(summary, indent=2))
(out / "request_counts.json").write_text(json.dumps(counts, indent=2))
print("request counts: " + json.dumps(counts))
if harness_fail_total:
    print(f"WARNING: {harness_fail_total} simulations hit harness exceptions (scored 0); rerun with RETRY_FAILED=1 to retry them")
line = f"RESULT tau2 pass^1={value:.2f} n={n_tasks}" if value is not None else f"RESULT tau2 pass^1=nan n={n_tasks}"
(out / "result.txt").write_text(line + "\n")
print(line)

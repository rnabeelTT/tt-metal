"""Launch the official `tau2 run` CLI (pinned tau2-bench) with one harness-level safety net.

  .venv/bin/python run_tau2.py <tau2 run args...>

Official behaviour: an uncaught exception in any simulation (server timeout after litellm's retries, a tool call
whose `arguments` is not valid JSON, ...) aborts the whole `tau2 run`, and re-running resumes from the save file.
Here each simulation is retried up to SIM_RETRIES (default 2) extra times; if it still raises, it is recorded as a
reward-0 simulation with termination_reason=agent_error and reward_info.info.note="harness_exception: ..." so the
run completes. summarize.py counts these separately (they are NOT silently mixed into the score: the RESULT line is
followed by a warning, and `RETRY_FAILED=1 tau2.sh ...` drops them from the save file so they are re-run).
Nothing else is changed: prompts, tools, user simulator, orchestrator and evaluator are the pinned upstream code.
"""
import os
import sys
import traceback
import uuid

import tau2.run as R
from tau2.data_model.simulation import RewardInfo, SimulationRun, TerminationReason
from tau2.utils.utils import get_now

SIM_RETRIES = int(os.environ.get("SIM_RETRIES", 2))
_orig_run_task = R.run_task


def run_task_with_retries(*a, **kw):
    task = kw.get("task") if "task" in kw else a[1]
    last = None
    for attempt in range(SIM_RETRIES + 1):
        try:
            return _orig_run_task(*a, **kw)
        except Exception as e:  # noqa: BLE001
            last = e
            print(f"[tau2 harness] task {task.id} attempt {attempt + 1} raised {type(e).__name__}: {str(e)[:300]}",
                  file=sys.stderr, flush=True)
    now = get_now()
    return SimulationRun(
        id=str(uuid.uuid4()), task_id=task.id, start_time=now, end_time=now, duration=0.0,
        termination_reason=TerminationReason.AGENT_ERROR,
        reward_info=RewardInfo(reward=0.0, info={"note": f"harness_exception: {type(last).__name__}: {str(last)[:2000]}",
                                                 "traceback": "".join(traceback.format_exception(last))[-4000:]}),
        messages=[], seed=kw.get("seed"),
    )


R.run_task = run_task_with_retries

if os.environ.get("TAU2_STUB_TASK_TAG") == "1":
    # SELF-TEST ONLY (stub_tau2.py gold mode): telecom tasks share identical user scenarios, so the stub user
    # cannot tell them apart; append the task id to the user simulator's system prompt. Never set for real runs.
    import threading

    from tau2.user.user_simulator import UserSimulator

    _cur = threading.local()
    _inner = R.run_task

    def run_task_tagged(*a, **kw):
        _cur.tag = f"{kw.get('domain', a[0] if a else '?')}/{(kw.get('task') or a[1]).id}"
        return _inner(*a, **kw)

    _orig_sp = UserSimulator.system_prompt.fget
    UserSimulator.system_prompt = property(
        lambda self: _orig_sp(self) + f"\n<stub_task_id>{getattr(_cur, 'tag', '?')}</stub_task_id>")
    R.run_task = run_task_tagged

if __name__ == "__main__":
    from tau2.cli import main

    sys.argv = ["tau2", "run"] + sys.argv[1:]
    main()

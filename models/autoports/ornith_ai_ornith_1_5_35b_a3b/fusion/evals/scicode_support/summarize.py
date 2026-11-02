"""Summarize a SciCode run from the per-step pass/fail logs and the inspect log.

usage: summarize.py <out_dir> <split> <with_background 0|1> <inspect_log_dir> [limit]

Reports two rulers over the same per-step results:
  * original  = eval/scripts/test_generated_code.py: only generated steps count
                (test: 288 steps / 65 problems; a problem is resolved when all of
                its generated steps pass, so 13, 62 and 76 are resolvable).
  * inspect   = eval/inspect_ai/scicode.py scorer: the 3 pre-filled steps stay in
                the denominator (291) and problems 13, 62, 76 can never resolve
                (upstream issue #63).
Prints one line starting with SCICODE and writes summary.json.
"""

import json
import os
import sys
from pathlib import Path

from datasets import load_dataset

PREFILLED = {("13", 5), ("62", 0), ("76", 2)}
DUMMY_TAIL = "\nprint('Hello, World!')\n"


def main() -> None:
    out, split, bg, log_dir = Path(sys.argv[1]), sys.argv[2], sys.argv[3] == "1", Path(sys.argv[4])
    bgdir = "with_background" if bg else "without_background"
    model_dirs = [p for p in (out / "eval_results").iterdir() if p.is_dir()]
    assert len(model_dirs) == 1, f"expected one model dir under {out/'eval_results'}: {model_dirs}"
    mdir = model_dirs[0]
    logs = mdir / "evaluation_logs" / bgdir
    code = mdir / "generated_code" / bgdir

    data = load_dataset("SciCode1/SciCode", split=split)
    if len(sys.argv) > 5 and sys.argv[5]:  # smoke runs (--limit): score only problems that ran
        ran = {f.name.split(".")[0] for f in code.glob("*.py")}
        data = data.filter(lambda p: p["problem_id"] in ran)
    gen_steps = gen_pass = all_steps = 0
    solved_orig, solved_insp, missing, timeouts, dummy = [], [], [], 0, 0
    per_problem = {}
    for prob in data:
        pid = prob["problem_id"]
        n_pass = n_gen = 0
        for idx, st in enumerate(prob["sub_steps"]):
            all_steps += 1
            if (pid, idx) in PREFILLED:
                continue
            n_gen += 1
            f = logs / f"{st['step_number']}.log"
            res = f.read_text().splitlines()[0] if f.is_file() else "missing"
            if res == "missing":
                missing.append(st["step_number"])
            timeouts += res == "time out"
            n_pass += res == "pass"
            g = code / f"{pid}.{idx + 1}.py"
            if g.is_file() and g.read_text(encoding="utf-8").endswith(DUMMY_TAIL):
                dummy += 1
        gen_steps += n_gen
        gen_pass += n_pass
        if n_pass == n_gen:
            solved_orig.append(pid)
            if n_gen == len(prob["sub_steps"]):
                solved_insp.append(pid)
        per_problem[pid] = f"{n_pass}/{n_gen}"

    # Response stats from the inspect log (stop reasons, empty answers, tokens).
    stats = {"requests": 0, "max_tokens_stops": 0, "empty_content": 0, "model_errors": 0,
             "output_tokens": 0, "reasoning_tokens": 0}
    try:
        from inspect_ai.log import list_eval_logs, read_eval_log

        logs_found = list_eval_logs(str(log_dir))
        if logs_found:
            log = read_eval_log(logs_found[0])
            for s in log.samples or []:
                for ev in s.events:
                    if getattr(ev, "event", None) != "model":
                        continue
                    stats["requests"] += 1
                    if ev.error:
                        stats["model_errors"] += 1
                        continue
                    o = ev.output
                    stats["max_tokens_stops"] += o.stop_reason == "max_tokens"
                    stats["empty_content"] += not (o.completion or "").strip()
                    if o.usage:
                        stats["output_tokens"] += o.usage.output_tokens or 0
                        stats["reasoning_tokens"] += o.usage.reasoning_tokens or 0
    except Exception as e:  # stats are informational only
        stats["log_error"] = repr(e)

    n_prob = len(data)
    summary = {
        "split": split,
        "with_background": bg,
        "prefilled_verbatim": os.environ.get("SCICODE_PREFILLED_VERBATIM", "0") == "1",
        "subproblem_acc": gen_pass / gen_steps,
        "subproblem_pass": gen_pass,
        "subproblem_total": gen_steps,
        "main_acc": len(solved_orig) / n_prob,
        "main_solved": solved_orig,
        "main_total": n_prob,
        "inspect_ruler": {
            "subproblem_acc": gen_pass / all_steps,
            "subproblem_total": all_steps,
            "main_acc": len(solved_insp) / n_prob,
        },
        "missing_step_logs": missing,
        "test_timeouts": timeouts,
        "dummy_responses_api_failures": dummy,
        "response_stats": stats,
        "per_problem": per_problem,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"SCICODE split={split} background={'with' if bg else 'without'} "
        f"prefilled_verbatim={os.environ.get('SCICODE_PREFILLED_VERBATIM', '0')} "
        f"subproblem={100 * summary['subproblem_acc']:.1f} ({gen_pass}/{gen_steps}) "
        f"main={100 * summary['main_acc']:.1f} ({len(solved_orig)}/{n_prob}) "
        f"inspect_ruler_sub={100 * gen_pass / all_steps:.1f} ({gen_pass}/{all_steps}) "
        f"inspect_ruler_main={100 * len(solved_insp) / n_prob:.1f} "
        f"requests={stats['requests']} max_tokens_stops={stats['max_tokens_stops']} "
        f"empty={stats['empty_content']} api_fail={dummy} timeouts={timeouts} missing={len(missing)}"
    )


if __name__ == "__main__":
    main()

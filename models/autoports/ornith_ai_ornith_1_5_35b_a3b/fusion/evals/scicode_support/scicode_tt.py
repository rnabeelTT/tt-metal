"""Thin inspect_ai task around the official SciCode inspect_ai port.

Prompts, the per-step solver loop, code extraction (extract_python_script /
get_function_from_code), the 3 pre-filled steps (13.6, 62.1, 76.3), the test
scripts and the 1800 s per-step timeout all come unchanged from
SciCode/eval/inspect_ai/scicode.py (loaded below by path).

One change: the official scorer is `async def` but runs every test
subprocess synchronously, which freezes inspect's event loop (no new model
requests are dispatched) for as long as any problem's tests run (up to
1800 s per step). Here the same ScicodeEvaluator.test_code is run in a worker
thread, so generation for other problems keeps going. Scoring semantics are
identical.

Must be run with cwd such that ../data is SciCode/eval/data (the official
module reads its templates and the pre-filled step code from there).
"""

import asyncio
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

from inspect_ai import Task, task
from inspect_ai.dataset import hf_dataset
from inspect_ai.scorer import Score, Target, mean, scorer
from inspect_ai.solver import TaskState

_OFFICIAL = Path(
    os.environ.get("SCICODE_REPO", Path.home() / "evals" / "SciCode"),
    "eval",
    "inspect_ai",
    "scicode.py",
)
_spec = importlib.util.spec_from_file_location("scicode_inspect_official", _OFFICIAL)
official = importlib.util.module_from_spec(_spec)
sys.modules["scicode_inspect_official"] = official
_spec.loader.exec_module(official)

# Opt-in (SCICODE_PREFILLED_VERBATIM=1, default off = official behaviour):
# upstream issues #49/#59. The curated code for the pre-filled steps 13.6 and
# 62.1 defines classes, but get_function_from_code(extract_function_name(header))
# resolves the class header to its first method and injects only a bare
# `__init__`, so later steps of problems 13 and 62 cannot pass. With the flag on,
# the three curated files are injected verbatim. Model-generated code is not
# affected either way (the solver keeps it in memory, unextracted).
if os.environ.get("SCICODE_PREFILLED_VERBATIM") == "1":
    _data = _OFFICIAL.parent.parent / "data"
    _prefilled = {(_data / f).read_text(encoding="utf-8") for f in ("13.6.txt", "62.1.txt", "76.3.txt")}
    _orig_get_function_from_code = official.get_function_from_code

    def _get_function_from_code(code_string, function_name):
        if code_string in _prefilled:
            return code_string
        return _orig_get_function_from_code(code_string, function_name)

    official.get_function_from_code = _get_function_from_code

_score_slots = asyncio.Semaphore(int(os.environ.get("SCICODE_SCORE_CONCURRENCY", "8")))
_tmpdir_lock = asyncio.Lock()


@scorer(metrics=[{"Problem Correctness": [mean()]}, official.sub_problem_correctness()])
def scicode_scorer_threaded(**params: dict[str, Any]):
    async def score(state: TaskState, target: Target):
        model_name = str(state.model).replace("/", "-")
        evaluator = official.ScicodeEvaluator(
            h5py_file=params["h5py_file"],
            code_dir=Path(params["output_dir"], model_name),
            log_dir=Path(params["output_dir"], model_name),
            with_background=params["with_background"],
        )
        async with _score_slots:
            # test_code names its scratch dir tmp_{time.time()}; space thread
            # starts so two problems can never get the same name.
            async with _tmpdir_lock:
                await asyncio.sleep(0.01)
            problem_correct, total_correct, total_steps = await asyncio.to_thread(
                evaluator.test_code, state.metadata
            )
        return Score(
            value={
                "Problem Correctness": problem_correct,
                "Total Correct": total_correct,
                "Total Steps": total_steps,
            }
        )

    return score


@task
def scicode_tt(
    split: str = "test",
    output_dir: str = "./tmp",
    with_background: bool = False,
    h5py_file: str = "../data/test_data.h5",
    mode: str = "normal",
):
    dataset = hf_dataset(
        "SciCode1/SciCode",
        split=split,
        sample_fields=official.record_to_sample,
    )
    # Longest step chains first: a problem's steps are sequential, so starting
    # the 14-step chains last would leave them alone on the server at the end.
    # Ordering does not change any prompt or score.
    dataset.sort(reverse=True, key=lambda s: len(s.metadata["sub_steps"]))
    return Task(
        dataset=dataset,
        solver=official.scicode_solver(
            output_dir=output_dir,
            with_background=with_background,
            mode=mode,
        ),
        scorer=scicode_scorer_threaded(
            output_dir=output_dir,
            with_background=with_background,
            h5py_file=h5py_file,
        ),
    )

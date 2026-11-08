"""Greedy (temperature 0) outputs of 8 fixed ~600-1000-token code prompts against the server on :20000, sent one at a time
(single-user prefill) and all at once (the batched prefill's wide 1024-bucket pairs). Prints whether each concurrent
output matches its sequential one and writes runs/<label>/greedy_long.json; `compare A B` compares two runs.

    python greedy_long.py run <label>
    python greedy_long.py compare <labelA> <labelB>
"""

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from greedy import RUNS, first_diff  # noqa: E402
import greedy  # noqa: E402

SRC = Path(__file__).resolve().parents[2] / "tt"
FILES = ["generator.py", "model.py", "prefill_batched.py", "mtp_multi.py", "generator_vllm.py", "model.py", "generator.py",
         "prefill_batched.py"]
CHARS = [2000, 2300, 2600, 2900, 2100, 2800, 2400, 3100]  # ~600-1000 tokens of Python each (all in the 1024 bucket)
PROMPTS = []
for i, (f, n) in enumerate(zip(FILES, CHARS)):
    text = (SRC / f).read_text()
    off = (i * 7919) % max(1, len(text) - n)
    PROMPTS.append(f"Here is part of a Python file:\n\n```python\n{text[off:off + n]}\n```\n\nExplain what this code does, "
                   f"step by step.")


def run(label):
    seq = [greedy.ask(p) for p in PROMPTS]
    with ThreadPoolExecutor(len(PROMPTS)) as ex:
        par = list(ex.map(greedy.ask, PROMPTS))
    out = RUNS / label
    out.mkdir(parents=True, exist_ok=True)
    (out / "greedy_long.json").write_text(json.dumps({"sequential": seq, "concurrent": par}, indent=1))
    same = sum(first_diff(a, b) is None for a, b in zip(seq, par))
    print(f"{label}: long prompts concurrent vs sequential {same}/{len(PROMPTS)} identical")


def compare(la, lb):
    A = json.loads((RUNS / la / "greedy_long.json").read_text())
    B = json.loads((RUNS / lb / "greedy_long.json").read_text())
    for mode in ("sequential", "concurrent"):
        same = sum(first_diff(x, y) is None for x, y in zip(A[mode], B[mode]))
        print(f"{mode}: {same}/{len(PROMPTS)} identical")


if __name__ == "__main__":
    {"run": lambda: run(sys.argv[2]), "compare": lambda: compare(sys.argv[2], sys.argv[3])}[sys.argv[1]]()

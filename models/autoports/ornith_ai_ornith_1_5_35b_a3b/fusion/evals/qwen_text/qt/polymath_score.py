"""Runs in .venv-polymath (the official repo's pinned deps). stdin: JSONL
{id, content, answer}; stdout: JSONL {id, extracted, correct}. Extraction and
equality are the official QwenLM/PolyMath eval/run_eval.py ones (first \\boxed{}
of the answer with spaces removed; scripts.math_equal)."""
import json
import sys
import warnings

warnings.filterwarnings("ignore")
REPO = sys.argv[1]  # .../PolyMath/eval
sys.path.insert(0, REPO)
import bundled.symeval as _s  # noqa: E402  upstream ships symeval without __init__ exports
import bundled.symeval.core as _c  # noqa: E402
_s.EvaluatorMath = _c.EvaluatorMath
from run_eval import extract_boxed_content  # noqa: E402
from scripts import math_equal  # noqa: E402

for line in sys.stdin:
    r = json.loads(line)
    ex = extract_boxed_content(r["content"] or "")
    ex = ex[0] if ex else None
    def eq(p, g):
        try:
            return bool(math_equal(p, g))
        except Exception:  # noqa: BLE001
            return False

    raw_ok = eq(ex, r["answer"])            # literally the official call
    # 9.5% of PolyMath references are wrapped in $...$ (e.g. "$e-1$"), which the official
    # math_equal never matches against an unwrapped \boxed{e-1}. Strip the math delimiters
    # (and, like the official prediction, spaces) from the reference and retry.
    g = r["answer"].strip().strip("$").strip()
    ok = raw_ok or (ex is not None and (eq(ex.strip("$"), g) or eq(ex.strip("$"), g.replace(" ", ""))))
    print(json.dumps({"id": r["id"], "extracted": ex, "correct": ok, "correct_official_raw": raw_ok},
                     ensure_ascii=False), flush=True)

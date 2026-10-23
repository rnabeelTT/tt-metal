"""rescore_mbpp.py <samples_mbpp_instruct_*.jsonl>: pass@1 of an mbpp_instruct run with a chat-aware code extractor.

mbpp_instruct's extractor prepends its gen_prefix fence ("```python\\n", an assistant prefill the chat-completions API does
not continue), so a chat answer's own ```python block extracts as "" and the task scores 0 through local-chat-completions.
This re-extracts the first ```python block of each stored answer (else the whole answer) and runs the stock tests through
lm-eval's own mbpp pass_at_1 (HF code_eval: a subprocess per sample, timeout). Run with lm-eval's venv and HF_ALLOW_CODE_EVAL=1.
"""

import json
import re
import sys

from lm_eval.tasks.mbpp.utils import pass_at_1

FENCE = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL)


def main():
    rows = [json.loads(l) for l in open(sys.argv[1])]
    preds, refs = [], []
    for r in rows:
        resp = r["resps"][0][0]
        blocks = FENCE.findall(resp)
        preds.append([blocks[0] if blocks else resp])
        refs.append("\n".join(r["doc"]["test_list"]))
    score = sum(pass_at_1(r, [p]) for r, p in zip(refs, preds)) / len(rows)  # per sample, as the task scores it
    print(f"mbpp (chat extractor) pass@1 {score:.4f} over {len(rows)}")


if __name__ == "__main__":
    main()

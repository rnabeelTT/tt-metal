import re

from lm_eval.tasks.humaneval.utils import pass_at_k  # noqa: F401  (the stock metric)

FENCE = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL)


def _prediction(resp, doc):
    blocks = FENCE.findall(resp)
    code = blocks[0] if blocks else resp
    header = doc["prompt"][: doc["prompt"].find("def ")] if "def " in doc["prompt"] else ""
    if re.search(rf"^\s*def\s+{re.escape(doc['entry_point'])}\s*\(", code, re.MULTILINE):
        return header + "\n" + code  # the prompt's imports / helpers + the model's full definition
    return doc["prompt"] + code


def build_predictions_chat(resps, docs):
    return [[_prediction(r, doc) for r in resp] for resp, doc in zip(resps, docs)]

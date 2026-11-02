"""BrowseComp (OpenAI; Qwen3.5-122B card 63.8, Ornith-1.5-35B 67.6, Ornith-1.5-122B 78.5) as a search agent.

Data: 1,266 questions, https://openaipublic.blob.core.windows.net/simple-evals/browse_comp_test_set.csv
(~/evals/data/browsecomp/browse_comp_test_set.csv), XOR-encrypted with the row's canary (decrypt as simple-evals).
Ids: bc_0000.. in CSV order.
Prompt: simple-evals browsecomp_eval.py QUERY_TEMPLATE (@652c89d) as the user turn; system = AGENT_SYSTEM
(Tongyi-style deep-research preamble + date); tools search + visit (common/search_agent.py), CTX_MODE fold.
Judge: simple-evals GRADER_TEMPLATE (official grader gpt-4.1-2025-04-14 in simple-evals; Claude agents here),
verdict {extracted_final_answer, reasoning, correct: yes|no, confidence}.
Metric: accuracy over all questions (no final answer = wrong) + calibration error (BrowseComp paper reports it).
Env: SAMPLE_N (fixed random subset, seed 1234; e.g. 200), LIMIT, AGENT_MAX_TURNS (100), CTX_MODE (fold), see
common/search_agent.py.
"""
import base64
import hashlib

import pandas as pd

from benches._search import AGENT_SYSTEM, HLE_STYLE_SCHEMA, accuracy_ingest, run_and_write, select, stub_gold
from common.search_agent import AgentCfg

BENCH_NAME = "browsecomp"
DATA = "~/evals/data/browsecomp/browse_comp_test_set.csv"

QUERY_TEMPLATE = """
{Question}

Your response should be in the following format:
Explanation: {{your explanation for your final answer}}
Exact Answer: {{your succinct, final answer}}
Confidence: {{your confidence score between 0% and 100% for your answer}}
""".strip()

GRADER_TEMPLATE = """
Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.


confidence: The extracted confidence score between 0|\\%| and 100|\\%| from [response]. Put 100 if there is no confidence score available.
""".strip()

INSTRUCTIONS = """BrowseComp judging (simple-evals browsecomp_eval.py; official grader gpt-4.1-2025-04-14, which parses
`correct: (yes|no)` from free text). Send judge_prompt as the only user message; return {"id", "extracted_final_answer",
"reasoning", "correct": "yes"|"no", "confidence": 0-100}."""


def derive_key(password: str, length: int) -> bytes:
    key = hashlib.sha256(password.encode()).digest()
    return key * (length // len(key)) + key[: length % len(key)]


def decrypt(ciphertext_b64: str, password: str) -> str:
    enc = base64.b64decode(ciphertext_b64)
    return bytes(a ^ b for a, b in zip(enc, derive_key(password, len(enc)))).decode()


def load_items():
    import os

    df = pd.read_csv(os.path.expanduser(DATA))
    return [{"id": f"bc_{i:04d}", "question": decrypt(r["problem"], r["canary"]), "answer": decrypt(r["answer"], r["canary"]),
             "topic": r.get("problem_topic")} for i, r in df.iterrows()]


def user_msg(it):
    return {"role": "user", "content": QUERY_TEMPLATE.format(Question=it["question"])}


def main(cfg):
    items = select(load_items(), cfg)
    acfg = AgentCfg.from_env(max_turns=100, ctx_mode="fold")
    stub_gold(items, user_msg, lambda it: f"Explanation: stub.\nExact Answer: {it['answer']}\nConfidence: 90%")
    run_and_write(cfg, acfg, BENCH_NAME, items,
                  messages_fn=lambda it: [{"role": "system", "content": AGENT_SYSTEM}, user_msg(it)],
                  question_fn=lambda it: it["question"],
                  prompt_fn=lambda it, ans: GRADER_TEMPLATE.format(question=it["question"], correct_answer=it["answer"], response=ans),
                  schema=HLE_STYLE_SCHEMA, instructions=INSTRUCTIONS, meta_fn=lambda it: {"topic": it["topic"]})


def ingest(cfg, verdicts, missing):
    accuracy_ingest(cfg, verdicts, missing, lambda v: str(v.get("correct", "")).strip().lower().startswith("yes"))

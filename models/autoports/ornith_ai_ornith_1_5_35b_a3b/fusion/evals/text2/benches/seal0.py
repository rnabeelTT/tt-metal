"""Seal-0 (SealQA; Qwen3.5-122B card 44.1) as a search agent.

Data: 111 questions, HF vtllms/sealqa seal-0.parquet (~/evals/data/seal0). Ids: seal0_000.. in file order.
Prompt: the question as the user turn (SealQA evaluates search-augmented models on the bare question); system =
AGENT_SYSTEM; tools search + visit; CTX_MODE fold. Note: many Seal-0 answers are time-sensitive ("fast-changing");
the gold answers are from the dataset release while the agent searches today's web (AGENT_DATE in the system prompt).
Judge: the official SealQA grading notebook GRADER_TEMPLATE (SimpleQA-style A/B/C; grading model gpt-4o-mini by
default), from https://colab.research.google.com/drive/1fJunutgxNaiU6-Met_wzyEdG-tXQLfrM (copy in
~/evals/src/sealqa_grader/grader.ipynb). Verdict {"grade": "A"|"B"|"C"}.
Metric: accuracy = share of A (CORRECT) over all questions; incorrect / not_attempted also reported.
"""
import os

import pandas as pd

from benches._search import AGENT_SYSTEM, accuracy_ingest, run_and_write, select, stub_gold
from common.search_agent import AgentCfg

BENCH_NAME = "seal0"
DATA = "~/evals/data/seal0/seal-0.parquet"

GRADER_TEMPLATE = """
Your job is to look at a question, a gold target, and a predicted answer, and then assign a grade of either ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"].
First, I will give examples of each grade, and then you will grade a new example.


The following are examples of CORRECT predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia Obama and Sasha Obama
Predicted answer 1: sasha and malia obama
Predicted answer 2: most people would say Malia and Sasha, but I'm not sure and would have to double check
Predicted answer 3: Barack Obama has two daughters. Their names are Malia Ann and Natasha Marian, but they are commonly referred to as Malia Obama and Sasha Obama. Malia was born on July 4, 1998, and Sasha was born on June 10, 2001.
```
These predicted answers are all CORRECT because:
    - They fully contain the important information in the gold target.
    - They do not contain any information that contradicts the gold target.
    - Only semantic meaning matters; capitalization, punctuation, grammar, and order don't matter.
    - Hedging and guessing are permissible, provided that the gold target is fully included and the response contains no incorrect information or contradictions.


The following are examples of INCORRECT predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia and Sasha
Predicted answer 1: Malia.
Predicted answer 2: Malia, Sasha, and Susan.
Predicted answer 3: Barack Obama does not have any children.
Predicted answer 4: I think it's either Malia and Sasha. Or it could be Malia and Jackie. Or it could be Joey and Malia.
Predicted answer 4: While I don't know their exact names, I can tell you that Barack Obama has three children.
Predicted answer 5: It's possible you may mean Betsy and Olivia. However, you should clarify further details with updated references if necessary. Is that the correct answer?
Predicted answer 6: It may be the case that Obama's child is named James. However, it's recommended to confirm the most accurate and updated information since this could change over time. This model may not always reflect the most current information.
```
These predicted answers are all INCORRECT because:
    - A factual statement in the answer contradicts the gold target. Incorrect statements that have some hedging (e.g., "it is possible that", "although i'm not sure, i think") are also considered incorrect.


The following are examples of NOT_ATTEMPTED predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia and Sasha
Predicted answer 1: I don't know.
Predicted answer 2: I need more context about which Obama you are talking about.
Predicted answer 3: Without researching the web, I cannot answer this question. However, I can tell you that Barack Obama has two children.
Predicted answer 4: Barack Obama has two children. I know that one of them is Malia, but I'm not sure about the other one.
```
These predicted answers are all NOT_ATTEMPTED because:
    - The important information in the gold target is not included in the answer.
    - No statements in the answer contradict the gold target.


Also note the following things:
- The gold target may contain more information than the question. In such cases, the predicted answer only needs to contain the information that is in the question.
    - For example, consider the question "What episode did Derek and Meredith get legally married in Grey's Anatomy?" with gold target "Season 7, Episode 20: White Wedding". Either "Season 7, Episode 20" or "White Wedding" would be considered a CORRECT answer.
- Do not punish predicted answers if they omit information that would be clearly inferred from the question.
    - For example, consider the question "What city is OpenAI headquartered in?" and the gold target "San Francisco, California". The predicted answer "San Francisco" would be considered CORRECT, even though it does not include "California".
    - Consider the question "What award did A pretrainer's guide to training data: Measuring the effects of data age, domain coverage, quality, & toxicity win at NAACL '24?", the gold target is "Outstanding Paper Award". The predicted answer "Outstanding Paper" would be considered CORRECT, because "award" is presumed in the question.
- Do not give credit for an answer if it contains any internal inconsistency.
    - For example, consider the question: "How many NBA players have scored 60 or more points in a regular season game since 2024?" with the gold answer "8". A response is INCORRECT if it states "8 players" but lists 7 or 9, or if it initially says "8 players" but later contradicts this by concluding 7 or 9.


Here is a new example. Simply reply with either CORRECT, INCORRECT, NOT ATTEMPTED. Don't apologize or correct yourself if there was a mistake; we are just trying to grade the answer.
```
Question: {question}
Gold target: {target}
Predicted answer: {predicted_answer}
```

Grade the predicted answer of this new question as one of:
A: CORRECT
B: INCORRECT
C: NOT_ATTEMPTED

Just return the letters "A", "B", or "C", with no text around it.
""".strip()

SCHEMA = {"type": "object", "properties": {"grade": {"enum": ["A", "B", "C"]}}, "required": ["grade"]}
INSTRUCTIONS = """Seal-0 grading (official SealQA notebook; grading model gpt-4o-mini, temperature 0; the first A/B/C in the
reply is taken, default C). Send judge_prompt as the only user message; return {"id", "grade": "A"|"B"|"C"}."""


def load_items():
    df = pd.read_parquet(os.path.expanduser(DATA))
    return [{"id": f"seal0_{i:03d}", "question": r["question"], "answer": r["answer"], "freshness": r["freshness"],
             "topic": r["topic"]} for i, r in df.iterrows()]


def user_msg(it):
    return {"role": "user", "content": it["question"]}


def main(cfg):
    items = select(load_items(), cfg)
    acfg = AgentCfg.from_env(max_turns=100, ctx_mode="fold")
    stub_gold(items, user_msg, lambda it: f"The answer is {it['answer']}.")
    run_and_write(cfg, acfg, BENCH_NAME, items,
                  messages_fn=lambda it: [{"role": "system", "content": AGENT_SYSTEM}, user_msg(it)],
                  question_fn=lambda it: it["question"],
                  prompt_fn=lambda it, ans: GRADER_TEMPLATE.format(question=it["question"], target=it["answer"], predicted_answer=ans),
                  schema=SCHEMA, instructions=INSTRUCTIONS,
                  meta_fn=lambda it: {"freshness": it["freshness"], "topic": it["topic"]})


def _grade(v):
    import re

    m = re.search(r"(A|B|C)", str(v.get("grade", "")))
    return m.group(0) if m else "C"


def ingest(cfg, verdicts, missing):
    def extra(rows):
        n = len(rows) or 1
        g = [_grade(r.get("verdict") or {}) for r in rows]
        return {"incorrect": round(100 * g.count("B") / n, 2), "not_attempted": round(100 * g.count("C") / n, 2)}

    accuracy_ingest(cfg, verdicts, missing, lambda v: _grade(v) == "A", calib=False, extra=extra)

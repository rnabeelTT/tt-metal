"""BrowseComp-ZH (Qwen3.5-122B card 69.9) as a search agent.

Data: 289 questions, PALIN2018/BrowseComp-ZH data/browsecomp-zh-encrypted.xlsx (@86abe63; copy in
~/evals/data/browsecomp_zh/), XOR-decrypted per row with its `canary` (official browsecomp-zh-decrypt.py logic).
Ids: bczh_000.. in file order.
Prompt: the official SYSTEM_PROMPT_CN answer-format block (Explanation / Exact Answer / Confidence) appended to the
question in the user turn; its first sentence ("answer from your own knowledge, don't refuse") is for tool-less
models and is dropped. System = AGENT_SYSTEM; tools search + visit; queries with CJK text go to the zh engine set
(baidu, sogou, 360search; fallback duckduckgo, yahoo) -- the paper validated questions on Baidu / Bing(CN) / Google.
Judge: official JUDGE_PROMPT_CN (run.py sends it with system "you are a helpful assistant!"; official judge GPT-4o).
Metric: accuracy (+ the repo's 5-bin calibration error, run_acc_calibration_error.py).
"""
import os

import pandas as pd

from benches._search import AGENT_SYSTEM, HLE_STYLE_SCHEMA, accuracy_ingest, run_and_write, select, stub_gold
from benches.browsecomp import decrypt
from common.search_agent import AgentCfg

BENCH_NAME = "browsecomp_zh"
DATA = "~/evals/data/browsecomp_zh/browsecomp-zh-encrypted.xlsx"

FORMAT_CN = """你的回复应遵循以下格式：
Explanation: {你对最终答案的解释}
Exact Answer: {你简洁的最终答案}
Confidence: {你对答案的置信度得分在 0% 到 100% 之间}"""

JUDGE_PROMPT_CN = """根据以下精确且明确的[response]，判断以下对[question]的[correct_answer]是否正确。

[question]:  {question}

[response]:  {response}

您的判断必须符合以下指定的格式和标准：

extracted_final_answer: 从[response]中提取的最终准确答案。如果无法从答案中提取出准确的最终答案，则将提取的答案填写为"None"。

[correct_answer]: {correct_answer}

reasoning: 根据[correct_answer]解释提取的最终答案正确或错误的原因， 仅关注[correct_answer]和提取的最终答案之间是否存在有意义的差异。请勿评论问题的任何背景，请勿尝试解决问题，请勿争论任何与[correct_answer]不同的答案，仅关注答案是否匹配。

correct: 如果提取的最终答案与上面给出的[correct_answer]相符，或者在数值问题的误差范围内，则回答"yes"。否则，例如，如果存在任何不一致、歧义、不等同，或者提取的答案不正确，则回答"no"。

confidence: 从[response]中提取的置信度分数，介于0% 到100% 之间。如果没有可用的置信度分数，则填写100%。

"""

INSTRUCTIONS = """BrowseComp-ZH judging (official run.py: GPT-4o, system "you are a helpful assistant!", user = judge_prompt
(JUDGE_PROMPT_CN), output parsed for extracted_final_answer / reasoning / correct / confidence). Return {"id",
"extracted_final_answer", "reasoning", "correct": "yes"|"no", "confidence": 0-100}."""

BINS = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 101)]


def load_items():
    df = pd.read_excel(os.path.expanduser(DATA))
    out = []
    for i, r in df.iterrows():
        pw = r["canary"]
        dec = {c: (decrypt(r[c], pw) if pd.notnull(r[c]) else "") for c in ("Topic", "Question", "Answer")}
        out.append({"id": f"bczh_{i:03d}", "question": dec["Question"], "answer": dec["Answer"], "topic": dec["Topic"]})
    return out


def user_msg(it):
    return {"role": "user", "content": it["question"] + "\n\n" + FORMAT_CN}


def main(cfg):
    items = select(load_items(), cfg)
    acfg = AgentCfg.from_env(max_turns=100, ctx_mode="fold")
    stub_gold(items, user_msg, lambda it: f"Explanation: stub.\nExact Answer: {it['answer']}\nConfidence: 90%")
    run_and_write(cfg, acfg, BENCH_NAME, items,
                  messages_fn=lambda it: [{"role": "system", "content": AGENT_SYSTEM}, user_msg(it)],
                  question_fn=lambda it: it["question"],
                  prompt_fn=lambda it, ans: JUDGE_PROMPT_CN.format(question=it["question"], correct_answer=it["answer"], response=ans),
                  schema=HLE_STYLE_SCHEMA, instructions=INSTRUCTIONS, meta_fn=lambda it: {"topic": it["topic"]})


def _ece(rows):
    st = [{"n": 0, "c": 0, "s": 0.0} for _ in BINS]
    total = 0
    for r in rows:
        v = r.get("verdict") or {}
        try:
            conf = int(str(v.get("confidence", 0)).split("%")[0])
        except ValueError:
            conf = 0
        b = st[min(conf // 20, len(BINS) - 1)]
        b["n"] += 1
        b["s"] += conf
        b["c"] += int(r["correct"])
        total += 1
    err = sum(b["n"] / total * abs(b["c"] / b["n"] - b["s"] / b["n"] / 100) for b in st if b["n"]) if total else 0
    return {"calibration_error_bczh_5bin": round(100 * err, 2)}


def ingest(cfg, verdicts, missing):
    accuracy_ingest(cfg, verdicts, missing, lambda v: str(v.get("correct", "")).strip().lower().startswith("yes"),
                    calib=False, extra=_ece)

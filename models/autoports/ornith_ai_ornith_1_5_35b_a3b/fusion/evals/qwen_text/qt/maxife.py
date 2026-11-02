"""MAXIFE (OPPO-Mente-Lab/MaXIFE, ACL 2025): 23 monolingual settings x 795 prompts
(English + 22 languages; the 22 english_to_* cross-lingual files are excluded,
matching the card's "English + multilingual original prompts, 23 settings").
Prompt = the dataset prompt verbatim. Scoring = the official repo code:
rule-based checkers (instructions.py, via instruction_evaluator.py's class map)
for keywords/length/format/repeat/marks/citation/emoji; style/tone/content/
language_switch instructions need an LLM judge (official: claude-3.5-sonnet,
T=0, verdict 0 / 0.7 / 1). Those judge requests are written to
judge_inputs.jsonl; put verdicts in <run>/judgments.jsonl ({"id","verdict"})
and rerun with --score-only. Metric: loose score (mean instruction score, the
paper's headline) per language, averaged over the 23 settings; strict
(share of instructions scoring >= 0.99) reported alongside."""
import glob
import json
import os
import sys

from .common import stratified_sample

REPO = os.path.expanduser("~/evals/src/MaXIFE")
MODEL_BASED = ("style:", "tone:", "content:", "language_switch:")
JUDGE_SYSTEM = ("You are a professional language evaluation assistant, skilled at determining whether text meets "
                "specific language requirements. Please only respond with one of three scores: 0, 0.7, or 1. Your "
                "response should be a single number without any additional text or explanations.")
_EV = None


def _official():
    global _EV
    if _EV is None:
        sys.path.insert(0, REPO)
        import logging
        from instruction_evaluator import InstructionEvaluator  # noqa: E402
        from prompt_template import PROMPT_TEMPLATES  # noqa: E402
        logging.getLogger().setLevel(logging.CRITICAL)
        _EV = (InstructionEvaluator(), PROMPT_TEMPLATES)
    return _EV


def build(bench):
    items = []
    files = [f for f in sorted(glob.glob(os.path.join(REPO, "data", "input", "*.jsonl")))
             if not any(s in os.path.basename(f) for s in ("english_to_", "only_questions", "not_finished", "errors_"))]
    assert len(files) == 23, files
    for f in files:
        name = os.path.basename(f)[:-6]
        for l in open(f, encoding="utf-8"):
            d = json.loads(l)
            items.append({"id": f"maxife-{name}-{int(d['key']):05d}", "group": name, "gold": None,
                          "messages": [{"role": "user", "content": d["prompt"]}],
                          "meta": {"language": d.get("language", "zh"), "key": d["key"],
                                   "instruction_id": d["instruction_id"], "kwargs": d["kwargs"]}})
    full = len(items)
    return stratified_sample(items, key=lambda x: x["group"]), {"full_size": full, "sample": "group", "avg": "macro"}


def score_batch(bench, rows, out, log):
    ev, templates = _official()
    judge_rows, res = [], []
    for r in rows:
        it, resp = r["item"], r["content"]
        lang = it["meta"]["language"]
        scores = []
        for k, (ins, kw) in enumerate(zip(it["meta"]["instruction_id"], it["meta"]["kwargs"])):
            jid = f"{it['id']}#{k}"
            if ins.startswith(MODEL_BASED):
                cls = ev.instruction_classes.get(ins)
                try:
                    inst = cls(lang)
                    inst.build_description(**kw)
                    text = inst.build_description()       # as model_based_evaluation.evaluate_with_model does
                    tmpl = templates[inst.__class__.__name__]
                    judge_rows.append({"id": jid, "question": it["messages"][0]["content"], "reference": None,
                                       "instruction": ins, "instruction_text": text, "model_answer": resp,
                                       "judge_model_official": "claude-3.5-sonnet", "temperature": 0,
                                       "allowed_verdicts": [0, 0.7, 1],
                                       "judge_messages": [{"role": "system", "content": JUDGE_SYSTEM},
                                                          {"role": "user", "content": tmpl.format(instruction=text, response=resp)}]})
                    scores.append({"ins": ins, "jid": jid, "score": None})
                except Exception as e:  # noqa: BLE001  official: any error -> 0
                    scores.append({"ins": ins, "jid": jid, "score": 0.0, "err": str(e)[:200]})
            else:
                s = ev.evaluate_instruction(ins, kw, resp, it["messages"][0]["content"], lang)
                scores.append({"ins": ins, "jid": jid, "score": float(s)})
        res.append({"instr": scores})
    with open(os.path.join(out, "judge_inputs.jsonl"), "w") as f:
        for j in judge_rows:
            f.write(json.dumps(j, ensure_ascii=False) + "\n")
    return res


def _verdict(v):
    """official process_model_response: a float in {0, 0.7, 1}, anything else -> 0."""
    try:
        x = float(str(v).strip())
    except ValueError:
        return 0.0
    return x if x in (0, 0.7, 1.0) else 0.0


def aggregate(bench, meta, rows):
    out = os.environ["QT_OUT"]
    jp = os.path.join(out, "judgments.jsonl")
    judg = {}
    if os.path.exists(jp):
        for l in open(jp):
            if l.strip():
                d = json.loads(l)
                judg[d["id"]] = _verdict(d["verdict"])
    per, per_rule, missing = {}, {}, 0
    for r in rows:
        g = r["item"]["group"]
        for s in r["score"]["instr"]:
            v = s["score"]
            if v is None:
                v = judg.get(s["jid"])
                if v is None:
                    missing += 1
                    continue
            else:
                per_rule.setdefault(g, []).append(v)
            per.setdefault(g, []).append(v)
    n_judge = sum(1 for r in rows for s in r["score"]["instr"] if s["score"] is None)

    def loose(d):
        return 100 * sum(sum(v) / len(v) for v in d.values()) / len(d) if d else None

    def strict(d):
        return 100 * sum(sum(x >= 0.99 for x in v) / len(v) for v in d.values()) / len(d) if d else None

    done = missing == 0
    return {"metric": "loose", "value": loose(per) if done else None, "strict": strict(per) if done else None,
            "rule_based_only_loose": loose(per_rule), "n_judge_requests": n_judge, "n_judge_missing": missing,
            "per_group": {g: round(100 * sum(v) / len(v), 2) for g, v in sorted(per.items())} if done else {},
            "pending": None if done else f"JUDGE_PENDING {os.path.join(out, 'judge_inputs.jsonl')}",
            "result_extra": ["strict", "rule_based_only_loose", "n_judge_requests", "n_judge_missing"]}


def gold_reply(bench, it):
    return "Fine."


def random_reply(bench, it, rng):
    return "Fine."

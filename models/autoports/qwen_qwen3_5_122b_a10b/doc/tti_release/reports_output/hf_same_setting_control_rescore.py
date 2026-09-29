"""Rescore the same-setting HF control texts with the EXACT lm-eval filters TTI's evals venv uses (lm_eval 0.4.10.dev0):
gpqa_diamond_cot_zeroshot flexible-extract = boxed_choice filter (not the older multi_choice_regex); ifeval via the task's
process_results in the same venv. Run with ~/tt-inference-server/.workflow_venvs/.venv_evals_common/bin/python."""
import copy
import json
import sys

from lm_eval.tasks import TaskManager

SRC, OUT = sys.argv[1], sys.argv[2]
r = json.load(open(SRC))
out = copy.deepcopy(r)
tm = TaskManager()
td = tm.load(["gpqa_diamond_cot_zeroshot", "ifeval"]) if hasattr(tm, "load") else None
if td is None:
    from lm_eval.tasks import get_task_dict

    td = get_task_dict(["gpqa_diamond_cot_zeroshot", "ifeval"], tm)
gp = td["gpqa_diamond_cot_zeroshot"]
fl = [f for f in gp.config.filter_list if f["name"] == "flexible-extract"][0]["filter"]
print(
    "TTI gpqa flexible-extract filter chain:",
    [(s["function"], {k: v for k, v in s.items() if k != "function"}) for s in fl],
)
base = "/home/raahem/tt-inference-server/workflow_logs/reports_output/release/Qwen__Qwen3.5-122B-A10B_p300x2_release/eval_id_qwen35-122b-autoport_Qwen3.5-122B-A10B_p300x2/Qwen__Qwen3.5-122B-A10B/samples_gpqa_diamond_cot_zeroshot_2026-09-26T11-26-17.990320.jsonl"
rows = {
    str(json.loads(l)["doc_id"]): json.loads(l) for l in open(base) if json.loads(l)["filter"] == "flexible-extract"
}
# build the filter objects exactly as lm-eval does from the yaml
from lm_eval.api.registry import get_filter

filters = [get_filter(s["function"])(**{k: v for k, v in s.items() if k != "function"}) for s in fl]


def apply(resps, docs):
    x = resps
    for f in filters:
        x = f.apply(x, docs)
    return x


g = out["gpqa_diamond_cot_zeroshot"]["docs"]
ids = sorted(g, key=int)
filtered = apply([[g[i]["hf_text"]] for i in ids], [rows[i]["doc"] for i in ids])
hf = dev = agree = 0
for i, f in zip(ids, filtered):
    exv = f[0] if isinstance(f, list) else f
    tgt = rows[i]["target"]
    ok = float(str(exv).strip() == tgt.strip())
    d = rows[i]["exact_match"]
    g[i]["hf_score"] = {
        "exact_match,flexible-extract": ok,
        "extracted_tti_filter": exv,
        "target": tgt,
        "old_regex_filter_score": r["gpqa_diamond_cot_zeroshot"]["docs"][i]["hf_score"].get(
            "exact_match,flexible-extract"
        ),
    }
    hf += ok
    dev += d
    agree += ok == d
n = len(ids)
out["gpqa_diamond_cot_zeroshot"].update(
    {
        "hf_exact_match_flexible": hf / n,
        "device_exact_match_same_docs": dev / n,
        "scoring": "lm_eval 0.4.10.dev0 (TTI evals venv) gpqa_diamond_cot_zeroshot flexible-extract filter chain applied to the HF texts; the first offline pass used lm_eval 0.4.13's multi_choice_regex and under-scored HF (24/40) — that value is kept per doc as old_regex_filter_score",
    }
)
out["gpqa_diamond_cot_zeroshot"].pop("note", None)
print(f"gpqa n={n} HF correct={int(hf)} ({100*hf/n:.2f}) device={int(dev)} ({100*dev/n:.2f}) agree={agree}")
# ifeval: recompute prompt-level strict with THIS venv's task (the letter_frequency checker is unseeded, so repeat runs vary by a doc or two)
ife = td["ifeval"]
d = out["ifeval"]["docs"]
ibase = "/home/raahem/tt-inference-server/workflow_logs/reports_output/release/Qwen__Qwen3.5-122B-A10B_p300x2_release/eval_id_qwen35-122b-autoport_Qwen3.5-122B-A10B_p300x2/Qwen__Qwen3.5-122B-A10B/samples_ifeval_2026-09-26T11-13-15.069013.jsonl"
idocs = {str(json.loads(l)["doc_id"]): json.loads(l)["doc"] for l in open(ibase)}
correct = 0
for k, v in d.items():
    m = ife.process_results(idocs[k], [v["hf_text"]])
    v["hf_score"]["prompt_level_strict_acc_tti_venv"] = float(m["prompt_level_strict_acc"])
    correct += m["prompt_level_strict_acc"]
out["ifeval"].pop("hf_score_tti_style", None)
out["ifeval"]["hf_prompt_level_strict_acc_tti_venv_recompute"] = correct / len(d)
out["ifeval"][
    "scoring"
] = f"control venv lm_eval 0.4.13 process_results: 98/109 = 89.91 (the entered reference); TTI venv lm_eval 0.4.10.dev0 recompute in this run: {int(correct)}/109 (the keywords:letter_frequency checker is unseeded; repeat runs give 97-99)"
print(f"ifeval TTI-venv recompute: {int(correct)}/109")
json.dump(out, open(OUT, "w"), indent=1)
print("wrote", OUT)

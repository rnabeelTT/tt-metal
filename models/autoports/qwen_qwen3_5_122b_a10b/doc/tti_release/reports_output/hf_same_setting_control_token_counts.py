"""Per-answer generated-token counts for the gpqa / ifeval eval samples (device: TTI's lm-eval samples jsonl; HF: the control
JSON), with the Qwen tokenizer, to state truncation at max_gen_toks honestly. Run in TTI's evals venv."""
import glob
import json
import sys

from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-122B-A10B")


def dev(path, cap, filt=None):
    rows = [json.loads(l) for l in open(path)]
    if filt:
        rows = [r for r in rows if r.get("filter") == filt]
    n = len(rows)
    cut = 0
    wrong = 0
    wrong_cut = 0
    toks = []
    for r in rows:
        t = len(tok.encode(r["resps"][0][0]))
        toks.append(t)
        c = t >= cap
        cut += c
        ok = r.get("exact_match", r.get("prompt_level_strict_acc"))
        if not ok:
            wrong += 1
            wrong_cut += c
    toks.sort()
    return n, cut, wrong, wrong_cut, toks[n // 2]


def hf(path, task, cap):
    d = json.load(open(path))[task]["docs"]
    n = len(d)
    cut = sum(1 for v in d.values() if v["hf_tokens"] >= cap)
    wrong = sum(
        1
        for v in d.values()
        if not v["hf_score"].get("exact_match,flexible-extract", v["hf_score"].get("prompt_level_strict_acc"))
    )
    wrong_cut = sum(
        1
        for v in d.values()
        if v["hf_tokens"] >= cap
        and not v["hf_score"].get("exact_match,flexible-extract", v["hf_score"].get("prompt_level_strict_acc"))
    )
    return n, cut, wrong, wrong_cut, sorted(v["hf_tokens"] for v in d.values())[n // 2]


base = "/home/raahem/tt-inference-server/workflow_logs/reports_output/"
rel11 = glob.glob(
    base
    + "release/Qwen__Qwen3.5-122B-A10B_p300x2_release/eval_id_*/Qwen__Qwen3.5-122B-A10B/samples_gpqa_diamond_cot_zeroshot_2026-09-29T09-42*.jsonl"
)[0]
run5 = glob.glob(
    base
    + "evals/Qwen__Qwen3.5-122B-A10B_p300x2_evals/eval_id_*/Qwen__Qwen3.5-122B-A10B/samples_gpqa_diamond_cot_zeroshot_2026-09-26T20-17*.jsonl"
)[0]
run5i = glob.glob(
    base
    + "evals/Qwen__Qwen3.5-122B-A10B_p300x2_evals/eval_id_*/Qwen__Qwen3.5-122B-A10B/samples_ifeval_2026-09-26T19-28*.jsonl"
)[0]
ctrl = sys.argv[1]
for label, res in [
    ("device gpqa ci-nightly (run 11), cap 4096", dev(rel11, 4096, "flexible-extract")),
    ("device gpqa full set (run 5), cap 4096", dev(run5, 4096, "flexible-extract")),
    ("device ifeval full set (run 5), cap 2048", dev(run5i, 2048)),
    ("HF gpqa ci-nightly control, cap 4096", hf(ctrl, "gpqa_diamond_cot_zeroshot", 4096)),
    ("HF ifeval ci-nightly control, cap 2048", hf(ctrl, "ifeval", 2048)),
]:
    n, cut, wrong, wrong_cut, med = res
    print(f"TOKENS {label}: n={n} at_cap={cut} wrong={wrong} wrong_and_at_cap={wrong_cut} median_tokens={med}")

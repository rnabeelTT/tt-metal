"""Fills the measured values into performance_head.md / quickstart.md from the q122_card_chain results (in place):
the scorecard subsets (runs/q122ev_{new,pub}/evals), the real-text median of the p300x2 card run and the single-user
decode speed of the p300x2-longctx run (MTP off).

    python fill.py && python assemble.py p300x2 longctx
"""

import glob
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = Path.home() / "ornith-fusion" / "runs"
METRICS = {
    "r1_gpqa_diamond": ("exact_match,none", "GPQA"),
    "ifeval": ("prompt_level_strict_acc,none", "IFE"),
    "livecodebench_v6": ("acc,none", "LCB"),
}


def evals(label):
    out = {}
    for task, (metric, key) in METRICS.items():
        files = sorted(glob.glob(str(RUNS / label / "evals" / task / "*" / "results_*.json")))
        if files:
            out[key] = json.load(open(files[-1]))["results"][task][metric]
    return out


def realtext_median(label):
    m = re.search(r"median ([0-9.]+)", (RUNS / label / "realtext.txt").read_text())
    return float(m.group(1))


def decode_tok_s(label, isl=128, osl=128):
    """1-user decode tok/s (1000 / median TPOT) of a bench row written by bench.sh."""
    f = RUNS / label / f"bench_{isl}_{osl}_u1.json"
    return 1000.0 / json.load(open(f))["median_tpot_ms"]


def main():
    new, pub = evals("q122ev_new"), evals("q122ev_pub")
    # GPQA from the full 198-question runs (runs/q122_gpqa.sh); the 40-question subset was too small to compare
    for vals, label in ((new, "q122gp_new"), (pub, "q122gp_pub")):
        full = evals(label)
        assert "GPQA" in full, f"no full GPQA run in {label}"
        vals["GPQA"] = full["GPQA"]
    head = (HERE / "performance_head.md").read_text()
    for key in ("GPQA", "IFE", "LCB"):
        for side, vals in (("NEW", new), ("PUB", pub)):
            if key in vals:
                head = head.replace("{%s_%s}" % (key, side), f"{vals[key]:.3g}")
    (HERE / "performance_head.md").write_text(head)
    qs = (HERE / "quickstart.md").read_text()
    qs = qs.replace("{SPEED_MTP}", f"{round(realtext_median('card_q122new_p300x2'))}")
    qs = qs.replace("{SPEED_LONGCTX}", f"{round(decode_tok_s('card_q122new_longctx'))}")
    (HERE / "quickstart.md").write_text(qs)
    left = re.findall(r"\{[A-Z_]+\}", head + qs)
    print("filled", {"new": new, "pub": pub}, "unfilled:", left)


if __name__ == "__main__":
    main()

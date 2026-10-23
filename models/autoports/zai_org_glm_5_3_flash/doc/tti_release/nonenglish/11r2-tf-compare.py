"""Stage 11 review #2: compare the device teacher-forcing rows (11r2-tf-device.py) with the HF rows (11r2-tf-hf.py, or
for doc106_s9 without an HF file: doc/tti_release/loopdiag/hfloop.json, HF top-5 as strings, mapped to ids).
Usage: 11r2-tf-compare.py TEXTS_JSON OUT_JSON HF_JSON|- TT_JSON...
Per text and TT row set: top-1 agreement (TT argmax == HF argmax), top-5 agreement (TT argmax in the HF top 5),
mean TT top-5 mass, mean TT p(HF argmax); over all scored positions (paragraphs: from position 1, i.e. the
'</think>' row excluded) and for doc106_s9 also positions 24-31 (the loop-entry window)."""
import json
import sys
from pathlib import Path

texts = {t["name"]: t for t in json.load(open(sys.argv[1]))}
out_path, hf_path, tts = sys.argv[2], sys.argv[3], sys.argv[4:]
hf = json.load(open(hf_path))["results"] if hf_path != "-" else {}
if "doc106_s9" not in hf:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(Path.home() / "glm53-flash-bringup/hfloop_overlay"))
    inv = {}
    for i in range(len(tok)):
        inv.setdefault(tok.decode([i]), i)
    hl = json.load(
        open(Path.home() / "tt-metal/models/autoports/zai_org_glm_5_3_flash/doc/tti_release/loopdiag/hfloop.json")
    )
    hf["doc106_s9"] = {
        "hf": [
            {
                "ids": [inv[s] for s, _ in r["hf_top5"]],
                "p": [p for _, p in r["hf_top5"]],
                "p_forced": r["hf_p"],
                "rank_forced": r["hf_rank"],
            }
            for r in hl["rows"]
        ]
    }
    # the served TT rows (vLLM logprobs, bundle p300x2) as one more row set
    t = texts["doc106_s9"]
    served = [
        {"ids": [i for i, _ in sorted(r, key=lambda x: -x[1])], "p": [p for _, p in sorted(r, key=lambda x: -x[1])]}
        for r in t["served_top5"]
    ]
else:
    served = None


def score(tt_rows, hf_rows, sel):
    n = len(sel)
    a1 = sum(tt_rows[i]["ids"][0] == hf_rows[i]["ids"][0] for i in sel)
    a5 = sum(tt_rows[i]["ids"][0] in hf_rows[i]["ids"][:5] for i in sel)
    m5 = sum(sum(tt_rows[i]["p"][:5]) for i in sel) / n
    ph = []
    for i in sel:
        h = hf_rows[i]["ids"][0]
        ph.append(tt_rows[i]["p"][tt_rows[i]["ids"].index(h)] if h in tt_rows[i]["ids"] else 0.0)
    hm5 = sum(sum(hf_rows[i]["p"][:5]) for i in sel) / n
    return {
        "n": n,
        "top1": round(a1 / n, 3),
        "top1_in_hf_top5": round(a5 / n, 3),
        "tt_top5_mass": round(m5, 3),
        "hf_top5_mass": round(hm5, 3),
        "tt_p_of_hf_top1": round(sum(ph) / n, 3),
    }


res = {}
sets = {}
for f in tts:
    d = json.load(open(f))
    pol = Path(f).stem.removeprefix("tt_")
    for name, paths in d["results"].items():
        for path, rows in paths.items():
            sets.setdefault(name, {})[f"{pol}/{path}"] = rows
if served:
    sets.setdefault("doc106_s9", {})["served_bundle/decode"] = served
for name, rs in sets.items():
    if name not in hf:
        continue
    H = hf[name]["hf"]
    N = len(H)
    windows = {"all": list(range(0 if name == "doc106_s9" else 1, N))}
    if name == "doc106_s9":
        windows["pos24_31"] = list(range(24, 32))
    res[name] = {k: {w: score(rows, H, sel) for w, sel in windows.items()} for k, rows in rs.items()}
Path(out_path).write_text(json.dumps(res, indent=1) + "\n")
for name, r in res.items():
    print(f"== {name}")
    for k, v in r.items():
        print(
            f"  {k:45s} "
            + " | ".join(
                f"{w}: top1 {s['top1']:.2f} in-hf5 {s['top1_in_hf_top5']:.2f} "
                f"m5 {s['tt_top5_mass']:.2f}/{s['hf_top5_mass']:.2f} pHF1 {s['tt_p_of_hf_top1']:.2f}"
                for w, s in v.items()
            )
        )

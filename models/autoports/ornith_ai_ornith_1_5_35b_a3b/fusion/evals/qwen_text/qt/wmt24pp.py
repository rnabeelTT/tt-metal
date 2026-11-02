"""WMT24++ (google/wmt24pp): 55 en->xx pairs x 998 segments. Following the
dataset card: drop is_bad_source segments (and the canary), reference = the
post-edit `target`. Qwen's "harder subset after difficulty labeling and
rebalancing" is not public, so this samples 1,000 segments spread evenly over
the 55 pairs (seed 1234). Metric: XCOMET-XXL segment scores, averaged per pair,
then over pairs (x100). Scoring runs on CPU in ~/evals/wmt24pp_comet/.venv."""
import glob
import json
import os
import shlex
import subprocess

import pandas as pd

from .common import raw, stratified_sample

COMET_PY = os.path.expanduser("~/evals/wmt24pp_comet/.venv/bin/python")
SCORER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "comet_score.py")

LANGUAGE_BY_CODE = {  # dataset README constants
    "ar_EG": "Arabic", "ar_SA": "Arabic", "bg_BG": "Bulgarian", "bn_IN": "Bengali", "ca_ES": "Catalan",
    "cs_CZ": "Czech", "da_DK": "Danish", "de_DE": "German", "el_GR": "Greek", "es_MX": "Spanish",
    "et_EE": "Estonian", "fa_IR": "Farsi", "fi_FI": "Finnish", "fil_PH": "Filipino", "fr_CA": "French",
    "fr_FR": "French", "gu_IN": "Gujarati", "he_IL": "Hebrew", "hi_IN": "Hindi", "hr_HR": "Croatian",
    "hu_HU": "Hungarian", "id_ID": "Indonesian", "is_IS": "Icelandic", "it_IT": "Italian", "ja_JP": "Japanese",
    "kn_IN": "Kannada", "ko_KR": "Korean", "lt_LT": "Lithuanian", "lv_LV": "Latvian", "ml_IN": "Malayalam",
    "mr_IN": "Marathi", "nl_NL": "Dutch", "no_NO": "Norwegian", "pa_IN": "Punjabi", "pl_PL": "Polish",
    "pt_BR": "Portuguese", "pt_PT": "Portuguese", "ro_RO": "Romanian", "ru_RU": "Russian", "sk_SK": "Slovak",
    "sl_SI": "Slovenian", "sr_RS": "Serbian", "sv_SE": "Swedish", "sw_KE": "Swahili", "sw_TZ": "Swahili",
    "ta_IN": "Tamil", "te_IN": "Telugu", "th_TH": "Thai", "tr_TR": "Turkish", "uk_UA": "Ukrainian",
    "ur_PK": "Urdu", "vi_VN": "Vietnamese", "zh_CN": "Mandarin", "zh_TW": "Mandarin", "zu_ZA": "Zulu"}
REGION_BY_CODE = {
    "ar_EG": "Egypt", "ar_SA": "Saudi Arabia", "bg_BG": "Bulgaria", "bn_IN": "India", "ca_ES": "Spain",
    "cs_CZ": "Czechia", "da_DK": "Denmark", "de_DE": "Germany", "el_GR": "Greece", "es_MX": "Mexico",
    "et_EE": "Estonia", "fa_IR": "Iran", "fi_FI": "Finland", "fil_PH": "Philippines", "fr_CA": "Canada",
    "fr_FR": "France", "gu_IN": "India", "he_IL": "Israel", "hi_IN": "India", "hr_HR": "Croatia",
    "hu_HU": "Hungary", "id_ID": "Indonesia", "is_IS": "Iceland", "it_IT": "Italy", "ja_JP": "Japan",
    "kn_IN": "India", "ko_KR": "South Korea", "lt_LT": "Lithuania", "lv_LV": "Latvia", "ml_IN": "India",
    "mr_IN": "India", "nl_NL": "Netherlands", "no_NO": "Norway", "pa_IN": "India", "pl_PL": "Poland",
    "pt_BR": "Brazil", "pt_PT": "Portugal", "ro_RO": "Romania", "ru_RU": "Russia", "sk_SK": "Slovakia",
    "sl_SI": "Slovenia", "sr_RS": "Serbia", "sv_SE": "Sweden", "sw_KE": "Kenya", "sw_TZ": "Tanzania",
    "ta_IN": "India", "te_IN": "India", "th_TH": "Thailand", "tr_TR": "Turkey", "uk_UA": "Ukraine",
    "ur_PK": "Pakistan", "vi_VN": "Vietnam", "zh_CN": "China", "zh_TW": "Taiwan", "zu_ZA": "South Africa"}

# Region-specific 0-shot request in the spirit of the WMT24++ paper's LLM prompt (its exact
# text is only in a figure); Qwen's own translation prompt is not published.
PROMPT = ("Translate the following English text into {lang} as used in {region}. "
          "Output only the translation, without any explanations or notes.\n\n{src}")


def build(bench):
    items, full = [], 0
    for f in sorted(glob.glob(raw("google__wmt24pp", "en-*.jsonl"))):
        lp = os.path.basename(f)[:-6]
        code = lp.split("-", 1)[1]
        df = pd.read_json(f, lines=True)
        full += len(df)
        for r in df.itertuples():
            if r.is_bad_source or r.domain == "canary":
                continue
            items.append({"id": f"wmt-{lp}-{int(r.segment_id):04d}", "group": lp, "gold": r.target,
                          "messages": [{"role": "user", "content": PROMPT.format(
                              lang=LANGUAGE_BY_CODE[code], region=REGION_BY_CODE[code], src=r.source)}],
                          "meta": {"lp": lp, "domain": r.domain, "src": r.source, "ref": r.target}})
    n_clean = len(items)
    items = stratified_sample(items, key=lambda x: x["group"])
    return items, {"full_size": full, "after_bad_source_filter": n_clean, "sample": "group", "avg": "macro"}


def score(bench, it, content):
    return {"mt_chars": len(content or ""), "empty": not (content or "").strip()}


def _run_comet(rows, out, log):
    model = os.environ.get("COMET_MODEL", "Unbabel/XCOMET-XXL")
    inp = os.path.join(out, "comet_input.jsonl")
    res = os.path.join(out, f"comet_{model.replace('/', '__')}.jsonl")
    with open(inp, "w") as f:
        for r in rows:
            f.write(json.dumps({"id": r["item"]["id"], "src": r["item"]["meta"]["src"],
                                "mt": (r["content"] or "").strip(), "ref": r["item"]["meta"]["ref"]},
                               ensure_ascii=False) + "\n")
    cmd = [COMET_PY, "-I", SCORER, "--model", model, "--input", inp, "--output", res,
           "--threads", os.environ.get("COMET_THREADS", "8"), "--precision", os.environ.get("COMET_PRECISION", "bf16"),
           "--batch-size", os.environ.get("COMET_BATCH", "8")]
    return model, res, cmd


def aggregate(bench, meta, rows):
    out = os.environ.get("QT_OUT")
    log = print
    model, res, cmd = _run_comet(rows, out, log)
    pending = None
    if os.environ.get("WMT_SCORE", "1") == "1":
        p = subprocess.run(["nice", "-n", "10"] + cmd, capture_output=True, text=True)
        if p.returncode != 0:
            open(os.path.join(out, "comet_error.log"), "w").write(p.stdout[-5000:] + "\n" + p.stderr[-20000:])
            pending = "SCORE_PENDING " + " ".join(shlex.quote(c) for c in cmd) + \
                      f"  # failed, see {out}/comet_error.log; then rerun with --score-only"
    elif not os.path.exists(res):
        pending = "SCORE_PENDING " + " ".join(shlex.quote(c) for c in cmd) + "  # then rerun with --score-only"
    per = {}
    if os.path.exists(res):
        sc = {d["id"]: d["score"] for d in map(json.loads, open(res))}
        for r in rows:
            if r["item"]["id"] in sc:
                per.setdefault(r["item"]["group"], []).append(sc[r["item"]["id"]])
    n_sc = sum(len(v) for v in per.values())
    val = 100 * sum(sum(v) / len(v) for v in per.values()) / len(per) if per and n_sc == len(rows) else None
    if per and n_sc != len(rows):
        pending = pending or f"SCORE_PENDING {n_sc}/{len(rows)} segments scored in {res}"
    metric = "xcomet_xxl" if model == "Unbabel/XCOMET-XXL" else model.split("/")[-1].lower().replace("-", "_")
    return {"metric": metric, "value": val, "comet_model": model, "n_empty": sum(r["score"]["empty"] for r in rows),
            "per_group": {g: round(100 * sum(v) / len(v), 2) for g, v in sorted(per.items())},
            "pending": pending, "result_extra": ["n_empty"]}


def gold_reply(bench, it):
    return it["meta"]["ref"]


def random_reply(bench, it, rng):
    # a fluent but wrong translation: another segment's reference in the same language
    return rng.choice(_POOL.setdefault(it["group"], [])) if _POOL.get(it["group"]) else "Lorem ipsum."


_POOL = {}


def init_stub(items):
    for it in items:
        _POOL.setdefault(it["group"], []).append(it["meta"]["ref"])

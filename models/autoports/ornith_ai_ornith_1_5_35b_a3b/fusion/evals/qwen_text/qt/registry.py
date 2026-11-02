"""bench name -> module implementing build/score/aggregate/gold_reply/random_reply."""
import importlib

_MODS = {
    **{b: "mcq" for b in ("mmlu_pro", "mmlu_redux", "ceval", "supergpqa", "mmmlu", "mmlu_prox", "include", "global_piqa")},
    "polymath": "polymath",
    "wmt24pp": "wmt24pp",
    "nova63": "nova63",
    "maxife": "maxife",
}

BENCHES = list(_MODS)


def module(bench):
    if bench not in _MODS:
        raise SystemExit(f"unknown bench {bench}; have {BENCHES}")
    return importlib.import_module(f"qt.{_MODS[bench]}")

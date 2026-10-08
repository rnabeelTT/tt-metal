"""Objective text-degradation metrics for served completions.

The failure this scores is NOT "the model is bad at spelling": it is a wrong token
sampled out of a quantization-noisy logit tail, which the model then writes around.
The artefacts that leaves are mechanical and countable:

  * a near-miss subword glued onto the right word   -> "interpreterterpreter", "Typpos", "taskss"
  * a word emitted twice                            -> "with with"
  * punctuation emitted twice                       -> ",,"
  * an opening delimiter whose closer was never sampled

``garble_per_1k`` is the gate: weighted artefact count per 1000 word units. It is meant
to be compared between two runs of the SAME prompt set, not read as an absolute.

False positives this deliberately suppresses, because a noisy gate is a useless gate:
  * curly apostrophes ("you’re") - normalised before the dictionary check
  * invented proper nouns ("Elara") - a capitalised word used more than once is a name,
    a one-off capitalised non-word ("Ambograph") is a garble
  * non-English answers - flagged with ``language_skip`` and excluded from the word-level
    score; the structural metrics stay, they are language independent
  * code - fenced, inline and indented blocks are stripped, and code keywords allowed
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path

_DICT_PATHS = ("/usr/share/dict/american-english", "/usr/share/dict/british-english", "/usr/share/dict/words")

# Weights: how much each artefact contributes. Intra-word duplication and adjacent word
# duplication are the least ambiguous signals and carry the most weight; a non-dictionary
# word is the noisiest signal and carries the least.
_WEIGHTS = {"intraword_dup": 3.0, "dup_adjacent": 3.0, "dup_punct": 1.0, "nonword": 1.0, "delim_imbalance": 1.0}

# Above this non-dictionary fraction the completion is almost certainly not English
# (the readiness prompt set includes a French request): keep structure, drop word scoring.
_FOREIGN_NONWORD_FRACTION = 0.35

_CODE_FENCE = re.compile(r"```.*?```", re.S)
_INLINE_CODE = re.compile(r"`[^`\n]*`")
_INDENTED_CODE = re.compile(r"^(?: {4}|\t).*$", re.M)
_URL = re.compile(r"https?://\S+")
# LaTeX control words and math spans are not prose: "\\mathrm", "$\\frac{a}{b}$"
_LATEX_CMD = re.compile(r"\\[A-Za-z]+")
_MATH_SPAN = re.compile(r"\$+[^$]*\$+|\\\[.*?\\\]|\\\(.*?\\\)", re.S)
# a "word unit" may be a hyphenated compound ("vllm-plugin") or carry a file extension
# ("community_model_config.py"); both are split into alphabetic parts for the dictionary check
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9_'.-]*")
_ALPHA = re.compile(r"[A-Za-z][A-Za-z']*")
_DUP_PUNCT = re.compile(r"([,.;:!?])\1")

_EXTENSIONS = {
    "py",
    "js",
    "ts",
    "tsx",
    "jsx",
    "json",
    "toml",
    "yaml",
    "yml",
    "md",
    "txt",
    "cpp",
    "hpp",
    "c",
    "h",
    "rs",
    "go",
    "sh",
    "cfg",
    "ini",
    "csv",
    "html",
    "css",
    "sql",
    "env",
    "lock",
    "log",
}

_EXTRA_VOCAB = {
    # general technical English the wordlists miss
    "ai",
    "api",
    "apis",
    "async",
    "await",
    "backend",
    "frontend",
    "config",
    "configs",
    "json",
    "yaml",
    "toml",
    "cli",
    "runtime",
    "wrapper",
    "wrappers",
    "parser",
    "parsers",
    "router",
    "routers",
    "endpoint",
    "endpoints",
    "bundle",
    "bundles",
    "catalog",
    "catalogs",
    "repo",
    "repos",
    "schema",
    "schemas",
    "dataset",
    "datasets",
    "tokenizer",
    "tokenizers",
    "quantization",
    "quantized",
    "inference",
    "serialization",
    "boolean",
    "enum",
    "metadata",
    "namespace",
    "middleware",
    "pydantic",
    "fastapi",
    "huggingface",
    "llm",
    "llms",
    "gpu",
    "gpus",
    "cpu",
    "cpus",
    "ttl",
    "stdout",
    "stderr",
    "env",
    "url",
    "urls",
    "uuid",
    "regex",
    "whitespace",
    "substring",
    "iterable",
    "lambda",
    "struct",
    "stdin",
    "mutex",
    "latency",
    "throughput",
    "scalar",
    "tensor",
    "tensors",
    # code keywords that survive the code strippers in prose explanations
    "def",
    "args",
    "kwargs",
    "init",
    "str",
    "int",
    "bool",
    "dict",
    "list",
    "repr",
    "len",
    "elif",
    "func",
    # the stack this port is served on
    "tt",
    "ttnn",
    "vllm",
    "blackhole",
    "wormhole",
    "tenstorrent",
    "dit",
    "mesh",
    "kv",
    "moe",
    "mtp",
    "bfp",
    "prefill",
    "decode",
    "sharded",
    "sharding",
    "dram",
    "ndjson",
    "openai",
    "qwen",
    "rope",
    "gdn",
    "ccl",
    # words about this failure mode itself
    "homophone",
    "homophones",
    "homograph",
    "homographs",
    "misspelling",
    "misspellings",
    "typo",
    "typos",
    # numerics / accelerator vocabulary that appears throughout answers about this model
    "activations",
    "activation",
    "quantisation",
    "dequantize",
    "dequantization",
    "mantissa",
    "mantissas",
    "exponent",
    "exponents",
    "logit",
    "logits",
    "softmax",
    "argmax",
    "autoregressive",
    "remediate",
    "numerics",
    "accumulator",
    "accumulators",
    "accumulate",
    "rounding",
    "saturating",
    "elementwise",
    "matmul",
    "matmuls",
    "kernel",
    "kernels",
    "pipelined",
    "preallocated",
    "precompute",
    "precomputed",
    "recompute",
    "recomputed",
    "lifecycle",
    "codebase",
    "runtime",
    "deterministic",
    "nondeterministic",
    "reproducibility",
    "throughputs",
    "sampler",
    "samplers",
    "nucleus",
    "greedy",
    "perplexity",
}


def _load_dictionary() -> set[str]:
    words: set[str] = set()
    for p in _DICT_PATHS:
        f = Path(p)
        if f.exists():
            words |= {w.strip().lower() for w in f.read_text(errors="ignore").splitlines() if w.strip()}
    return words | _EXTRA_VOCAB


_DICT = _load_dictionary()


def _normalize(text: str) -> str:
    """Curly punctuation to ASCII so "you’re" matches the wordlist's "you're"."""
    return unicodedata.normalize("NFKC", text).replace("’", "'").replace("‘", "'")


def _is_plausible_identifier(w: str) -> bool:
    """Code-ish tokens are not spelling errors: snake_case, camelCase, digits, short ALLCAPS."""
    return bool(
        "_" in w
        or any(c.isdigit() for c in w)
        or (w[1:] != w[1:].lower() and w != w.upper())  # interior capital: camelCase / PascalCase
        or (w.isupper() and len(w) <= 6)
    )


def _intraword_dup(w: str) -> bool:
    """A word containing an immediately repeated run of >= 3 characters.

    "interpreterterpreter" -> "terpreter" twice. Requires the word to be >= 8 chars so
    ordinary English doubling does not trip it.
    """
    lw = w.lower()
    if len(lw) < 8:
        return False
    for n in range(3, len(lw) // 2 + 1):
        for i in range(len(lw) - 2 * n + 1):
            if lw[i : i + n] == lw[i + n : i + 2 * n]:
                return True
    return False


# Hyphenated compounds are split into parts, so bare prefixes/suffixes must not count as
# non-words ("pre-allocated" -> "pre" + "allocated", "fine-grained" -> "fine" + "grained").
_AFFIXES = {
    "pre",
    "post",
    "non",
    "sub",
    "multi",
    "inter",
    "intra",
    "semi",
    "pseudo",
    "quasi",
    "anti",
    "auto",
    "co",
    "re",
    "un",
    "de",
    "bi",
    "tri",
    "uni",
    "micro",
    "macro",
    "mini",
    "meta",
    "ultra",
    "grained",
    "aware",
    "wise",
    "like",
    "based",
    "driven",
    "specific",
    "side",
    "level",
    "time",
    "point",
    "wide",
    "free",
    "safe",
    "ready",
    "only",
    "style",
    "ish",
}


def score_text(text: str, prompt: str = "") -> dict:
    """Artefact counts + ``garble_per_1k`` for one completion."""
    text = _normalize(text)
    allow = {m.group(0).lower() for m in _WORD.finditer(_normalize(prompt))}
    allow |= {p.lower() for u in list(allow) for p in _ALPHA.findall(u)}

    prose = _CODE_FENCE.sub(" ", text)
    prose = _MATH_SPAN.sub(" ", prose)
    prose = _LATEX_CMD.sub(" ", prose)
    prose = _URL.sub(" ", _INDENTED_CODE.sub(" ", _INLINE_CODE.sub(" ", prose)))

    units = [u for u in (m.group(0).strip(".-'") for m in _WORD.finditer(prose)) if u]
    n_words = len(units)

    # a capitalised word used more than once is a name the answer introduced, not a garble
    seen: dict[str, int] = {}
    for u in units:
        seen[u.lower()] = seen.get(u.lower(), 0) + 1

    nonword: list[str] = []
    intraword: list[str] = []
    for unit in units:
        if _intraword_dup(unit):
            intraword.append(unit)
            continue
        if unit.lower() in allow or _is_plausible_identifier(unit):
            continue
        parts = [m.group(0) for m in _ALPHA.finditer(unit)]
        for i, part in enumerate(parts):
            lp = part.lower().strip("'")
            if len(lp) < 2 or lp in allow or lp in _DICT:
                continue
            if len(parts) > 1 and lp in _AFFIXES:
                continue  # a bare affix of a hyphenated compound
            if i == len(parts) - 1 and lp in _EXTENSIONS and "." in unit:
                continue
            if part[0].isupper() and seen.get(unit.lower(), 0) > 1:
                continue  # recurring proper noun
            nonword.append(part)

    dup_adjacent = [
        units[i] for i in range(1, len(units)) if units[i].lower() == units[i - 1].lower() and len(units[i]) > 1
    ]
    dup_punct = _DUP_PUNCT.findall(prose)

    imbalance = 0
    for o, c in (("(", ")"), ("[", "]"), ("{", "}")):
        imbalance += abs(text.count(o) - text.count(c))
    imbalance += text.count("`") % 2

    language_skip = bool(n_words) and (len(nonword) / n_words) > _FOREIGN_NONWORD_FRACTION
    counts = {
        "nonword": 0 if language_skip else len(nonword),
        "intraword_dup": len(intraword),
        "dup_adjacent": len(dup_adjacent),
        "dup_punct": len(dup_punct),
        "delim_imbalance": imbalance,
    }
    weighted = sum(_WEIGHTS[k] * v for k, v in counts.items())
    return {
        "n_words": n_words,
        **counts,
        "language_skip": language_skip,
        "garble_per_1k": round(1000.0 * weighted / n_words, 2) if n_words else 0.0,
        "examples": {
            "nonword": [] if language_skip else nonword[:12],
            "intraword_dup": intraword[:12],
            "dup_adjacent": dup_adjacent[:12],
        },
    }


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print(json.dumps(score_text(sys.stdin.read()), indent=1))
        return 0
    for path in argv:
        print(f"== {path}")
        print(json.dumps(score_text(Path(path).read_text()), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

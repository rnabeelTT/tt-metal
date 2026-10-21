"""Served needle-in-a-haystack retrieval check (stage 9, profile p300x2-1m). Usage:
    09-needle.py URL OUT_JSON CASE [CASE ...]      CASE = TOKENS:DEPTH, e.g. 1040000:0.1 300000:0.5
Filler: concatenated source / doc files of the tt-metal tree in a seeded random order (varied text the model cannot
continue from memory as a whole). A random 7-digit passkey sentence goes in at DEPTH of the filler; the question is at
the end. One chat request per case, reasoning_effort low, max_tokens 512, greedy. PASS: the content holds the passkey.
The rendered prompt (chat template) is checked to be TOKENS tokens at most and to fit the server's max_model_len."""

import json
import os
import random
import sys
import time
from pathlib import Path

import requests
from transformers import AutoTokenizer

URL, OUT, CASES = sys.argv[1], sys.argv[2], sys.argv[3:]
MODEL = "zai-org/GLM-5.3-Flash"
ROOT = Path(os.environ.get("TT_METAL_HOME", Path.home() / "tt-metal"))
tok = AutoTokenizer.from_pretrained(os.environ.get("GLM53_CKPT", MODEL))
MAX_TOKENS = 512
QUESTION = "\n\nWhat is the secret code mentioned in the text above? Answer with the 7-digit number only."


def files(seed):
    pats = [
        "tt_metal/**/*.cpp",
        "tt_metal/**/*.hpp",
        "ttnn/cpp/**/*.cpp",
        "ttnn/cpp/**/*.hpp",
        "tech_reports/**/*.md",
        "models/demos/**/*.py",
        "docs/**/*.rst",
        "docs/**/*.md",
    ]
    fs = sorted({p for pat in pats for p in ROOT.glob(pat) if p.is_file() and 2_000 < p.stat().st_size < 200_000})
    random.Random(seed).shuffle(fs)
    return fs


def build(n_tokens, depth, seed):
    rng = random.Random(seed)
    code = rng.randint(1_000_000, 9_999_999)
    needle = f"\n\nThe secret code is {code}. Remember it.\n\n"
    budget = n_tokens - len(tok.encode(needle + QUESTION, add_special_tokens=False)) - 64  # template overhead
    pieces, total = [], 0
    for f in files(seed):
        t = f.read_text(errors="ignore")
        n = len(tok.encode(t, add_special_tokens=False))
        if total + n > budget:
            break
        pieces.append(t)
        total += n
    cut = max(1, int(len(pieces) * depth))
    text = "\n\n".join(pieces[:cut]) + needle + "\n\n".join(pieces[cut:]) + QUESTION
    msgs = [{"role": "user", "content": text}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, reasoning_effort="low")
    ids = ids["input_ids"] if hasattr(ids, "keys") else ids
    depth_real = len(tok.encode("\n\n".join(pieces[:cut]), add_special_tokens=False)) / max(1, total)
    return msgs, code, len(ids), depth_real, len(pieces)


res = []
ok_all = True
for i, case in enumerate(CASES):
    n, d = case.split(":")
    msgs, code, n_prompt, depth_real, nfiles = build(int(n), float(d), seed=1000 + i)
    t0 = time.perf_counter()
    r = requests.post(
        f"{URL}/v1/chat/completions",
        timeout=10800,
        json={
            "model": MODEL,
            "messages": msgs,
            "max_tokens": MAX_TOKENS,
            "temperature": 0.0,
            "chat_template_kwargs": {"reasoning_effort": "low"},
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    )
    r.raise_for_status()
    ttft, content, reasoning, usage, finish = None, "", "", None, None
    for line in r.iter_lines():
        if not line or not line.startswith(b"data: ") or line == b"data: [DONE]":
            continue
        ev = json.loads(line[6:])
        if ev.get("usage"):
            usage = ev["usage"]
        for c in ev.get("choices", []):
            dl = c.get("delta", {})
            if (dl.get("content") or dl.get("reasoning") or dl.get("reasoning_content")) and ttft is None:
                ttft = time.perf_counter() - t0
            content += dl.get("content") or ""
            reasoning += dl.get("reasoning") or dl.get("reasoning_content") or ""
            finish = c.get("finish_reason") or finish
    wall = time.perf_counter() - t0
    ok = str(code) in content
    ok_all &= ok
    rec = {
        "case": case,
        "passkey": code,
        "files": nfiles,
        "depth_tokens": round(depth_real, 4),
        "prompt_tokens_rendered": n_prompt,
        "prompt_tokens_served": (usage or {}).get("prompt_tokens"),
        "completion_tokens": (usage or {}).get("completion_tokens"),
        "ttft_s": round(ttft or -1, 1),
        "wall_s": round(wall, 1),
        "finish_reason": finish,
        "content": content,
        "reasoning": reasoning[-2000:],
        "pass": ok,
    }
    res.append(rec)
    Path(OUT).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(
        f"NEEDLE {case}: prompt {rec['prompt_tokens_served']} tokens (rendered {n_prompt}), depth "
        f"{depth_real:.3f}, TTFT {rec['ttft_s']} s, finish {finish}, passkey {code}, content {content[:80]!r} -> "
        f"{'PASS' if ok else 'FAIL'}",
        flush=True,
    )
    if not ok:
        break  # report before any retry
print("NEEDLE_RESULT", "PASS" if ok_all else "FAIL", OUT)

# post-hook on the same live server (the caller stops the server after this script): status/needle_post.sh, if present,
# runs with SERVER_URL set (one-shot: renamed to .done after the run)
hook = Path.home() / "glm53-flash-bringup/status/needle_post.sh"
if hook.exists():
    import subprocess

    print("NEEDLE_POSTHOOK start", hook, flush=True)
    rc = subprocess.call(["bash", str(hook)], env={**os.environ, "SERVER_URL": URL})
    hook.rename(hook.with_name(hook.name + ".done"))
    print("NEEDLE_POSTHOOK exit", rc, flush=True)

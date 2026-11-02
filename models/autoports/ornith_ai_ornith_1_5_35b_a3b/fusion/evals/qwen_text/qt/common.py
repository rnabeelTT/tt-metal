"""Shared pieces of the qwen_text harness: frozen item files, seeded sampling,
the async chat-completions client (resumable), and answer extraction."""
import asyncio
import hashlib
import json
import os
import random
import re
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(HERE, "hf_data")      # raw HF downloads (untrusted data, never executed)
DATA = os.path.join(HERE, "data")        # frozen items + sample ids per bench (committed by prepare.py)
SEED = 1234
SAMPLE_N = 1000

# The card's "Standardize Output Format" sentence for multiple choice, verbatim
# (Qwen3.5-122B-A10B README, "Best Practices" item 3).
MCQ_JSON_INSTR = 'Please show your choice in the `answer` field with only the choice letter, e.g., `"answer": "C"`.'
# The card's math sentence, verbatim.
MATH_INSTR = "Please reason step by step, and put your final answer within \\boxed{}."

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def raw(*p):
    return os.path.join(RAW, *p)


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def stratified_sample(items, key, n=SAMPLE_N, seed=SEED):
    """Fixed sample of n items spread evenly over groups (key(item)). Each group
    gets floor(n/G); the remainder goes to groups picked by the seeded RNG. Items
    inside a group are drawn by the same RNG from the id-sorted list, so the
    result depends only on the data and the seed."""
    rng = random.Random(seed)
    groups = {}
    for it in sorted(items, key=lambda x: x["id"]):
        groups.setdefault(key(it), []).append(it)
    names = sorted(groups)
    base, rem = divmod(n, len(names))
    extra = set(rng.sample(names, rem)) if rem else set()
    out = []
    for g in names:
        k = min(len(groups[g]), base + (1 if g in extra else 0))
        out += rng.sample(groups[g], k)
    return sorted(out, key=lambda x: x["id"])


def simple_sample(items, n=SAMPLE_N, seed=SEED):
    rng = random.Random(seed)
    return sorted(rng.sample(sorted(items, key=lambda x: x["id"]), n), key=lambda x: x["id"])


def mcq_block(options, letters=LETTERS):
    return "\n".join(f"{letters[i]}. {o}" for i, o in enumerate(options))


# ---------- answer extraction ----------
_JSON_ANS = re.compile(r'["\u201c\u201d\']?answer["\u201c\u201d\']?\s*[:\uff1a]\s*["\u201c\u201d\'(\[]?\s*\(?([A-Z])(?![A-Za-z])\)?\s*["\u201c\u201d\')\]]?', re.I)
_BOXED = re.compile(r"\\boxed\{\s*(?:\\text\{)?\s*\(?([A-Z])\)?\s*\}?\s*\}")
_ANS_IS = re.compile(r"[Aa][Nn][Ss][Ww][Ee][Rr]\s*(?:is|:)\s*\**\s*\(?([A-Z])\)?(?![A-Za-z])")


def extract_choice(text, n_options):
    """Letter picked by the model. Primary: the card's `"answer": "X"` field (last
    occurrence). Fallbacks (logged as such): \\boxed{X}, 'answer is X', a bare
    letter. Returns (letter|None, method)."""
    if not text:
        return None, "empty"
    valid = LETTERS[:n_options]
    for name, rx in (("json", _JSON_ANS), ("boxed", _BOXED), ("answer_is", _ANS_IS)):
        ms = [m.group(1).upper() for m in rx.finditer(text)]
        ms = [m for m in ms if m in valid]
        if ms:
            return ms[-1], name
    t = text.strip().strip("*.()[]{}\"' ").upper()
    if len(t) == 1 and t in valid:
        return t, "bare"
    return None, "none"


def last_boxed(text):
    """Content of the last \\boxed{...} (brace-balanced), or None."""
    if not text:
        return None
    i = max(text.rfind("\\boxed{"), text.rfind("\\boxed {"))
    if i < 0:
        return None
    j = text.find("{", i)
    depth, k = 0, j
    while k < len(text):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return text[j + 1:k]
        k += 1
    return None


def msg_key(messages):
    return hashlib.sha1(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


# ---------- client ----------
async def run_requests(items, out_path, base_url, model, conc, gen, timeout, retries, log):
    """POST each item's messages to {base_url}/chat/completions with `conc` in
    flight. One JSON line per finished item is appended to out_path; items already
    there with ok=True are skipped (resume)."""
    import httpx

    done = {}
    if os.path.exists(out_path):
        for r in read_jsonl(out_path):
            if r.get("ok"):
                done[r["id"]] = r
    todo = [it for it in items if it["id"] not in done]
    log(f"requests: {len(items)} total, {len(done)} already done, {len(todo)} to send")
    if not todo:
        return
    sem = asyncio.Semaphore(conc)
    lock = asyncio.Lock()
    t0 = time.time()
    n_done = [0]
    toks = [0]
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": "Bearer " + os.environ.get("OPENAI_API_KEY", "EMPTY")}
    limits = httpx.Limits(max_connections=conc + 4, max_keepalive_connections=conc + 4)
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=60), limits=limits) as client:
        f = open(out_path, "a")

        async def one(it):
            body = {"model": model, "messages": it["messages"], **gen}
            body.update(it.get("gen_override", {}))
            err = None
            async with sem:
                for a in range(retries + 1):
                    try:
                        r = await client.post(url, json=body, headers=headers)
                        if r.status_code != 200:
                            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
                        d = r.json()
                        ch = d["choices"][0]
                        m = ch.get("message") or {}
                        reasoning = m.get("reasoning")
                        if reasoning is None:
                            reasoning = m.get("reasoning_content")
                        rec = {"id": it["id"], "ok": True, "content": m.get("content") or "",
                               "reasoning": reasoning, "finish_reason": ch.get("finish_reason"),
                               "usage": d.get("usage"), "attempts": a + 1}
                        break
                    except Exception as e:  # noqa: BLE001
                        err = f"{type(e).__name__}: {e}"
                        await asyncio.sleep(min(60, 5 * 2 ** a))
                else:
                    rec = {"id": it["id"], "ok": False, "error": err}
            async with lock:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                n_done[0] += 1
                toks[0] += ((rec.get("usage") or {}).get("completion_tokens") or 0)
                if n_done[0] % 25 == 0 or n_done[0] == len(todo):
                    el = time.time() - t0
                    log(f"  {n_done[0]}/{len(todo)} done, {el/60:.1f} min, {toks[0]/max(el,1e-9):.0f} out tok/s"
                        + ("" if rec["ok"] else f"  last error: {rec['error'][:200]}"))

        await asyncio.gather(*(one(it) for it in todo))
        f.close()


def load_responses(path):
    """Latest record per id (ok records win over failures)."""
    out = {}
    if not os.path.exists(path):
        return out
    for r in read_jsonl(path):
        if r["id"] not in out or r.get("ok"):
            out[r["id"]] = r
    return out


def eprint(*a):
    print(*a, file=sys.stderr, flush=True)

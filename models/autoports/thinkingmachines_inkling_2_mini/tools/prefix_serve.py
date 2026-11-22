# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Prefix cache through a running Inkling-2-Mini vLLM server (stdlib only; run on the host), greedy, streamed:

  1. a long prompt (a document + a question) cold, then the same prompt again: TTFT drops, the answer is identical;
  2. the next chat turn (prompt 1 + its answer + a new question): resumes from the cached prompt;
  3. a different question about the same document (a shared prefix, different tail);
  4. N users at once, each with the shared document and their own question (packed prefill).

usage: python3 tools/prefix_serve.py [port=8000] [doc_tokens~=4000] [users=4]"""

import concurrent.futures as cf
import json
import os
import sys
import time
import urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
DOC = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
USERS = int(sys.argv[3]) if len(sys.argv) > 3 else 4
URL = f"http://localhost:{PORT}/v1/chat/completions"
MODEL = os.environ.get("MODEL", "inkling-2-mini")


def document(n_tokens):
    paras = []
    i = 0
    while len(" ".join(paras)) < n_tokens * 4:
        paras.append(f"Section {i}. The warehouse in district {i % 17} stores {100 + 7 * i} crates of item "
                     f"{chr(65 + i % 26)}{i}. Its manager, employee number {1000 + 13 * i}, inspects them every "
                     f"{2 + i % 5} days and reports to the regional office in city {i % 11}.")
        i += 1
    return "\n".join(paras)


def chat(messages, max_tokens=64):
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": 0, "stream": True,
            "chat_template_kwargs": {"reasoning_effort": "none"}, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    ttft, text, usage = None, "", {}
    with urllib.request.urlopen(req, timeout=1200) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            ev = json.loads(line[6:])
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices", []):
                d = ch.get("delta", {}).get("content")
                if d:
                    if ttft is None:
                        ttft = time.time() - t0
                    text += d
    return text, ttft or (time.time() - t0), usage.get("prompt_tokens", 0)


def main():
    doc = document(DOC)
    q1 = [{"role": "user", "content": doc + "\n\nHow many crates does section 12 store? Answer in one sentence."}]
    a1, t1, n1 = chat(q1)
    print(f"1a cold   : {n1} prompt tokens, TTFT {t1:.2f}s: {a1[:100]!r}", flush=True)
    a1b, t1b, _ = chat(q1)
    print(f"1b again  : TTFT {t1b:.2f}s, identical answer: {a1b == a1}", flush=True)
    q2 = q1 + [{"role": "assistant", "content": a1}, {"role": "user", "content": "And who manages it?"}]
    a2, t2, n2 = chat(q2)
    print(f"2  turn 2 : {n2} prompt tokens, TTFT {t2:.2f}s: {a2[:100]!r}", flush=True)
    q3 = [{"role": "user", "content": doc + "\n\nWhich city does the manager of section 30 report to?"}]
    a3, t3, n3 = chat(q3)
    print(f"3  same doc, new question: {n3} prompt tokens, TTFT {t3:.2f}s: {a3[:100]!r}", flush=True)
    qs = [[{"role": "user", "content": doc + f"\n\nHow often is section {5 + 9 * u} inspected?"}] for u in range(USERS)]
    t0 = time.time()
    with cf.ThreadPoolExecutor(USERS) as ex:
        res = list(ex.map(chat, qs))
    print(f"4  {USERS} users, shared doc: TTFT {', '.join(f'{r[1]:.2f}' for r in res)} s (wall {time.time() - t0:.1f}s)",
          flush=True)
    ok = a1b == a1 and t1b < 0.5 * t1
    print("PASS" if ok else "FAIL", flush=True)


if __name__ == "__main__":
    main()

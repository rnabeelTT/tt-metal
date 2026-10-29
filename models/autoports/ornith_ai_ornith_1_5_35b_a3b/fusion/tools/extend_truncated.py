"""Extend a finished text2 run to a larger token budget, exactly as if it had run at that budget.

    extend_truncated.py <run_dir> <new_label> --budget 81920 [--conc 16]

Writes a new run dir (sibling of <run_dir>, named <new_label>/<bench>) with responses.jsonl / reasoning.jsonl
holding, per item:
- text items that finished (finish_reason "stop") within the old budget: copied as is (a run at the larger budget
  produces the same distribution for them);
- text items cut at the old budget ("length"): the generation is CONTINUED from where it stopped: the server's own
  chat template (via /tokenize, add_generation_prompt) + the saved reasoning tokens go to /v1/completions with the
  remaining budget, so the result is one generation of up to --budget tokens (only the KV prefix is re-prefilled);
- image items: regenerated from scratch at the new budget through the chat API (the completions endpoint takes no
  images), all of them, so their distribution is exact too.
config.json records max_tokens=<budget> plus "extended_from". Then score / judge the new run as usual.
"""
import argparse
import concurrent.futures as cf
import json
import os
import sys
import threading
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def post(base, path, body, timeout):
    req = urllib.request.Request(base.rstrip("/").removesuffix("/v1") + path, data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("new_label")
    ap.add_argument("--budget", type=int, default=81920)
    ap.add_argument("--conc", type=int, default=16)
    ap.add_argument("--timeout", type=float, default=21600)
    a = ap.parse_args()
    src = Path(a.run_dir)
    cfg = json.load(open(src / "config.json"))
    old_budget = cfg["max_tokens"]
    assert a.budget > old_budget, "budget must grow"
    bench = cfg["bench"]
    out = Path(cfg["out"]).parent.parent / a.new_label / bench
    out.mkdir(parents=True, exist_ok=True)
    base, model = cfg["base_url"], cfg["model"]
    samp = {"temperature": cfg["temperature"], "top_p": cfg["top_p"], "top_k": cfg["top_k"]}
    if cfg.get("presence_penalty") is not None:
        samp["presence_penalty"] = cfg["presence_penalty"]

    mod = __import__(f"benches.{bench}", fromlist=["load_items", "messages"])
    items = {it["id"]: it for it in mod.load_items()}
    resp = {json.loads(l)["id"]: json.loads(l) for l in open(src / "responses.jsonl")}
    reas = {json.loads(l)["id"]: json.loads(l)["reasoning"] for l in open(src / "reasoning.jsonl")}
    missing = [i for i in items if i not in resp]
    assert not missing, f"{len(missing)} items have no response in {src}; finish the run first"

    done = set()
    if (out / "responses.jsonl").exists():
        done = {json.loads(l)["id"] for l in open(out / "responses.jsonl")}
    lock = threading.Lock()
    fr = open(out / "responses.jsonl", "a")
    fz = open(out / "reasoning.jsonl", "a")

    def emit(row, reasoning):
        with lock:
            fr.write(json.dumps(row) + "\n"); fr.flush()
            fz.write(json.dumps({"id": row["id"], "reasoning": reasoning}) + "\n"); fz.flush()

    todo = {"copy": [], "continue": [], "regen": []}
    for i, it in items.items():
        if i in done:
            continue
        if it.get("image"):
            todo["regen"].append(i)
        elif resp[i]["finish_reason"] == "length":
            todo["continue"].append(i)
        else:
            todo["copy"].append(i)
    print({k: len(v) for k, v in todo.items()}, "already done", len(done), flush=True)
    for i in todo["copy"]:
        r = dict(resp[i]); r["extend"] = "copied"
        emit(r, reas.get(i, ""))

    def cont(i):
        msgs = mod.messages(items[i])
        head = post(base, "/tokenize", {"model": model, "messages": msgs, "add_generation_prompt": True}, 600)["tokens"]
        tail = post(base, "/tokenize", {"model": model, "prompt": reas[i], "add_special_tokens": False}, 600)["tokens"]
        left = a.budget - len(tail)
        t0 = time.time()
        if left <= 0:
            text, fin, ntok = "", "length", 0
        else:
            d = post(base, "/v1/completions", {"model": model, "prompt": head + tail, "max_tokens": left,
                                               "stop": ["<|im_end|>"], **samp}, a.timeout)
            text, fin = d["choices"][0]["text"], d["choices"][0]["finish_reason"]
            ntok = d["usage"]["completion_tokens"]
        more, sep, content = text.partition("</think>")
        content = content.strip() if sep else ""
        if not sep:
            fin = "length"
        row = {"id": i, "content": content, "tool_calls": None, "finish_reason": fin,
               "usage": {"completion_tokens": len(tail) + ntok, "continued_tokens": ntok},
               "latency": resp[i].get("latency", 0) + time.time() - t0, "error": None, "extend": "continued"}
        emit(row, reas[i] + more)

    def regen(i):
        t0 = time.time()
        d = post(base, "/v1/chat/completions", {"model": model, "messages": mod.messages(items[i]),
                                                "max_tokens": a.budget, **samp}, a.timeout)
        m = d["choices"][0]["message"]
        row = {"id": i, "content": m.get("content") or "", "tool_calls": m.get("tool_calls"),
               "finish_reason": d["choices"][0]["finish_reason"], "usage": d.get("usage"),
               "latency": time.time() - t0, "error": None, "extend": "regenerated"}
        emit(row, m.get("reasoning") or m.get("reasoning_content") or "")

    jobs = [(cont, i) for i in todo["continue"]] + [(regen, i) for i in todo["regen"]]
    n = 0
    with cf.ThreadPoolExecutor(a.conc) as ex:
        futs = {ex.submit(f, i): i for f, i in jobs}
        for fut in cf.as_completed(futs):
            n += 1
            try:
                fut.result()
            except Exception as e:  # leave it out; a rerun picks it up
                print("FAILED", futs[fut], repr(e)[:300], flush=True)
            if n % 25 == 0 or n == len(jobs):
                print(f"[extend] {n}/{len(jobs)} {time.strftime('%T')}", flush=True)
    ncfg = dict(cfg, label=a.new_label, out=str(out), max_tokens=a.budget,
                extended_from={"run": str(src), "max_tokens": old_budget, "copied": len(todo["copy"]),
                               "continued": len(todo["continue"]), "regenerated": len(todo["regen"])})
    json.dump(ncfg, open(out / "config.json", "w"), indent=2)
    for f in ("sample_ids.json",):
        if (src / f).exists() and not (out / f).exists():
            (out / f).write_text((src / f).read_text())
    have = sum(1 for _ in open(out / "responses.jsonl"))
    print(f"EXTENDED {bench} {have}/{len(items)} -> {out}", flush=True)


if __name__ == "__main__":
    main()

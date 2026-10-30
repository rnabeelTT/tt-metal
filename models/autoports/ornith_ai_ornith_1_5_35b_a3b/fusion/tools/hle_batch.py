"""hle_batch.py: judge an HLE run incrementally, reusing verdicts already given (keyed by sha256 of the judge prompt).
  build <run_dir> <workdir> <store> [--text-only] : judge inputs for finished answers not yet in the store -> <workdir>/inputs.jsonl
  merge <workdir> <store>                         : append <workdir>/verdicts_*.jsonl to the store (validated ones only)
  tally <run_dir> <store>                         : HLE accuracy the official way: every sampled question counts, an answer
                                                    without a verdict (unfinished, empty, not judged) is wrong"""
import hashlib, json, sys, glob
from pathlib import Path
sys.path.insert(0, str(Path.home() / "evals/text2"))
sha = lambda s: hashlib.sha256(s.encode()).hexdigest()


def store_keys(store):
    p = Path(store)
    return {json.loads(l)["prompt_sha256"] for l in open(p)} if p.exists() else set()


def build(run_dir, work, store, text_only):
    from benches import hle
    items = {it["id"]: it for it in hle.load_items()}
    have = store_keys(store)
    rows = []
    for l in open(Path(run_dir) / "responses.jsonl"):
        r = json.loads(l); it = items[r["id"]]; ans = r.get("content") or ""
        if r["finish_reason"] != "stop" or not ans.strip() or (text_only and it["image"]):
            continue
        jp = hle.JUDGE_PROMPT.format(question=it["question"], correct_answer=it["answer"], response=ans)
        if sha(jp) not in have:
            rows.append({"id": it["id"], "judge_prompt": jp, "verdict_schema": hle.SCHEMA})
    Path(work).mkdir(parents=True, exist_ok=True)
    with open(Path(work) / "inputs.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    print(f"{len(rows)} new items to judge -> {work}/inputs.jsonl ({len(have)} already in store)")


def merge(work, store):
    prompts = {}
    for c in glob.glob(f"{work}/chunk_*.jsonl"):
        for l in open(c):
            r = json.loads(l); prompts[r["id"]] = r["judge_prompt"]
    have = store_keys(store); n = 0
    with open(store, "a") as f:
        for v in sorted(glob.glob(f"{work}/verdicts_*.jsonl")):
            for l in open(v):
                r = json.loads(l)
                if r["id"] not in prompts or r.get("correct") not in ("yes", "no"):
                    continue
                k = sha(prompts[r["id"]])
                if k in have:
                    continue
                f.write(json.dumps({"prompt_sha256": k, **r}) + "\n"); have.add(k); n += 1
    print(f"merged {n} verdicts; store now {len(have)}")


def tally(run_dir, store):
    import math
    from benches import hle
    run_dir = Path(run_dir)
    items = {it["id"]: it for it in hle.load_items()}
    ids = json.loads((run_dir / "sample_ids.json").read_text())["ids"] if (run_dir / "sample_ids.json").exists() else list(items)
    v = {}
    for l in open(store):
        r = json.loads(l); v[r["prompt_sha256"]] = r
    fin = {"stop": 0, "length": 0, "missing": 0}; ok = judged = 0
    resp = {json.loads(l)["id"]: json.loads(l) for l in open(run_dir / "responses.jsonl")}
    for i in ids:
        r = resp.get(i)
        if r is None:
            fin["missing"] += 1; continue
        fin["stop" if r["finish_reason"] == "stop" else "length"] += 1
        ans = r.get("content") or ""
        if not ans.strip():
            continue
        it = items[i]
        x = v.get(sha(hle.JUDGE_PROMPT.format(question=it["question"], correct_answer=it["answer"], response=ans)))
        if x:
            judged += 1; ok += x["correct"] == "yes"
    n = len(ids); acc = 100 * ok / n; hw = 196 * math.sqrt(acc / 100 * (1 - acc / 100) / n)
    print(f"accuracy {acc:.2f} +/- {hw:.2f} (n = {n}; correct {ok}, judged {judged}; finished {fin['stop']}, "
          f"cut at the token cap {fin['length']}, no response {fin['missing']}; unjudged count as wrong)")


if __name__ == "__main__":
    if sys.argv[1] == "tally":
        tally(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "build":
        build(sys.argv[2], sys.argv[3], sys.argv[4], "--text-only" in sys.argv)
    else:
        merge(sys.argv[2], sys.argv[3])

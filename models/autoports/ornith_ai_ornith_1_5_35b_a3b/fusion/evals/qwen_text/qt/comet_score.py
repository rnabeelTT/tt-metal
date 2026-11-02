"""Runs in ~/evals/wmt24pp_comet/.venv (unbabel-comet, CPU torch).
Scores {id, src, mt, ref} JSONL with a COMET model; writes {id, score} JSONL.
Resumable: ids already in --output are skipped (scored in chunks)."""
import argparse
import json
import os
import time

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Unbabel/XCOMET-XXL")
ap.add_argument("--input", required=True)
ap.add_argument("--output", required=True)
ap.add_argument("--threads", type=int, default=8)
ap.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
ap.add_argument("--batch-size", type=int, default=8)
ap.add_argument("--chunk", type=int, default=64)
a = ap.parse_args()

import torch  # noqa: E402
from comet import download_model, load_from_checkpoint  # noqa: E402

torch.set_num_threads(a.threads)
rows = [json.loads(l) for l in open(a.input)]
done = set()
if os.path.exists(a.output):
    done = {json.loads(l)["id"] for l in open(a.output)}
todo = [r for r in rows if r["id"] not in done]
print(f"{len(rows)} segments, {len(done)} already scored, {len(todo)} to score with {a.model} ({a.precision}, {a.threads} threads)", flush=True)
if todo:
    t0 = time.time()
    model = load_from_checkpoint(download_model(a.model))
    if a.precision == "bf16":
        model = model.to(torch.bfloat16)
    model.eval()
    print(f"loaded in {time.time() - t0:.0f}s", flush=True)
    with open(a.output, "a") as f:
        for i in range(0, len(todo), a.chunk):
            ch = todo[i:i + a.chunk]
            t1 = time.time()
            with torch.autocast("cpu", dtype=torch.bfloat16, enabled=a.precision == "bf16"):
                pred = model.predict([{"src": r["src"], "mt": r["mt"], "ref": r["ref"]} for r in ch],
                                     batch_size=a.batch_size, gpus=0, progress_bar=False, num_workers=1)
            for r, s in zip(ch, pred.scores):
                f.write(json.dumps({"id": r["id"], "score": float(s)}) + "\n")
            f.flush()
            print(f"  {i + len(ch)}/{len(todo)} scored, {len(ch) / (time.time() - t1):.2f} seg/s", flush=True)

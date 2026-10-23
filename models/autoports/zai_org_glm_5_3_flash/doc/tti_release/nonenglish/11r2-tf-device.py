"""Stage 11 review #2 (P2 non-English): device teacher forcing of fixed token texts under one precision policy
(GLM53_PRECISION), both paths:
  prefill  one forward over prompt + text[:-1] (Glm53Generator.prefill_logits, all rows)
  decode   prefill of the prompt, then the text forced one token per traced decode step (decode_forward,
           enable_trace=True, host logits)
Per scored position: the top-20 token ids + probabilities (softmax of the bf16 logits, T = 1) and p(forced token).
Usage (device, from a pinned snapshot): 11r2-tf-device.py TEXTS_JSON OUT_JSON [--paths prefill,decode]
Guard: the weight cache key directory must exist with the same checkpoint-file fingerprint (no new conversion)."""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch

ap = argparse.ArgumentParser()
ap.add_argument("texts")
ap.add_argument("out")
ap.add_argument("--paths", default="prefill,decode")
ap.add_argument("--seq-len", type=int, default=4096)
ap.add_argument("--topk", type=int, default=20)
a = ap.parse_args()
os.environ.setdefault("HF_MODEL", "zai-org/GLM-5.3-Flash")

import models.autoports.zai_org_glm_5_3_flash.tt as _tt  # noqa: E402
import ttnn  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tools import s7_probe as s7  # noqa: E402

tt_dir = Path(_tt.__file__).parent
md5 = {f.name: hashlib.md5(f.read_bytes()).hexdigest()[:12] for f in sorted(tt_dir.glob("*.py"))}
print("TFAB CODE", tt_dir, json.dumps(md5), flush=True)

# cache guard: the existing bundle cache directory, never a fresh conversion
from models.autoports.zai_org_glm_5_3_flash.tt import weight_cache as wc  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.ckpt import CheckpointReader  # noqa: E402

rd = CheckpointReader(os.environ["GLM53_CKPT"])
roots = [p for p in Path(os.path.expanduser(os.environ["GLM53_CACHE_DIR"])).iterdir() if p.is_dir()]
want = f"ckpt{wc.checkpoint_digest(rd)}"
hit = [p for p in roots if p.name.endswith(want)]
if not hit:
    sys.exit(f"TFAB guard: no cache dir ending {want} in {roots}")
man = json.loads((hit[0] / "manifest.json").read_text())
if man["checkpoint_files"] != wc.checkpoint_file_stats(rd):
    sys.exit("TFAB guard: checkpoint file fingerprint differs from the cache manifest (would convert again)")
print("TFAB cache guard OK", hit[0], flush=True)

items = json.loads(Path(a.texts).read_text())
rec = {
    "precision_env": os.environ.get("GLM53_PRECISION", ""),
    "code_md5": md5,
    "package": str(tt_dir),
    "cache": str(hit[0]),
    "ckpt": os.environ["GLM53_CKPT"],
    "results": {},
}


def rows(logits, gen):
    """logits [N, V] (fp32) predicting gen[0..N-1] -> per-position records."""
    lp = torch.log_softmax(logits.float(), dim=-1)
    top = torch.topk(lp, a.topk, dim=-1)
    out = []
    for i in range(logits.shape[0]):
        out.append(
            {
                "ids": top.indices[i].tolist(),
                "p": [round(float(x), 6) for x in top.values[i].exp()],
                "p_forced": round(float(lp[i, gen[i]].exp()), 6),
                "rank_forced": int((lp[i] > lp[i, gen[i]]).sum()),
            }
        )
    return out


mesh = s7._open_mesh(int(os.environ.get("GLM53_TRACE_REGION", 64 << 20)))
try:
    from models.autoports.zai_org_glm_5_3_flash.tt.generator import Glm53Generator
    from models.autoports.zai_org_glm_5_3_flash.tt.precision import PrecisionOverrides

    t0 = time.time()
    g = Glm53Generator(mesh, max_batch_size=1, max_seq_len=a.seq_len)
    rec["build_s"] = round(time.time() - t0)
    m = g.model
    rec["precision"] = PrecisionOverrides.from_env().record()
    rec["weight_cache"] = (
        {"root": str(m.weight_cache.root), "hits": m.weight_cache.hits, "stored": m.weight_cache.misses}
        if m.weight_cache is not None
        else None
    )
    print(f"TFAB built in {rec['build_s']} s; cache {rec['weight_cache']}", flush=True)
    V = m.vocab
    for it in items:
        P, gen = it["prompt_ids"], it["gen_ids"]
        res = {}
        if "prefill" in a.paths:
            lg = g.prefill_logits(P + gen[:-1])
            lg = lg.reshape(-1, lg.shape[-1])[:, :V]
            res["prefill"] = rows(lg[len(P) - 1 :], gen)
            g.reset()
        if "decode" in a.paths:
            g.reset()
            first = g.prefill_forward(torch.tensor([P]), prompt_lens=[len(P)])
            L = [first.reshape(-1, first.shape[-1])[-1, :V]]
            for i in range(len(gen) - 1):
                out = g.decode_forward(
                    torch.tensor([[gen[i]]]),
                    torch.tensor([len(P) + i]),
                    enable_trace=True,
                    on_device_sampling=False,
                    reload_inputs=True,
                )
                L.append(out.reshape(-1)[:V])
            res["decode"] = rows(torch.stack(L), gen)
            g.reset()
        rec["results"][it["name"]] = res
        msg = {k: round(sum(r["rank_forced"] == 0 for r in v) / len(v), 3) for k, v in res.items()}
        print(f"TFAB {it['name']}: forced-token top-1 rate {msg}", flush=True)
    if os.environ.get("TFAB_PERF", "1") == "1":  # decode step time at 1 slot (model trace + sampler, s7_probe)
        s7.run_decode(g, [1], 50, rec)
    g.teardown()
finally:
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rec) + "\n")
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

"""Stage 11 review #2 (P2 non-English): HF CPU reference for 11r2-tf-device.py (same texts, same output rows).
One forward per text (tools/hf_full_ref.py FullModel: unmodified HF Glm5NextTextModel, bf16, E192 experts via the
mask overlay). Usage (HF venv, CPU, under systemd-run MemoryMax): 11r2-tf-hf.py TEXTS_JSON CKPT OUT_JSON"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import torch

ap = argparse.ArgumentParser()
ap.add_argument("texts")
ap.add_argument("ckpt")
ap.add_argument("out")
ap.add_argument("--expert-cache-gb", type=float, default=12.0)
ap.add_argument("--threads", type=int, default=16)
ap.add_argument("--topk", type=int, default=20)
ap.add_argument(
    "--batch", action="store_true", help="one right-padded forward over all texts (each expert loaded once)"
)
ap.add_argument(
    "--expert-fmt",
    default=None,
    help="bfp4 | bfp8: round every routed expert weight as the port stores it "
    "(tests/hf_ref/bfp.py, blocks of 16 along the output dim): the HF reference with TT's expert grid",
)
a = ap.parse_args()
torch.set_num_threads(a.threads)
TT = Path.home() / "tt-metal"
spec = importlib.util.spec_from_file_location(
    "hf_full_ref", TT / "models/autoports/zai_org_glm_5_3_flash/tools/hf_full_ref.py"
)
R = importlib.util.module_from_spec(spec)
sys.modules["hf_full_ref"] = R
spec.loader.exec_module(R)

if a.expert_fmt:
    bspec = importlib.util.spec_from_file_location(
        "bfp", TT / "models/autoports/zai_org_glm_5_3_flash/tests/hf_ref/bfp.py"
    )
    B = importlib.util.module_from_spec(bspec)
    bspec.loader.exec_module(B)
    _orig_load = R.ExpertCache._load

    def fast_round(w, fmt):
        """B.bfp_round(w, fmt, dim=-2) in float32 arithmetic (exp2 of an integer is exact); checked bit-exact below."""
        Wb = B.MANT_BITS[fmt]
        x = w.float().t().contiguous()
        n = x.shape[-1]
        x = x.reshape(x.shape[0], n // 16, 16)
        bits = x.view(torch.int32)
        exp = (bits >> 23) & 0xFF
        shared = exp.amax(dim=-1, keepdim=True)
        diff = shared - exp
        m = ((bits & 0x7FFFFF) | (1 << 23)) >> diff.clamp(max=31)
        shift = 24 - Wb
        rv = m & ((1 << shift) - 1)
        m = m >> shift
        tie = 1 << (shift - 1)
        m = (m + ((rv > tie) | ((rv == tie) & ((m & 1) == 1))).int()).clamp(max=(1 << Wb) - 1)
        m = torch.where((exp == 0) | (diff >= 24), 0, m)
        val = m.float() * torch.exp2((shared - 127 - (Wb - 1)).float())
        val = torch.where(bits < 0, -val, val)
        return val.reshape(x.shape[0], n).t().to(w.dtype)

    _t = (torch.randn(256, 64) * torch.logspace(-3, 2, 64)).to(torch.bfloat16)
    assert torch.equal(
        fast_round(_t, a.expert_fmt).view(torch.int16), B.bfp_round(_t, a.expert_fmt, dim=-2).view(torch.int16)
    )

    def _load_q(self, layer, e):
        gu, dn = _orig_load(self, layer, e)
        return fast_round(gu, a.expert_fmt), fast_round(dn, a.expert_fmt)

    R.ExpertCache._load = _load_q
    R.log(f"routed experts rounded to {a.expert_fmt} (blocks along the output dim)")

items = json.loads(Path(a.texts).read_text())
fm = R.FullModel(Path(a.ckpt), a.expert_cache_gb)
rec = {"ckpt": a.ckpt, "expert_fmt": a.expert_fmt, "results": {}}
seqs = [it["prompt_ids"] + it["gen_ids"][:-1] for it in items]
if a.batch:
    Lmax = max(map(len, seqs))
    ids = torch.tensor([q + [q[-1]] * (Lmax - len(q)) for q in seqs])  # right padding: causal, rows < len unaffected
    t0 = time.time()
    with torch.no_grad():
        LG = fm.logits(ids).float()
    R.log(f"batched forward {tuple(ids.shape)} in {time.time() - t0:.0f} s")
for k, it in enumerate(items):
    P, gen = it["prompt_ids"], it["gen_ids"]
    t0 = time.time()
    if a.batch:
        lg = LG[k, len(P) - 1 : len(seqs[k])]
    else:
        with torch.no_grad():
            lg = fm.logits(torch.tensor([seqs[k]]))[0, len(P) - 1 :].float()
    lp = torch.log_softmax(lg, dim=-1)
    top = torch.topk(lp, a.topk, dim=-1)
    rows = []
    for i in range(lg.shape[0]):
        rows.append(
            {
                "ids": top.indices[i].tolist(),
                "p": [round(float(x), 6) for x in top.values[i].exp()],
                "p_forced": round(float(lp[i, gen[i]].exp()), 6),
                "rank_forced": int((lp[i] > lp[i, gen[i]]).sum()),
            }
        )
    rec["results"][it["name"]] = {"hf": rows}
    R.log(f"{it['name']}: {len(seqs[k])} tokens in {time.time() - t0:.0f} s")
    Path(a.out).write_text(json.dumps(rec) + "\n")
rec["peak_rss_gb"] = R.peak_rss_gb()
Path(a.out).write_text(json.dumps(rec) + "\n")
print("HF done", rec["peak_rss_gb"])

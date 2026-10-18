# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 6 weight-cache check (tt/weight_cache.py): build the model, time the weight load, record a byte digest of the
device weights and the logits of one prefill + one eager decode step; --compare checks runs for bit-identity.

  run:     GLM53_CACHE_DIR=<dir | off> GLM53_TP=4 GLM53_LAYERS=0,3,4 \
           python -m models.autoports.zai_org_glm_5_3_flash.tools.weight_cache_check --tag cold --out DIR
  compare: python -m models.autoports.zai_org_glm_5_3_flash.tools.weight_cache_check --out DIR --compare nocache cold warm

Weight digests: each tensor is dumped from the device (ttnn.dump_tensor: header + per-device shard bytes, so bfp4 /
bfp8 blocks are compared as stored) and the data region hashed. --hash-layers limits the digest to some layers.
"""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch

import ttnn

WEIGHTS = {
    "kda": ("input_projection", "output_projection", "decay_output_projection", "decay_bias_flat", "decay_scale_flat"),
    "dsa": ("w_q_a", "w_q_b", "w_kv_a", "w_uk", "w_uv_t", "w_o", "w_iq", "w_ik", "w_iw", "w_ig"),
    "moe": ("up_gate", "down", "shared_packed", "shared_w2", "router_w", "router_bias"),
    "dense": ("wg", "wu", "wgu", "wd"),  # wg / wu, or wgu once packed (fused decoder)
}


def _inner(m):
    return m.__dict__.get("_inner", m)


def named_weights(model, layers):
    out = {"embed_w": model.embed_w, "lm_head": model.lm_head}
    for d in model.layers:
        if layers is not None and d.layer_idx not in layers:
            continue
        attn = _inner(d.attn)
        src = attn.weights if d.attn_kind == "kda" else attn
        for n in WEIGHTS[d.attn_kind]:
            out[f"L{d.layer_idx}.{d.attn_kind}.{n}"] = getattr(src, n, None)
        if d.attn_kind == "kda":
            for j, t in enumerate(attn.weights.convolution_taps):
                out[f"L{d.layer_idx}.kda.conv_tap_{j}"] = t
        mlp = _inner(d.mlp)
        for n in WEIGHTS[d.mlp_kind]:
            out[f"L{d.layer_idx}.{d.mlp_kind}.{n}"] = getattr(mlp, n, None)
    # skip tensors a later stage replaced and freed (dumping a deallocated tensor segfaults)
    return {k: v for k, v in out.items() if v is not None and v.is_allocated()}


def digest(t, tmp: Path) -> dict:
    ttnn.dump_tensor(tmp, t)
    raw = tmp.read_bytes()
    tmp.unlink()
    hs = int.from_bytes(raw[:8], "little")
    return {
        "dtype": str(t.dtype),
        "layout": str(t.layout),
        "shape": list(t.shape),
        "mem": str(t.memory_config().buffer_type),
        "data_sha": hashlib.sha256(raw[8 + hs :]).hexdigest()[:20],
        "header_sha": hashlib.sha256(raw[8 : 8 + hs]).hexdigest()[:20],
        "bytes": len(raw),
    }


def evict_page_cache(base: str) -> int:
    """Drop the cache files from the page cache (POSIX_FADV_DONTNEED, no root needed): a warm load then reads the disk,
    as after a reboot. Returns the bytes evicted."""
    n = 0
    for f in Path(os.path.expanduser(base)).rglob("*.tensorbin"):
        fd = os.open(f, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            n += os.fstat(fd).st_size
        finally:
            os.close(fd)
    return n


def run(a):
    if a.evict and os.environ.get("GLM53_CACHE_DIR"):
        print(f"evicted {evict_page_cache(os.environ['GLM53_CACHE_DIR']) / 1e9:.1f} GB from the page cache", flush=True)
    from models.autoports.zai_org_glm_5_3_flash.tests.harness import close_mesh
    from models.autoports.zai_org_glm_5_3_flash.tt.generator import L1_SMALL_SIZE, Glm53Generator

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(
        ttnn.MeshShape(1, int(os.environ.get("GLM53_TP", "4"))),
        l1_small_size=L1_SMALL_SIZE,
        trace_region_size=512 << 20,
    )
    try:
        t0 = time.time()
        g = Glm53Generator(mesh, max_batch_size=a.batch, max_seq_len=4096)
        t_gen = time.time() - t0
        m = g.model
        wc = m.weight_cache
        rec = {
            "tag": a.tag,
            "layers": m.layer_indices,
            "cache_dir": str(wc.root) if wc else None,
            "cache_hits": wc.hits if wc else 0,
            "cache_stored": wc.misses if wc else 0,
            "weight_load_s": round(m.load_seconds, 1),
            "generator_build_s": round(t_gen, 1),
        }
        print(f"ok load {json.dumps(rec)}", flush=True)
        layers = None if a.hash_layers is None else {int(s) for s in a.hash_layers.split(",")}
        tmp = out / f".{a.tag}.digest.tensorbin"
        rec["weights"] = {k: digest(t, tmp) for k, t in named_weights(m, layers).items()}
        print(f"ok digests ({len(rec['weights'])} tensors)", flush=True)

        prompt = torch.arange(1000, 1075).reshape(1, -1)
        logits = g.prefill_forward(prompt, prompt_lens=[75])
        pre = logits[0, -1, : m.vocab].float().clone()
        first = int(torch.argmax(pre))
        pos = [-1] * a.batch
        pos[0] = 75
        m.inputs.set_positions(pos)
        g._write_tokens([first])
        lg = m.decode_step(g._tok_view())
        dec = m.logits_to_host(lg)[0, 0, 0, : m.vocab].float().clone()
        ttnn.deallocate(lg)
        rec["prefill_argmax"], rec["decode_argmax"] = first, int(torch.argmax(dec))
        torch.save({"prefill": pre, "decode": dec}, out / f"{a.tag}.logits.pt")
        (out / f"{a.tag}.json").write_text(json.dumps(rec, indent=1))
        print(f"ok logits prefill argmax {first}, decode argmax {rec['decode_argmax']}", flush=True)
        g.teardown()
    finally:
        close_mesh(mesh)


def compare(a):
    out = Path(a.out)
    recs = {t: json.loads((out / f"{t}.json").read_text()) for t in a.compare}
    lg = {t: torch.load(out / f"{t}.logits.pt") for t in a.compare}
    base = a.compare[0]
    ok = True
    for t in a.compare:
        r = recs[t]
        print(
            f"{t}: weight load {r['weight_load_s']} s, generator build {r['generator_build_s']} s, "
            f"cache hits {r['cache_hits']} stored {r['cache_stored']} ({r['cache_dir']})"
        )
    for t in a.compare[1:]:
        common = sorted(set(recs[base]["weights"]) & set(recs[t]["weights"]))
        diff = [k for k in common if recs[base]["weights"][k]["data_sha"] != recs[t]["weights"][k]["data_sha"]]
        hdr = [k for k in common if recs[base]["weights"][k]["header_sha"] != recs[t]["weights"][k]["header_sha"]]
        same_logits = all(torch.equal(lg[base][k], lg[t][k]) for k in ("prefill", "decode"))
        print(
            f"{base} vs {t}: {len(common) - len(diff)}/{len(common)} weights byte-identical"
            f"{' (differ: ' + ', '.join(diff) + ')' if diff else ''}; headers differ: {hdr or 'none'}; "
            f"prefill+decode logits bit-identical: {same_logits}"
        )
        ok &= not diff and not hdr and same_logits
    print("RESULT", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tag")
    p.add_argument("--out", required=True)
    p.add_argument("--hash-layers", default=None, help="comma list of layer ids to digest (default: all built)")
    p.add_argument("--compare", nargs="+")
    p.add_argument("--batch", type=int, default=32, help="decode slots (the full model fits 8 at 4096 context)")
    p.add_argument("--evict", action="store_true", help="drop the cache files from the page cache first")
    a = p.parse_args()
    compare(a) if a.compare else run(a)

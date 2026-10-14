# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""HF reference for one GLM-5.3-Flash decoder layer (runs in the HF venv, transformers >= 5.16, never imports ttnn).

The TTNN environment pins transformers 5.12, which has no `glm5_next`, so the pytest harness calls this script as a
subprocess (see tests/hf_ref/client.py). The layer is the unmodified HF `Glm5NextTextDecoderLayer`, including both
mHC wrappers; its input and output are the 4-stream residual X [B, S, 4, 4096] bf16.

Request (torch.save dict, --request):
  layer:     decoder layer index (0..44)
  weights:   "real" | {"synthetic": seed, "stats": path}
  x:         [B, S_total, 4, 4096] bf16 input stream (or "tokens": [B, S_total] int64 -> embedding + layers 0..layer-1
             with real weights produce the input stream; recorded as result["x"])
  segments:  list of segment lengths; the first is the prefill, each later one continues with the HF cache
             (length 1 = decode step, longer = chunked continuation)
  mask:      optional [B, S_total] bool padding mask (True = real token)
  capture:   bool, also return layer state (KDA conv/recurrent state, DSA latent / indexer rows, top-k indices)
Result (torch.save dict, --out): y [B, S_total, 4, 4096] bf16, per-segment state when capture is set.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve()
AUTOPORT = HERE.parents[2]


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ckpt = _load_module("glm53_ckpt", AUTOPORT / "tt" / "ckpt.py")
synth = _load_module("glm53_synthetic", AUTOPORT / "tt" / "synthetic.py")

from transformers.cache_utils import DynamicCache  # noqa: E402
from transformers.models.glm5_next import modeling_glm5_next as M  # noqa: E402
from transformers.models.glm5_next.configuration_glm5_next import Glm5NextConfig  # noqa: E402

FP32_STRICT = tuple(getattr(M.Glm5NextPreTrainedModel, "_keep_in_fp32_modules_strict", None) or ())

RENAMES = [
    ("self_attn.f_a_proj.", "self_attn.forget_gate.f_a_proj."),
    ("self_attn.f_b_proj.", "self_attn.forget_gate.f_b_proj."),
    ("self_attn.dt_bias", "self_attn.forget_gate.dt_bias"),
    ("self_attn.A_log", "self_attn.forget_gate.A_log"),
    ("hc_attn_fn", "attn_hc.fn"),
    ("hc_attn_base", "attn_hc.base"),
    ("hc_attn_scale", "attn_hc.scale"),
    ("hc_ffn_fn", "ffn_hc.fn"),
    ("hc_ffn_base", "ffn_hc.base"),
    ("hc_ffn_scale", "ffn_hc.scale"),
]


def text_config(ckpt_dir: Path, num_experts: int = 192):
    raw = json.loads((ckpt_dir / "config.json").read_text())
    raw["text_config"]["n_routed_experts"] = num_experts  # E192 (the parent config.json says 288 until the build ends)
    raw["text_config"]["num_nextn_predict_layers"] = 0
    cfg = Glm5NextConfig(**raw)
    tc = cfg.text_config
    tc._attn_implementation = "sdpa"
    return tc


def to_hf_names(sd: dict[str, torch.Tensor], num_experts: int) -> dict[str, torch.Tensor]:
    """Checkpoint suffixes of one layer -> HF module state dict names (transformers conversion_mapping glm5_next)."""
    out = {}
    convs = {}
    gate, up, down = {}, {}, {}
    for name, t in sd.items():
        if name.startswith("mlp.experts."):
            _, _, e, proj, _ = name.split(".")
            {"gate_proj": gate, "up_proj": up, "down_proj": down}[proj][int(e)] = t
            continue
        if name in ("self_attn.q_conv1d.weight", "self_attn.k_conv1d.weight", "self_attn.v_conv1d.weight"):
            convs[name.split(".")[1][0]] = t
            continue
        for a, b in RENAMES:
            if name.startswith(a) or a in name:
                name = name.replace(a, b)
                break
        out[name] = t
    if convs:
        out["self_attn.conv1d.weight"] = torch.cat([convs["q"], convs["k"], convs["v"]], dim=0)
    if gate:
        out["mlp.experts.gate_up_proj"] = torch.stack(
            [torch.cat([gate[e], up[e]], dim=0) for e in range(num_experts)], dim=0
        )
        out["mlp.experts.down_proj"] = torch.stack([down[e] for e in range(num_experts)], dim=0)
    return out


def hf_dtype(name: str, t: torch.Tensor) -> torch.Tensor:
    """HF load semantics with dtype=bfloat16: everything bf16 except _keep_in_fp32_modules_strict."""
    if any(k in name for k in FP32_STRICT):
        return t.float()
    return t.to(torch.bfloat16)


def build_layer(tc, layer: int, sd_ckpt: dict[str, torch.Tensor]):
    with torch.device("meta"):
        mod = M.Glm5NextTextDecoderLayer(tc, layer)
    hf_sd = {n: hf_dtype(n, t) for n, t in to_hf_names(sd_ckpt, tc.num_local_experts).items()}
    missing = set(mod.state_dict()) - set(hf_sd)
    extra = set(hf_sd) - set(mod.state_dict())
    if missing or extra:
        raise RuntimeError(f"layer {layer}: missing {sorted(missing)} extra {sorted(extra)}")
    mod.load_state_dict(hf_sd, assign=True, strict=True)
    # buffers (e_score_correction_bias is a Buffer in the router) keep fp32 like HF
    mod.eval()
    return mod


def layer_weights(spec, layer: int, rd):
    if spec == "real":
        return rd.layer_state(layer, dtype=torch.bfloat16)
    stats = json.loads(Path(spec["stats"]).read_text())
    return synth.synthetic_layer_state(stats, layer, seed=int(spec["synthetic"]))


@torch.no_grad()
def run_long(req: dict, rd, tc, layer: int, mod) -> dict:
    """Long-context reference without materializing the stream: the prompt's first n_prompt - tail rows only fill the
    layer cache (DSA: latent + indexer rows through the HF submodules, no attention / MLP; KDA: the full layer forward,
    output discarded - the recurrence needs it), then the HF layer runs on the last `tail` prompt rows and `decode`
    single-token steps. Input rows come from synthetic.synthetic_stream_segment (same generator as the TT test)."""
    spec = req["long"]
    stats = json.loads(Path(spec["stats"]).read_text())
    N, tail, D, seg = int(spec["n_prompt"]), int(spec["tail"]), int(spec["decode"]), int(spec.get("seg", 2048))
    seed = int(spec["seed"])
    gen = lambda a, n: synth.synthetic_stream_segment(stats, layer, seed, a, n).unsqueeze(0)  # [1, n, 4, H]
    cache = DynamicCache(config=tc)
    block = tc.layer_types[layer]
    fill_end = N - tail
    a = 0
    t0 = time.time()
    while a < fill_end:
        n = min(seg, fill_end - a)
        x = gen(a, n)
        m = torch.ones(1, n, dtype=torch.bool)
        if block == "linear_attention":  # the recurrence needs the attention block; the MLP does not touch the cache
            _, _, h = mod.attn_hc(x)
            mod.self_attn(hidden_states=mod.input_layernorm(h), cache_params=cache, attention_mask=m)
        else:
            _, _, h = mod.attn_hc(x)
            h = mod.input_layernorm(h)
            at = mod.self_attn
            c = at.kv_a_layernorm(at.kv_a_proj_with_mqa(h))
            cache.update(c.view(1, 1, n, -1), c.new_zeros(1, 1, n, 0), layer)
            ix = at.indexer
            k = ix.k_norm(ix.wk(h))
            gsc = torch.nn.functional.linear(h, ix.index_kpool_compress_gate)
            cache.update_indexer(torch.cat([k, gsc, m.to(k.dtype)[..., None]], dim=-1), layer)
        a += n
        if (a // seg) % 32 == 0:
            print(f"fill {a}/{fill_end} {time.time() - t0:.0f} s", flush=True)
    xt = gen(fill_end, tail + D)
    ys = []
    y, _ = mod(
        xt[:, :tail], attention_mask=torch.ones(1, tail, dtype=torch.bool), past_key_values=cache, use_cache=True
    )
    ys.append(y)
    for i in range(D):
        y, _ = mod(
            xt[:, tail + i : tail + i + 1],
            attention_mask=torch.ones(1, 1, dtype=torch.bool),
            past_key_values=cache,
            use_cache=True,
        )
        ys.append(y)
    return {"x": xt, "y": torch.cat(ys, dim=1), "fill_seconds": time.time() - t0}


def capture_state(cache, layer: int, block_type: str) -> dict:
    lay = cache.layers[layer]
    if block_type == "linear_attention":
        return {
            "conv_state": lay.conv_states[0].detach().clone(),
            "recurrent_state": lay.recurrent_states[0].detach().clone(),
        }
    return {"latent": lay.keys.detach().clone(), "indexer_rows": lay.indexer_keys.detach().clone()}


@torch.no_grad()
def run(req: dict) -> dict:
    t0 = time.time()
    ckpt_dir = Path(req.get("ckpt") or os.environ["GLM53_CKPT"])
    rd = ckpt.CheckpointReader(ckpt_dir)
    tc = text_config(ckpt_dir, rd.num_experts)
    layer = int(req["layer"])
    torch.manual_seed(0)

    result = {}
    if "long" in req:
        mod = build_layer(tc, layer, layer_weights(req["weights"], layer, rd))
        res = run_long(req, rd, tc, layer, mod)
        res["seconds"] = time.time() - t0
        res["block_type"] = tc.layer_types[layer]
        return res
    if "x" in req:
        x = req["x"].to(torch.bfloat16)
    else:  # real-token input stream: embedding + layers 0..layer-1 (real weights), full prefill
        ids = req["tokens"]
        emb = rd.top("embed_tokens.weight")[ids]
        x = emb.unsqueeze(2).expand(-1, -1, tc.hc_mult, -1).contiguous()
        cache = DynamicCache(config=tc)
        m = torch.ones(ids.shape, dtype=torch.bool)
        for li in range(layer):
            mod = build_layer(tc, li, rd.layer_state(li))
            x, _ = mod(x, attention_mask=m, past_key_values=cache, use_cache=True, position_embeddings=None)
            del mod
        result["x"] = x.clone()

    mod = build_layer(tc, layer, layer_weights(req["weights"], layer, rd))
    block_type = tc.layer_types[layer]
    taps = {}
    if req.get("taps"):  # sub-block inputs / outputs of the FIRST segment (diagnostics)

        def tap(name, which):
            def hook(m, args, out):
                if name in taps:
                    return
                o = out[0] if isinstance(out, tuple) and which == "out0" else out
                if which == "in0":
                    o = args[0] if args else None
                if which == "hc":
                    o = out[2]  # (post, comb, collapsed)
                taps[name] = o.detach().clone() if torch.is_tensor(o) else o

            return hook

        mod.attn_hc.register_forward_hook(tap("attn_collapsed", "hc"))
        mod.input_layernorm.register_forward_hook(tap("attn_in", "out"))
        mod.self_attn.register_forward_hook(tap("attn_out", "out0"))
        mod.ffn_hc.register_forward_hook(tap("X1", "in0"))
        mod.ffn_hc.register_forward_hook(tap("ffn_collapsed", "hc"))
        mod.post_attention_layernorm.register_forward_hook(tap("ffn_in", "out"))
        mod.mlp.register_forward_hook(tap("ffn_out", "out"))
    B, S_total = x.shape[:2]
    segments = list(req.get("segments") or [S_total])
    assert sum(segments) == S_total, (segments, S_total)
    mask = req.get("mask")
    cache = DynamicCache(config=tc)
    ys, states, topks = [], [], []
    s0 = 0
    for seg in segments:
        xs = x[:, s0 : s0 + seg]
        ms = mask[:, s0 : s0 + seg] if mask is not None else torch.ones(B, seg, dtype=torch.bool)
        y, topk = mod(xs, attention_mask=ms, past_key_values=cache, use_cache=True, position_embeddings=None)
        ys.append(y)
        if req.get("capture"):
            states.append(capture_state(cache, layer, block_type))
            topks.append(topk.clone() if topk is not None else None)
        s0 += seg
    result["y"] = torch.cat(ys, dim=1)
    if req.get("capture"):
        result["states"] = states
        result["topk"] = topks
    if taps:
        result["taps"] = taps
    result["seconds"] = time.time() - t0
    result["block_type"] = block_type
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--request", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    req = torch.load(args.request, weights_only=False)
    res = run(req)
    tmp = args.out + ".tmp"
    torch.save(res, tmp)
    os.replace(tmp, args.out)
    print(f"hf_layer_ref layer {req['layer']} done in {res['seconds']:.1f} s", flush=True)


if __name__ == "__main__":
    main()

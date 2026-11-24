# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""CPU BF16 reference for tencent/Hy4-flash (text backbone), the quality yardstick of the TT port.

Runs the HF modeling code itself (transformers PR 50903, ManaEstras/transformers@c2ec850: `HYV4FlashTextModel.forward`
with its masks, indexer and caches), loaded lazily so it fits in host RAM: the attention / router / norm weights
(~13 GB) are read into RAM in the dtypes `from_pretrained(dtype=bfloat16)` gives them (fp32 for
`_keep_in_fp32_modules_strict`), while the routed experts (290 GB) and the Engram tables (127 GB) stay memory-mapped
from the safetensors shards: only the experts / rows a token touches are read, through the page cache.

  python ref_hy4.py tf  <texts.json> <out dir>      teacher forced: per sequence top-32 log-probs, NLL, argmax==next
  python ref_hy4.py gen <prompts.json> <out dir>    greedy generation (REF_MAX_NEW tokens) from chat prompts

tf outputs (same format as ref_m27 / ref_inkling):
  ref_<name>.npz   top_ids [T, 32] int32, top_lp [T, 32] fp16, lse (log-sum-exp of the fp32 logits) [T], nll [T-1],
                   acc [T-1] (argmax == the real next token)
  routing.npz      names, layers [41] (MoE layers 1..41), ids [B, 41, T, 16] uint16, w [B, 41, T, 16] fp16 (the
                   weights as applied: renormalised and x routed_scaling_factor), per length group
  gold_<name>.pt   (REF_GOLD=1, first sequence of each group) streams entering every layer [L, T, 4, H] bf16 and the
                   final normed hidden [T, H]: the fixtures the device layer tests compare against
gen outputs: gen.json {name: {"prompt_ids", "ids", "text"}}.

Texts: {name: [token ids]}; sequences of equal length run as one batch. Long sequences run in query chunks
(REF_CHUNK, default 4096) against the HF DynamicCache, as HF's own chunked prefill: past 2048 tokens of context the
indexed layers' compressed indexer selects the recalled groups.

env: REF_THREADS (8), REF_ATTN (fp32 | hf: hf = transformers' eager attention, whose softmax runs in bf16; fp32 =
     the same math with q / k / v / sinks in fp32, as the serving kernels accumulate; default fp32), REF_CHUNK,
     REF_GOLD, REF_MAX_NEW (64), HF_HUB, HY4_REPO
"""

import json
import os
import re
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engram_store import EngramHasher, snapshot, tensor_memmap, weight_map  # noqa: E402

from transformers import AutoConfig  # noqa: E402
from transformers.cache_utils import DynamicCache  # noqa: E402
from transformers.models.hy_v4_flash import modeling_hy_v4_flash as M  # noqa: E402

torch.set_num_threads(int(os.environ.get("REF_THREADS", "8")))
SNAP = snapshot()
WMAP = weight_map()
FP32_KEEP = M.HYV4FlashPreTrainedModel._keep_in_fp32_modules_strict
IMAGE_TOKEN_ID, VIDEO_TOKEN_ID = 120120, 120124

# transformers conversion_mapping "hy_v4_flash" (text tower)
RENAMES = [
    (r"\.self_attn\.learnable_sink_param$", ".self_attn.sinks"),
    (r"\.self_attn\.linear_gate\.weight$", ".self_attn.gate_proj.weight"),
    (r"\.hc_attn_layer\.hc_pre\.hc_(fn|base|scale)$", r".attn_hc.\1"),
    (r"\.hc_mlp_layer\.hc_pre\.hc_(fn|base|scale)$", r".ffn_hc.\1"),
    (r"^\.hc_head\.hc_head_(fn|base|scale)$", r".hc_head.hc_\1"),
]
SKIP = ("model.dfly_layers.",)  # the DFly2 drafter: served separately (dflash port), not part of the backbone
LAZY = ("mlp.experts.gate_up_proj", "mlp.experts.down_proj", "multi_head_embedding.embedding.weight")


def config():
    cfg = AutoConfig.from_pretrained(SNAP).text_config
    cfg._attn_implementation = "eager"  # the only attention the model supports (sinks + indexer mask)
    cfg._experts_implementation = "eager"  # the per-expert loop: touches only the routed experts' pages
    return cfg


def module_name(ckpt_key):
    name = "." + ckpt_key
    for pat, rep in RENAMES:
        name = re.sub(pat, rep, name)
    return name[1:]


def ckpt_tensor(key, lazy=False):
    """Checkpoint tensor; lazy: a zero-copy view of the memory-mapped shard (read-only)."""
    mm = tensor_memmap(key, WMAP)
    a = mm if lazy else np.array(mm)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # non-writable memmap: never written
        t = torch.from_numpy(a)
    if a.dtype == np.uint16:
        t = t.view(torch.bfloat16)
    return t


def materialize(module, prefix, cfg):
    """Fill a meta-initialised module from the checkpoint keys under `prefix` (HF dtypes; experts / Engram tables
    memory-mapped) and recompute its non-persistent buffers. Raises on any unmatched name."""
    module.to_empty(device="cpu")
    params = dict(module.named_parameters()) | dict(module.named_buffers())
    loaded = set()
    for key in WMAP:
        if not key.startswith(prefix) or key.startswith(SKIP):
            continue
        name = module_name(key[len(prefix) :]) if prefix else module_name(key)
        if name not in params:
            raise RuntimeError(f"{key} -> {name}: no such parameter")
        lazy = name.endswith(LAZY)
        t = ckpt_tensor(key, lazy=lazy)
        p = params[name]
        if t.shape != p.shape:
            raise RuntimeError(f"{key}: checkpoint {tuple(t.shape)} vs module {tuple(p.shape)}")
        if t.is_floating_point() and not lazy:
            t = t.to(torch.float32 if any(s in name for s in FP32_KEEP) else torch.bfloat16)
        owner, _, leaf = name.rpartition(".")
        mod = module.get_submodule(owner) if owner else module
        if isinstance(getattr(mod, leaf), torch.nn.Parameter):
            setattr(mod, leaf, torch.nn.Parameter(t, requires_grad=False))
        else:
            mod._buffers[leaf] = t
        loaded.add(name)
    # non-persistent buffers: recompute as `_init_weights` does
    for mname, mod in module.named_modules():
        pre = mname + "." if mname else ""
        if isinstance(mod, M.HYV4FlashRotaryEmbedding):
            fresh = M.HYV4FlashRotaryEmbedding(cfg)
            for b, v in fresh.named_buffers():
                mod._buffers[b] = v
                loaded.add(pre + b)
        elif isinstance(mod, M.HYV4FlashEngramLayer):
            mod.head_vocab_sizes = torch.tensor(mod.config_head_vocab_sizes)
            mod.hash_multipliers = mod.draw_hash_multipliers(int(mod.lookup_table.max()) + 1)
            loaded |= {pre + "head_vocab_sizes", pre + "hash_multipliers"}
        elif isinstance(mod, M.HYV4FlashEngramEmbedding):
            mod.offsets = mod.head_offsets()
            loaded.add(pre + "offsets")
    missing = [n for n in params if n not in loaded]
    if missing:
        raise RuntimeError(f"{prefix}: not loaded {missing}")
    return module.eval()


def fp32_eager_attention(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    """transformers' eager attention with sinks, in fp32 (its softmax otherwise runs in bf16)."""
    out, w = _HF_EAGER(module, query.float(), key.float(), value.float(), attention_mask, scaling, dropout, **kwargs)
    return out.to(query.dtype), w


_HF_EAGER = M.eager_attention_forward
if os.environ.get("REF_ATTN", "fp32") == "fp32":
    M.eager_attention_forward = fp32_eager_attention  # looked up at call time by HYV4FlashAttention.forward


class Ref:
    def __init__(self):
        self.cfg = cfg = config()
        t0 = time.time()
        with torch.device("meta"):
            tm = M.HYV4FlashTextModel(cfg)
        self.tm = materialize(tm, "model.", cfg)
        # the vision tower, drafter and lm_head are outside model.layers / model.*: lm_head separately (fp32)
        self.lm_head = ckpt_tensor("lm_head.weight").float()  # [V, H], as `_keep_in_fp32_modules_strict` lm_head
        self.hashers = {li: EngramHasher(li) for li in (1, 11, 21, 31)}
        self.pad_id = cfg.engram_pad_id
        print(f"reference built in {time.time() - t0:.0f}s", flush=True)

    def engram_inputs(self, ids, start=0, history=None):
        """[B, T] ids (one document per row, positions start..) -> ngram_tokens [B, T, 3], engram_mask [B, T]
        (HYV4FlashModel.build_engram_inputs; the hashing itself stays in the HF layer)."""
        h = next(iter(self.hashers.values()))
        ng = torch.stack(
            [
                torch.from_numpy(h.ngram_tokens(r.numpy(), start, None if history is None else history[b].numpy()))
                for b, r in enumerate(ids)
            ]
        )
        mask = (ids != IMAGE_TOKEN_ID) & (ids != VIDEO_TOKEN_ID) & (ids != self.pad_id)
        return ng, mask

    def logits(self, hidden):
        return hidden.float() @ self.lm_head.T  # HF: lm_head(h.to(fp32)), fp32 weight

    @torch.no_grad()
    def forward_chunked(self, ids, chunk, hooks=None):
        """Teacher-forced forward of [B, T] ids through HYV4FlashTextModel in query chunks against one cache.
        Returns the final normed hidden [B, T, H]."""
        B, T = ids.shape
        cache = DynamicCache(config=self.cfg)
        chunk = T if chunk <= 0 else chunk
        outs = []
        for s in range(0, T, chunk):
            e = min(T, s + chunk)
            ng, em = self.engram_inputs(ids[:, s:e], s, ids[:, :s] if s else None)
            pos = torch.arange(s, e).view(1, 1, -1).expand(3, B, -1)
            out = self.tm(
                input_ids=ids[:, s:e], position_ids=pos, cache=cache, use_cache=True, ngram_tokens=ng, engram_mask=em
            )
            outs.append(out.last_hidden_state)
            print(f"  chunk {s}-{e} done", flush=True)
        return torch.cat(outs, 1)


def record_routing(tm, store):
    """Wrap every MoE router to record (ids, weights) per call into store[li] (a list, one entry per call)."""
    for li, layer in enumerate(tm.layers):
        if not isinstance(layer.mlp, M.HYV4FlashMoE):
            continue
        gate = layer.mlp.gate
        fwd = gate.forward

        def rec(h, li=li, fwd=fwd):
            logits, w, idx = fwd(h)
            store.setdefault(li, []).append((idx.to(torch.int32), w.float()))
            return logits, w, idx

        gate.forward = rec


def record_gold(tm, store):
    """Record the streams entering every decoder layer (forward pre-hooks)."""
    hs = []
    for li, layer in enumerate(tm.layers):

        def pre(mod, args, kwargs, li=li):
            x = args[0] if args else kwargs["hidden_states"]
            store.setdefault(li, []).append(x.detach().clone())

        hs.append(layer.register_forward_pre_hook(pre, with_kwargs=True))
    return hs


@torch.no_grad()
def teacher_forced(texts_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    seqs = json.load(open(texts_path))
    ref = Ref()
    cfg, tm = ref.cfg, ref.tm
    chunk = int(os.environ.get("REF_CHUNK", "4096"))
    gold_on = bool(os.environ.get("REF_GOLD"))
    moe_layers = [i for i, layer in enumerate(tm.layers) if isinstance(layer.mlp, M.HYV4FlashMoE)]
    groups = {}
    for n, v in seqs.items():
        groups.setdefault(len(v), []).append(n)
    for T, names in sorted(groups.items()):
        ids = torch.tensor([seqs[n] for n in names])
        B = len(names)
        routes, gold = {}, {}
        record_routing(tm, routes)
        hooks = record_gold(tm, gold) if gold_on else []
        t0 = time.time()
        h = ref.forward_chunked(ids, chunk)
        for hk in hooks:
            hk.remove()
        for layer in tm.layers:  # unwrap the routers
            if isinstance(layer.mlp, M.HYV4FlashMoE):
                layer.mlp.gate.forward = layer.mlp.gate.forward.__defaults__[1]
        dt = time.time() - t0
        print(f"group T={T} B={B}: {dt:.0f}s ({dt / len(tm.layers):.1f}s per layer)", flush=True)
        K = cfg.num_experts_per_tok
        r_ids = np.zeros((B, len(moe_layers), T, K), np.uint16)
        r_w = np.zeros((B, len(moe_layers), T, K), np.float16)
        for j, li in enumerate(moe_layers):
            idx = torch.cat([c[0].view(B, -1, K) for c in routes[li]], 1)
            w = torch.cat([c[1].view(B, -1, K) for c in routes[li]], 1)
            r_ids[:, j], r_w[:, j] = idx.numpy(), w.numpy()
        tag = f"_T{T}" if len(groups) > 1 else ""
        np.savez(os.path.join(out_dir, f"routing{tag}.npz"), names=np.array(names), layers=np.array(moe_layers),
                 ids=r_ids, w=r_w)
        for b, n in enumerate(names):
            lg = ref.logits(h[b])
            lse = torch.logsumexp(lg, -1)
            lp = lg - lse[:, None]
            nxt = ids[b, 1:]
            nll = -lp[:-1].gather(1, nxt[:, None])[:, 0]
            top = lp.topk(32, -1)
            acc = lp[:-1].argmax(-1) == nxt
            np.savez(
                os.path.join(out_dir, f"ref_{n}.npz"),
                top_ids=top.indices.numpy().astype(np.int32),
                top_lp=top.values.numpy().astype(np.float16),
                lse=lse.numpy(),
                nll=nll.numpy(),
                acc=acc.numpy(),
            )
            print(f"REF {n} T={T} nll {float(nll.mean()):.3f} acc {float(acc.float().mean()):.3f}", flush=True)
        if gold_on:
            x_in = torch.stack([torch.cat([c[0] for c in gold[li]], 1) for li in range(len(tm.layers))])  # [L,T,4,H]
            torch.save({"name": names[0], "ids": ids[0].clone(), "x_in": x_in, "hidden": h[0].clone()},
                       os.path.join(out_dir, f"gold_{names[0]}.pt"))


@torch.no_grad()
def generate(prompts_path, out_dir):
    """Greedy decoding, one token at a time through the HF cache (prompt prefilled in REF_CHUNK chunks)."""
    os.makedirs(out_dir, exist_ok=True)
    prompts = json.load(open(prompts_path))
    ref = Ref()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(SNAP)
    max_new = int(os.environ.get("REF_MAX_NEW", "64"))
    eos = {ref.cfg.eos_token_id} if isinstance(ref.cfg.eos_token_id, int) else set(ref.cfg.eos_token_id)
    res = {}
    for name, p in prompts.items():
        ids = torch.tensor([p])
        cache = DynamicCache(config=ref.cfg)
        T = ids.shape[1]
        t0 = time.time()
        ng, em = ref.engram_inputs(ids)
        pos = torch.arange(T).view(1, 1, -1).expand(3, 1, -1)
        h = ref.tm(input_ids=ids, position_ids=pos, cache=cache, use_cache=True, ngram_tokens=ng,
                   engram_mask=em).last_hidden_state
        nxt = int(ref.logits(h[0, -1]).argmax())
        out = [nxt]
        t1 = time.time()
        allids = list(p) + [nxt]
        while len(out) < max_new and nxt not in eos:
            cur = torch.tensor([[nxt]])
            n = len(allids) - 1
            ng, em = ref.engram_inputs(cur, n, torch.tensor([allids[:-1]]))
            pos = torch.full((3, 1, 1), n)
            h = ref.tm(input_ids=cur, position_ids=pos, cache=cache, use_cache=True, ngram_tokens=ng,
                       engram_mask=em).last_hidden_state
            nxt = int(ref.logits(h[0, -1]).argmax())
            out.append(nxt)
            allids.append(nxt)
        text = tok.decode(out)
        dt = time.time() - t1
        print(f"GEN {name}: prefill {t1 - t0:.0f}s, {len(out)} tokens at {dt / max(1, len(out) - 1):.1f}s/token\n"
              f"{text!r}", flush=True)
        res[name] = {"prompt_ids": p, "ids": out, "text": text}
        json.dump(res, open(os.path.join(out_dir, "gen.json"), "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    mode = sys.argv[1]
    {"tf": teacher_forced, "gen": generate}[mode](*sys.argv[2:4])

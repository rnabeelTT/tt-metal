# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Full-model HF reference for GLM-5.3-Flash E192 on CPU (runs in the HF venv, never imports ttnn).

The stock readiness generator (readiness_check/generate.py) cannot run this model: it calls
AutoModelForCausalLM.from_pretrained, which would materialize the FP8 checkpoint as ~450 GB of bf16 on a 249 GB
host. This tool builds the unmodified HF `Glm5NextTextModel` (transformers glm5_next) with every non-routed-expert
weight resident in bf16 (HF load semantics: bf16 except `_keep_in_fp32_modules_strict`), and replaces each MoE
layer's `mlp.experts` with `LazyExperts`: the eager `Glm5NextTextExperts.forward` (the implementation the per-layer
reference tests/hf_ref/hf_layer_ref.py runs) whose per-expert weights are dequantized from the mmapped FP8 shards
(W = fp8 * scale_inv[i // 128, j // 128], fp32 -> bf16, bit-identical to tt/ckpt.py dequant_fp8) only when a forward
step routes a token to them, with a byte-bounded LRU of dequantized experts. The forward is the HF text model's own:
embed -> 4 hc streams -> 45 decoder layers (KDA / DSA state in the HF DynamicCache) -> hc_head (stream mean) ->
final norm -> untied lm_head (bf16).

Modes:
  --mode validate   short chat prompt: per-layer bitwise check against hf_layer_ref.build_layer (same input stream,
                    fresh cache), last-position logits, greedy text.
  --mode readiness  mirror of `python -m readiness_check.generate --prompt-source aime24 --chat-template --gen-len 100
                    --top-k 100`: first AIME24 prompt, tokenizer chat template with add_generation_prompt, greedy HF
                    continuation (DynamicCache prefill + decode), then the teacher-forcing batch-prefill pass over
                    prompt + continuation whose top-k rows are saved with readiness_check.schema.save_reference.
                    Writes <output> and <output without .refpt>.json (metadata, completion text, timings).
  --mode autoregressive  HF side of readiness_check.run_autoregressive (whose in-process from_pretrained of the
                    parent checkpoint cannot run here): the prompt file tokenized exactly as the runner does
                    (tokenizer.encode(text.strip(), add_special_tokens=True)), greedy model.generate-equivalent decode of
                    --max-new-tokens (default run_autoregressive.DEFAULT_MAX_NEW_TOKENS = 256) stopping on the
                    generation_config.json eos ids (what model.generate stops on). Writes --tokens-out (JSON: token
                    ids, text, prompt ids, metadata) for `tools/readiness.py autoregressive --hf-tokens`.
  --mode score-trajectory  HF teacher-forced scoring of another generator's free-running trajectory: one forward
                    over prompt + generated ids (--trajectory: an autoregressive_meta.json, side `tt`), per position the
                    HF argmax / top-5 / logit margin vs the token that generator emitted -> --score-out JSON. Tells a
                    near-tie divergence (and a loop the HF model would also follow from the same context) from a bug.
  --mode qualitative  shared qualitative suite (readiness_check/vllm_prompts.txt, split on blank lines as
                    run_vllm_server does), each prompt one user message rendered with the tokenizer chat template
                    (add_generation_prompt=True); writes <out-dir>/qualitative_prompt_format.json and
                    rendered_prompts.json, then (unless --render-only) greedy HF controls of --max-new-tokens tokens
                    per prompt to <out-dir>/hf_outputs.json (rewritten after every prompt).

Run:
  GLM53_CKPT=~/glm53-flash-bringup/ckpt/glm53_flash_e192_fp8 ~/glm53-flash-bringup/.venv_hf/bin/python -I \
      models/autoports/zai_org_glm_5_3_flash/tools/hf_full_ref.py --mode readiness \
      --output models/autoports/zai_org_glm_5_3_flash/readiness_aime24_chat.refpt
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shlex
import sys
import time
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve()
AUTOPORT = HERE.parents[1]
TT_METAL_HOME = HERE.parents[4]

HF_MODEL_ID = "zai-org/GLM-5.3-Flash"
DEFAULT_AIME24 = TT_METAL_HOME / "models" / "demos" / "deepseek_v3" / "demo" / "aime_under_8k_prompts.json"
BLOCK = 128


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# The per-layer reference (imported, never modified: its AST is part of the layer-reference cache key).
H = _load_module("glm53_hf_layer_ref", AUTOPORT / "tests" / "hf_ref" / "hf_layer_ref.py")
M = H.M
DynamicCache = H.DynamicCache


def _readiness_root() -> Path:
    for cand in (
        os.environ.get("TT_MODEL_BRINGUP_ROOT"),
        str(Path.home() / ".claude/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.3"),
        str(Path.home() / "skills/plugins/tt-model-bringup"),
    ):
        if cand and (Path(cand) / "runtime" / "readiness_check" / "schema.py").exists():
            return Path(cand) / "runtime"
    raise RuntimeError("readiness_check runtime not found (set TT_MODEL_BRINGUP_ROOT)")


def rss_gb() -> float:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 2**20
    return -1.0


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] [rss {rss_gb():.1f} GB] {msg}", flush=True)


# --------------------------------------------------------------------------------------------- lazy routed experts
def dequant_fp8_blocked(w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Same arithmetic as tt/ckpt.py dequant_fp8 (fp32 product, one rounding to bf16) without the repeat_interleave
    scale expansion; the expert shapes are multiples of 128."""
    o, i = w.shape
    if o % BLOCK or i % BLOCK:
        return H.ckpt.dequant_fp8(w, s, torch.bfloat16)
    out = w.float().view(o // BLOCK, BLOCK, i // BLOCK, BLOCK) * s.float()[:, None, :, None]
    return out.view(o, i).to(torch.bfloat16)


class ExpertCache:
    """(layer, expert) -> (gate_up [2I, H], down [H, I]) bf16, LRU bounded by bytes."""

    def __init__(self, rd, budget_gb: float):
        self.rd = rd
        self.budget = int(budget_gb * 2**30)
        self.bytes = 0
        self.store: OrderedDict = OrderedDict()
        self.hits = self.misses = 0
        self.load_s = 0.0

    def _load(self, layer: int, e: int):
        rd = self.rd
        ws = []
        for proj in ("gate_proj", "up_proj", "down_proj"):
            name = rd._expert_name(layer, e, f"{proj}.weight")
            w = rd.raw(name)
            ws.append(
                dequant_fp8_blocked(w, rd.raw(name + "_scale_inv"))
                if w.dtype == torch.float8_e4m3fn
                else w.to(torch.bfloat16)
            )
        return torch.cat([ws[0], ws[1]], dim=0), ws[2]

    def get(self, layer: int, e: int):
        key = (layer, e)
        v = self.store.get(key)
        if v is not None:
            self.hits += 1
            self.store.move_to_end(key)
            return v
        self.misses += 1
        t0 = time.time()
        v = self._load(layer, e)
        self.load_s += time.time() - t0
        nb = sum(t.numel() * t.element_size() for t in v)
        if self.budget > 0:
            self.store[key] = v
            self.bytes += nb
            while self.bytes > self.budget and self.store:
                _, old = self.store.popitem(last=False)
                self.bytes -= sum(t.numel() * t.element_size() for t in old)
        return v

    def stats(self) -> dict:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "resident": len(self.store),
            "resident_gb": round(self.bytes / 2**30, 2),
            "load_s": round(self.load_s, 1),
        }


class LazyExperts(nn.Module):
    """Glm5NextTextExperts (eager path) with weights fetched per routed expert from ExpertCache."""

    def __init__(self, cache: ExpertCache, layer: int, num_experts: int, swiglu_limit: float):
        super().__init__()
        self.cache = cache
        self.layer = layer
        self.num_experts = num_experts
        self.swiglu_limit = swiglu_limit

    _apply_gate = M.Glm5NextTextExperts._apply_gate

    def forward(self, hidden_states, top_k_index, top_k_weights):
        # verbatim Glm5NextTextExperts.forward, gate_up_proj[e] / down_proj[e] -> cache.get(layer, e)
        final = torch.zeros_like(hidden_states)
        with torch.no_grad():
            mask = F.one_hot(top_k_index, num_classes=self.num_experts + 1).permute(2, 1, 0)
            hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            gate_up, down = self.cache.get(self.layer, int(expert_idx))
            top_k_pos, token_idx = torch.where(mask[expert_idx])
            current = self._apply_gate(F.linear(hidden_states[token_idx], gate_up))
            current = F.linear(current, down) * top_k_weights[token_idx, top_k_pos, None]
            final.index_add_(0, token_idx, current.to(final.dtype))
        return final


# ---------------------------------------------------------------------------------------------------- full model
class FullModel(nn.Module):
    def __init__(self, ckpt_dir: Path, expert_cache_gb: float):
        super().__init__()
        t0 = time.time()
        self.rd = H.ckpt.CheckpointReader(ckpt_dir)
        tc = H.text_config(ckpt_dir, self.rd.num_experts)
        tc.use_cache = True
        self.tc = tc
        self.experts = ExpertCache(self.rd, expert_cache_gb)
        with torch.device("meta"):
            model = M.Glm5NextTextModel(tc)
            lm_head = nn.Linear(tc.hidden_size, tc.vocab_size, bias=False)
        for li, layer in enumerate(model.layers):
            if isinstance(layer.mlp, M.Glm5NextTextMoE):
                layer.mlp.experts = LazyExperts(self.experts, li, tc.num_local_experts, tc.swiglu_limit)
            sd = H.to_hf_names(self.rd.layer_state(li, experts=False), tc.num_local_experts)
            hf_sd = {n: H.hf_dtype(n, t) for n, t in sd.items()}
            want = set(layer.state_dict())
            if want != set(hf_sd):
                raise RuntimeError(f"layer {li}: missing {sorted(want - set(hf_sd))} extra {sorted(set(hf_sd) - want)}")
            layer.load_state_dict(hf_sd, assign=True, strict=True)
            if li % 9 == 0:
                log(f"loaded layer {li} ({tc.layer_types[li]}, {tc.mlp_layer_types[li]})")
        top = {
            "embed_tokens.weight": self.rd.top("embed_tokens.weight"),
            "norm.weight": self.rd.top("norm.weight"),
        }
        model.embed_tokens.load_state_dict({"weight": top["embed_tokens.weight"]}, assign=True)
        model.norm.load_state_dict({"weight": top["norm.weight"]}, assign=True)
        lm_head.load_state_dict({"weight": self.rd.get("lm_head.weight", torch.bfloat16)}, assign=True)
        for name, t in list(model.named_parameters()) + list(model.named_buffers()):
            if t.is_meta:
                raise RuntimeError(f"unloaded tensor {name}")
        self.model = model.eval()
        self.lm_head = lm_head.eval()
        self.build_s = time.time() - t0
        log(f"model built in {self.build_s:.0f} s")

    @torch.no_grad()
    def logits(self, input_ids, cache=None, **kw):
        """HF CausalLM forward: text model -> lm_head on every position (bf16 logits)."""
        if cache is None:
            out = self.model(input_ids=input_ids, **kw)
        else:
            out = self.model(input_ids=input_ids, past_key_values=cache, use_cache=True, **kw)
        return self.lm_head(out.last_hidden_state)

    @torch.no_grad()
    def greedy(self, prompt: torch.Tensor, max_new: int, stop_ids: list[int], top_k: int = 0):
        """model.generate(do_sample=False, use_cache=True) equivalent: DynamicCache prefill, then one token per
        step with a growing all-ones attention mask; argmax of fp32 last-position logits; the stop token is
        emitted and ends generation. Returns (tokens, per-step seconds, per-step top-k of the decode logits)."""
        ids = prompt.view(1, -1)
        cache = DynamicCache(config=self.tc)
        mask = torch.ones_like(ids)
        out, step_s, step_topk = [], [], []
        t0 = time.time()
        logits = self.logits(ids, cache, attention_mask=mask)[:, -1, :].float()
        step_s.append(time.time() - t0)
        log(f"prefill {ids.shape[1]} tokens: {step_s[-1]:.1f} s {self.experts.stats()}")
        for i in range(max_new):
            if top_k:
                step_topk.append(torch.topk(logits[0], top_k).indices.to(torch.int32))
            nxt = int(torch.argmax(logits[0]))
            out.append(nxt)
            if nxt in stop_ids or i == max_new - 1:
                break
            mask = torch.cat([mask, mask.new_ones(1, 1)], dim=1)
            t0 = time.time()
            logits = self.logits(torch.tensor([[nxt]]), cache, attention_mask=mask)[:, -1, :].float()
            step_s.append(time.time() - t0)
            if i % 10 == 0 or i < 3:
                log(f"decode {i + 1}: {step_s[-1]:.1f} s tok {nxt} {self.experts.stats()}")
        return out, step_s, step_topk


# ---------------------------------------------------------------------------------------------- readiness mirror
def chat_tokens(tok, text: str) -> list[int]:
    r = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, tokenize=True)
    if hasattr(r, "keys") and "input_ids" in r:  # transformers 5 returns a BatchEncoding
        r = r["input_ids"]
    return [int(t) for t in r]


def stop_ids_like_generate(tok, config) -> list[int]:
    """readiness_check.generate._generation_stop_ids with the HF model config object (Glm5NextConfig)."""

    def norm(x):
        if x is None:
            return []
        if isinstance(x, int):
            return [x]
        return [int(t) for t in x if t is not None]

    ids = norm(tok.eos_token_id) + norm(getattr(config, "eos_token_id", None))
    eot = getattr(tok, "eot_token_id", None)
    if eot is None:
        vocab = tok.get_vocab()
        for t in ("<|eot_id|>", "<end_of_turn>"):
            if vocab and t not in vocab:
                continue
            c = tok.convert_tokens_to_ids(t)
            if isinstance(c, int) and c >= 0 and c != tok.unk_token_id:
                eot = c
                break
    ids += norm(eot)
    out = []
    for t in ids:
        if t not in out:
            out.append(t)
    return out


def hf_revision(ckpt_dir: Path):
    meta = ckpt_dir / ".cache" / "huggingface" / "download" / "config.json.metadata"
    if meta.exists():
        return meta.read_text().splitlines()[0].strip()
    return None


def run_readiness(args, fm: FullModel, tok):
    from readiness_check.schema import Reference, ReferenceEntry, save_reference  # noqa: E402

    items = json.loads(Path(args.aime24_prompts_file).read_text())
    item = items[args.aime24_prompt_index]
    assert item.get("task") in {"aime24", "r1_aime24"}, item.get("task")
    prompt_text = item["prompt"]
    ptoks = chat_tokens(tok, prompt_text)
    prompt = torch.tensor(ptoks, dtype=torch.long)
    top_cfg = H.Glm5NextConfig(**json.loads((Path(args.ckpt) / "config.json").read_text()))
    stop_ids = stop_ids_like_generate(tok, top_cfg)
    log(f"aime24 prompt {args.aime24_prompt_index}: {len(ptoks)} chat-template tokens; stop ids {stop_ids}")

    t_gen = time.time()
    gen, step_s, step_topk = fm.greedy(prompt, args.gen_len, stop_ids, top_k=args.top_k)
    t_gen = time.time() - t_gen
    gen_t = torch.tensor(gen, dtype=torch.long)
    completion = tok.decode(gen, skip_special_tokens=False)
    log(f"generated {len(gen)} tokens in {t_gen:.0f} s:\n{completion}")

    # teacher-forcing pass (generate._generate_one_entry): one forward over prompt + continuation, no attention mask
    t_tf = time.time()
    full = torch.cat([prompt, gen_t]).unsqueeze(0)
    logits = fm.logits(full)
    P, G = len(ptoks), len(gen)
    pred = logits[0, P - 1 : P + G - 1, :]
    k = min(args.top_k, pred.shape[-1])
    topk = torch.topk(pred, k=k, dim=-1).indices.to(torch.int32)
    t_tf = time.time() - t_tf
    log(f"teacher-forcing pass over {full.shape[1]} tokens: {t_tf:.0f} s")
    tf_top1 = topk[:, 0].tolist()
    dec_top1 = [int(t[0]) for t in step_topk]
    agree = sum(int(a == b) for a, b in zip(tf_top1, gen)) / G
    top5_agree = sum(int(g in topk[i, :5].tolist()) for i, g in enumerate(gen)) / G

    entry = ReferenceEntry(
        prompt_text=tok.decode(ptoks, skip_special_tokens=False),
        prompt_tokens=prompt.unsqueeze(0),
        generated_tokens=gen_t.unsqueeze(0),
        topk_tokens=topk,
        tf_prompt_len=P,
    )
    pad = tok.pad_token_id
    pad = int(pad) if (pad is not None and pad != tok.bos_token_id) else int(stop_ids[0])
    ref = Reference(
        k=args.top_k,
        hf_model_id=HF_MODEL_ID,
        entries=[entry],
        token_ids_meta={
            "bos_id": int(tok.bos_token_id) if tok.bos_token_id is not None else None,
            "eos_id": int(stop_ids[0]),
            "pad_id": pad,
        },
    )
    out = save_reference(ref, args.output)
    decode_s = step_s[1:]
    import transformers

    meta = {
        "reference": str(out),
        "format": "readiness_v1 (readiness_check.schema.save_reference)",
        "hf_model_id": HF_MODEL_ID,
        "hf_revision": hf_revision(Path(args.ckpt)),
        "checkpoint": str(Path(args.ckpt).resolve()),
        "tokenizer": f"{type(tok).__name__} from {Path(args.ckpt).resolve()} (copied from {HF_MODEL_ID} at the revision above)",
        "prompt_source": "aime24",
        "aime24_prompts_file": str(Path(args.aime24_prompts_file).resolve()),
        "aime24_prompt_index": args.aime24_prompt_index,
        "chat_template": True,
        "add_generation_prompt": True,
        "gen_len": args.gen_len,
        "top_k": args.top_k,
        "num_entries": 1,
        "command": " ".join(shlex.quote(a) for a in [sys.executable] + sys.orig_argv[1:]),
        "generator": "tools/hf_full_ref.py: mirror of readiness_check.generate (aime24, --chat-template, greedy "
        "model.generate continuation, then batch-prefill teacher-forcing top-k); the stock script cannot load the model",
        "note": "Routed experts are E192-masked (192 of 288 per MoE layer, mask "
        "autotrust/GLM5.3-Flash-E192-DGX-Spark, checkpoint renumbered 0..191, routers pruned to the kept rows) and "
        "FP8 e4m3 (block 128) weights are dequantized to bf16 (fp32 product, one rounding). Non-expert weights "
        "resident in bf16 (fp32: e_score_correction_bias, conv1d, dt_bias, A_log); experts dequantized lazily per "
        "forward step. HF Glm5NextTextModel, attn sdpa, experts eager, CPU.",
        "stop_ids": stop_ids,
        "stop_ids_note": "as readiness_check.generate computes them (tokenizer eos + model.config.eos_token_id, which "
        "the composite Glm5NextConfig does not define); generation_config.json lists [154820, 154827, 154829]",
        "prompt_tokens": P,
        "generated_tokens": G,
        "stopped_on_eos": bool(gen and gen[-1] in stop_ids),
        "completion_text": completion,
        "full_text": tok.decode(ptoks + gen, skip_special_tokens=False),
        "teacher_forcing_top1_equals_greedy": agree,
        "teacher_forcing_top5_contains_greedy": top5_agree,
        "decode_vs_tf_top1_mismatch_positions": [i for i, (a, b) in enumerate(zip(dec_top1, tf_top1)) if a != b],
        "timing": {
            "build_s": round(fm.build_s, 1),
            "prefill_s": round(step_s[0], 1),
            "decode_steps": len(decode_s),
            "decode_s_per_token_mean": round(sum(decode_s) / max(1, len(decode_s)), 2),
            "decode_s_per_token_min": round(min(decode_s), 2) if decode_s else None,
            "decode_s_per_token_max": round(max(decode_s), 2) if decode_s else None,
            "generation_total_s": round(t_gen, 1),
            "teacher_forcing_s": round(t_tf, 1),
            "threads": torch.get_num_threads(),
            "expert_cache_gb": args.expert_cache_gb,
            "expert_cache": fm.experts.stats(),
            "peak_rss_gb": round(peak_rss_gb(), 1),
        },
        "versions": {"transformers": transformers.__version__, "torch": torch.__version__},
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    meta_path = Path(
        str(args.output)[: -len(".refpt")] + ".json"
        if str(args.output).endswith(".refpt")
        else str(args.output) + ".json"
    )
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    log(f"saved {out} and {meta_path}; TF top1==greedy {agree:.3f}, top5 {top5_agree:.3f}")


def generation_eos_ids(ckpt_dir: Path) -> list[int]:
    """The stop ids model.generate uses: generation_config.json eos_token_id (from_pretrained loads it)."""
    gc = json.loads((Path(ckpt_dir) / "generation_config.json").read_text())
    e = gc.get("eos_token_id")
    return [int(e)] if isinstance(e, int) else [int(t) for t in (e or [])]


def _common_meta(args, tok, fm: FullModel) -> dict:
    import transformers

    return {
        "hf_model_id": HF_MODEL_ID,
        "hf_revision": hf_revision(Path(args.ckpt)),
        "checkpoint": str(Path(args.ckpt).resolve()),
        "tokenizer": f"{type(tok).__name__} from {Path(args.ckpt).resolve()}",
        "command": " ".join(shlex.quote(a) for a in [sys.executable] + sys.orig_argv[1:]),
        "note": "CPU HF reference (tools/hf_full_ref.py): unmodified HF Glm5NextTextModel, E192-masked routed experts "
        "FP8->bf16 dequantized lazily, non-expert weights bf16; greedy argmax of fp32 last-position logits with a "
        "DynamicCache (model.generate(do_sample=False) equivalent).",
        "versions": {"transformers": transformers.__version__, "torch": torch.__version__},
        "threads": torch.get_num_threads(),
        "expert_cache_gb": args.expert_cache_gb,
    }


def run_autoregressive_ref(args, fm: FullModel, tok):
    prompt_file = Path(args.prompt_file).resolve()
    prompt_text = prompt_file.read_text(encoding="utf-8").strip()  # as run_autoregressive
    ids = tok.encode(prompt_text, add_special_tokens=True)
    stop_ids = generation_eos_ids(Path(args.ckpt))
    log(f"autoregressive prompt {len(ids)} tokens, max_new_tokens {args.max_new_tokens}, stop ids {stop_ids}")
    t0 = time.time()
    gen, step_s, _ = fm.greedy(torch.tensor(ids, dtype=torch.long), args.max_new_tokens, stop_ids)
    total = time.time() - t0
    text = tok.decode(gen, skip_special_tokens=False)
    log(f"generated {len(gen)} tokens in {total:.0f} s:\n{text}")
    out = {
        "source": "tools/hf_full_ref.py --mode autoregressive (precomputed HF side of readiness_check.run_autoregressive)",
        **_common_meta(args, tok, fm),
        "prompt_file": str(prompt_file),
        "prompt_text": prompt_text,
        "prompt_tokenization": "tokenizer.encode(prompt_file.read_text().strip(), add_special_tokens=True)",
        "prompt_token_ids": [int(t) for t in ids],
        "max_new_tokens": args.max_new_tokens,
        "stop_ids": stop_ids,
        "stopped_on_eos": bool(gen and gen[-1] in stop_ids),
        "token_ids": [int(t) for t in gen],
        "num_tokens": len(gen),
        "text": text,
        "timing": {
            "prefill_s": round(step_s[0], 1),
            "decode_s_per_token_mean": round(sum(step_s[1:]) / max(1, len(step_s) - 1), 2),
            "generation_total_s": round(total, 1),
            "expert_cache": fm.experts.stats(),
            "peak_rss_gb": round(peak_rss_gb(), 1),
        },
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    dst = Path(args.tokens_out)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    log(f"saved {dst}")


def load_qualitative_prompts(path: Path) -> list[str]:
    """Same split as readiness_check.run_vllm_server._run_qualitative_prompts."""
    return [p.strip() for p in Path(path).read_text().split("\n\n") if p.strip()]


def render_qualitative(args, tok) -> list[dict]:
    prompts = load_qualitative_prompts(args.prompts_file)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rendered = []
    for i, p in enumerate(prompts):
        msgs = [{"role": "user", "content": p}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = chat_tokens(tok, p)
        assert ids == tok.encode(text, add_special_tokens=False), f"prompt {i}: tokenize=True != encode(rendered)"
        rendered.append({"id": f"q{i + 1}", "prompt": p, "messages": msgs, "rendered_text": text, "token_ids": ids})
    import transformers

    fmt = {
        "hf_model_id": HF_MODEL_ID,
        "hf_revision": hf_revision(Path(args.ckpt)),
        "checkpoint": str(Path(args.ckpt).resolve()),
        "tokenizer_class": type(tok).__name__,
        "chat_template_present": bool(getattr(tok, "chat_template", None)),
        "chat_template_source": str((Path(args.ckpt) / "chat_template.jinja").resolve()),
        "prompt_mode": "chat",
        "rendering": "tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False, "
        "add_generation_prompt=True) (default kwargs: reasoning_effort -> 'max', thinking on: the generation prompt "
        "ends '<|assistant|><think>'); token ids = apply_chat_template(..., tokenize=True) == "
        "encode(rendered, add_special_tokens=False)",
        "roles": "single user message per prompt, no system message (the template inserts '<|system|>Reasoning Effort: Max')",
        "prompt_source": str(Path(args.prompts_file).resolve()),
        "prompt_split": "blank-line separated, as readiness_check.run_vllm_server",
        "generation": {"do_sample": False, "greedy": True, "stop_ids": generation_eos_ids(Path(args.ckpt))},
        "transformers": transformers.__version__,
    }
    (out_dir / "qualitative_prompt_format.json").write_text(json.dumps(fmt, indent=2, ensure_ascii=False) + "\n")
    (out_dir / "rendered_prompts.json").write_text(json.dumps(rendered, indent=2, ensure_ascii=False) + "\n")
    log(f"rendered {len(rendered)} prompts -> {out_dir}")
    return rendered


def run_qualitative_ref(args, fm: FullModel, tok, rendered: list[dict]):
    out_dir = Path(args.out_dir)
    stop_ids = generation_eos_ids(Path(args.ckpt))
    dst = out_dir / "hf_outputs.json"
    results = json.loads(dst.read_text()) if (args.resume and dst.exists()) else []
    done = {r["id"] for r in results}
    for r in rendered:
        if r["id"] in done or (args.prompt_ids and r["id"] not in args.prompt_ids.split(",")):
            continue
        log(f"{r['id']}: {len(r['token_ids'])} prompt tokens: {r['prompt'][:60]!r}")
        t0 = time.time()
        gen, step_s, _ = fm.greedy(torch.tensor(r["token_ids"], dtype=torch.long), args.max_new_tokens, stop_ids)
        text = tok.decode(gen, skip_special_tokens=False)
        log(f"{r['id']}: {len(gen)} tokens in {time.time() - t0:.0f} s:\n{text}")
        results.append(
            {
                "id": r["id"],
                "prompt": r["prompt"],
                "prompt_token_ids": r["token_ids"],
                "max_new_tokens": args.max_new_tokens,
                "stopped_on_eos": bool(gen and gen[-1] in stop_ids),
                "token_ids": gen,
                "text": text,
                "prefill_s": round(step_s[0], 1),
                "decode_s_per_token_mean": round(sum(step_s[1:]) / max(1, len(step_s) - 1), 2),
            }
        )
        meta = {"source": "tools/hf_full_ref.py --mode qualitative", **_common_meta(args, tok, fm)}
        dst.write_text(json.dumps({"meta": meta, "outputs": results}, indent=2, ensure_ascii=False) + "\n")
    log(f"saved {dst}")


@torch.no_grad()
def run_score_trajectory(args, fm: FullModel, tok):
    meta = json.loads(Path(args.trajectory).read_text())
    ptoks = [int(t) for t in meta["prompt_token_ids"]]
    gen = [int(t) for t in meta[args.trajectory_side]["token_ids"]]
    ref = [int(t) for t in meta.get("hf", {}).get("token_ids", [])]
    full = torch.tensor([ptoks + gen[:-1]], dtype=torch.long)
    t0 = time.time()
    logits = fm.logits(full)[0, len(ptoks) - 1 :].float()  # row i predicts gen[i]
    log(f"scored {full.shape[1]} tokens in {time.time() - t0:.0f} s")
    rows = []
    for i, g in enumerate(gen):
        top = torch.topk(logits[i], 5)
        ids = [int(x) for x in top.indices]
        rows.append(
            {
                "i": i,
                "emitted": g,
                "emitted_str": tok.decode([g]),
                "hf_top5": ids,
                "hf_top5_str": [tok.decode([x]) for x in ids],
                "hf_top5_logits": [round(float(v), 3) for v in top.values],
                "emitted_logit": round(float(logits[i, g]), 3),
                "emitted_rank": int((logits[i] > logits[i, g]).sum()),
                "hf_top1_margin": round(float(top.values[0] - top.values[1]), 3),
            }
        )
    n = len(rows)
    out = {
        "source": "tools/hf_full_ref.py --mode score-trajectory",
        **_common_meta(args, tok, fm),
        "trajectory": str(Path(args.trajectory).resolve()),
        "side": args.trajectory_side,
        "num_tokens": n,
        "hf_top1_agreement": sum(r["emitted_rank"] == 0 for r in rows) / n,
        "hf_top5_agreement": sum(r["emitted_rank"] < 5 for r in rows) / n,
        "worst_ranks": sorted(((r["emitted_rank"], r["i"]) for r in rows), reverse=True)[:10],
        "hf_greedy_first_divergence": next((i for i, (a, b) in enumerate(zip(ref, gen)) if a != b), None),
        "rows": rows,
    }
    Path(args.score_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.score_out).write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    log(f"top1 agreement {out['hf_top1_agreement']:.3f} top5 {out['hf_top5_agreement']:.3f} worst {out['worst_ranks']}")


def peak_rss_gb() -> float:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) / 2**20
    return -1.0


# ------------------------------------------------------------------------------------------------------- validate
@torch.no_grad()
def run_validate(args, fm: FullModel, tok):
    rd, tc = fm.rd, fm.tc
    # 1) fast dequant == tt/ckpt.py dequant_fp8 (bitwise) on one expert of each projection
    for proj in ("gate_proj", "up_proj", "down_proj"):
        name = rd._expert_name(3, 0, f"{proj}.weight")
        a = dequant_fp8_blocked(rd.raw(name), rd.raw(name + "_scale_inv"))
        b = rd.expert(3, 0, proj)
        assert torch.equal(a, b), proj
    log("dequant_fp8_blocked == ckpt.dequant_fp8 (bitwise)")

    ids = torch.tensor([chat_tokens(tok, args.prompt)])
    log(f"validate prompt {ids.shape[1]} tokens: {tok.decode(ids[0])!r}")
    taps = {}

    def hook(li):
        def f(m, a, kw, out):
            taps[li] = (a[0].detach().clone(), out[0].detach().clone())

        return f

    layers = (
        list(range(tc.num_hidden_layers))
        if args.check_layers == "all"
        else [int(x) for x in args.check_layers.split(",")]
    )
    hs = [fm.model.layers[li].register_forward_hook(hook(li), with_kwargs=True) for li in layers]
    cache = DynamicCache(config=tc)
    t0 = time.time()
    logits = fm.logits(ids, cache, attention_mask=torch.ones_like(ids))
    log(f"prefill {time.time() - t0:.1f} s {fm.experts.stats()}")
    for h in hs:
        h.remove()
    last = logits[0, -1].float()
    top = torch.topk(last, 10)
    log(
        "last-position top-10: "
        + ", ".join(f"{tok.decode([int(i)])!r}:{v:.2f}" for v, i in zip(top.values, top.indices))
    )

    # the per-layer reference exactly as hf_layer_ref.run builds it: a fresh text config (building the full
    # Glm5NextTextModel runs post_init, which resolves tc._experts_implementation None -> "grouped_mm"; the standalone
    # reference keeps None, i.e. the eager experts loop that LazyExperts reproduces)
    tc_ref = H.text_config(Path(args.ckpt), rd.num_experts)
    log(
        f"experts implementation: full-model config {tc._experts_implementation!r} (replaced by LazyExperts, eager), "
        f"reference config {tc_ref._experts_implementation!r}"
    )
    results = {}
    for li in layers:
        x, y = taps[li]
        ref = H.build_layer(tc_ref, li, rd.layer_state(li))
        yr, _ = ref(
            x,
            attention_mask=torch.ones(x.shape[:2], dtype=torch.bool),
            past_key_values=DynamicCache(config=tc_ref),
            use_cache=True,
            position_embeddings=None,
        )
        diff = (y.float() - yr.float()).abs().max().item()
        results[li] = {
            "bitwise": bool(torch.equal(y, yr)),
            "max_abs_diff": diff,
            "type": f"{tc.layer_types[li]}/{tc.mlp_layer_types[li]}",
        }
        log(f"layer {li} vs hf_layer_ref.build_layer: {results[li]}")
        del ref

    # every checked layer bitwise equal on the full model's own stream => chaining the per-layer reference over
    # those layers reproduces the full model exactly
    gen, step_s, _ = fm.greedy(
        ids[0],
        args.gen_len,
        stop_ids_like_generate(tok, H.Glm5NextConfig(**json.loads((Path(args.ckpt) / "config.json").read_text()))),
    )
    text = tok.decode(gen, skip_special_tokens=False)
    log(f"greedy {len(gen)} tokens, decode mean {sum(step_s[1:]) / max(1, len(step_s) - 1):.1f} s/token:\n{text}")
    out = {
        "prompt": args.prompt,
        "prompt_tokens": ids[0].tolist(),
        "last_top10": [(int(i), float(v)) for v, i in zip(top.values, top.indices)],
        "layers": results,
        "greedy_tokens": gen,
        "greedy_text": text,
        "prefill_s": step_s[0],
        "decode_s": step_s[1:],
        "expert_cache": fm.experts.stats(),
        "peak_rss_gb": peak_rss_gb(),
    }
    if args.validate_out:
        Path(args.validate_out).write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode", choices=("validate", "readiness", "autoregressive", "qualitative", "score-trajectory"), required=True
    )
    ap.add_argument(
        "--ckpt",
        default=os.environ.get("GLM53_CKPT", str(Path.home() / "glm53-flash-bringup/ckpt/glm53_flash_e192_fp8")),
    )
    ap.add_argument("--output", type=Path, default=AUTOPORT / "readiness_aime24_chat.refpt")
    ap.add_argument("--aime24-prompts-file", type=Path, default=DEFAULT_AIME24)
    ap.add_argument("--aime24-prompt-index", type=int, default=0)
    ap.add_argument("--gen-len", type=int, default=100)
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--expert-cache-gb", type=float, default=48.0)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--prompt", default="What is 2+3?")
    ap.add_argument("--check-layers", default="0,3,4")
    ap.add_argument("--validate-out", default=None)
    ap.add_argument(
        "--prompt-file", type=Path, default=None, help="autoregressive: prompt file (default: the runner's)"
    )
    ap.add_argument("--max-new-tokens", type=int, default=256, help="autoregressive / qualitative token budget")
    ap.add_argument("--tokens-out", type=Path, default=AUTOPORT / "doc/full_model/autoregressive/hf_tokens.json")
    ap.add_argument(
        "--prompts-file", type=Path, default=None, help="qualitative: prompt suite (default vllm_prompts.txt)"
    )
    ap.add_argument("--out-dir", type=Path, default=AUTOPORT / "doc/full_model/qualitative")
    ap.add_argument("--render-only", action="store_true", help="qualitative: write the prompt artifacts, no model")
    ap.add_argument("--prompt-ids", default="", help="qualitative: comma-separated subset (q1,q2,...)")
    ap.add_argument(
        "--trajectory", type=Path, default=AUTOPORT / "doc/full_model/autoregressive/autoregressive_meta.json"
    )
    ap.add_argument("--trajectory-side", default="tt")
    ap.add_argument(
        "--score-out", type=Path, default=AUTOPORT / "doc/full_model/autoregressive/hf_scores_tt_trajectory.json"
    )
    ap.add_argument("--resume", action="store_true", help="qualitative: keep outputs already in hf_outputs.json")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    sys.path.insert(0, str(_readiness_root()))
    torch.manual_seed(0)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.ckpt)
    rc = _readiness_root() / "readiness_check"
    if args.prompt_file is None:
        args.prompt_file = rc / "autoregressive_prompt.txt"
    if args.prompts_file is None:
        args.prompts_file = rc / "vllm_prompts.txt"
    if args.mode == "qualitative":
        rendered = render_qualitative(args, tok)
        if args.render_only:
            return
    fm = FullModel(Path(args.ckpt), args.expert_cache_gb)
    if args.mode == "validate":
        run_validate(args, fm, tok)
    elif args.mode == "autoregressive":
        run_autoregressive_ref(args, fm, tok)
    elif args.mode == "qualitative":
        run_qualitative_ref(args, fm, tok, rendered)
    elif args.mode == "score-trajectory":
        run_score_trajectory(args, fm, tok)
    else:
        run_readiness(args, fm, tok)


if __name__ == "__main__":
    main()

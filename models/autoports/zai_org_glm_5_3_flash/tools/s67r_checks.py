# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage-6 / stage-7 review #1 rework checks on ONE generator build (full model or GLM53_LAYERS reduced model).

  ready      teacher forcing + prefill check (readiness_check runners, tools/s7_probe.run_ready)
  ar         readiness autoregressive run + greedy determinism (tools/full_model_checks.run_with, qualitative skipped)
  batch      batch > 1: distinct real-text prompts of different non-aligned lengths prefilled into non-adjacent slots,
             decoded together (traced, on-device greedy, token feedback on device) == each prompt's solo run (logits
             rows + tokens); the same prompt in two slots gives identical rows; a slot prefilled while others are mid-
             decode leaves their outputs unchanged. Raw logits rows saved (--save-pt) for cross-build A/B (skip on/off)
  ptchange   page table changed between two TRACED decode steps (a new, permuted block at a 128-token boundary): the
             cache write lands in the new block (latent rows read back), the old block is untouched, logits == an eager
             (untraced, host-logits) decode on the same table sequence; an unchanged table passed every step costs no
             device copy (counter)
  sampling   traced split sampler with alternating greedy / top-k-top-p / penalties params (serving decode path): greedy
             token == argmax of the step's logits, sampled token inside the top-k set, every die's token buffer holds the
             same token (lockstep), a fixed seed reproduces
  chunkplan  real-text prompts of the given lengths: the cost chunk plan vs the legacy plan on the same build (last-row
             logits PCC / max diff, top-1 / top-5, greedy continuation)
  deep       long prompts (--deep lengths): TTFT, then traced token-out decode at the deep position (ms / token); with
             the block-sharded DSA cache, sink-weight headroom per DSA layer (eager prefill + eager decode steps)
  decode     traced decode timing per active-slot count (tools/s7_probe.run_decode)

usage: GLM53_TP=4 python -m models.autoports.zai_org_glm_5_3_flash.tools.s67r_checks --slots 8 --seq-len 4096 \
           --do batch,ptchange --out FILE.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import torch

import ttnn

MODEL_DIR = Path(__file__).resolve().parents[1]
TEXT_FILES = (
    "tech_reports/LLMs/llms.md",
    "tech_reports/prog_examples/matmul_multi_core_optimized/data_reuse.md",
    "README.md",
    "CONTRIBUTING.md",
    "LICENSE",
)


# ------------------------------------------------------------------------------------------------ provenance
def provenance() -> dict:
    root = Path(os.environ.get("TT_METAL_HOME", MODEL_DIR.parents[2]))
    sha = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(root), "status", "--short", "--", str(MODEL_DIR / "tt")], capture_output=True, text=True
    ).stdout.split("\n")
    md5 = {p.name: hashlib.md5(p.read_bytes()).hexdigest() for p in sorted((MODEL_DIR / "tt").glob("*.py"))}
    return {
        "git_head": sha,
        "tt_dirty": [d for d in dirty if d],
        "tt_md5": md5,
        "env": {k: v for k, v in os.environ.items() if k.startswith("GLM53_")},
    }


# ------------------------------------------------------------------------------------------------ helpers
def text_ids(tok, n: int) -> torch.Tensor:
    """n tokens of natural text (repo docs, concatenated, no repetition below the corpus size)."""
    root = Path(os.environ.get("TT_METAL_HOME", MODEL_DIR.parents[2]))
    text = ""
    for f in TEXT_FILES:
        p = root / f
        if p.exists():
            text += p.read_text(errors="ignore") + "\n\n"
    ids = tok.encode(text, add_special_tokens=False)
    if len(ids) < n:
        ids = ids * (-(-n // len(ids)))
    return torch.tensor(ids[:n], dtype=torch.int64)


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double().flatten(), b.double().flatten()
    a, b = a - a.mean(), b - b.mean()
    d = a.norm() * b.norm()
    return float((a @ b) / d) if d > 0 else float("nan")


def cmp_rows(a: torch.Tensor, b: torch.Tensor) -> dict:
    t5a, t5b = set(a.topk(5).indices.tolist()), set(b.topk(5).indices.tolist())
    return {
        "max_abs_diff": float((a.float() - b.float()).abs().max()),
        "pcc": round(pcc(a, b), 6),
        "top1_equal": int(a.argmax()) == int(b.argmax()),
        "top5_overlap": len(t5a & t5b) / 5,
        "bit_identical": bool(torch.equal(a, b)),
    }


def first_div(x, y):
    for i, (p, q) in enumerate(zip(x, y)):
        if p != q:
            return i
    return None if len(x) == len(y) else min(len(x), len(y))


def logits_rows(g) -> torch.Tensor:
    m = g.model
    return m.logits_to_host(g.logits_out)[0, 0, : g.max_batch_size, : m.vocab].clone()


def prefill_one(g, ids: torch.Tensor, slot: int) -> int:
    lg = g.prefill_forward(ids.reshape(1, -1), prompt_lens=[int(ids.numel())], slots=[slot])
    return int(torch.argmax(lg[0, -1, : g.model.vocab]))


# ------------------------------------------------------------------------------------------------ batch > 1
def run_slots(g, assign, steps, mid=None):
    """assign [(slot, ids)] prefilled up front (in order); mid = (step, slot, ids): prefilled before that step while
    the others are mid-decode. Traced greedy decode with token feedback on device; the host reloads tokens /
    positions only after a prefill. Returns {slot: {"tokens", "rows"}} (rows: the slot's logits row per step)."""
    g.reset()
    cur = {}
    for s, ids in assign:
        cur[s] = [prefill_one(g, ids, s), int(ids.numel())]
    out = {s: {"tokens": [cur[s][0]], "rows": []} for s in cur}
    reload = True
    for i in range(steps):
        if mid is not None and i == mid[0]:
            s, ids = mid[1], mid[2]
            cur[s] = [prefill_one(g, ids, s), int(ids.numel())]
            out[s] = {"tokens": [cur[s][0]], "rows": []}
            reload = True
        sl = sorted(cur)
        toks = g.decode_forward(
            torch.tensor([[cur[s][0]] for s in sl]),
            torch.tensor([cur[s][1] for s in sl]),
            slots=sl,
            reload_inputs=reload,
        )
        rows = logits_rows(g)
        for j, s in enumerate(sl):
            out[s]["rows"].append(rows[s].to(torch.bfloat16))
            cur[s] = [int(toks[j]), cur[s][1] + 1]
            out[s]["tokens"].append(cur[s][0])
        reload = False
    return out


def _cmp_runs(a, b) -> dict:
    """Two runs of one slot: token sequences and per-step logits rows (up to the first token divergence)."""
    fd = first_div(a["tokens"], b["tokens"])
    n = len(a["rows"]) if fd is None else min(fd, len(a["rows"]))
    per = [cmp_rows(a["rows"][i], b["rows"][i]) for i in range(n)]
    return {
        "tokens_equal": a["tokens"] == b["tokens"],
        "first_token_divergence": fd,
        "steps_compared": n,
        "bit_identical_steps": sum(p["bit_identical"] for p in per),
        "min_pcc": min((p["pcc"] for p in per), default=None),
        "max_abs_diff": max((p["max_abs_diff"] for p in per), default=None),
        "top1_equal_steps": sum(p["top1_equal"] for p in per),
    }


def run_batch(g, slots, lengths, steps, save_pt=None) -> dict:
    tok = g.tokenizer
    corpus = text_ids(tok, sum(lengths) + 64)
    prompts, o = [], 0
    for L in lengths:  # distinct passages of the corpus
        prompts.append(corpus[o : o + L])
        o += L
    rec = {"slots": list(slots), "lengths": list(lengths), "steps": steps}
    solo = [run_slots(g, [(s, p)], steps)[s] for s, p in zip(slots, prompts)]
    joint = run_slots(g, list(zip(slots, prompts)), steps)
    rec["joint_vs_solo"] = {str(s): _cmp_runs(joint[s], solo[i]) for i, s in enumerate(slots)}
    # same prompt in two slots (the first and the last of `slots`)
    sa, sb = slots[0], slots[-1]
    same = run_slots(g, [(sa, prompts[0]), (sb, prompts[0])], steps)
    rec["same_prompt_two_slots"] = {"slots": [sa, sb], **_cmp_runs(same[sa], same[sb])}
    rec["same_prompt_vs_solo"] = _cmp_runs(same[sb], solo[0])
    # prefill of the last slot while the others are mid-decode
    k = max(1, steps // 3)
    mid = run_slots(g, list(zip(slots[:-1], prompts[:-1])), steps, mid=(k, slots[-1], prompts[-1]))
    rec["mid_prefill"] = {"step": k, "slot": slots[-1]}
    rec["mid_prefill_others_vs_solo"] = {str(s): _cmp_runs(mid[s], solo[i]) for i, s in enumerate(slots[:-1])}
    late = solo[-1]
    rec["mid_prefill_late_slot_vs_solo"] = _cmp_runs(
        mid[slots[-1]], {"tokens": late["tokens"][: steps - k + 1], "rows": late["rows"][: steps - k]}
    )
    rec["texts"] = {str(s): tok.decode(joint[s]["tokens"]) for s in slots} if tok is not None else None
    if save_pt:
        torch.save({"solo": solo, "joint": joint, "slots": list(slots), "lengths": list(lengths)}, save_pt)
    ok = all(v["tokens_equal"] for v in rec["joint_vs_solo"].values()) and rec["same_prompt_two_slots"]["tokens_equal"]
    rec["pass_tokens"] = bool(ok)
    print("S67R BATCH " + json.dumps(rec), flush=True)
    g.reset()
    return rec


# ------------------------------------------------------------------------------------------------ page-table change
def latent_rows(dsa, blk: int, n: int = 128) -> torch.Tensor:
    """Host copy of the cache rows of global block `blk` (its owner die under the block-sharded layout): the latent
    rows, or with an fp8 latent pool (to_torch cannot read fp8) the block's 32 pooled index keys (bf16; written through
    the same page table by the same decode step). Reads the die's whole pool to the host: a device slice here would
    build a new program after the decode capture (its program-cache buffers would be live across replays)."""
    if dsa.latent.dtype == ttnn.fp8_e4m3:
        h = ttnn.to_torch(ttnn.get_device_tensors(dsa.pooled)[0]).float()
        return h.reshape(-1, 32, h.shape[-1])[blk].clone()
    sharded = getattr(dsa, "dsa_shard", False)
    die, r0 = (blk % dsa.tp, (blk // dsa.tp) * 128) if sharded else (0, blk * 128)
    h = ttnn.to_torch(ttnn.get_device_tensors(dsa.latent)[die]).float()
    W = int(h.shape[-1])
    return h.reshape(-1, W)[r0 : r0 + n, :512].clone()


def run_ptchange(g, slot: int, L: int, steps: int) -> dict:
    m = g.model
    dsa = next(d.attn for d in m.layers if d.attn_kind == "dsa")
    ids = text_ids(g.tokenizer, L)
    B = (L // 128 + 1) * 128  # first 128-token boundary after the prompt
    assert L < B < L + steps, (L, B, steps)
    col = B // 128
    base = g.page_table.clone()
    new = base.clone()
    nb = base.shape[1]
    # the new block: a permuted id that `slot` does not use at these positions (its own last column's block, swapped)
    new[slot, col], new[slot, nb - 1] = base[slot, nb - 1], base[slot, col]
    old_blk, new_blk = int(base[slot, col]), int(new[slot, col])
    calls = {"n": 0}
    real_set = m.set_page_table

    def counted(pt):
        calls["n"] += 1
        return real_set(pt)

    def run(traced: bool, forced=None):
        g._set_page_rows(base[slot : slot + 1], [slot])
        g.reset()
        tok0 = prefill_one(g, ids, slot)
        snap_old = latent_rows(dsa, old_blk)
        snap_new = latent_rows(dsa, new_blk)
        toks, rows = [tok0], []
        m.set_page_table = counted
        c0 = calls["n"]
        unchanged_copies = 0
        try:
            pos = L
            for i in range(steps):
                pt_rows = (new if pos >= B else base)[slot : slot + 1]
                before = calls["n"]
                tk = toks[-1] if forced is None else forced[i]
                if traced:
                    out = g.decode_forward(
                        torch.tensor([[tk]]),
                        torch.tensor([pos]),
                        page_table=pt_rows,
                        slots=[slot],
                        reload_inputs=(i == 0),
                    )
                    rows.append(logits_rows(g)[slot])
                    toks.append(int(out[0]))
                else:
                    out = g.decode_forward(
                        torch.tensor([[tk]]),
                        torch.tensor([pos]),
                        page_table=pt_rows,
                        slots=[slot],
                        enable_trace=False,
                        on_device_sampling=False,
                        reload_inputs=True,
                    )
                    rows.append(out[0].clone())
                    toks.append(int(torch.argmax(out[0])))
                if pos != B:
                    unchanged_copies += calls["n"] - before
                pos += 1
        finally:
            m.set_page_table = real_set
        k = L + steps - B  # tokens written at positions >= B
        if dsa.latent.dtype == ttnn.fp8_e4m3:  # pooled keys: one row per 4 positions (the open pool included)
            k = -(-k // 4)
        after_old, after_new = latent_rows(dsa, old_blk), latent_rows(dsa, new_blk)
        return {
            "tokens": toks,
            "rows": rows,
            "copies_total": calls["n"] - c0,
            "copies_on_unchanged_steps": unchanged_copies,
            "new_block_rows_written": int((after_new[:k] != snap_new[:k]).any(dim=1).sum()),
            "new_block_rows_nonzero": int((after_new[:k].abs().sum(dim=1) > 0).sum()),
            "old_block_unchanged": bool(torch.equal(after_old, snap_old)),
            "rows_written_expected": k,
        }

    tr = run(True)
    ea = run(False, forced=tr["tokens"][:-1])  # eager, teacher-forced with the traced tokens
    per = [cmp_rows(tr["rows"][i], ea["rows"][i]) for i in range(steps)]
    g._set_page_rows(base[slot : slot + 1], [slot])
    g.reset()
    rec = {
        "slot": slot,
        "prompt_len": L,
        "boundary": B,
        "old_block": old_blk,
        "new_block": new_blk,
        "traced": {k: v for k, v in tr.items() if k != "rows"},
        "eager": {k: v for k, v in ea.items() if k != "rows"},
        "traced_vs_eager_min_pcc": min(p["pcc"] for p in per),
        "traced_vs_eager_max_abs_diff": max(p["max_abs_diff"] for p in per),
        "traced_vs_eager_top1_equal_steps": sum(p["top1_equal"] for p in per),
        "steps": steps,
    }
    rec["pass"] = bool(
        tr["copies_total"] == 1
        and tr["copies_on_unchanged_steps"] == 0
        and tr["new_block_rows_written"] == tr["rows_written_expected"]
        and tr["old_block_unchanged"]
        and rec["traced_vs_eager_top1_equal_steps"] == steps
    )
    print("S67R PTCHANGE " + json.dumps(rec), flush=True)
    return rec


# ------------------------------------------------------------------------------------------------ sampling
def _sp(n, temperature=0.0, top_k=1, top_p=1.0, rep=1.0, seed=None):
    from types import SimpleNamespace

    return SimpleNamespace(
        temperature=[temperature] * n,
        top_k=[top_k] * n,
        top_p=[top_p] * n,
        presence_penalty=[0.0] * n,
        frequency_penalty=[0.0] * n,
        repetition_penalty=[rep] * n,
        seed=[seed] * n,
        enable_log_probs=[False] * n,
        num_logprobs=None,
    )


def run_sampling(g, slots, L: int, steps: int) -> dict:
    from models.autoports.zai_org_glm_5_3_flash.tt.serving_state import rows_to_slots, rows_to_slots_tokens

    m = g.model
    SR, n = m.SR, len(slots)
    ids = text_ids(g.tokenizer, L)
    modes = {
        "greedy": dict(),
        "topk": dict(temperature=0.8, top_k=20, top_p=0.9),
        "penalty": dict(temperature=0.8, top_k=20, top_p=1.0, rep=1.3),
    }
    plan = ["greedy", "topk", "greedy", "penalty", "topk", "greedy"] * (-(-steps // 6))
    plan = plan[:steps]
    sm = m.sampling
    if not any(k.penalties_on and v["id"] is not None for k, v in sm._trace_states.items()):
        plan = [p for p in plan if p != "penalty"]  # this build captured no penalties sampler trace

    def episode(plan, seed=None):
        g.reset()
        cur = {s: [prefill_one(g, ids, s), L] for s in slots}
        recs = []
        prompt = rows_to_slots_tokens(ids.reshape(1, -1).repeat(n, 1), slots, SR)
        prev = None
        for i, mode in enumerate(plan):
            sp_rows = _sp(n, seed=seed if mode != "greedy" else None, **modes[mode])
            sl = list(slots)
            toks = g.decode_forward(
                torch.tensor([[cur[s][0]] for s in sl]),
                torch.tensor([cur[s][1] for s in sl]),
                slots=sl,
                serving=True,
                sampling_params=rows_to_slots(sp_rows, sl, SR),
                reload_inputs=(i == 0),
                reload_sampling_params=(mode != prev),
                reset_sampling_state=(i == 0 or mode != prev),
                prompt_tokens=prompt,
                output_tokens=None,
                read_from_device=True,
            )
            prev = mode
            rows = logits_rows(g)  # penalized in place on penalty steps (what the sampler drew from)
            dies = [
                ttnn.to_torch(t).reshape(-1)[: g.max_batch_size].tolist() for t in ttnn.get_device_tensors(g.tok_buf)
            ]
            r = {"mode": mode, "tokens": [int(t) for t in toks.tolist()], "dies_agree": all(d == dies[0] for d in dies)}
            for j, s in enumerate(sl):
                row, t = rows[s], int(toks[j])
                if mode == "greedy":
                    r.setdefault("greedy_eq_argmax", []).append(t == int(row.argmax()))
                else:
                    k = modes[mode]["top_k"]
                    ok = t in set(row.topk(k).indices.tolist())
                    if not ok and mode == "penalty":  # rows may be the raw logits: apply the repetition penalty on host
                        rep = modes[mode]["rep"]
                        pen = row.float().clone()
                        idx = torch.unique(ids)
                        v = pen[idx]
                        pen[idx] = torch.where(v > 0, v / rep, v * rep)
                        ok = t in set(pen.topk(k).indices.tolist())
                        r.setdefault("penalty_checked_on_host_penalized_logits", []).append(True)
                    r.setdefault("in_topk", []).append(ok)
                cur[s] = [t, cur[s][1] + 1]
            recs.append(r)
        return recs

    a = episode(plan)
    seeded_plan = ["topk"] * 6
    s1 = episode(seeded_plan, seed=1234)
    s2 = episode(seeded_plan, seed=1234)
    s3 = episode(seeded_plan, seed=99)
    # back to greedy for later checks (params, seeds and penalty state)
    g._sstate.apply_decode(
        rows_to_slots(_sp(n), list(slots), SR),
        [-1] * SR,
        reload_params=True,
        reset_state=True,
        host_authoritative=True,
    )
    g.reset()
    rec = {
        "slots": list(slots),
        "prompt_len": L,
        "alternating": a,
        "greedy_steps_ok": all(all(r.get("greedy_eq_argmax", [True])) for r in a),
        "sampled_in_topk": all(all(r.get("in_topk", [True])) for r in a),
        "dies_agree_all_steps": all(r["dies_agree"] for r in a + s1 + s2 + s3),
        "seed_reproduces": [r["tokens"] for r in s1] == [r["tokens"] for r in s2],
        "seed_changes_draws": [r["tokens"] for r in s1] != [r["tokens"] for r in s3],
        "seeded_tokens": [[r["tokens"] for r in s1], [r["tokens"] for r in s2], [r["tokens"] for r in s3]],
    }
    rec["pass"] = bool(
        rec["greedy_steps_ok"] and rec["sampled_in_topk"] and rec["dies_agree_all_steps"] and rec["seed_reproduces"]
    )
    print("S67R SAMPLING " + json.dumps(rec), flush=True)
    return rec


# ------------------------------------------------------------------------------------------------ chunk plans
def run_chunkplan(g, lengths, gen: int, save_pt=None) -> dict:
    m = g.model
    corpus = text_ids(g.tokenizer, max(lengths))
    out = {}
    rows = {"ids": corpus}
    saved = m.prefill_plan
    try:
        for L in lengths:
            if L + gen - 1 > g.max_seq_len:
                continue
            ids = corpus[:L]
            res = {}
            for plan in ("cost", "legacy"):
                m.prefill_plan = plan
                g.reset()
                t0 = time.perf_counter()
                lg = g.prefill_forward(ids.reshape(1, -1), prompt_lens=[L])[0, -1, : m.vocab]
                dt = time.perf_counter() - t0
                toks = g.generate(ids.tolist(), gen)
                res[plan] = (lg, toks, m.chunk_plan(L), dt)
            (la, ta, pa, da), (lb, tb, pb, db) = res["cost"], res["legacy"]
            rows[L] = {"cost": la.to(torch.bfloat16), "legacy": lb.to(torch.bfloat16)}
            r = {
                "plan_cost": pa,
                "plan_legacy": pb,
                "prefill_ms_cost": round(da * 1e3, 1),
                "prefill_ms_legacy": round(db * 1e3, 1),
                **cmp_rows(la, lb),
                "greedy_tokens_equal": ta == tb,
                "greedy_first_divergence": first_div(ta, tb),
                "tokens_cost": ta,
                "tokens_legacy": tb,
            }
            out[L] = r
            print(f"S67R CHUNKPLAN L={L}: " + json.dumps(r), flush=True)
    finally:
        m.prefill_plan = saved
        g.reset()
        if save_pt:  # last-row logits of both plans (+ the prompt ids) for an HF control (s67r_hf_lastrow.py)
            torch.save(rows, save_pt)
    return out


# ------------------------------------------------------------------------------------------------ deep positions
def _dsa_stats(m, on: bool):
    for d in m.layers:
        if d.attn_kind == "dsa" and getattr(d.attn, "dsa_shard", False):
            d.attn.stats_on = on


def _read_stats(m) -> dict:
    import math

    out = {}
    for d in m.layers:
        a = d.attn
        if d.attn_kind == "dsa" and getattr(a, "dsa_shard", False):
            st = dict(a.wstats)
            if st["count"]:
                st["headroom_nats_lower_bound"] = round(88.7 + math.log(st["min_w"]), 2) if st["min_w"] > 0 else None
            out[str(a.layer_idx)] = st
            a.wstats = {"min_w": float("inf"), "zeros": 0, "count": 0, "calls": 0}
    return out


def run_deep(g, lengths, steps: int, stats: bool) -> dict:
    from models.autoports.zai_org_glm_5_3_flash.tools.memory_probe import copy_prompt

    m = g.model
    out = {}
    for L in lengths:
        if L + steps > g.max_seq_len:
            continue
        ids, pas = copy_prompt(g.tokenizer, L)
        g.reset()
        if stats:
            _dsa_stats(m, True)
        ttnn.synchronize_device(g.mesh)
        t0 = time.perf_counter()
        lg = g.prefill_forward(ids.reshape(1, -1), prompt_lens=[L])
        ttft = time.perf_counter() - t0
        r = {"ttft_s": round(ttft, 2), "plan_chunks": len(m.chunk_plan(L))}
        first = int(torch.argmax(lg[0, -1, : m.vocab]))
        if stats:  # eager decode steps (the stats read w to the host: never inside a trace replay)
            tk, pos = first, L
            for _ in range(4):
                o = g.decode_forward(
                    torch.tensor([[tk]]), torch.tensor([pos]), enable_trace=False, on_device_sampling=False
                )
                tk, pos = int(torch.argmax(o[0])), pos + 1
            _dsa_stats(m, False)
            r["sink_stats"] = _read_stats(m)
            g.reset()
            lg = g.prefill_forward(ids.reshape(1, -1), prompt_lens=[L])
            first = int(torch.argmax(lg[0, -1, : m.vocab]))
        toks = [first]
        out1 = g.decode_forward(torch.tensor([[first]]), torch.tensor([L]), reload_inputs=True)
        toks.append(int(out1[0]))
        ttnn.synchronize_device(g.mesh)
        t0 = time.perf_counter()
        for i in range(steps):
            o = g.decode_forward(torch.tensor([[0]]), torch.tensor([0]), reload_inputs=False)
            toks.append(int(o[0]))
        dt = (time.perf_counter() - t0) / steps
        expect = [pas[(L + j) % len(pas)] for j in range(len(toks))]
        r.update(
            {
                "decode_ms_per_token_generate_loop": round(dt * 1e3, 2),
                "decode_positions": [L + 1, L + steps],
                "copy_accuracy": round(sum(int(a == b) for a, b in zip(toks, expect)) / len(toks), 3),
            }
        )
        out[L] = r
        print(f"S67R DEEP L={L}: " + json.dumps(r), flush=True)
    g.reset()
    return out


# ------------------------------------------------------------------------------------------------ allocation audit
def run_allocaudit(g, L: int = 161, steps: int = 8) -> dict:
    """Device tensors created from the host (from_torch / zeros / ones / full / as_tensor with a device, empty) AFTER
    the decode capture, by call site, over one teacher-forcing-like episode (reset, prefill, traced decode) and one
    all-logits prefill (the prefill-check path). A persistent tensor created after the capture can sit in the freed
    trace intermediates and be overwritten by replays (seen under DRAM pressure: teacher forcing 14/32/90)."""
    import traceback

    names = ["from_torch", "zeros", "ones", "full", "as_tensor", "empty", "zeros_like", "ones_like", "full_like"]
    sites = {}
    real = {n: getattr(ttnn, n) for n in names}

    def wrap(n):
        def f(*a, **k):
            dev = k.get("device") is not None or n.endswith("_like") or n in ("empty",)
            if dev:
                st = [fr for fr in traceback.extract_stack()[:-1] if "zai_org_glm_5_3_flash" in fr.filename]
                key = n + " @ " + " <- ".join(f"{Path(fr.filename).name}:{fr.lineno}:{fr.name}" for fr in st[-3:][::-1])
                sites[key] = sites.get(key, 0) + 1
            return real[n](*a, **k)

        return f

    v = lambda: int(ttnn.get_memory_view(g.mesh, ttnn.BufferType.DRAM).total_bytes_allocated_per_bank)
    ids = text_ids(g.tokenizer, L + steps)
    a0 = v()
    for n in names:
        setattr(ttnn, n, wrap(n))
    try:
        g.generate(ids[:L].tolist(), steps)
        a1 = v()
        g.prefill_logits(ids[: L + steps].tolist())
        g.reset()
        a2 = v()
        g.generate(ids[:L].tolist(), steps)
        a3 = v()
    finally:
        for n in names:
            setattr(ttnn, n, real[n])
    rec = {
        "alloc_per_bank": [a0, a1, a2, a3],
        "growth_bytes_per_die": (a3 - a0) * 8,
        "post_capture_device_creations": dict(sorted(sites.items(), key=lambda kv: -kv[1])),
    }
    print("S67R ALLOCAUDIT " + json.dumps(rec), flush=True)
    g.reset()
    return rec


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slots", type=int, default=1)
    ap.add_argument("--seq-len", type=int, default=None, help="None = the generator's default for --slots")
    ap.add_argument("--layout", default=None, choices=[None, "replicated", "sharded"])
    ap.add_argument("--do", default="ready")
    ap.add_argument("--batch-slots", default=None, help="e.g. 0,3,6 (default: 0, S/2+1, S-1)")
    ap.add_argument("--batch-lengths", default="161,300,517")
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--save-pt", default=None)
    ap.add_argument("--chunk-lengths", default="513,600,1000,1153,2049,2600,10001")
    ap.add_argument("--chunk-gen", type=int, default=24)
    ap.add_argument("--deep", default="32768,100000")
    ap.add_argument("--deep-steps", type=int, default=32)
    ap.add_argument("--active", default="1")
    ap.add_argument("--prompts", default="128,161,512,2048,8192")
    ap.add_argument("--reference", default=str(MODEL_DIR / "readiness_aime24_chat.refpt"))
    ap.add_argument("--report-dir", default=None)
    ap.add_argument("--ar-dir", default=str(MODEL_DIR / "doc" / "optimized_full_model"))
    ap.add_argument("--trace-region", type=int, default=int(os.environ.get("GLM53_TRACE_REGION", 64 << 20)))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.environ.setdefault("HF_MODEL", "zai-org/GLM-5.3-Flash")
    from models.autoports.zai_org_glm_5_3_flash.tools import s7_probe as P
    from models.autoports.zai_org_glm_5_3_flash.tt.generator import Glm53Generator

    _dmode = os.environ.get("S67R_DEALLOC", "")
    _dcount = {"calls": 0}
    if _dmode in ("noop", "noforce", "audit"):  # diagnostic: tensor lifetime (deallocate defaults to force=True)
        import traceback as _tb

        _real_dealloc = ttnn.deallocate
        _shared_sites = {}

        def _dealloc(t, *x, **k):
            _dcount["calls"] += 1
            if _dmode == "noop":
                return None
            k["force"] = False
            r = _real_dealloc(t, *x, **k)
            if _dmode == "audit" and isinstance(t, ttnn.Tensor) and t.is_allocated():
                # force=True (the default) would have freed a buffer that another tensor (a view) still holds
                st = [
                    f
                    for f in _tb.extract_stack()[:-1]
                    if "zai_org_glm_5_3_flash" in f.filename
                    or "models/common" in f.filename
                    or "tt_transformers" in f.filename
                ]
                key = " <- ".join(f"{Path(f.filename).name}:{f.lineno}:{f.name}" for f in st[-3:][::-1])
                _shared_sites[key] = _shared_sites.get(key, 0) + 1
            return r

        ttnn.deallocate = _dealloc
        import atexit

        def _report():
            print(f"S67R_DEALLOC mode={_dmode} calls={_dcount['calls']}", flush=True)
            for k, v in sorted(_shared_sites.items(), key=lambda kv: -kv[1]):
                print(f"S67R_SHARED_FREE {v} x {k}", flush=True)

        atexit.register(_report)
    todo = a.do.split(",")
    rec = {"args": vars(a), "layers": os.environ.get("GLM53_LAYERS", "all"), "provenance_start": provenance()}
    print("S67R PROVENANCE " + json.dumps(rec["provenance_start"]), flush=True)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    def save():
        out.write_text(json.dumps(rec, indent=1, default=str) + "\n")

    mesh = P._open_mesh(a.trace_region)
    try:
        t0 = time.time()
        g = Glm53Generator(mesh, max_batch_size=a.slots, max_seq_len=a.seq_len, dsa_layout=a.layout)
        rec["build_s"] = round(time.time() - t0, 1)
        rec["plan"] = g.plan
        rec["skip_inactive"] = getattr(g.model, "skip_inactive", None)
        rec["moe_split"] = getattr(g.model, "moe_split", None)
        rec["lockstep"] = os.environ.get("GLM53_LOCKSTEP", "1")
        v = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
        rec["dram_free_after_build_per_die"] = int(v.total_bytes_free_per_bank) * int(v.num_banks)
        print(f"S67R built S={a.slots} plan={g.plan} in {rec['build_s']} s", flush=True)
        save()
        S = a.slots
        if "allocaudit" in todo:
            rec["allocaudit"] = run_allocaudit(g)
            save()
        if "ready" in todo:
            if os.environ.get("S67R_EAGER_TF") == "1":  # diagnostic: teacher forcing on eager decode
                real_gen = g.generate
                g.generate = lambda *x, **k: real_gen(*x, **{**k, "enable_trace": False, "on_device_sampling": False})
            P.run_ready(g, a.reference, a.report_dir, rec)
            save()
        if "ar" in todo:
            from types import SimpleNamespace

            from models.autoports.zai_org_glm_5_3_flash.tools import full_model_checks as fmc

            rc = Path(os.environ["TT_MODEL_BRINGUP_ROOT"]) / "runtime" / "readiness_check"
            ca = SimpleNamespace(
                prompt_file=rc / "autoregressive_prompt.txt",
                hf_tokens=fmc.DOC / "autoregressive" / "hf_tokens.json",
                qual_max_new_tokens=0,
                skip_autoregressive=False,
                wait_hf_min=0.0,
                skip_qualitative=True,
            )
            rec["ar"] = fmc.run_with(g, mesh, ca, Path(a.ar_dir))
            g.reset()
            save()
        if "batch" in todo and S > 1:
            slots = [int(x) for x in a.batch_slots.split(",")] if a.batch_slots else sorted({0, S // 2 + 1, S - 1})
            rec["batch"] = run_batch(
                g, slots, [int(x) for x in a.batch_lengths.split(",")][: len(slots)], a.steps, a.save_pt
            )
            save()
        if "ptchange" in todo:
            rec["ptchange"] = run_ptchange(g, slot=(1 if S > 1 else 0), L=250, steps=12)
            save()
        if "sampling" in todo:
            rec["sampling"] = run_sampling(g, [0] if S == 1 else [0, S - 1], L=97, steps=12)
            save()
        if "chunkplan" in todo:
            cp_pt = str(Path(a.out).with_suffix("")) + "_chunkplan_rows.pt"
            rec["chunkplan"] = run_chunkplan(g, [int(x) for x in a.chunk_lengths.split(",")], a.chunk_gen, cp_pt)
            save()
        if "ttft" in todo:
            P.run_ttft(g, [int(x) for x in a.prompts.split(",")], rec)
            save()
        if "decode" in todo:
            P.run_decode(g, [int(x) for x in a.active.split(",")], 20, rec)
            save()
        if "deep" in todo:
            rec["deep"] = run_deep(
                g, [int(x) for x in a.deep.split(",")], a.deep_steps, getattr(g.model, "dsa_shard", False)
            )
            save()
        g.teardown()
    finally:
        rec["provenance_end"] = provenance()
        rec["provenance_unchanged"] = rec["provenance_end"]["tt_md5"] == rec["provenance_start"]["tt_md5"]
        save()
        print(
            "S67R_RESULT "
            + json.dumps({k: v for k, v in rec.items() if k not in ("provenance_start",)}, default=str)[:20000],
            flush=True,
        )
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()

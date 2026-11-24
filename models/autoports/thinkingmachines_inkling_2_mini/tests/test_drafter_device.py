# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device test of tt/drafter.py (DSpark drafter) against reference/dspark_ref.py on real text, eager and traced.

Fixture /cache/fixtures/dspark_taps.pt (tools/dspark_taps.py, host CPU): per text the HF target's taps [T, 4, H], the
anchors and the CPU reference's block hidden / Markov drafts / top-16 step logits.

  1. eager: per text the context is committed incrementally (commit(taps[last:p], last)), draft(ids[p], p, d=7) at
     every anchor (p = a + 1): draft tokens identical to the reference (all, per depth, d <= 3), first-mismatch analysis
     (device token's reference logit vs the reference top-1: a near-tie or a real miss), block hidden error
  2. sliding window shrunk to 256 vs the CPU reference with the same window
  3. traced step (ctrl + commit of 4 rows + draft d=3) replayed along every anchor: tokens vs the eager ones / the
     reference; traced time per step and per phase (each phase captured alone)

usage: ~/inkling2-work/jobs/die.sh 3 drafter "" python -u models/autoports/thinkingmachines_inkling_2_mini/tests/test_drafter_device.py
env: DRAFT_W_DT / DRAFT_HEAD_DT / DRAFT_FOLD_DT (bf16 | bfp8, default bfp8), FIXTURE, DRAFT_D (traced d, default 3),
     MESH=N (a 1 x N mesh: the drafter replicated, its vocab sharded unless DRAFT_VSHARD=0), DRAFT_VSHARD_SIM=N (one
     die, N simulated vocab slices), DRAFT_SKIP_EAGER=1
"""

import faulthandler
import os
import signal
import sys
import time
import types

import torch

import ttnn
from models.autoports.thinkingmachines_inkling_2_mini.tt import clean_exit
from models.autoports.thinkingmachines_inkling_2_mini.tt import dmm as dmm_mod
from models.autoports.thinkingmachines_inkling_2_mini.tt import drafter as drafter_mod
from models.autoports.thinkingmachines_inkling_2_mini.tt import model as model_mod

FIXTURE = os.environ.get("FIXTURE", "/cache/fixtures/dspark_taps.pt")
DTR = int(os.environ.get("DRAFT_D", "3"))


class _Count:
    """ttnn proxy counting calls that return a device tensor (op count)."""

    def __init__(self, mod, box):
        self._m, self._box = mod, box

    def __getattr__(self, a):
        v = getattr(self._m, a)
        if isinstance(v, types.ModuleType):
            return _Count(v, self._box)
        if callable(v) and not isinstance(v, type):
            def f(*args, **kw):
                r = v(*args, **kw)
                if isinstance(r, ttnn.Tensor) or a == "generic_op":
                    self._box[a] = self._box.get(a, 0) + 1
                return r
            return f
        return v


def count_ops(fn):
    box = {}
    mods = (drafter_mod, dmm_mod, model_mod)
    saved = [m.ttnn for m in mods]
    for m in mods:
        m.ttnn = _Count(ttnn, box)
    try:
        fn()
    finally:
        for m, s in zip(mods, saved):
            m.ttnn = s
    return box


def cpu_block_ref(fx_cfg, taps, ids, p, window):
    """The reference block hidden with a given sliding window (reference/dspark_ref.py without its LM-head fold)."""
    from safetensors.torch import load_file

    from models.autoports.thinkingmachines_inkling_2_mini.reference import dspark_ref as R
    from models.autoports.thinkingmachines_inkling_2_mini.tt.weights import Checkpoint, load_config

    cfg = R.MTPConfig(**{**fx_cfg, "sliding_window": window})
    c = load_config()
    ck = Checkpoint(c.snap)
    ref = R.DSparkDrafter.__new__(R.DSparkDrafter)
    ref.cfg = cfg
    ref.w = load_file(os.path.join(c.snap, "mtp.safetensors"))
    ref.embed_weight = ck.get("model.llm.embed.weight")
    ref.embed_norm_weight = ck.get("model.llm.embed_norm.weight")
    ref.target_eps = c.rms_norm_eps
    ref.fuse_alpha = torch.softmax(ref.w["fuse_weight"].float(), dim=-1).to(torch.bfloat16)
    d = cfg.head_dim
    inv = 1.0 / (cfg.rope_theta ** (torch.arange(0, d, 2, dtype=torch.float64) / d))
    ang = torch.arange(taps.shape[0] + 16, dtype=torch.float64)[:, None] * inv[None, :]
    ref.cos, ref.sin = ang.cos().float(), ang.sin().float()
    kv = ref.context_kv(taps[:p], torch.arange(p))
    return ref.block_hidden(kv, p, int(ids[p]))


def timed(mesh, fn, n=20):
    """Device time of fn's programs: captured alone in a trace, replayed n times (ms per replay)."""
    fn()
    ttnn.synchronize_device(mesh)
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    fn()
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
    t0 = time.time()
    for _ in range(n):
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    ms = (time.time() - t0) / n * 1e3
    ttnn.release_trace(mesh, tid)
    return ms


def main():
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    torch.manual_seed(0)
    fx = torch.load(FIXTURE)
    n = int(os.environ.get("MESH", "1"))
    if n > 1:  # (the vocab-sharded drafter's all-gather)
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, n), trace_region_size=256 << 20)
    try:
        run(mesh, fx)
    finally:
        clean_exit.close(mesh)


def analyse(tag, rows, r_of):
    """rows: [(name, j, p, got)] -> match stats vs the reference (first-mismatch analysis)."""
    n_tok = n_same = n_all = 0
    depth = None
    near, real = [], []
    for name, j, p, got in rows:
        r = r_of[name]
        ref = r["markov"][j].tolist()[: len(got)]
        same = [g == x for g, x in zip(got, ref)]
        depth = depth or [[0, 0] for _ in same]
        n_tok += len(same)
        n_same += sum(same)
        n_all += all(same)
        for k, s in enumerate(same):
            depth[k][0] += s
            depth[k][1] += 1
        if not all(same):
            k = same.index(False)
            tv, ti = r["top_v"][j][k], r["top_i"][j][k]
            hit = (ti == got[k]).nonzero()
            gap = (tv[0] - tv[hit[0, 0]]).item() if len(hit) else float("inf")
            (near if gap <= 0.25 else real).append((name, p, k, got[k], ref[k], round(gap, 4)))
    print(f"[{tag}] draft tokens identical to the CPU reference: {n_same}/{n_tok} = {n_same / n_tok:.4f}; anchors "
          f"fully identical {n_all}/{len(rows)}; per depth " + " ".join(f"{a / b:.3f}" for a, b in depth), flush=True)
    print(f"[{tag}] first mismatches: {len(near)} near-ties (ref gap <= 0.25): {near}; other: {real}", flush=True)


def run(mesh, fx):
    t0 = time.time()
    maxT = max(r["taps"].shape[0] for r in fx["texts"].values())
    dr = drafter_mod.DSparkDrafter(mesh, None, max_len=maxT + 64)
    print(f"drafter up in {time.time() - t0:.0f}s", flush=True)
    gamma = dr.B - 1
    r_of = fx["texts"]

    if os.environ.get("DRAFT_SKIP_EAGER") == "1":
        return run_traced(mesh, dr, fx, {})
    # ---- 1. eager
    rows, rows3, herr = [], [], []
    for name, r in r_of.items():
        ids, taps = r["ids"], r["taps"]
        last = 0
        for j, a in enumerate(r["anchors"]):
            p = a + 1
            dr.commit(taps[last:p], last)
            last = p
            _, got, h = dr.draft(int(ids[p]), p, None, gamma, return_hidden=True)
            rows.append((name, j, p, got))
            hd = dr._host(h).float().reshape(-1, dr.H)[: dr.B]
            hr = r["hidden"][j].float()
            herr.append(((hd - hr).abs().max().item(),
                         torch.corrcoef(torch.stack([hd.flatten(), hr.flatten()]))[0, 1].item()))
        print(f"[{name}] eager done ({time.time() - t0:.0f}s)", flush=True)
    analyse("eager d=7", rows, r_of)
    analyse("eager d<=3", [(n, j, p, g[:3]) for n, j, p, g in rows], r_of)
    print(f"block hidden vs ref: max |diff| {max(e for e, _ in herr):.4f}, min pcc {min(c for _, c in herr):.6f}",
          flush=True)

    # ---- 2. sliding window 256
    name, r = next(iter(r_of.items()))
    ids, taps = r["ids"], r["taps"]
    p = int(r["anchors"][-1]) + 1
    dr.window = 256
    dr.commit(taps[:p], 0)
    _, _, h = dr.draft(int(ids[p]), p, None, gamma, return_hidden=True)
    hd = dr._host(h).float().reshape(-1, dr.H)[: dr.B]
    hw = cpu_block_ref(fx["cfg"], taps, ids, p, 256).float()
    print(f"window 256 at p={p}: |dev - ref(256)| max {(hd - hw).abs().max():.4f}, "
          f"|ref(256) - ref(1024)| max {(hw - r['hidden'][-1].float()).abs().max():.4f}", flush=True)
    dr.window = 1024
    run_traced(mesh, dr, fx, {(n, j): g for n, j, _, g in rows})


def run_traced(mesh, dr, fx, eager):
    r_of = fx["texts"]
    # ---- 3. traced step: commit the 4 rows before the anchor + draft
    RCm = drafter_mod.RC
    tb = ttnn.from_torch(torch.zeros(1, 4, 8, dr.H), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                         mesh_mapper=dr.rep,
                         memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def put_taps(rows_):
        x = torch.zeros(1, 4, 8, dr.H, dtype=torch.bfloat16)
        x[0, :, : rows_.shape[0]] = rows_.transpose(0, 1)
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                                        mesh_mapper=dr.rep), tb)

    name0, r0 = next(iter(r_of.items()))
    dr.commit(r0["taps"][: 64], 0)
    put_taps(r0["taps"][64:64 + RCm])
    dr.set_inputs(int(r0["ids"][64 + RCm]), 64, RCm)
    print("traced: first body", flush=True)
    dr.capture(tb, DTR)
    print("captured", flush=True)
    trows, same_eager = [], 0
    for name, r in r_of.items():
        ids, taps = r["ids"], r["taps"]
        for j, a in enumerate(r["anchors"]):
            p = a + 1
            dr.commit(taps[: p - RCm], 0)  # (eager: the context before the step's 4 rows)
            put_taps(taps[p - RCm:p])
            dr.set_inputs(int(ids[p]), p - RCm, RCm)
            dr.replay()
            got = dr.tokens_host()
            trows.append((name, j, p, got))
            same_eager += (name, j) in eager and got == eager[(name, j)][:DTR]
    analyse(f"traced d={DTR}", trows, r_of)
    print(f"traced tokens == eager tokens: {same_eager}/{len(trows)}", flush=True)
    for n_acc in (1, 2, 3):  # commits of fewer rows (rows >= n written past the context)
        r = r0
        p = int(r["anchors"][-1]) + 1
        dr.commit(r["taps"][: p - n_acc], 0)
        put_taps(r["taps"][p - n_acc:p])
        dr.set_inputs(int(r["ids"][p]), p - n_acc, n_acc)
        dr.replay()
        print(f"traced commit n={n_acc}: {dr.tokens_host()} ref {r['markov'][-1].tolist()[:DTR]}", flush=True)

    # timing: the whole step replayed back to back
    n = 50
    ttnn.synchronize_device(mesh)
    t1 = time.time()
    for _ in range(n):
        dr.replay()
    ttnn.synchronize_device(mesh)
    t_step = (time.time() - t1) / n * 1e3
    print(f"TRACED step (ctrl + commit {RCm} rows + draft d={DTR}, ctx {p}, max_len {dr.max_len}): {t_step:.3f} ms",
          flush=True)
    dr.release()
    h = dr.block_hidden()
    t4 = ttnn.slice(tb, [0, 0, 0, 0], [1, 4, RCm, dr.H])
    ph = {
        "ctrl": timed(mesh, dr._ctrl),
        "commit (4 rows)": timed(mesh, lambda: dr._commit_rows(t4, dr.idx_c, dr.in_p0, dr.idx_n)),
        "block forward": timed(mesh, dr.block_hidden),
        f"base logits (rows 1..{DTR})": timed(mesh, lambda: dr._base_logits(h, DTR)),
        f"draft total (d={DTR})": timed(mesh, lambda: dr._draft(DTR)),
        "one Markov step program": timed(mesh, lambda: dr._step(ttnn.slice(h, [0, 0, 1, 0], [1, 1, 2, dr.R]), 1,
                                                                dr.prev[0])),
    }
    ph["Markov steps (rest)"] = ph[f"draft total (d={DTR})"] - ph["block forward"] - ph[f"base logits (rows 1..{DTR})"]
    for k, v in ph.items():
        print(f"  phase {k:28s} {v:7.3f} ms", flush=True)
    for li in range(dr.L):
        q = ttnn.from_torch(torch.zeros(1, dr.NH, dr.B, dr.D), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                            mesh_mapper=dr.rep)
        kv = ttnn.from_torch(torch.zeros(1, dr.NKV, dr.B, dr.D), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                             device=mesh, mesh_mapper=dr.rep)
        print(f"  phase attention layer {li} (S={dr.S[li]}) {timed(mesh, lambda: dr._attention(li, q, kv, kv)):7.3f} ms",
              flush=True)
    ops = count_ops(lambda: dr._traced_body(tb, DTR, True))
    print(f"ops per traced step: {sum(ops.values())}  {dict(sorted(ops.items(), key=lambda x: -x[1]))}", flush=True)
    run_sampled(mesh, dr, fx, tb, put_taps)
    print("RESULT done", flush=True)


def run_sampled(mesh, dr, fx, tb, put_taps):
    """4. sampled drafts (capture(sample=True)): zero noise = the top-1 of the softcapped logits (vs the greedy
    reference), candidates vs the reference's top-16 step logits, the empirical token distribution under T = 1 Gumbel
    noise vs q = softmax(candidate logits), traced time."""
    RCm = drafter_mod.RC
    r_of = fx["texts"]
    name0, r0 = next(iter(r_of.items()))
    p = int(r0["anchors"][0]) + 1
    dr.commit(r0["taps"][: p - RCm], 0)
    put_taps(r0["taps"][p - RCm:p])
    dr.set_inputs(int(r0["ids"][p]), p - RCm, RCm)
    dr.set_gumbel(torch.zeros(DTR, drafter_mod.TOPK))
    dr.capture(tb, DTR, sample=True)
    same, n, vdiff, idmatch = 0, 0, 0.0, []
    for name, r in r_of.items():
        ids, taps = r["ids"], r["taps"]
        for j, a in enumerate(r["anchors"]):
            p = a + 1
            dr.commit(taps[: p - RCm], 0)
            put_taps(taps[p - RCm:p])
            dr.set_inputs(int(ids[p]), p - RCm, RCm)
            dr.set_gumbel(torch.zeros(DTR, drafter_mod.TOPK))
            dr.replay()
            got = dr.tokens_host()
            ref = r["markov"][j].tolist()[:DTR]
            same += sum(g == x for g, x in zip(got, ref))
            n += DTR
            ci, cl = dr.cand_host(DTR)
            for k in range(DTR):
                if got[:k] != ref[:k]:
                    break
                ti, tv = r["top_i"][j][k], r["top_v"][j][k].float()
                m = min(len(ti), 8)
                idmatch.append(int((ci[k, :m] == ti[:m]).sum()) / m)
                vdiff = max(vdiff, (cl[k, 0] - tv[0]).abs().item())
    print(f"[sampled, zero noise] tokens identical to the reference: {same}/{n}; top-8 candidate ids matching the "
          f"reference's (steps on the reference path): {sum(idmatch) / len(idmatch):.3f}; |top-1 logit - ref| max "
          f"{vdiff:.4f}", flush=True)
    # distribution: one anchor, T = 1, N draws: first token frequency vs q
    N = 400
    p = int(r0["anchors"][-1]) + 1
    dr.commit(r0["taps"][: p - RCm], 0)
    put_taps(r0["taps"][p - RCm:p])
    cnt = {}
    g = torch.Generator().manual_seed(1)
    t0 = time.time()
    for _ in range(N):
        dr.set_inputs(int(r0["ids"][p]), p - RCm, RCm)
        u = torch.rand(DTR, drafter_mod.TOPK, generator=g).clamp_(1e-12, 1 - 1e-7)
        dr.set_gumbel(-torch.log(-torch.log(u)))
        dr.replay()
        tok = dr.tokens_host()[0]
        cnt[tok] = cnt.get(tok, 0) + 1
    ci, cl = dr.cand_host(DTR)
    q = torch.softmax(cl[0].double(), -1)
    emp = torch.tensor([cnt.get(int(i), 0) / N for i in ci[0]], dtype=torch.float64)
    tv = 0.5 * (q - emp).abs().sum().item() + 0.5 * sum(v for t, v in cnt.items() if t not in set(ci[0].tolist())) / N
    print(f"[sampled, T=1, {N} draws] first token: q top-3 {[round(x, 3) for x in q[:3].tolist()]} empirical "
          f"{[round(x, 3) for x in emp[:3].tolist()]}; total variation {tv:.3f} (sampling noise ~{(1 / N) ** 0.5:.3f})",
          flush=True)
    nrep = 50
    ttnn.synchronize_device(mesh)
    t1 = time.time()
    for _ in range(nrep):
        dr.replay()
    ttnn.synchronize_device(mesh)
    print(f"TRACED sampled step (d={DTR}): {(time.time() - t1) / nrep * 1e3:.3f} ms", flush=True)
    dr.release()


if __name__ == "__main__":
    sys.exit(main())

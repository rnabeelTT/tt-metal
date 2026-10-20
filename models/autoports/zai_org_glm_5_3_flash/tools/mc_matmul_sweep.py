# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage-5 op-level geometry / fidelity sweep of the material dense matmuls at their TP4 per-device shapes (one die: the
per-device shapes and DRAM bandwidth are those of each die of the (1, 4) mesh).

Decode rows (M = 32): ttnn default, 1D multicast (cores x in0_block_w, incl. in0 blocks > 8), DRAM-sharded (activation
cores x in0_block_w x readers per bank, Blackhole multi-reader), each best candidate again at HiFi2. Device time per call
from 100 traced replays. Prefill rows (M = 2048): ttnn default, the explicit 2D config (tt/multichip_decoder.prefill_pc)
with in0_block_w variants, HiFi2. Every candidate's error against an fp64 product of the same (dequantized) weight.

  python -m models.autoports.zai_org_glm_5_3_flash.tools.mc_matmul_sweep --out sweep.jsonl [--roles kda_in,dsa_o]

Stage-8 options (defaults keep the stage-5 behaviour): --wdtype bfp4 sweeps the same geometries on bfp4 weights (host
round-to-nearest), --fidelity / --alt-fidelity set the base and the second-trial compute fidelity (default HiFi4 /
HiFi2); every decode record carries `out_sha` (sha256 of the output bytes) so bit-identical candidates can be told
apart; role lm_head = the LM head [4096, 38720] per die (no prefill).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time

import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tt.multichip_decoder import DSMatmul, prefill_pc
from models.demos.blackhole.qwen36.tt import tp_common as tpc

B8, B16, F32 = ttnn.bfloat8_b, ttnn.bfloat16, ttnn.float32
# role: (K, N, weight dtype, decode input dtype, prefill input dtype or None)
ROLES = {
    "kda_in": (4096, 8336, B8, B16, B16),
    "kda_out": (2048, 4096, B8, B16, F32),
    "dsa_o": (4096, 4096, B8, B16, B16),
    "dsa_qb": (1536, 4096, B8, B16, B16),
    "dsa_wqb": (1536, 4096, B16, B16, None),
    "pack_a": (4096, 2048, B8, B16, B16),
    "shared_gu": (4096, 1024, B8, B16, B16),
    "shared_dn": (512, 4096, B8, B16, B16),
    "mlp_gu": (4096, 6144, B8, B16, B16),
    "mlp_dn": (3072, 4096, B8, B16, B16),
    "lm_head": (4096, 38720, B8, B16, None),
}
WDT = {"bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}
FID = {f: getattr(ttnn.MathFidelity, f) for f in ("LoFi", "HiFi2", "HiFi3", "HiFi4")}
OPTS = {"wdtype": None, "base": None, "alt": None}  # stage-8 overrides (main)


def ckc(fid, l1_acc):
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=fid, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=l1_acc
    )


HIFI4, HIFI2 = ttnn.MathFidelity.HiFi4, ttnn.MathFidelity.HiFi2


def dev_t(dev, t, dtype, mc=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc)


def time_traced(dev, fn, reps=100):
    out = fn()  # compile
    ttnn.deallocate(out)
    tid = ttnn.begin_trace_capture(dev, cq_id=0)
    out = fn()
    ttnn.end_trace_capture(dev, tid, cq_id=0)
    ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
    t0 = time.time()
    for _ in range(reps):
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(dev)
    us = (time.time() - t0) / reps * 1e6
    res = ttnn.to_torch(out).float()
    ttnn.release_trace(dev, tid)
    ttnn.deallocate(out)
    return us, res


def time_eager(dev, fn, reps=5):
    out = fn()
    ttnn.synchronize_device(dev)
    res = ttnn.to_torch(out).float()
    ttnn.deallocate(out)
    ts = []
    for _ in range(reps):
        t0 = time.time()
        out = fn()
        ttnn.synchronize_device(dev)
        ts.append((time.time() - t0) * 1e6)
        ttnn.deallocate(out)
    return min(ts), res


def err(res, ref, N):
    r = res.reshape(-1, res.shape[-1])[: ref.shape[0], :N].double()
    return float((r - ref).abs().max() / ref.abs().max()), float((r * ref).sum() / (ref * ref).sum())


def divisors(n, cands):
    return [c for c in cands if n % c == 0]


def sweep_role(dev, name, out, do_prefill):
    K, N, wdt, xdt, pdt = ROLES[name]
    wdt = OPTS["wdtype"] or wdt
    BASE, ALT = OPTS["base"] or HIFI4, OPTS["alt"] or HIFI2
    g = torch.Generator().manual_seed(hash(name) % 1000)
    w_host = torch.randn(K, N, generator=g) / K**0.5
    w = dev_t(dev, w_host, wdt)
    w_deq = ttnn.to_torch(w).double().reshape(K, N)
    grid = dev.compute_with_storage_grid_size()
    kt, nt = K // 32, N // 32
    rec = lambda **kw: out.write(json.dumps({"role": name, "K": K, "N": N, "wdtype": str(wdt), **kw}) + "\n") or print(
        "MM", json.dumps({"role": name, **kw})
    )

    # ---- decode rows
    x_host = torch.randn(1, 1, 32, K, generator=g)
    x = dev_t(dev, x_host, xdt)
    ref = x_host[0, 0].to(torch.bfloat16).double() @ w_deq if xdt == B16 else x_host[0, 0].double() @ w_deq
    best = {}

    def run(tag, fn, fam, fid):
        try:
            us, res = time_traced(dev, fn)
            e, sc = err(res, ref, N)
            sha = hashlib.sha256(res.reshape(-1, res.shape[-1])[:32, :N].contiguous().numpy().tobytes()).hexdigest()[
                :16
            ]
            rec(mode="decode", cand=tag, family=fam, fidelity=str(fid), us=us, rel_err=e, scale=sc, out_sha=sha)
            if fid == BASE and (fam not in best or us < best[fam][0]):
                best[fam] = (us, tag)
        except Exception as ex:  # noqa: BLE001 - record the refusal (L1 / config) and continue
            rec(mode="decode", cand=tag, family=fam, fidelity=str(fid), error=f"{type(ex).__name__}: {str(ex)[:300]}")

    run("default", lambda: ttnn.linear(x, w, compute_kernel_config=ckc(BASE, False)), "default", BASE)
    fns = {}
    for cores in sorted({c for c in (16, 32, 48, 64, 80, 96, 110) if c <= grid.x * grid.y}):
        pcn = -(-nt // cores)
        cores_used = -(-nt // pcn)
        for ib in divisors(kt, (4, 8, 16, 32, 64)):
            tag = f"1d_c{cores_used}_ib{ib}"
            if tag in fns:
                continue
            try:
                pc = tpc.create_matmul_1d_decode_progcfg(
                    32, K, N, cores_used, fp32_acc=True, grid_w=grid.x, in0_block_w=ib
                )
            except Exception as ex:  # noqa: BLE001
                rec(mode="decode", cand=tag, family="1d", error=str(ex)[:200])
                continue
            f = (
                lambda pc_: (
                    lambda fid: (lambda: ttnn.linear(x, w, compute_kernel_config=ckc(fid, True), program_config=pc_))
                )
            )(pc)
            fns[tag] = ("1d", f)
            run(tag, f(BASE), "1d", BASE)
    # DRAM-sharded: N padded to 8 banks x 32
    Np = tpc._roundup(N, 32 * tpc.DRAM_CORES)
    wp = torch.zeros(K, Np)
    wp[:, :N] = w_host
    w_ds = dev_t(dev, wp, wdt, tpc.create_dram_sharded_mem_config(K, Np))
    tiles_bank = Np // 32 // tpc.DRAM_CORES
    for ac in divisors(kt, (2, 4, 8, 16, 32)):
        shard = kt // ac
        for ib in sorted(
            {shard, shard // 2 if shard % 2 == 0 else shard, min(shard, 8) if shard % min(shard, 8) == 0 else shard}
        ):
            for r in [r for r in (1, 2, 3) if tiles_bank % r == 0]:
                tag = f"ds_a{ac}_ib{ib}_r{r}"
                if tag in fns:
                    continue
                try:
                    mm = DSMatmul(w_ds, act_cores=ac, in0_block_w=ib, readers=r, convert=False)
                except Exception as ex:  # noqa: BLE001
                    rec(mode="decode", cand=tag, family="ds", error=str(ex)[:200])
                    continue
                f = (lambda mm_: (lambda fid: (lambda: mm_(x, ckc(fid, False)))))(mm)
                fns[tag] = ("ds", f)
                run(tag, f(BASE), "ds", BASE)
    for fam, (us, tag) in best.items():  # fidelity trial on each family's best
        if tag in fns:
            run(tag, fns[tag][1](ALT), fam, ALT)
    ttnn.deallocate(w_ds)
    ttnn.deallocate(x)

    # ---- prefill rows
    if do_prefill and pdt is not None:
        xp_host = torch.randn(1, 1, 2048, K, generator=g)
        xp = dev_t(dev, xp_host, pdt)
        refp = (xp_host[0, 0].to(torch.bfloat16).double() if pdt == B16 else xp_host[0, 0].double()) @ w_deq

        def runp(tag, fn, fid):
            try:
                us, res = time_eager(dev, fn)
                e, sc = err(res, refp, N)
                rec(mode="prefill", cand=tag, fidelity=str(fid), us=us, rel_err=e, scale=sc)
            except Exception as ex:  # noqa: BLE001
                rec(mode="prefill", cand=tag, fidelity=str(fid), error=f"{type(ex).__name__}: {str(ex)[:300]}")

        runp("default", lambda: ttnn.linear(xp, w, compute_kernel_config=ckc(HIFI4, False)), HIFI4)
        for ib in [None] + divisors(kt, (2, 4, 8)):
            pc = prefill_pc(xp, K, N, ib)
            tag = f"2d_ib{pc.in0_block_w}"
            runp(
                tag,
                lambda pc_=pc: ttnn.linear(xp, w, compute_kernel_config=ckc(HIFI4, False), program_config=pc_),
                HIFI4,
            )
            runp(
                tag,
                lambda pc_=pc: ttnn.linear(xp, w, compute_kernel_config=ckc(HIFI2, False), program_config=pc_),
                HIFI2,
            )
        ttnn.deallocate(xp)
    ttnn.deallocate(w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--roles", default=",".join(ROLES))
    ap.add_argument("--no-prefill", action="store_true")
    ap.add_argument("--wdtype", choices=sorted(WDT), default=None, help="stage 8: weight dtype of every role")
    ap.add_argument("--fidelity", choices=sorted(FID), default=None, help="stage 8: base fidelity (default HiFi4)")
    ap.add_argument("--alt-fidelity", choices=sorted(FID), default=None, help="second trial (default HiFi2)")
    args = ap.parse_args()
    OPTS.update(wdtype=WDT.get(args.wdtype), base=FID.get(args.fidelity), alt=FID.get(args.alt_fidelity))
    out = open(args.out, "a")
    dev = ttnn.open_device(device_id=0, trace_region_size=64 << 20)
    try:
        for name in args.roles.split(","):
            sweep_role(dev, name, out, not args.no_prefill)
            out.flush()
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()

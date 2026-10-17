# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage-5 A/B of the multichip decode options (tt/multichip_decoder.MCOpts) on one real layer, one build.

The options are read at run time, so every variant runs on the same layer: set the options, eager step (compiles the
variant's programs and builds its weights), capture the decode trace, warm replay, then time `--replays` back-to-back
replays (device-bound step time) and `--replays` refreshed steps (host input copy + replay, as tests/test_perf_decoder).
Unit checks (--units) compare each option's output with the stage-4 path on the same input (norm, packed DSA
projections, DRAM-sharded router, fast_reduce_nc mix sums, ring vs linear all-gather).

  python -m models.autoports.zai_org_glm_5_3_flash.tools.mc_decode_ab --kind dsa_moe --batch 1 \
      --variants "base:;norm:norm_sharded=1;pack:dsa_pack=1" --out ab.jsonl
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time

os.environ.setdefault("GLM53_TP", "4")

import torch  # noqa: E402

import ttnn  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tests.harness import (  # noqa: E402
    H4,
    DecodeRunner,
    build_layer,
    close_mesh,
    open_mesh,
)
from models.autoports.zai_org_glm_5_3_flash.tt import multichip_decoder as mcd  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.ckpt import CheckpointReader  # noqa: E402

KINDS = {"kda_dense": 0, "dsa_moe": 3, "kda_moe": 4}


class Stage4Base(mcd.MultichipDecoder):
    """Built on the stage-4 options (MCOpts()) plus GLM53_MC, not on DEFAULT_MC: every variant toggles at run time, so
    the build must keep the separate weights (ds_shared off unless GLM53_MC asks for it)."""

    @classmethod
    def from_state_dict(cls, *a, **kw):
        kw.setdefault("mc", mcd.MCOpts())
        return mcd.MultichipDecoder.from_state_dict.__func__(cls, *a, **kw)


def pcc(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def set_opts(dec, spec: str):
    base = mcd.MCOpts()
    kw = {}
    for item in [i for i in spec.split(",") if i.strip()]:
        k, v = item.split("=")
        typ = type(getattr(base, k.strip()))
        kw[k.strip()] = (v.strip().lower() in ("1", "true", "on")) if typ is bool else typ(v.strip())
    new = dataclasses.replace(base, **kw)
    for f in dataclasses.fields(new):  # in place: the CCL, DSA and router hooks hold this object
        setattr(dec.mc, f.name, getattr(new, f.name))


def rep(mesh, t, dtype=ttnn.bfloat16):
    return ttnn.from_torch(
        t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh)
    )


def dev0(t):
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()


def units(dec, mesh, out, usable):
    """Each option vs the stage-4 path on one input (device 0 copy of replicated results)."""
    g = torch.Generator().manual_seed(7)
    res = {}
    # norm: [1,1,32,4096] bf16, the attention input norm weight
    h = torch.randn(1, 1, 32, 4096, generator=g) * 0.05
    a = mcd.OptimizedDecoder._norm(dec, rep(mesh, h), dec.input_norm_w)
    b = mcd.sharded_rms_norm(rep(mesh, h), dec.input_norm_w, dec.eps, dec.norm_ckc)
    ta, tb = dev0(a), dev0(b)
    res["norm"] = {"pcc": pcc(ta, tb), "max_abs": float((ta - tb).abs().max()), "bit_equal": bool(torch.equal(ta, tb))}
    # mix sums: fp32 partials [1,64,32,64]
    p = torch.randn(1, 64, 32, 64, generator=g) * 0.3
    pt = rep(mesh, p, ttnn.float32)
    ref = p.double().sum(dim=1, keepdim=True)
    s_sum = dev0(ttnn.sum(pt, dim=1, keepdim=True))
    s_nc = dev0(ttnn.experimental.fast_reduce_nc(pt, dims=[1], output=None, compute_kernel_config=dec.attn_hc.ckc))
    rel = lambda x: float(((x.double() - ref).abs().max() / ref.abs().max()))
    res["mix_sum"] = {"ttnn_sum_rel_err": rel(s_sum), "fast_reduce_nc_rel_err": rel(s_nc)}
    # ring vs linear all-gather (bf16 [1,1,32,1024] per device)
    if dec.ccl is not None:
        x = torch.randn(1, 1, 32, 4096, generator=g)
        xs = ttnn.from_torch(
            x,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1),
        )
        outs = {}
        for topo in ("linear", "ring"):
            if topo == "ring" and not str(usable).endswith("Ring"):
                continue
            dec.mc.topology = topo
            xi = ttnn.clone(xs)
            outs[topo] = dev0(dec.ccl.all_gather(xi, dim=3, dtype=ttnn.bfloat16, decode=True))
        dec.mc.topology = "linear"
        if len(outs) == 2:
            res["ag_ring_equal_linear"] = bool(torch.equal(outs["linear"], outs["ring"]))
    # DSA packed projections vs separate matmuls
    if dec.attn_kind == "dsa":
        at = dec.attn
        x = rep(mesh, torch.randn(1, 1, 32, 4096, generator=g) * 0.5)
        names = [
            ("w_q_a", {}),
            ("w_kv_a", {}),
            ("w_ik", {}),
            ("w_ig", {"dtype": ttnn.float32}),
            ("w_iw", {"dtype": ttnn.float32}),
        ]
        dec.mc.dsa_pack = False
        sep = {n: dev0(at._mm(x, getattr(at, n), **kw)) for n, kw in names}
        dec.mc.dsa_pack = True
        pk = {n: dev0(at._mm(x, getattr(at, n), **kw)) for n, kw in names}
        at.pack_flush()
        dec.mc.pack_b_1d = True
        pk1 = {n: dev0(at._mm(x, getattr(at, n), **kw)) for n, kw in names}
        at.pack_flush()
        dec.mc.pack_b_1d = False
        dec.mc.dsa_pack = False
        xr = dev0(x).double()[0, 0]
        res["dsa_pack"] = {}
        for n, _ in names:
            ref = xr @ dev0(getattr(at, n)).double().reshape(4096, -1)  # the device weight (dequantized), fp64
            e = lambda t: float((t.double()[0, 0] - ref).abs().max() / ref.abs().max())
            res["dsa_pack"][n] = {
                "pcc": pcc(sep[n], pk[n]),
                "max_abs": float((sep[n] - pk[n]).abs().max()),
                "sep_rel_err": e(sep[n]),
                "pack_rel_err": e(pk[n]),
                "pack_b1d_rel_err": e(pk1[n]),
            }
        # sharded q_a / kv_a norms
        for n, w, K in (("q_a_norm", at.q_a_norm, 1536), ("kv_a_norm", at.kv_a_norm, 512)):
            v = torch.randn(1, 1, 32, K, generator=g)
            a = dev0(mcd.GlmDSA._rms(at, rep(mesh, v), w))
            b = dev0(mcd.sharded_rms_norm(rep(mesh, v), w, at.eps, at.ckc))
            res[n] = {"pcc": pcc(a, b), "max_abs": float((a - b).abs().max())}
    if dec.mlp_kind == "moe":
        m = dec.mlp
        x = rep(mesh, torch.randn(1, 1, 32, 4096, generator=g) * 0.5)
        dec.mc.router_ds = False
        a = dev0(m._router_logits(x, ttnn.L1_MEMORY_CONFIG, ttnn.float32))
        dec.mc.router_ds = True
        b = dev0(m._router_logits(x, ttnn.L1_MEMORY_CONFIG, ttnn.float32))
        dec.mc.router_ds, dec.mc.router_1d = False, True
        c = dev0(m._router_logits(x, ttnn.L1_MEMORY_CONFIG, ttnn.float32))
        dec.mc.router_ds, dec.mc.router_1d = False, False
        ref = dev0(x).double()[0, 0] @ dev0(m.router_w).double().reshape(4096, -1)
        e = lambda t: float((t.double()[0, 0] - ref).abs().max() / ref.abs().max())
        res["router_ds"] = {
            "pcc": pcc(a, b),
            "max_abs": float((a - b).abs().max()),
            "shape": list(b.shape),
            "il_rel_err": e(a),
            "ds_rel_err": e(b),
            "il1d_rel_err": e(c),
        }
    print("UNITS", json.dumps(res))
    out.write(json.dumps({"units": res}) + "\n")
    out.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--variants", required=True, help="name:spec;name:spec (spec = MCOpts key=value list)")
    ap.add_argument("--replays", type=int, default=30)
    ap.add_argument("--units", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = open(args.out, "a")
    rd = CheckpointReader()
    mesh = open_mesh()
    try:
        t0 = time.time()
        dec = build_layer(
            mesh,
            rd,
            KINDS[args.kind],
            batch=args.batch,
            prefill_lengths=(128,),
            max_context=4096,
            decoder_cls=Stage4Base,
        )
        print(f"build {args.kind} batch {args.batch}: {time.time() - t0:.0f} s; links {dec.tt_ccl.get_num_links()}")
        set_opts(dec, "topology=auto")
        try:
            usable = dec.ccl.topology()
        except Exception as e:  # noqa: BLE001
            print(f"get_usable_topology failed: {e}")
            usable = "unknown"
        print(f"usable topology (auto): {usable}")
        out.write(
            json.dumps(
                {
                    "kind": args.kind,
                    "batch": args.batch,
                    "usable_topology": str(usable),
                    "links": dec.tt_ccl.get_num_links(),
                }
            )
            + "\n"
        )
        set_opts(dec, "")
        if args.units:
            units(dec, mesh, out, usable)
        B = args.batch
        rows = {s: torch.randn(H4, generator=torch.Generator().manual_seed(s)) * 0.02 for s in range(B)}
        for item in [v for v in args.variants.split(";") if v.strip()]:
            name, spec = item.split(":", 1)
            if "ring" in spec and not str(usable).endswith("Ring"):
                print(f"skip {name}: no ring on this fabric")
                continue
            set_opts(dec, spec)
            rec = {"kind": args.kind, "batch": B, "variant": name, "spec": spec}
            runner = DecodeRunner(dec, mesh, B)
            try:
                y = runner.eager_step(rows, [128] * B)
                rec["eager_finite"] = bool(torch.isfinite(y).all())
                runner.capture(rows, [129] * B)
                runner.traced_step(rows, [129] * B)
                ttnn.synchronize_device(mesh)
                t = time.time()
                for _ in range(args.replays):
                    ttnn.execute_trace(mesh, runner.trace_id, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh)
                rec["replay_us"] = (time.time() - t) / args.replays * 1e6
                t = time.time()
                for i in range(args.replays):
                    runner.traced_step(rows, [130 + i] * B, sync=False)
                ttnn.synchronize_device(mesh)
                rec["step_us"] = (time.time() - t) / args.replays * 1e6
                yt = runner.traced_step(rows, [130 + args.replays] * B)
                rec["traced_finite"] = bool(torch.isfinite(yt).all())
            except Exception as e:  # noqa: BLE001 - record and continue with the next variant
                rec["error"] = f"{type(e).__name__}: {str(e)[:400]}"
            finally:
                runner.release()
            print("AB", json.dumps(rec))
            out.write(json.dumps(rec) + "\n")
            out.flush()
    finally:
        close_mesh(mesh)


if __name__ == "__main__":
    main()

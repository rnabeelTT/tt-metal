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


_FID = {
    "LoFi": ttnn.MathFidelity.LoFi,
    "HiFi2": ttnn.MathFidelity.HiFi2,
    "HiFi3": ttnn.MathFidelity.HiFi3,
    "HiFi4": ttnn.MathFidelity.HiFi4,
}
# fid.<group>: the module compute config a fidelity trial replaces (attribute owner, attribute name)
_FID_GROUPS = {
    "attn": lambda d: (d.attn, "ckc"),  # KDA projections + o-norm / DSA projections + norms
    "shared": lambda d: (d.mlp, "shared_ckc"),
    "routed": lambda d: (d.mlp, "routed_ckc"),
    "mlp": lambda d: (d.mlp, "ckc"),  # dense MLP
}


def _apply_fid(dec, group, fid):
    owner, attr = _FID_GROUPS[group](dec)
    store = dec.__dict__.setdefault("_orig_ckc", {})
    key = (group, attr)
    if key not in store:
        store[key] = getattr(owner, attr)
    base = store[key]
    new = (
        base
        if fid is None
        else ttnn.WormholeComputeKernelConfig(
            math_fidelity=fid,
            math_approx_mode=base.math_approx_mode,
            fp32_dest_acc_en=base.fp32_dest_acc_en,
            packer_l1_acc=base.packer_l1_acc,
        )
    )
    setattr(owner, attr, new)
    if group == "routed" and hasattr(owner, "_cfg_grouped"):
        owner._cfg_grouped = None


def set_opts(dec, spec: str):
    """spec: MCOpts key=value list, plus env.NAME=value (process environment, read by the ops / modules at call time),
    moe.FIELD=value (the MoE OptimizationConfig; resets the cached sparse-matmul configs) and fid.GROUP=HiFi2|... (a
    module compute config: attn / shared / routed / mlp). Every key not named returns to its stage-4 / built value."""
    base = dec.__dict__.get("_mc_base") or mcd.MCOpts()
    kw, env, moe, fid = {}, {}, {}, {}
    for item in [i for i in spec.split(",") if i.strip()]:
        k, v = item.split("=")
        k, v = k.strip(), v.strip()
        if k.startswith("env."):
            env[k[4:]] = v
        elif k.startswith("moe."):
            moe[k[4:]] = v
        elif k.startswith("fid."):
            fid[k[4:]] = _FID[v]
        else:
            typ = type(getattr(base, k))
            kw[k] = (v.lower() in ("1", "true", "on")) if typ is bool else typ(v)
    if kw.get("moe_grouped_in0"):  # construction-time in MultichipDecoder: apply through the MoE config here
        moe.setdefault("grouped_gate_up_in0", str(kw["moe_grouped_in0"]))
        moe.setdefault("grouped_down_in0", str(kw["moe_grouped_in0"]))
    new = dataclasses.replace(base, **kw)
    for f in dataclasses.fields(new):  # in place: the CCL, DSA and router hooks hold this object
        setattr(dec.mc, f.name, getattr(new, f.name))
    env0 = dec.__dict__.setdefault("_env0", {})  # the construction's values (e.g. DEFAULT_MC's MHC_SINKHORN_*)
    for k in set(env) - set(env0):
        env0[k] = os.environ.get(k)
    for k in dec.__dict__.get("_env_keys", set()) - set(env):
        if env0.get(k) is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = env0[k]
    for k, v in env.items():
        os.environ[k] = v
    dec._env_keys = set(env)
    if dec.mlp_kind == "moe":
        m = dec.mlp
        if "_opt0" not in dec.__dict__:
            dec._opt0 = m.opt

        def conv(k, v):
            old = getattr(dec._opt0, k)
            if isinstance(old, bool):
                return v.lower() in ("1", "true", "on")
            return int(v) if old is None else type(old)(v)

        cur = dataclasses.replace(dec._opt0, **{k: conv(k, v) for k, v in moe.items()})
        if cur != m.opt:
            m.opt = cur
            m._cfg_grouped = None
    for g in _FID_GROUPS:
        if g in fid or (g, _FID_GROUPS[g](dec)[1]) in dec.__dict__.get("_orig_ckc", {}):
            try:
                _apply_fid(dec, g, fid.get(g))
            except AttributeError:
                pass


def reset_state(dec, B):
    """Fresh (zero) decode state of the layer (determinism runs replay the same sequence from the same state)."""
    a = dec.attn
    if dec.attn_kind == "dsa":
        for n in ("latent", "pooled", "page_table", "acc_m", "acc_den", "acc_num"):
            ttnn.deallocate(getattr(a, n))
        a.latent = None
    else:
        st = dec.decode_state
        ttnn.deallocate(st.recurrent)
        for c in st.conv:
            ttnn.deallocate(c)
    dec.allocate_decode_state(B)


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
        dec.mc.pack_a_1d = True
        pka = {n: dev0(at._mm(x, getattr(at, n), **kw)) for n, kw in names}
        at.pack_flush()
        dec.mc.pack_a_1d = dec.mc.pack_b_1d = False
        dec.mc.dsa_pack = False
        # dsa_1d: q_b (bfp8) / wq_b (bf16) on [32, 1536] inputs
        xq = rep(mesh, torch.randn(1, 1, 32, 1536, generator=g) * 0.5)
        xqr = dev0(xq).double()[0, 0]
        res["dsa_1d"] = {}
        for n in ("w_q_b", "w_iq"):
            wt = getattr(at, n)
            ref = xqr @ dev0(wt).double().reshape(1536, -1)
            e1 = lambda t, ref=ref: float((t.double()[0, 0] - ref).abs().max() / ref.abs().max())
            dec.mc.dsa_1d = False
            a1 = dev0(at._mm(xq, wt))
            dec.mc.dsa_1d = True
            b1 = dev0(at._mm(xq, wt))
            dec.mc.dsa_1d = False
            res["dsa_1d"][n] = {"default_rel_err": e1(a1), "1d_rel_err": e1(b1), "pcc": pcc(a1, b1)}
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
                "pack_a1d_b1d_rel_err": e(pka[n]),
            }
        # sharded q_a / kv_a norms
        for n, w, K in (("q_a_norm", at.q_a_norm, 1536), ("kv_a_norm", at.kv_a_norm, 512)):
            v = torch.randn(1, 1, 32, K, generator=g)
            a = dev0(mcd.GlmDSA._rms(at, rep(mesh, v), w))
            b = dev0(mcd.sharded_rms_norm(rep(mesh, v), w, at.eps, at.ckc))
            res[n] = {"pcc": pcc(a, b), "max_abs": float((a - b).abs().max())}
    if dec.attn_kind == "kda":  # kda_conv_mixed: bf16 history straight into addcmul vs the fp32 typecast path
        at = dec.attn
        st = dec.decode_state
        W = int(st.conv[0].shape[-1])
        Bp = int(st.conv[0].shape[-2])
        cur = rep(mesh, torch.randn(1, 1, Bp, W, generator=g))
        saved = [ttnn.clone(c) for c in st.conv]
        for c in st.conv:
            ttnn.copy(rep(mesh, torch.randn(1, 1, Bp, W, generator=g)), c)
        dec.mc.kda_conv_mixed = False
        a = dev0(at._conv_fold(cur, st))
        dec.mc.kda_conv_mixed = True
        b = dev0(at._conv_fold(cur, st))
        dec.mc.kda_conv_mixed = False
        for c, s_ in zip(st.conv, saved):
            ttnn.copy(s_, c)
        res["kda_conv_mixed"] = {"bit_equal": bool(torch.equal(a, b)), "max_abs": float((a - b).abs().max())}
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


def default_spec() -> str:
    """DEFAULT_MC as a set_opts spec on the stage-4 base (ds_shared is construction-time and stays off), with the
    sinkhorn fidelity as its environment variable."""
    base, d = mcd.MCOpts(), mcd.DEFAULT_MC
    items = []
    for f in dataclasses.fields(d):
        v = getattr(d, f.name)
        if v == getattr(base, f.name) or f.name in ("ds_shared", "sinkhorn_perm_fidelity", "sinkhorn_faces"):
            continue
        items.append(f"{f.name}={int(v) if isinstance(v, bool) else v}")
    if d.sinkhorn_perm_fidelity:
        items.append(f"env.MHC_SINKHORN_PERM_FIDELITY={d.sinkhorn_perm_fidelity}")
    if d.sinkhorn_faces:
        items.append("env.MHC_SINKHORN_FACES=1")
    return ",".join(items)


def run_variant(dec, mesh, B, rows, args, rec, pos0):
    runner = DecodeRunner(dec, mesh, B)
    try:
        y = runner.eager_step(rows, [pos0] * B)
        rec["eager_finite"] = bool(torch.isfinite(y).all())
        runner.capture(rows, [pos0 + 1] * B)
        runner.traced_step(rows, [pos0 + 1] * B)
        ttnn.synchronize_device(mesh)
        t = time.time()
        for _ in range(args.replays):
            ttnn.execute_trace(mesh, runner.trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        rec["replay_us"] = (time.time() - t) / args.replays * 1e6
        t = time.time()
        for i in range(args.replays):
            runner.traced_step(rows, [pos0 + 2 + i] * B, sync=False)
        ttnn.synchronize_device(mesh)
        rec["step_us"] = (time.time() - t) / args.replays * 1e6
        yt = runner.traced_step(rows, [pos0 + 2 + args.replays] * B)
        rec["traced_finite"] = bool(torch.isfinite(yt).all())
    except Exception as e:  # noqa: BLE001 - record and continue with the next variant
        rec["error"] = f"{type(e).__name__}: {str(e)[:400]}"
    finally:
        runner.release()


def sequence(dec, mesh, B, rows, n, pos0):
    """One eager step and n traced steps from a fresh state: the outputs (every variant on the same inputs)."""
    reset_state(dec, B)
    runner = DecodeRunner(dec, mesh, B)
    try:
        ys = [runner.eager_step(rows, [pos0] * B)]
        runner.capture(rows, [pos0 + 1] * B)
        ys += [runner.traced_step(rows, [pos0 + 1 + i] * B) for i in range(n)]
    finally:
        runner.release()
    return ys


def compare(ref, got):
    """Outputs of a variant vs the first variant's: bit identity, worst relative error and PCC over all steps."""
    a, b = torch.stack([r.double() for r in ref]), torch.stack([g.double() for g in got])
    fin = bool(torch.isfinite(b).all())
    d = (a - b).abs()
    am, bm = a - a.mean(), b - b.mean()
    return {
        "steps": len(ref),
        "bit_identical": all(torch.equal(x, y) for x, y in zip(ref, got)),
        "max_rel": float(d.max() / a.abs().max()) if fin else None,
        "pcc": float((am * bm).sum() / (am.norm() * bm.norm())) if fin else None,
        "rms_ratio": float(b.norm() / a.norm()) if fin else None,
        "finite": fin,
    }


def determinism(dec, mesh, B, rows, n, pos0):
    """Two runs of n traced replays from a fresh state with the same inputs: every replay's output bit-identical."""
    seqs = []
    for _ in range(2):
        reset_state(dec, B)
        runner = DecodeRunner(dec, mesh, B)
        try:
            runner.eager_step(rows, [pos0] * B)
            runner.capture(rows, [pos0 + 1] * B)
            seqs.append([runner.traced_step(rows, [pos0 + 1 + i] * B) for i in range(n)])
        finally:
            runner.release()
    diff = [i for i in range(n) if not torch.equal(seqs[0][i], seqs[1][i])]
    finite = all(bool(torch.isfinite(t).all()) for t in seqs[0])
    return {"replays": n, "bit_identical": not diff, "first_diff": diff[:5], "finite": finite}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--variants", required=True, help="name:spec;name:spec (spec = MCOpts key=value list)")
    ap.add_argument("--replays", type=int, default=30)
    ap.add_argument("--units", action="store_true")
    ap.add_argument("--determinism", type=int, default=0, help="per variant: 2 x N replays from a fresh state")
    ap.add_argument("--max-context", type=int, default=4096)
    ap.add_argument("--pos", type=int, default=128, help="decode position of the timed steps")
    ap.add_argument(
        "--base",
        default="stage4",
        choices=("stage4", "default"),
        help="build and variant base: stage-4 options (run-time toggles) or DEFAULT_MC (the shipped build)",
    )
    ap.add_argument("--compare", type=int, default=0, help="per variant: n traced steps vs the first variant's")
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
            max_context=args.max_context,
            decoder_cls=Stage4Base if args.base == "stage4" else mcd.MultichipDecoder,
        )
        if args.base == "default":
            dec._mc_base = dataclasses.replace(dec.mc)
        print(f"build {args.kind} batch {args.batch}: {time.time() - t0:.0f} s; links {dec.tt_ccl.get_num_links()}")
        set_opts(dec, "topology=auto" if args.base == "stage4" else "")
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
                    "max_context": args.max_context,
                    "pos": args.pos,
                }
            )
            + "\n"
        )
        set_opts(dec, "")
        if args.units:
            units(dec, mesh, out, usable)
        B = args.batch
        rows = {s: torch.randn(H4, generator=torch.Generator().manual_seed(s)) * 0.02 for s in range(B)}
        ref_seq = None
        for item in [v for v in args.variants.split(";") if v.strip()]:
            name, spec = item.split(":", 1) if ":" in item else (item, "")  # bash read strips a trailing ":"
            if name == "fin" and not spec and args.base == "stage4":  # "fin:" = DEFAULT_MC as run-time options
                spec = default_spec()
            if "ring" in spec and not str(usable).endswith("Ring"):
                print(f"skip {name}: no ring on this fabric")
                continue
            set_opts(dec, spec)
            rec = {
                "kind": args.kind,
                "batch": B,
                "variant": name,
                "spec": spec,
                "pos": args.pos,
                "max_context": args.max_context,
            }
            run_variant(dec, mesh, B, rows, args, rec, args.pos)
            if args.compare and "error" not in rec:
                try:
                    seq = sequence(dec, mesh, B, rows, args.compare, args.pos)
                    if ref_seq is None:
                        ref_seq, rec["compare_ref"] = seq, True
                    else:
                        rec["vs_first"] = compare(ref_seq, seq)
                except Exception as e:  # noqa: BLE001
                    rec["vs_first"] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
            if args.determinism and "error" not in rec:
                try:
                    rec["determinism"] = determinism(dec, mesh, B, rows, args.determinism, args.pos)
                except Exception as e:  # noqa: BLE001
                    rec["determinism"] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
            print("AB", json.dumps(rec))
            out.write(json.dumps(rec) + "\n")
            out.flush()
    finally:
        close_mesh(mesh)


if __name__ == "__main__":
    main()

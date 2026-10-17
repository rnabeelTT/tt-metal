# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage-2 review #1 probe: the graph-fusing candidates the review listed, current graph vs candidate, one die.

Sections (synthetic inputs at the decoder's real shapes; host-timed, synchronized, n calls, us per call):
  heads  DSA head split  [1,1,rows,H*D] -> [1,H,rows,D] (reshape + permute vs nlp_create_qkv_heads, num_kv_heads 0)
         DSA head concat [1,H,rows,D] -> [1,1,rows,H*D] (permute + reshape vs nlp_concat_heads)
         H = 64 (one die) and 16 (TP4), D = 256, rows 1 / 32 / 2048; indexer split H = 32, D = 128, rows 2048.
         Pure data movement: the check is bit-exactness (torch.equal) against the current graph.
  mlp    dense SwiGLU MLP (4096 -> 12288 -> 4096, bfp8, HiFi4): two linears + minimum / silu / clamp / multiply vs
         one multiply with the clamps and SiLU as fused input activations, vs a packed [gate | up] linear.
  moe    routed-expert activation chain on the static dispatch buffer [1,G,32,2*Nl] (decode 32 users: G 195; prefill
         2048: G 699): minimum + tensor maximum + swiglu + routing-weight multiply vs two slices + one multiply with
         fused activations + the routing-weight multiply.
  kdanorm KDA decode gated RMSNorm: rms_norm + reshape(gate) + typecast + sigmoid + multiply + reshape + typecast vs
         rms_norm + reshape + multiply(sigmoid fused) vs permute + kda.sigmoid_gated_rms_norm (B % 32 == 0 only).
  kdaconv KDA decode short conv (fp32): 4 multiply + 3 add + silu vs multiply + 3 addcmul + silu.

Usage: python -m models.autoports.zai_org_glm_5_3_flash.tools.fusing_probe [section ...]
"""

import sys
import time

import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tests.harness import open_mesh, pcc

DRAM = ttnn.DRAM_MEMORY_CONFIG
L = 10.0  # swiglu_limit


def _time(mesh, fn, n):
    fn()
    ttnn.synchronize_device(mesh)
    t = time.perf_counter()
    for _ in range(n):
        o = fn()
    ttnn.synchronize_device(mesh)
    del o
    return (time.perf_counter() - t) / n * 1e6


def _run(mesh, name, variants, n, ref=None, exact_vs=None):
    """variants: {label: fn}. Prints us per call, PCC vs ref (torch) and bit-exactness vs variant `exact_vs`."""
    outs = {}
    for label, fn in variants.items():
        try:
            out = ttnn.to_torch(fn()).float()
            us = _time(mesh, fn, n)
        except Exception as e:  # noqa: BLE001
            print(f"{name:34} {label:22}: error {str(e).splitlines()[0][:140]}", flush=True)
            continue
        outs[label] = out
        msg = f"{name:34} {label:22}: {us:9.1f} us"
        if ref is not None and out.numel() == ref.numel():
            msg += f"  PCC vs torch {pcc(out, ref):.7f}"
        if exact_vs is not None and exact_vs in outs and label != exact_vs:
            a = outs[exact_vs]
            same = a.shape == out.shape and torch.equal(a, out)
            msg += f"  bit-exact vs {exact_vs}: {same}"
            if not same and a.shape == out.shape:
                msg += f" (PCC {pcc(out, a):.7f}, max|d| {(out - a).abs().max().item():.3e})"
            elif not same:
                msg += f" (shape {tuple(out.shape)} vs {tuple(a.shape)})"
        print(msg, flush=True)
    return outs


def heads(mesh, g):
    for H, D, rows_list in ((64, 256, (1, 32, 2048)), (16, 256, (1, 32, 2048)), (32, 128, (2048,))):
        for rows in rows_list:
            n = 10 if rows >= 2048 else 50
            q = torch.randn(1, 1, rows, H * D, generator=g).bfloat16()
            tq = ttnn.from_torch(q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=DRAM)
            ref = q.float().reshape(1, rows, H, D).permute(0, 2, 1, 3)
            _run(
                mesh,
                f"split  H{H} D{D} rows {rows}",
                {
                    "reshape+permute": lambda: ttnn.permute(
                        ttnn.reshape(tq, (1, rows, H, D)), (0, 2, 1, 3), memory_config=DRAM
                    ),
                    "nlp_create_qkv_heads": lambda: ttnn.experimental.nlp_create_qkv_heads(
                        tq, num_heads=H, num_kv_heads=0, transpose_k_heads=False, memory_config=DRAM
                    )[0],
                },
                n,
                ref=ref,
                exact_vs="reshape+permute",
            )
            ttnn.deallocate(tq)
            if D != 256:
                continue
            o = torch.randn(1, H, rows, D, generator=g).bfloat16()
            to = ttnn.from_torch(o, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=DRAM)
            ref = o.float().permute(0, 2, 1, 3).reshape(1, 1, rows, H * D)
            _run(
                mesh,
                f"concat H{H} D{D} rows {rows}",
                {
                    "permute+reshape": lambda: ttnn.reshape(
                        ttnn.permute(to, (0, 2, 1, 3), memory_config=DRAM), (1, 1, rows, H * D)
                    ),
                    "nlp_concat_heads": lambda: ttnn.experimental.nlp_concat_heads(to, memory_config=DRAM),
                },
                n,
                ref=ref,
                exact_vs="permute+reshape",
            )
            ttnn.deallocate(to)


def _concat_grouped(t, rows, H, D, groups):
    """nlp_concat_heads per group of H / groups heads (its input CB holds every head of a row block: 64 x 256 needs
    2.2 MB > 1.5 MB L1), then the group outputs side by side."""
    x = ttnn.experimental.nlp_concat_heads(ttnn.reshape(t, (groups, H // groups, rows, D)), memory_config=DRAM)
    parts = [ttnn.slice(x, (i, 0, 0, 0), (i + 1, 1, rows, H // groups * D), memory_config=DRAM) for i in range(groups)]
    return ttnn.concat(parts, dim=-1, memory_config=DRAM)


def heads2(mesh, g):
    for H, rows in ((64, 64), (64, 128), (64, 2048), (16, 64), (16, 128)):
        D = 256
        n = 10 if rows >= 2048 else 50
        q = torch.randn(1, 1, rows, H * D, generator=g).bfloat16()
        tq = ttnn.from_torch(q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=DRAM)
        _run(
            mesh,
            f"split  H{H} D{D} rows {rows}",
            {
                "reshape+permute": lambda: ttnn.permute(
                    ttnn.reshape(tq, (1, rows, H, D)), (0, 2, 1, 3), memory_config=DRAM
                ),
                "nlp_create_qkv_heads": lambda: ttnn.experimental.nlp_create_qkv_heads(
                    tq, num_heads=H, num_kv_heads=0, transpose_k_heads=False, memory_config=DRAM
                )[0],
            },
            n,
            exact_vs="reshape+permute",
        )
        ttnn.deallocate(tq)
        o = torch.randn(1, H, rows, D, generator=g).bfloat16()
        to = ttnn.from_torch(o, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=DRAM)
        variants = {
            "permute+reshape": lambda: ttnn.reshape(
                ttnn.permute(to, (0, 2, 1, 3), memory_config=DRAM), (1, 1, rows, H * D)
            ),
        }
        if H <= 16:
            variants["nlp_concat_heads"] = lambda: ttnn.experimental.nlp_concat_heads(to, memory_config=DRAM)
        else:
            variants["concat_heads x2 groups"] = lambda: _concat_grouped(to, rows, H, D, 2)
            variants["concat_heads x4 groups"] = lambda: _concat_grouped(to, rows, H, D, 4)
        _run(mesh, f"concat H{H} D{D} rows {rows}", variants, n, exact_vs="permute+reshape")
        ttnn.deallocate(to)


def _uwp(op, *p):
    return ttnn.UnaryWithParam(op, *p) if len(p) != 1 else ttnn.UnaryWithParam((op, p[0]))


def mlp(mesh, g):
    hid, inter = 4096, 12288
    ckc = ttnn.init_device_compute_kernel_config(
        mesh.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    wg = torch.randn(hid, inter, generator=g) * (3.0 / hid**0.5)
    wu = torch.randn(hid, inter, generator=g) * (3.0 / hid**0.5)
    up_ = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=DRAM)
    twg, twu, twgu = up_(wg), up_(wu), up_(torch.cat([wg, wu], dim=1))
    a_act = [_uwp(ttnn.UnaryOpType.HARDTANH, -L, L)]
    b_act = [_uwp(ttnn.UnaryOpType.MINIMUM, L), ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU)]
    lin = lambda x, w: ttnn.linear(x, w, compute_kernel_config=ckc, memory_config=DRAM)
    for rows in (32, 2048):
        n = 10 if rows >= 2048 else 50
        x = torch.randn(1, 1, rows, hid, generator=g).bfloat16()
        tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=DRAM)
        gw, uw = ttnn.to_torch(twg).float(), ttnn.to_torch(twu).float()
        gr, ur = x.float() @ gw, x.float() @ uw
        ref = torch.nn.functional.silu(gr.clamp(max=L)) * ur.clamp(-L, L)

        def cur():
            gg = ttnn.silu(ttnn.minimum(lin(tx, twg), L), memory_config=DRAM)
            uu = ttnn.clamp(lin(tx, twu), min=-L, max=L, memory_config=DRAM)
            return ttnn.multiply(gg, uu, memory_config=DRAM)

        def fused_act():
            return ttnn.multiply(
                lin(tx, twu),
                lin(tx, twg),
                input_tensor_a_activations=a_act,
                input_tensor_b_activations=b_act,
                memory_config=DRAM,
            )

        def packed():
            gu = lin(tx, twgu)
            gg = ttnn.slice(gu, (0, 0, 0, 0), (1, 1, rows, inter), memory_config=DRAM)
            uu = ttnn.slice(gu, (0, 0, 0, inter), (1, 1, rows, 2 * inter), memory_config=DRAM)
            return ttnn.multiply(
                uu, gg, input_tensor_a_activations=a_act, input_tensor_b_activations=b_act, memory_config=DRAM
            )

        _run(
            mesh,
            f"mlp rows {rows}",
            {
                "2 linear (alone)": lambda: (lin(tx, twg), lin(tx, twu))[1],
                "packed linear (alone)": lambda: lin(tx, twgu),
                "current": cur,
                "fused-act multiply": fused_act,
                "packed + fused-act": packed,
            },
            n,
            ref=ref,
            exact_vs="current",
        )


def moe(mesh, g):
    Nl = 2048
    lo = torch.cat([torch.full((Nl,), -L), torch.full((Nl,), -3.0e38)]).reshape(1, 1, 1, 2 * Nl)
    t_lo = ttnn.from_torch(lo, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
    a_act = [_uwp(ttnn.UnaryOpType.HARDTANH, -L, L)]
    b_act = [_uwp(ttnn.UnaryOpType.MINIMUM, L), ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU)]
    for label, G in (("decode 32 users", 195), ("prefill 2048", 699)):
        n = 10 if G > 300 else 30
        Rd = G * 32
        f = (torch.randn(1, G, 32, 2 * Nl, generator=g) * 4).bfloat16()
        w = torch.rand(1, 1, Rd, 1, generator=g).bfloat16()
        tf = ttnn.from_torch(f, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=DRAM)
        tw = ttnn.from_torch(w, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=DRAM)
        ff = f.float()
        ref = (ff[..., :Nl].clamp(-L, L) * torch.nn.functional.silu(ff[..., Nl:].clamp(max=L))).reshape(
            1, 1, Rd, Nl
        ) * w.float()

        def cur():
            c = ttnn.maximum(ttnn.minimum(tf, L), t_lo)
            s = ttnn.swiglu(c, memory_config=DRAM)
            return ttnn.multiply(ttnn.reshape(s, (1, 1, Rd, Nl)), tw, memory_config=DRAM)

        def fused_act():
            up = ttnn.slice(tf, (0, 0, 0, 0), (1, G, 32, Nl), memory_config=DRAM)
            gate = ttnn.slice(tf, (0, 0, 0, Nl), (1, G, 32, 2 * Nl), memory_config=DRAM)
            s = ttnn.multiply(
                up, gate, input_tensor_a_activations=a_act, input_tensor_b_activations=b_act, memory_config=DRAM
            )
            return ttnn.multiply(ttnn.reshape(s, (1, 1, Rd, Nl)), tw, memory_config=DRAM)

        _run(
            mesh,
            f"moe act {label} G{G}",
            {"current": cur, "slices + fused-act": fused_act},
            n,
            ref=ref,
            exact_vs="current",
        )
        ttnn.deallocate(tf)


def kdanorm(mesh, g):
    H, V, P, off = 64, 128, 32960, 24576 + 128  # ogate column offset in the fused projection (W + K, tile aligned)
    eps = 1e-5
    ckc = ttnn.init_device_compute_kernel_config(
        mesh.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    wv = (1 + 0.1 * torch.randn(V, generator=g)).bfloat16()
    w_row = ttnn.from_torch(wv.float().reshape(1, 1, 1, V), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh)
    w_vec = ttnn.from_torch(wv.reshape(V), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
    sig = [ttnn.UnaryWithParam(ttnn.UnaryOpType.SIGMOID)]
    for B in (1, 32):
        o = torch.randn(1, 1, B * H, V, generator=g)
        proj = torch.randn(1, 1, B, P, generator=g).bfloat16()
        to = ttnn.from_torch(o, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=DRAM)
        tp = ttnn.from_torch(proj, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=DRAM)
        og = ttnn.slice(tp, (0, 0, 0, off), (1, 1, B, off + H * V), memory_config=DRAM)
        ref = (o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + eps) * wv.float()).reshape(1, 1, B, H * V)
        ref = ref * torch.sigmoid(proj.float()[..., off : off + H * V])
        rms = lambda: ttnn.rms_norm(to, epsilon=eps, weight=w_row, memory_config=DRAM, compute_kernel_config=ckc)

        def cur():
            x = rms()
            s = ttnn.sigmoid(ttnn.typecast(ttnn.reshape(og, (1, 1, B * H, V)), ttnn.float32), memory_config=DRAM)
            x = ttnn.multiply(x, s, memory_config=DRAM)
            return ttnn.typecast(ttnn.reshape(x, (1, 1, B, H * V)), ttnn.bfloat16)

        def gate_fused():
            x = ttnn.reshape(rms(), (1, 1, B, H * V))
            return ttnn.multiply(x, og, input_tensor_b_activations=sig, dtype=ttnn.bfloat16, memory_config=DRAM)

        def op():
            x = ttnn.permute(ttnn.reshape(to, (B, H, 1, V)), (1, 2, 0, 3), memory_config=DRAM)  # [H,1,B,V]
            x = ttnn.reshape(x, (H, B, V))
            return ttnn.experimental.kda.sigmoid_gated_rms_norm(
                x,
                ttnn.reshape(tp, (1, B, P)),
                w_vec,
                H,
                epsilon=eps,
                memory_config=DRAM,
                compute_kernel_config=ckc,
                output_dtype=ttnn.bfloat16,
                gate_column_offset=off,
            )

        _run(
            mesh,
            f"kda gated norm B{B}",
            {"current": cur, "gate fused in multiply": gate_fused, "permute + gated op": op},
            50,
            ref=ref,
            exact_vs="current",
        )


def kdaconv(mesh, g):
    W = 64 * (128 + 128 + 128)
    for B in (1, 32):
        cur_t = torch.randn(1, 1, B, W, generator=g).bfloat16()
        hist = [torch.randn(1, 1, B, W, generator=g).bfloat16() for _ in range(3)]
        taps = [(0.3 * torch.randn(1, 1, 1, W, generator=g)).bfloat16() for _ in range(4)]
        dev = lambda t: ttnn.from_torch(
            t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=DRAM
        )
        tc, th, tt_ = dev(cur_t), [dev(h) for h in hist], [dev(t) for t in taps]
        acc = cur_t.float() * taps[3].float() + sum(h.float() * t.float() for h, t in zip(hist, taps[:3]))
        ref = torch.nn.functional.silu(acc)

        def cur():
            a = ttnn.multiply(tc, tt_[3], dtype=ttnn.float32, memory_config=DRAM)
            for j in range(3):
                a = ttnn.add(
                    a, ttnn.multiply(th[j], tt_[j], dtype=ttnn.float32, memory_config=DRAM), memory_config=DRAM
                )
            return ttnn.silu(a, memory_config=DRAM)

        def addcmul():
            a = ttnn.multiply(tc, tt_[3], dtype=ttnn.float32, memory_config=DRAM)
            for j in range(3):
                a = ttnn.addcmul(a, th[j], tt_[j], memory_config=DRAM)
            return ttnn.silu(a, memory_config=DRAM)

        def addcmul_f32():
            a = ttnn.multiply(tc, tt_[3], dtype=ttnn.float32, memory_config=DRAM)
            for j in range(3):
                a = ttnn.addcmul(a, ttnn.typecast(th[j], ttnn.float32), ttnn.typecast(tt_[j], ttnn.float32))
            return ttnn.silu(a, memory_config=DRAM)

        _run(
            mesh,
            f"kda conv B{B}",
            {"current": cur, "addcmul (bf16 b, c)": addcmul, "addcmul (fp32 b, c)": addcmul_f32},
            50,
            ref=ref,
            exact_vs="current",
        )


def kda2(mesh, g):
    """Adapted retries: gated norm with the gate kept in [B, H*V] order; conv with fp32 tap constants; in-place
    history shift (where into the history buffer, no copies); main's packed-decode qkv_causal_conv1d_silu."""
    H, V, P, off, eps = 64, 128, 32960, 24576 + 128, 1e-5
    W = 64 * 384
    ckc = ttnn.init_device_compute_kernel_config(
        mesh.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    wv = (1 + 0.1 * torch.randn(V, generator=g)).bfloat16()
    w_row = ttnn.from_torch(wv.float().reshape(1, 1, 1, V), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh)
    sig = [ttnn.UnaryWithParam(ttnn.UnaryOpType.SIGMOID)]
    dev = lambda t, dt=ttnn.bfloat16: ttnn.from_torch(
        t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=DRAM
    )
    for B in (1, 32):
        o = torch.randn(1, 1, B * H, V, generator=g)
        proj = torch.randn(1, 1, B, P, generator=g).bfloat16()
        to, tp = dev(o, ttnn.float32), dev(proj)
        og = ttnn.slice(tp, (0, 0, 0, off), (1, 1, B, off + H * V), memory_config=DRAM)
        ref = (o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + eps) * wv.float()).reshape(1, 1, B, H * V)
        ref = ref * torch.sigmoid(proj.float()[..., off : off + H * V])
        rms = lambda: ttnn.rms_norm(to, epsilon=eps, weight=w_row, memory_config=DRAM, compute_kernel_config=ckc)

        def cur():
            x = rms()
            s_ = ttnn.sigmoid(ttnn.typecast(ttnn.reshape(og, (1, 1, B * H, V)), ttnn.float32), memory_config=DRAM)
            x = ttnn.multiply(x, s_, memory_config=DRAM)
            return ttnn.typecast(ttnn.reshape(x, (1, 1, B, H * V)), ttnn.bfloat16)

        def gate_rows():
            x = ttnn.reshape(rms(), (1, 1, B, H * V))
            s_ = ttnn.sigmoid(ttnn.typecast(og, ttnn.float32), memory_config=DRAM)
            return ttnn.typecast(ttnn.multiply(x, s_, memory_config=DRAM), ttnn.bfloat16)

        def gate_rows_act():
            x = ttnn.reshape(rms(), (1, 1, B, H * V))
            s_ = ttnn.typecast(og, ttnn.float32)
            return ttnn.typecast(
                ttnn.multiply(x, s_, input_tensor_b_activations=sig, memory_config=DRAM), ttnn.bfloat16
            )

        _run(
            mesh,
            f"kda gated norm B{B}",
            {"current": cur, "gate in [B,H*V]": gate_rows, "gate [B,H*V] sigmoid-act": gate_rows_act},
            50,
            ref=ref,
            exact_vs="current",
        )

        cur_t = torch.randn(1, 1, B, W, generator=g).bfloat16()
        hist = [torch.randn(1, 1, B, W, generator=g).bfloat16() for _ in range(3)]
        taps = [(0.3 * torch.randn(1, 1, 1, W, generator=g)).bfloat16() for _ in range(4)]
        tc, th, tt_ = dev(cur_t), [dev(h) for h in hist], [dev(t) for t in taps]
        t32 = [dev(t.float(), ttnn.float32) for t in taps]

        def conv_cur():
            a = ttnn.multiply(tc, tt_[3], dtype=ttnn.float32, memory_config=DRAM)
            for j in range(3):
                a = ttnn.add(
                    a, ttnn.multiply(th[j], tt_[j], dtype=ttnn.float32, memory_config=DRAM), memory_config=DRAM
                )
            return ttnn.silu(a, memory_config=DRAM)

        def conv_addcmul32():
            a = ttnn.multiply(tc, t32[3], dtype=ttnn.float32, memory_config=DRAM)
            for j in range(3):
                a = ttnn.addcmul(a, ttnn.typecast(th[j], ttnn.float32), t32[j])
            return ttnn.silu(a, memory_config=DRAM)

        _run(
            mesh,
            f"kda conv B{B}",
            {"current": conv_cur, "addcmul (fp32 tap consts)": conv_addcmul32},
            50,
            exact_vs="current",
        )

        # history shift: current (3 where + 3 copy) vs where straight into the history buffers
        act = dev((torch.rand(1, 1, B, 1, generator=g) > 0.3).float())
        bufs = [dev(h) for h in hist]

        def shift_cur():
            new = [
                ttnn.where(act, bufs[1], bufs[0], memory_config=DRAM),
                ttnn.where(act, bufs[2], bufs[1], memory_config=DRAM),
                ttnn.where(act, tc, bufs[2], memory_config=DRAM),
            ]
            for n_, b_ in zip(new, bufs):
                ttnn.copy(n_, b_)
            return bufs[0]

        def shift_inplace():
            ttnn.where(act, bufs[1], bufs[0], output_tensor=bufs[0])
            ttnn.where(act, bufs[2], bufs[1], output_tensor=bufs[1])
            ttnn.where(act, tc, bufs[2], output_tensor=bufs[2])
            return bufs[0]

        for label, fn in (("current", shift_cur), ("where in place", shift_inplace)):
            for b_, h in zip(bufs, hist):
                ttnn.copy_host_to_device_tensor(ttnn.from_torch(h, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT), b_)
            try:
                fn()
                got = [ttnn.to_torch(b_).float() for b_ in bufs]
                us = _time(mesh, fn, 50)
            except Exception as e:  # noqa: BLE001
                print(f"kda shift B{B} {label:22}: error {str(e).splitlines()[0][:140]}", flush=True)
                continue
            a_ = ttnn.to_torch(act).float().reshape(1, 1, B, 1) > 0
            exp = [
                torch.where(a_, hist[1], hist[0]),
                torch.where(a_, hist[2], hist[1]),
                torch.where(a_, cur_t, hist[2]),
            ]
            ok = all(torch.equal(x.float(), y.float()) for x, y in zip(got, exp))
            print(f"kda shift B{B} {label:22}: {us:9.1f} us  one step exact: {ok}", flush=True)

        # main's packed decode conv (models/demos/qwen38_27b_qb2/tt/decode_conv.py), B % 8 == 0 only; the op alone, with
        # its own row-major history [B, 3, C] (the port keeps three TILE [1,1,B,W] rows) and bf16 outputs
        if B % 8:
            print(f"kda packed conv B{B}: not applicable (B % 8 != 0)", flush=True)
            continue
        q_w = k_w = v_w = 64 * 128
        rm = lambda t: ttnn.from_torch(
            t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, memory_config=DRAM
        )
        h_rm = rm(torch.stack([h[0, 0] for h in hist], dim=1))  # [B, 3, W]
        x_rm = rm(cur_t.reshape(B, 1, W))
        tap_list = [
            ttnn.from_torch(t.reshape(1, 1, W), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh) for t in taps
        ]
        start = ttnn.from_torch(
            torch.zeros(1, dtype=torch.int64), device=mesh, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
        )

        def packed():
            joined = ttnn.reshape(ttnn.concat([h_rm, x_rm], dim=1), [1, B * 4, W])
            outs = ttnn.experimental.kda.qkv_causal_conv1d_silu(
                joined,
                h_rm[:1],
                *tap_list,
                q_w,
                k_w,
                v_w,
                program_config=ttnn.QkvCausalConv1dSiluProgramConfig(channel_chunk_size=256),
                actual_start=start,
                predecessor_carry=h_rm[:1],
            )
            res = []
            for out_, w_ in zip(outs, (q_w, k_w, v_w)):
                out_ = ttnn.reshape(ttnn.to_layout(out_, ttnn.ROW_MAJOR_LAYOUT), [B, 4, w_])[:, 3:4, :]
                res.append(ttnn.to_layout(out_, ttnn.TILE_LAYOUT))
            return res[0]

        try:
            packed()
            print(f"kda packed conv B{B} (op + main's adapters): {_time(mesh, packed, 50):9.1f} us", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"kda packed conv B{B}: error {str(e).splitlines()[0][:160]}", flush=True)


def _ckc(mesh):
    return ttnn.init_device_compute_kernel_config(
        mesh.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )


def _err(out, ref):
    """signed mean and max |.| of the relative error against a float64 reference."""
    d = (out.double() - ref.double()) / ref.double().abs().clamp_min(1e-30)
    return f"signed mean rel err {d.mean().item():+.2e}, max |rel err| {d.abs().max().item():.2e}"


def finalize(mesh, g):
    """Review #2 P1-a: the mhc_mix finalize (sum over the K chunks, Gram diagonals, rsqrt scale) on the op's partials
    [1, C, T, 64] fp32: current graph vs moreh_sum over the chunks vs the rsqrt chain as input activations of the final
    multiply."""
    from models.autoports.zai_org_glm_5_3_flash.tt.fused_decoder import mix_chunks

    K, eps = 16384, 1e-6
    ckc = _ckc(mesh)
    for T in (1, 32, 2048):
        Tp = -(-T // 32) * 32
        C = mix_chunks(Tp, K // 32)
        part = torch.randn(1, C, Tp, 64, generator=g, dtype=torch.float64) * 0.3
        part[..., 32:] = part[..., 32:].abs() * 40  # Gram partials: positive diagonal-scale magnitudes
        part = part[:, :, :T].float()
        mask = torch.zeros(1, 1, T, 32)
        mask[0, 0, torch.arange(T), torch.arange(T) % 32] = 1.0
        mask64 = torch.cat([torch.zeros(1, 1, T, 32), mask], dim=-1)
        dev = lambda t: ttnn.from_torch(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=DRAM)
        tp_, tm, tm64 = dev(part), dev(mask), dev(mask64)
        tot = part.double().sum(1, keepdim=True)
        ss = (tot[..., 32:] * mask.double()).sum(-1, keepdim=True)
        ref = tot[..., :32] * torch.rsqrt(ss / K + eps)
        act = [
            ttnn.UnaryWithParam((ttnn.UnaryOpType.MUL_UNARY_SFPU, 1.0 / K)),
            ttnn.UnaryWithParam((ttnn.UnaryOpType.ADD_UNARY_SFPU, eps)),
            ttnn.UnaryWithParam(ttnn.UnaryOpType.RSQRT),
        ]

        def tail(tot_, fold, mask_mode):
            mixes = ttnn.slice(tot_, (0, 0, 0, 0), (1, 1, T, 32))
            if mask_mode == "64":
                ss_ = ttnn.sum(ttnn.where(tm64, tot_, 0.0, memory_config=DRAM), dim=-1, keepdim=True)
            else:
                gram = ttnn.slice(tot_, (0, 0, 0, 32), (1, 1, T, 64))
                ss_ = ttnn.sum(ttnn.where(tm, gram, 0.0, memory_config=DRAM), dim=-1, keepdim=True)
            if fold:
                return ttnn.multiply(mixes, ss_, input_tensor_b_activations=act, memory_config=DRAM)
            sc = ttnn.rsqrt(ttnn.add(ttnn.multiply(ss_, 1.0 / K), eps), memory_config=DRAM)
            return ttnn.multiply(mixes, sc, memory_config=DRAM)

        sum1 = lambda: ttnn.sum(tp_, dim=1, keepdim=True, memory_config=DRAM)
        msum = lambda: ttnn.moreh_sum(tp_, dim=1, keepdim=True, memory_config=DRAM, compute_kernel_config=ckc)
        outs = _run(
            mesh,
            f"mhc finalize T{T} C{C}",
            {
                "current": lambda: tail(sum1(), False, "32"),
                "chunk sum only (sum)": sum1,
                "chunk sum only (moreh)": msum,
                "moreh_sum": lambda: tail(msum(), False, "32"),
                "rsqrt chain folded": lambda: tail(sum1(), True, "32"),
                "64-wide mask": lambda: tail(sum1(), False, "64"),
                "moreh + fold + 64 mask": lambda: tail(msum(), True, "64"),
            },
            50 if T < 2048 else 20,
            exact_vs="current",
        )
        for k_, o_ in outs.items():
            if o_.shape[-1] == 32:
                print(f"    {k_:24}: vs float64 {_err(o_, ref.float())}", flush=True)


def relayout(mesh, g):
    """Review #2 P1-b / P1-e / P1-c / P1-d: tile-row relayouts at decode rows, single-core RMSNorm, KDA prefill
    slice + untilize."""
    from models.autoports.zai_org_glm_5_3_flash.tt.multichip_decoder import sharded_rms_norm

    ckc = _ckc(mesh)
    RM, TL = ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT
    # (b) KDA o [1,1,B*H,V] fp32 -> [1,1,B,H*V]
    H, V = 64, 128
    for B in (1, 32):
        o = torch.randn(1, 1, B * H, V, generator=g)
        to = ttnn.from_torch(o, dtype=ttnn.float32, layout=TL, device=mesh, memory_config=DRAM)
        _run(
            mesh,
            f"kda o relayout B{B}",
            {
                "reshape (TILE)": lambda: ttnn.reshape(to, (1, 1, B, H * V)),
                "untilize+view+tilize": lambda: ttnn.to_layout(
                    ttnn.reshape(ttnn.to_layout(to, RM, memory_config=DRAM), (1, 1, B, H * V)), TL, memory_config=DRAM
                ),
            },
            50,
            exact_vs="reshape (TILE)",
        )
    # (e) DSA decode head split / concat at rows 1 and 32: grouped concat, row-major permute
    D = 256
    for Hh in (64, 16):
        for rows in (1, 32):
            q = torch.randn(1, 1, rows, Hh * D, generator=g).bfloat16()
            tq = ttnn.from_torch(q, dtype=ttnn.bfloat16, layout=TL, device=mesh, memory_config=DRAM)
            _run(
                mesh,
                f"split  H{Hh} rows {rows}",
                {
                    "reshape+permute": lambda: ttnn.permute(
                        ttnn.reshape(tq, (1, rows, Hh, D)), (0, 2, 1, 3), memory_config=DRAM
                    ),
                    "row-major permute": lambda: ttnn.to_layout(
                        ttnn.permute(
                            ttnn.reshape(ttnn.to_layout(tq, RM, memory_config=DRAM), (1, rows, Hh, D)),
                            (0, 2, 1, 3),
                            memory_config=DRAM,
                        ),
                        TL,
                        memory_config=DRAM,
                    ),
                },
                50,
                exact_vs="reshape+permute",
            )
            o = torch.randn(1, Hh, rows, D, generator=g).bfloat16()
            to = ttnn.from_torch(o, dtype=ttnn.bfloat16, layout=TL, device=mesh, memory_config=DRAM)
            v = {
                "permute+reshape": lambda: ttnn.reshape(
                    ttnn.permute(to, (0, 2, 1, 3), memory_config=DRAM), (1, 1, rows, Hh * D)
                ),
                "row-major permute": lambda: ttnn.to_layout(
                    ttnn.reshape(
                        ttnn.permute(ttnn.to_layout(to, RM, memory_config=DRAM), (0, 2, 1, 3), memory_config=DRAM),
                        (1, 1, rows, Hh * D),
                    ),
                    TL,
                    memory_config=DRAM,
                ),
            }
            if Hh > 16:
                v["concat_heads x4 groups"] = lambda: _concat_grouped(to, rows, Hh, D, 4)
            else:
                v["nlp_concat_heads"] = lambda: ttnn.experimental.nlp_concat_heads(to, memory_config=DRAM)
            _run(mesh, f"concat H{Hh} rows {rows}", v, 50, exact_vs="permute+reshape")
    # (c) RMSNorm of the mhc_pre output [1,1,B,4096] bf16: interleaved (one core per tile row) vs width-sharded
    w = ttnn.from_torch(
        (1 + 0.1 * torch.randn(1, 1, 1, 4096, generator=g)).bfloat16().reshape(1, 1, 128, 32),
        dtype=ttnn.bfloat16,
        layout=RM,
        device=mesh,
    )
    for B in (1, 32):
        x = torch.randn(1, 1, B, 4096, generator=g).bfloat16()
        tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=TL, device=mesh, memory_config=DRAM)
        _run(
            mesh,
            f"rms_norm B{B}",
            {
                "interleaved": lambda: ttnn.rms_norm(
                    tx, epsilon=1e-5, weight=w, memory_config=DRAM, compute_kernel_config=ckc
                ),
                "width-sharded (32 cores)": lambda: sharded_rms_norm(tx, w, 1e-5, ckc),
                "width-sharded (16 cores)": lambda: sharded_rms_norm(tx, w, 1e-5, ckc, max_cores=16),
            },
            50,
            exact_vs="interleaved",
        )
    # (d) KDA prefill: qkv columns [0, 24576) of the [1, T, 32960] projection, row-major for qkv_causal_conv1d_silu
    T, P, W = 2048, 32960, 24576
    pr = torch.randn(1, T, P, generator=g).bfloat16()
    tpr = ttnn.from_torch(pr, dtype=ttnn.bfloat16, layout=TL, device=mesh, memory_config=DRAM)
    _run(
        mesh,
        "kda prefill qkv slice+untilize",
        {
            "slice + to_layout": lambda: ttnn.to_layout(
                ttnn.slice(tpr, (0, 0, 0), (1, T, W), memory_config=DRAM), RM, memory_config=DRAM
            ),
            "untilize_with_unpadding": lambda: ttnn.untilize_with_unpadding(
                tpr, output_tensor_end=(0, T - 1, W - 1), memory_config=DRAM
            ),
        },
        20,
        exact_vs="slice + to_layout",
    )


def kdaprefill(mesh, g):
    """Review #2 P1-d, adapted retry: the KDA prefill qkv columns straight out of a narrower matmul with `untilize_out`
    (row-major output for qkv_causal_conv1d_silu) + a second matmul for the other columns, against the fused projection
    + slice + untilize."""
    T, Hd, P, W = 2048, 4096, 32960, 24576
    ckc = ttnn.init_device_compute_kernel_config(
        mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True, packer_l1_acc=True
    )
    grid = mesh.compute_with_storage_grid_size()
    x = (torch.randn(1, 1, T, Hd, generator=g) * 0.05).bfloat16()
    w = (torch.randn(Hd, P, generator=g) * 0.02).bfloat16()
    up = lambda t, dt: ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=DRAM)
    tx, tw = up(x, ttnn.bfloat16), up(w, ttnn.bfloat8_b)
    tq, tr = up(w[:, :W], ttnn.bfloat8_b), up(w[:, W:], ttnn.bfloat8_b)
    lin = lambda a, b, **kw: ttnn.linear(a, b, compute_kernel_config=ckc, memory_config=DRAM, **kw)

    def cur():
        pr = lin(tx, tw)
        q = ttnn.to_layout(
            ttnn.slice(pr, (0, 0, 0, 0), (1, 1, T, W), memory_config=DRAM), ttnn.ROW_MAJOR_LAYOUT, memory_config=DRAM
        )
        return q

    variants = {"fused proj + slice + untilize": cur, "fused proj alone": lambda: lin(tx, tw)}
    Nt = W // 32
    for pn, sw in ((8, 4), (8, 8), (7, 7)):
        if Nt % pn and pn != 7:
            continue
        for kind in ("1d", "reuse"):
            try:
                if kind == "1d":
                    pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                        compute_with_storage_grid_size=(grid.x, grid.y),
                        in0_block_w=2,
                        out_subblock_h=1,
                        out_subblock_w=sw,
                        per_core_M=T // 32,
                        per_core_N=pn,
                        fuse_batch=True,
                        fused_activation=None,
                        mcast_in0=True,
                        untilize_out=True,
                    )
                else:
                    pc = ttnn.MatmulMultiCoreReuseProgramConfig(
                        compute_with_storage_grid_size=(grid.x, grid.y),
                        in0_block_w=2,
                        out_subblock_h=1,
                        out_subblock_w=sw,
                        per_core_M=T // 32,
                        per_core_N=pn,
                        untilize_out=True,
                    )
            except Exception as e:  # noqa: BLE001
                print(f"kda prefill {kind} pn {pn} sw {sw}: config error {str(e).splitlines()[0][:120]}", flush=True)
                continue
            variants[f"qkv untilize_out {kind} N{pn}/{sw} (+rest)"] = lambda pc=pc: (
                lin(tx, tr),
                lin(tx, tq, program_config=pc),
            )[1]
    _run(mesh, "kda prefill 2048 qkv", variants, 10, exact_vs="fused proj + slice + untilize")


def chunksum(mesh, g):
    """Review #3 P2-2 (i): the mhc_mix chunk sum over dim 1 of the partials [1, C, T, 64] fp32: ttnn.sum (transpose +
    reduce + transpose, current) vs ttnn.experimental.fast_reduce_nc(dims=[1]); time, bit-exactness and error vs
    float64."""
    from models.autoports.zai_org_glm_5_3_flash.tt.fused_decoder import mix_chunks

    ckc = _ckc(mesh)
    for T in (1, 32, 2048):
        Tp = -(-T // 32) * 32
        C = mix_chunks(Tp, 16384 // 32)
        part = torch.randn(1, C, Tp, 64, generator=g, dtype=torch.float64) * 0.3
        part[..., 32:] = part[..., 32:].abs() * 40
        part = part[:, :, :T].float()
        tp_ = ttnn.from_torch(part, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=DRAM)
        ref = part.double().sum(1, keepdim=True)
        outs = _run(
            mesh,
            f"chunk sum T{T} C{C}",
            {
                "ttnn.sum (current)": lambda: ttnn.sum(tp_, dim=1, keepdim=True, memory_config=DRAM),
                "fast_reduce_nc": lambda: ttnn.experimental.fast_reduce_nc(
                    tp_, dims=[1], memory_config=DRAM, compute_kernel_config=ckc
                ),
            },
            50 if T < 2048 else 20,
        )
        # fast_reduce_nc reports the tile-padded row count (32) at T 1: compare the logical rows
        crop = {k_: o_[..., :T, :].reshape(ref.shape) for k_, o_ in outs.items()}
        base = crop.get("ttnn.sum (current)")
        for k_, o_ in crop.items():
            same = base is not None and torch.equal(o_, base)
            print(f"    {k_:24}: vs float64 {_err(o_, ref.float())}; bit-exact vs ttnn.sum: {same}", flush=True)


def kdaprefill2(mesh, g):
    """Review #3 P2-2 (ii): the narrower-matmul pattern at the model's numerics (bf16 x bf16, HiFi4, fp32 accumulation,
    default program configs): fused projection [4096 x 32960] + slice of the qkv columns + untilize, vs a qkv projection
    [4096 x 24576] + a rest projection [4096 x 8384] + untilize (no slice)."""
    T, Hd, P, W = 2048, 4096, 32960, 24576
    ckc = ttnn.init_device_compute_kernel_config(
        mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True
    )
    x = (torch.randn(1, T, Hd, generator=g) * 0.05).bfloat16()
    w = (torch.randn(Hd, P, generator=g) * 0.02).bfloat16()
    up = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=DRAM)
    tx, tw, tq, tr = up(x), up(w), up(w[:, :W].contiguous()), up(w[:, W:].contiguous())
    lin = lambda a, b: ttnn.linear(a, b, compute_kernel_config=ckc, memory_config=DRAM)
    RM = ttnn.ROW_MAJOR_LAYOUT

    def cur():
        pr = lin(tx, tw)
        return ttnn.to_layout(ttnn.slice(pr, (0, 0, 0), (1, T, W), memory_config=DRAM), RM, memory_config=DRAM)

    def split():
        rest = lin(tx, tr)
        return ttnn.to_layout(lin(tx, tq), RM, memory_config=DRAM)

    _run(
        mesh,
        "kda prefill 2048 bf16 HiFi4",
        {
            "fused + slice + untilize": cur,
            "qkv + rest + untilize": split,
            "fused proj alone": lambda: lin(tx, tw),
            "qkv proj alone": lambda: lin(tx, tq),
            "rest proj alone": lambda: lin(tx, tr),
        },
        10,
        exact_vs="fused + slice + untilize",
    )


SECTIONS = {
    "chunksum": chunksum,
    "kdaprefill2": kdaprefill2,
    "kdaprefill": kdaprefill,
    "finalize": finalize,
    "relayout": relayout,
    "kda2": kda2,
    "heads": heads,
    "heads2": heads2,
    "mlp": mlp,
    "moe": moe,
    "kdanorm": kdanorm,
    "kdaconv": kdaconv,
}


def main():
    names = sys.argv[1:] or list(SECTIONS)
    mesh = open_mesh()
    g = torch.Generator().manual_seed(0)
    try:
        for s in names:
            print(f"=== {s}", flush=True)
            SECTIONS[s](mesh, g)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()

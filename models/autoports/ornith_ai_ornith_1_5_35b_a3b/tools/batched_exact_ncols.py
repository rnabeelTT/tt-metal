# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Where does the batched short-prompt prefill (tt/prefill_batched.py) leave the single-admission prefill's numerics?

Per layer and per user: the batched pass's output rows vs the production single-user masked-bucket pass (eager, same
page-table rows), and the batched pass's rows for user 0 with other users present vs user 0 alone (composition).

    python -m models.autoports.ornith_ai_ornith_1_5_35b_a3b.tools.batched_exact [--layers 0,3] [--lens 100,57,128,20]
"""
import argparse
import os

import torch

import ttnn
from models.autoports.ornith_ai_ornith_1_5_35b_a3b.tt.generator import build_generator
from models.autoports.ornith_ai_ornith_1_5_35b_a3b.tt.prefill_batched import BatchedPrefill

MODEL_DIR = "models/autoports/ornith_ai_ornith_1_5_35b_a3b"


def host(t, mesh):
    return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).float()  # [ndev*..., ...]


def diff(a, b):
    a, b = a.float(), b.float()
    d = (a - b).abs()
    return f"exact={bool(torch.equal(a, b))} max|d|={float(d.max()):.4g} frac_neq={float((d > 0).float().mean()):.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="0,3")
    ap.add_argument("--lens", default="100,57,128,20")
    ap.add_argument("--probe", action="store_true", help="sub-op outputs inside the first layer")
    ap.add_argument("--quiet", action="store_true", help="one line per layer unless a user differs")
    a = ap.parse_args()
    os.environ.setdefault("MESH_DEVICE", "P300x2")
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, int(os.environ.get("MESH_COLS", 4))), l1_small_size=24576, trace_region_size=256 << 20)
    try:
        B = 4
        gen = build_generator(MODEL_DIR, mesh, max_batch_size=B, max_seq_len=4096,
                              layer_indices=[int(x) for x in a.layers.split(",")] if a.layers not in ("", "all") else None,
                              warm_decode_trace=False, warm_prefill_trace=False)
        model = gen.model
        bp = BatchedPrefill(model, B=B)
        lens = [int(x) for x in a.lens.split(",")]
        n = len(lens)
        T = max(lens)
        bucket = model._mask_bucket_for(T)
        toks = torch.zeros(B, T, dtype=torch.int64)
        g = torch.Generator().manual_seed(7)
        for u in range(n):
            toks[u, : lens[u]] = torch.randint(1000, 120000, (lens[u],), generator=g)

        # ---- batched: per-layer outputs
        rec = []

        def wrap(fn):
            def w(*args, **kw):
                y = fn(*args, **kw)
                rec.append(host(y, mesh))
                return y
            return w

        # sub-op probes inside the first layer (--probe): every call's output(s), in call order, per pass
        probes = []

        def pwrap(obj, name):
            f = getattr(obj, name)

            def w(*args, **kw):
                y = f(*args, **kw)
                for i, t in enumerate(y if isinstance(y, (tuple, list)) else (y,)):
                    if isinstance(t, ttnn.Tensor):
                        h = host(t, mesh)
                        probes.append((f"{name}[{i}]", h.reshape(mesh.get_num_devices(), -1, h.shape[-1])))
                return y

            setattr(obj, name, w)

        if a.probe:
            l0 = model.layers[int(os.environ.get("PROBE_LAYER", 0))]
            for obj, name in ((l0, "_decode_norm"), (l0.attention, "_project_qkvzab"), (l0.feed_forward, "forward"),
                              (l0.attention, "forward_prefill"), (l0.attention, "forward_prefill_batched")):
                if hasattr(obj, name):
                    pwrap(obj, name)
        if os.environ.get("PROBE_MMRS") == "1":  # the GDN / attention out-proj (matmul + reduce-scatter): its input and output
            import models.demos.blackhole.qwen36.tt.tp_common as _tpc
            _f = _tpc.matmul_reduce_scatter_prefill

            def _mm(x, *args, **kw):
                hx = host(x, mesh)
                probes.append(("mmrs_in", hx.reshape(mesh.get_num_devices(), -1, hx.shape[-1])))
                y = _f(x, *args, **kw)
                hy = host(y, mesh)
                probes.append(("mmrs_out", hy.reshape(mesh.get_num_devices(), -1, hy.shape[-1])))
                return y

            _tpc.matmul_reduce_scatter_prefill = _mm
        bp._attn_layer = wrap(bp._attn_layer)
        bp._gdn_layer = wrap(bp._gdn_layer)
        gen.reset()
        got, _, _ = bp.prefill(toks[:n], lens, gen.page_table[:n], list(range(n)))
        bat = list(rec)
        rec.clear()
        pb = list(probes)
        probes.clear()
        gen.reset()
        got1, _, _ = bp.prefill(toks[:1], lens[:1], gen.page_table[:1], [0])
        bat1 = list(rec)
        rec.clear()
        probes.clear()

        # ---- single: per-layer outputs per user
        orig = [layer.forward for layer in model.layers]
        for layer, f in zip(model.layers, orig):
            layer.forward = wrap(f)
        sing, logits, ps = [], [], []
        for u in range(n):
            gen.reset()
            lg = gen.prefill_forward(toks[u : u + 1, : lens[u]], page_table=gen.page_table[u : u + 1], prompt_lens=[lens[u]])
            logits.append(torch.as_tensor(lg).float().reshape(-1))
            sing.append(list(rec))
            rec.clear()
            ps.append(list(probes))
            probes.clear()
        for layer, f in zip(model.layers, orig):
            layer.forward = f

        print(f"BEXACT bucket {bucket} lens {lens}", flush=True)
        if a.probe:
            pb_mix = [p for p in pb if "forward_prefill" not in p[0]]
            for u in range(n):
                pu = [p for p in ps[u] if "forward_prefill" not in p[0]]
                bu = model._mask_bucket_for(lens[u])
                for (nb, tb), (ns, ts) in zip(pb_mix, pu):
                    rb = tb[:, u * bucket : u * bucket + lens[u]]
                    rs = ts[:, : lens[u]]
                    print(f"BEXACT probe user {u} (single bucket {bu}) {nb}/{ns} rows b{tuple(tb.shape)} s{tuple(ts.shape)}: "
                          f"{diff(rb, rs) if rb.shape == rs.shape else 'shape'}", flush=True)
        for li in range(len(model.layers)):
            kind = "attn" if model.layers[li].is_full_attention else "gdn"
            xb = bat[li]  # [ndev, 1, B*bucket, d]
            xb1 = bat1[li]
            worst = []
            for u in range(n):
                L = lens[u]
                xs = sing[u][li]  # [ndev, 1, bucket_u, d]
                rb = xb[:, :, u * bucket : u * bucket + L]
                rs = xs[:, :, :L]
                msg = f"BEXACT layer {li} ({kind}) user {u}: batched vs single {diff(rb, rs) if rb.shape == rs.shape else ('shape', rb.shape, rs.shape)}"
                if u == 0:
                    msg += f" | with-others vs alone {diff(rb, xb1[:, :, :L])}"
                if not a.quiet or not torch.equal(rb, rs):
                    print(msg, flush=True)
                worst.append(torch.equal(rb, rs))
            if a.quiet:
                print(f"BEXACT layer {li} ({kind}): {sum(worst)}/{len(worst)} users exact", flush=True)
        for u in range(n):
            print(f"BEXACT logits user {u}: {diff(got[u, 0], logits[u])} argmax {int(got[u,0].argmax())} vs {int(logits[u].argmax())}",
                  flush=True)
        print(f"BEXACT logits user 0 with-others vs alone: {diff(got[0, 0], got1[0, 0])}", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()

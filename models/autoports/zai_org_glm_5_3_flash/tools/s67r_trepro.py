# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Decode vs max_seq_len repro (stage-6/7 rework): the full model's teacher forcing passes at max_seq_len 16384 / 65536
and fails at 102400 (dense-phase decode positions ~160-260; prefill is fine). The DSA decode path at positions <= 2050
must not depend on the cache size. This builds the GLM53_LAYERS model (default 0,3,4) at each --lengths value, prefills
one real-text prompt into slot 0 and decodes --steps steps eagerly (host logits, teacher-forced with the first build's
tokens), then compares every build's per-step logits with the first build's.

usage: GLM53_TP=4 GLM53_LAYERS=0,3,4 python -m models.autoports.zai_org_glm_5_3_flash.tools.s67r_trepro \
           --lengths 16384,65536,81920,102400 --out FILE.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

import ttnn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", default="16384,65536,81920,102400")
    ap.add_argument("--prompt", type=int, default=161)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--traced", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--save-pt", default=None)
    ap.add_argument("--dump-layers", type=int, default=0)
    ap.add_argument("--gen", type=int, default=0)
    ap.add_argument("--host-sample", type=int, default=0)
    ap.add_argument("--state-dump", type=int, default=0)
    ap.add_argument("--forced", default=None, help="teacher-forced token file (.pt from a previous --save-pt run)")
    a = ap.parse_args()
    os.environ.setdefault("GLM53_LAYERS", "0,3,4")
    from models.autoports.zai_org_glm_5_3_flash.tools import s7_probe as P
    from models.autoports.zai_org_glm_5_3_flash.tools.s67r_checks import cmp_rows, provenance, text_ids
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
    capturing = {"on": False}
    _b, _e = ttnn.begin_trace_capture, ttnn.end_trace_capture

    def _begin(*x, **k):
        capturing["on"] = True
        return _b(*x, **k)

    def _end(*x, **k):
        r = _e(*x, **k)
        capturing["on"] = False
        return r

    ttnn.begin_trace_capture, ttnn.end_trace_capture = _begin, _end

    def _sync(dev):
        if not capturing["on"]:
            ttnn.synchronize_device(dev)

    if os.environ.get("S67R_SYNC_CCL") == "1":  # diagnostic: a device sync after every collective
        import models.autoports.zai_org_glm_5_3_flash.tt.model as Mm
        import models.autoports.zai_org_glm_5_3_flash.tt.multichip_decoder as Mmc

        def synced(fn):
            def f(*x, **k):
                out = fn(*x, **k)
                _sync(x[1] if len(x) > 1 and not isinstance(x[1], int) else k.get("mesh_device"))
                return out

            return f

        for mod in (Mm, Mmc):
            for nm in ("tt_all_gather", "tt_all_reduce"):
                if hasattr(mod, nm):
                    setattr(mod, nm, synced(getattr(mod, nm)))
    if os.environ.get("S67R_SYNC_LAYER") == "1":  # diagnostic: a device sync after every decoder layer (all steps)
        from models.autoports.zai_org_glm_5_3_flash.tt.multichip_decoder import MultichipDecoder

        orig_df = MultichipDecoder.decode_forward

        def df(self, X, *x, **k):
            Y = orig_df(self, X, *x, **k)
            _sync(Y.device())
            return Y

        MultichipDecoder.decode_forward = df
    rec = {"args": vars(a), "provenance": provenance(), "runs": {}}
    mesh = P._open_mesh(64 << 20)
    ref = None
    try:
        for T in [int(x) for x in a.lengths.split(",")]:
            g = Glm53Generator(mesh, max_batch_size=1, max_seq_len=T)
            ids = text_ids(g.tokenizer, a.prompt)
            g.reset()
            lg = g.prefill_forward(ids.reshape(1, -1), prompt_lens=[a.prompt])[0, -1, : g.model.vocab]
            toks = [int(lg.argmax())] if ref is None else ref["tokens"]
            if a.forced:
                toks = torch.load(a.forced)["tokens"]
            rows = [lg.clone()]
            pos = a.prompt
            dumps = []
            if a.dump_layers:  # every layer's decode output of the first decode step (die 0), for a divergence scan
                for li, d in zip(g.model.layer_indices, g.model.layers):
                    orig = d.decode_forward

                    def wrapped(X, *x, _o=orig, _li=li, **k):
                        Y = _o(X, *x, **k)
                        if len(dumps) < len(g.model.layers):
                            if a.dump_layers == 2:  # device copy now, host read after the step (no sync mid-step)
                                dumps.append((_li, ttnn.clone(Y)))
                            else:
                                dumps.append((_li, ttnn.to_torch(ttnn.get_device_tensors(Y)[0]).float().clone()))
                        return Y

                    d.decode_forward = wrapped
            if (
                a.gen
            ):  # the generate() / teacher-forcing flow: traced decode + on-device greedy sampling + token feedback
                forced = torch.load(a.forced)["tokens"] if a.forced else None
                rows = [lg.clone()]

                def hook(i, pred):
                    rows.append(g.model.logits_to_host(g.logits_out)[0, 0, 0, : g.model.vocab].clone())
                    return forced[i + 1] if forced is not None and i + 1 < len(forced) else None

                toks = g.generate(ids.tolist(), a.steps + 1, next_input=hook, on_device_sampling=not a.host_sample)
                rows = rows[: a.steps + 1]
                a_steps = 0
            else:
                a_steps = a.steps
            hooks_fired = {"state": 0}

            def state_snapshot():
                hooks_fired["state"] += 1
                out = {}
                for k, t in g._trace_tensors().items():
                    if k.endswith((".latent", ".pooled", ".lpt")) or k == "logits_out" and t is None:
                        continue
                    if not t.is_allocated():
                        out[k] = None
                        continue
                    out[k] = torch.stack([ttnn.to_torch(d).float() for d in ttnn.get_device_tensors(t)])
                return out

            states = {"pre": state_snapshot()} if a.state_dump == 1 else {}
            for i in range(a_steps):
                if a.state_dump and i == 1:
                    states["post1"] = state_snapshot()
                    if a.state_dump == 2:  # + the step-1 latent row and pool-key tile of every DSA layer (die 0)
                        p1 = a.prompt
                        for li, d in zip(g.model.layer_indices, g.model.layers):
                            if d.attn_kind == "dsa":
                                at = d.attn
                                blk = int(at.page_table_host[0, p1 // 128])
                                lat = ttnn.to_torch(ttnn.get_device_tensors(at.latent)[0]).float()
                                states["post1"][f"l{li}.latent_row"] = lat.reshape(-1, lat.shape[-1])[
                                    blk * 128 + p1 % 128
                                ].clone()
                                pk = ttnn.to_torch(ttnn.get_device_tensors(at.pooled)[0]).float()
                                states["post1"][f"l{li}.pooled_blk"] = pk.reshape(-1, 32, pk.shape[-1])[blk].clone()
                                del lat, pk
                o = g.decode_forward(
                    torch.tensor([[toks[i]]]),
                    torch.tensor([pos]),
                    enable_trace=bool(a.traced),
                    on_device_sampling=False,
                    reload_inputs=True,
                )
                rows.append(o[0].clone())
                if ref is None and not a.forced:
                    toks.append(int(o[0].argmax()))
                pos += 1
            dumps = [
                (li, ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float().clone() if isinstance(t, ttnn.Tensor) else t)
                for li, t in dumps
            ]
            r = {"T": T, "MB": g.model.inputs.MB, "tokens": toks}
            if a.save_pt:
                torch.save(
                    {
                        "tokens": toks,
                        "rows": [r.to(torch.bfloat16) for r in rows],
                        "layer_dumps": dumps,
                        "states": states if a.state_dump else None,
                        "hooks_fired": hooks_fired,
                        "addresses": {k: v for k, v in g.trace_addresses().items()} if g.trace_id is not None else None,
                    },
                    f"{a.save_pt}_{T}.pt",
                )
            if ref is None:
                ref = {"tokens": toks, "rows": rows, "T": T}
            else:
                r["vs_first"] = [cmp_rows(x, y) for x, y in zip(rows, ref["rows"])]
            rec["runs"][str(T)] = r
            print(f"S67R TREPRO T={T}: " + json.dumps({k: v for k, v in r.items() if k != "tokens"}), flush=True)
            g.teardown()
            del g
    finally:
        Path(a.out).write_text(json.dumps(rec, indent=1, default=str) + "\n")
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()

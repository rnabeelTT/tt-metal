# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic of the block-sharded DSA cache (tt/dsa_shard.py): the b1 case of tests/test_dsa_shard.py (layer 3, one
slot, 6001-token synthetic prefill, 3 decode steps), run replicated and sharded with EAGER decode steps (traced=0) or
eager + traced (traced=1). Records per decode step the attention output o_lat [16 heads x 512] of every die (both
layouts), and in the sharded layout the per-die sink weight w / real mass u per (head, die) and the sink logit.

usage: GLM53_TP=4 python -m models.autoports.zai_org_glm_5_3_flash.tools.dsa_shard_probe --out FILE [--traced 1]
"""

import argparse
import dataclasses

import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tests.harness import (
    DecodeRunner,
    build_layer,
    close_mesh,
    open_mesh,
    prefill_user,
)
from models.autoports.zai_org_glm_5_3_flash.tests.test_dsa_shard import _synthetic
from models.autoports.zai_org_glm_5_3_flash.tt import dsa_shard
from models.autoports.zai_org_glm_5_3_flash.tt.ckpt import CheckpointReader
from models.autoports.zai_org_glm_5_3_flash.tt.dsa import GlmDSA
from models.autoports.zai_org_glm_5_3_flash.tt.multichip_decoder import DEFAULT_MC, MCOpts, MultichipDecoder

REC = {"on": False, "o": [], "uw": [], "sd": []}


def _host(t):
    return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(t.device(), dim=0)).float()


_orig_attend = GlmDSA._sparse_attend


def _rep_attend(self, q_abs, index_f32, cache_batch_idx=None, latent_rows=None):
    out = _orig_attend(self, q_abs, index_f32, cache_batch_idx, latent_rows)
    if REC["on"]:
        REC["o"].append(_host(out))
    return out


_orig_shard = dsa_shard.ShardedDSAMixin._shard_attend


def _shard_attend(self, q_abs, index_f32, cache_batch_idx, c, rows):
    out = _orig_shard(self, q_abs, index_f32, cache_batch_idx, c, rows)
    if REC["on"]:
        REC["o"].append(_host(out))
    return out


_orig_sdpa = ttnn.transformer.sparse_sdpa


def _sdpa(*a, **kw):
    out = _orig_sdpa(*a, **kw)
    if REC["on"] and int(out.shape[-1]) == dsa_shard.KCOL:
        o = _host(out)  # [4 dies, 64 heads, rows, 544]
        REC["sd"].append(o[..., 512:515])
    return out


_orig_rs = dsa_shard.ShardedDSAMixin._reduce_scatter_heads


def _rs(self, t):
    if REC["on"]:
        REC["uw"].append(_host(t)[..., :1] if t.shape[-1] == 32 else None)  # the mass / w input (per die, pre-sum)
    return _orig_rs(self, t)


def run(mesh, rd, shard, traced, x, L):
    mc = dataclasses.replace(MCOpts.from_env(DEFAULT_MC), dsa_shard=shard)

    class C(MultichipDecoder):
        @classmethod
        def from_state_dict(cls, *a, **kw):
            kw["mc"] = mc
            return super().from_state_dict(*a, **kw)

    dec = build_layer(mesh, rd, 3, "real", batch=1, prefill_lengths=(128, 2048), max_context=8192, decoder_cls=C)
    perm = torch.randperm(dec.attn.NB - 1, generator=torch.Generator().manual_seed(11))[:64].reshape(1, 64)
    dec.set_page_table(perm.to(torch.int32))
    prefill_user(dec, mesh, x, L, 0)
    run_ = DecodeRunner(dec, mesh, 1)
    outs, o_lat, mass = [], [], []
    for i in range(3):
        REC["o"].clear()
        REC["uw"].clear()
        REC["sd"].clear()
        REC["on"] = True
        if i == 0 or not traced:
            o = run_.eager_step({0: x[L + i]}, [L + i])
            if traced:
                REC["on"] = False
                run_.capture({0: x[L + 1]}, [L + 1])
        else:
            o = run_.traced_step({0: x[L + i]}, [L + i])
        REC["on"] = False
        outs.append(o[0] - x[L + i])
        o_lat.append(REC["o"][0] if REC["o"] else None)
        mass.append(([u for u in REC["uw"] if u is not None], list(REC["sd"])))
    run_.release()
    del run_, dec
    return outs, o_lat, mass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--traced", type=int, default=0)
    a = ap.parse_args()
    GlmDSA._sparse_attend = _rep_attend
    dsa_shard.ShardedDSAMixin._shard_attend = _shard_attend
    dsa_shard.ShardedDSAMixin._reduce_scatter_heads = _rs
    ttnn.transformer.sparse_sdpa = _sdpa
    rd = CheckpointReader()
    mesh = open_mesh()
    L = 6001
    x = _synthetic(80, L + 3)
    try:
        res = {}
        for shard in (False, True):
            res[shard] = run(mesh, rd, shard, bool(a.traced), x, L)
    finally:
        close_mesh(mesh)
    from models.autoports.zai_org_glm_5_3_flash.tests.harness import pcc

    for i in range(3):
        o_r, o_s = res[False][1][i], res[True][1][i]
        print(f"step {i}: layer update PCC {pcc(res[True][0][i], res[False][0][i]):.6f}")
        if o_r is not None and o_s is not None:
            # [4 dies, 16, 1, 512] (replicated: the 16 local heads of each die)
            o_r = o_r.reshape(64, -1)
            o_s = o_s.reshape(64, -1)
            ph = [pcc(o_s[h], o_r[h]) for h in range(64)]
            bad = [(h, round(p, 4)) for h, p in enumerate(ph) if p < 0.999]
            print(f"  o_lat per head PCC min {min(ph):.6f}; heads < 0.999: {bad}")
        ms, uws = res[True][2][i]
        for m in ms:
            print(f"  mass/w per die: min {float(m.min()):.3e} max {float(m.max()):.3e} zeros {int((m == 0).sum())}")
        for uw in uws:  # [4 dies, 64 heads, rows, 3]: u, w, kill column
            u, w = uw[..., 0].reshape(4, 64), uw[..., 1].reshape(4, 64)
            print(
                f"  u per (die, head): min {float(u.min()):.3e} max {float(u.max()):.3e}; w: min {float(w.min()):.3e} "
                f"max {float(w.max()):.3e}; w == 0: {int((w == 0).sum())}; u + w - 1 max {float((u + w - 1).abs().max()):.3e}"
            )
            for h in range(64):
                print(
                    f"    head {h}: u {[f'{float(v):.3e}' for v in u[:, h]]} w {[f'{float(v):.3e}' for v in w[:, h]]}"
                )
    torch.save({"res": res}, a.out)


if __name__ == "__main__":
    main()

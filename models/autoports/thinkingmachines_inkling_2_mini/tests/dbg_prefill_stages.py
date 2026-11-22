# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Debug: one layer's prefill stages, each fed its golden input, against the golden internals (reference/golden.py).
usage: tools/run_dev.sh python models/autoports/thinkingmachines_inkling_2_mini/tests/dbg_prefill_stages.py  (LI=0)"""

import math
import os

import torch

import ttnn
from models.autoports.thinkingmachines_inkling_2_mini.tt.model import Inkling2Model, _bf, _f32

LI = int(os.environ.get("LI", "0"))
S5 = math.sqrt(5.0)


def pcc(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def main():
    g = torch.load(os.environ.get("FIX", "/cache/fixtures/golden_t256.pt"))
    I = {k: v[0][0] if v[0].dim() == 3 else v[0] for k, v in g["internals"].items() if k.startswith(f"layers.{LI}.")}
    T = g["hidden"].shape[1]
    N = int(os.environ.get("MESH", "1"))
    if N > 1:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, N))
    rep_m = ttnn.ReplicateTensorToMesh(mesh) if N > 1 else None
    m = Inkling2Model(mesh, max_seq_len=1024, expert_dtype=ttnn.bfloat16, prefill_chunk=256, layers=[LI])
    L, st, ops = m.layers[LI], m.apps[0], m.ops
    dev = lambda t: ttnn.from_torch(t.reshape(1, 1, T, -1).to(torch.bfloat16), dtype=ttnn.bfloat16,  # noqa: E731
                                    layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep_m)
    d0 = lambda t: ttnn.get_device_tensors(t)[0] if N > 1 else t  # noqa: E731
    host = lambda t: ttnn.to_torch(d0(t)).float().reshape(T, -1)  # noqa: E731
    p = f"layers.{LI}."
    rep = lambda n, got, want: print(f"{n:34s} pcc {pcc(got, want):.5f}  maxerr {(got - want).abs().max():.4f}",  # noqa
                                     flush=True)
    if L.sliding:
        xn = I[p + "input_layernorm"]
        from models.autoports.thinkingmachines_inkling_2_mini.tt.attention import attention

        st.reset()
        att = host(attention(L.attn, st.cache, dev(xn), dev(xn), dev(xn), 0, tt_ccl=m.tt_ccl))
        rep("attention (o_proj out)", att * S5, I[p + "self_attn"])
        # the same rows through the rows (decode) path, 8 at a time
        st.reset()
        outs = []
        for j in range(0, T, 8):
            sl = ttnn.slice(dev(xn), [0, 0, j, 0], [1, 1, j + 8, xn.shape[-1]])
            outs.append(ttnn.to_torch(d0(attention(L.attn, st.cache, sl, sl, sl, j, tt_ccl=m.tt_ccl))).float()
                        .reshape(8, -1))
        rep("attention via rows path", torch.cat(outs) * S5, I[p + "self_attn"])
    st.attn_conv.reset()
    c = host(st.conv("attn", dev(I[p + "self_attn"] / S5), False, T))
    rep("post-attention conv (prefill kernel)", c, I[p + "post_attention_convolution"])
    st.attn_conv.reset()
    c = host(st.conv("attn", dev(I[p + "self_attn"] / S5), True, T))
    rep("post-attention conv (rows/ttnn)", c, I[p + "post_attention_convolution"])
    mi = I[p + "post_attention_layernorm"]
    mo = host(L.moe.forward(dev(mi)))
    rep("moe (masked prefill path)", mo * S5, I[p + "mlp"])
    outs = []
    for j in range(0, T, 8):
        sl = ttnn.slice(dev(mi), [0, 0, j, 0], [1, 1, j + 8, mi.shape[-1]])
        outs.append(ttnn.to_torch(d0(L.moe.forward(sl))).float().reshape(8, -1))
    rep("moe (gather path, 8 rows)", torch.cat(outs) * S5, I[p + "mlp"])
    st.mlp_conv.reset()
    c = host(st.conv("mlp", dev(I[p + "mlp"] / S5), False, T))
    rep("post-mlp conv (prefill kernel)", c, I[p + "post_mlp_convolution"])
    ttnn.close_mesh_device(mesh)
    print("== done")


if __name__ == "__main__":
    main()

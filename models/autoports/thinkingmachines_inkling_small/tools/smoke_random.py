# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Integration smoke test with random weights (no checkpoint shards needed beyond config/tokenizer): builds the real
Inkling model classes through tt/model.create_model on a fake state dict with the checkpoint's shapes, for layers
0 (dense, sliding) and 5 (MoE, global), then runs a few one-row decode steps and one rows-mode group. Checks only that
everything constructs, runs and stays finite.

usage (in the image): tools/run_dev.sh python models/autoports/thinkingmachines_inkling_small/tools/smoke_random.py
"""

import torch

import ttnn
from models.autoports.thinkingmachines_inkling_small.tt import model_config as MC
from models.autoports.thinkingmachines_inkling_small.tt.model import create_model
from models.demos.blackhole.qwen36.utils.substate import LazyState

H, V, E, S, I, ID = 4096, 201024, 256, 2, 2048, 16384


class FakeSD(dict):
    def __init__(self, cfg):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        r = lambda *s, sc=0.02: (torch.randn(*s, generator=g) * sc).to(torch.bfloat16)  # noqa: E731
        self["tok_embeddings.weight"] = r(V, H, sc=1.0)
        self["norm.weight"] = torch.ones(H, dtype=torch.bfloat16) / 16
        self["output.weight"] = r(V, H)
        self._layers = {}
        for li in range(cfg["num_hidden_layers"]):
            sliding = li in cfg["local_layer_ids"]
            nh, nkv, hd = 32, 8, 128
            d = {
                "input_layernorm.weight": torch.ones(H, dtype=torch.bfloat16),
                "post_attention_layernorm.weight": torch.ones(H, dtype=torch.bfloat16),
                "attn_sconv.weight": r(H, 1, 4, sc=0.2), "mlp_sconv.weight": r(H, 1, 4, sc=0.2),
                "self_attn.q_proj.weight": r(nh * hd, H), "self_attn.k_proj.weight": r(nkv * hd, H),
                "self_attn.v_proj.weight": r(nkv * hd, H), "self_attn.r_proj.weight": r(nh * 16, H),
                "self_attn.o_proj.weight": r(H, nh * hd), "self_attn.q_norm.weight": torch.ones(hd, dtype=torch.bfloat16),
                "self_attn.k_norm.weight": torch.ones(hd, dtype=torch.bfloat16),
                "self_attn.k_sconv.conv1d.weight": r(nkv * hd, 1, 4, sc=0.2),
                "self_attn.v_sconv.conv1d.weight": r(nkv * hd, 1, 4, sc=0.2),
                "self_attn.rel_logits_proj.proj": r(16, 512 if sliding else 1024, sc=0.5),
            }
            if li < cfg["dense_mlp_idx"]:
                d.update({"mlp.gate_proj.weight": r(ID, H), "mlp.up_proj.weight": r(ID, H), "mlp.down_proj.weight": r(H, ID)})
            else:
                d.update({"mlp.gate.weight": r(E + S, H), "mlp.gate.e_score_correction_bias": torch.zeros(E),
                          "mlp.gate.global_scale": torch.ones(1),
                          "mlp.experts.gate_up_proj": None, "mlp.experts.down_proj": None})
            self._layers[li] = d

    def layer(self, li):
        d = self._layers[li]
        g = torch.Generator().manual_seed(li)

        def get(k):
            sub = k.split(".", 2)[2]
            if sub == "mlp.experts.gate_up_proj":
                return (torch.randn(E + S, 2 * I, H, generator=g) * 0.02).to(torch.bfloat16)
            if sub == "mlp.experts.down_proj":
                return (torch.randn(E + S, H, I, generator=g) * 0.02).to(torch.bfloat16)
            return d[sub]

        return LazyState(get, [f"layers.{li}.{k}" for k in d])

    def __getitem__(self, k):
        if k.startswith("layers."):
            li = int(k.split(".")[1])
            return self.layer(li)[k]
        return dict.__getitem__(self, k)

    def __contains__(self, k):
        if k.startswith("layers."):
            li, sub = int(k.split(".")[1]), k.split(".", 2)[2]
            return sub in self._layers.get(li, {})
        return dict.__contains__(self, k)


def main():
    orig = MC.Qwen35MoeModelArgs.load_state_dict
    MC.Qwen35MoeModelArgs.load_state_dict = lambda self: FakeSD(self.inkling | {"num_hidden_layers": self.n_layers_full})
    _init = MC.Qwen35MoeModelArgs.__init__

    def init(self, *a, **k):
        _init(self, *a, **k)
        self.n_layers_full = self.n_layers

    MC.Qwen35MoeModelArgs.__init__ = init
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), l1_small_size=24576, trace_region_size=64_000_000)
    try:
        args, model = create_model(mesh, max_seq_len=4096, layer_indices=[0, 5])
        rep = ttnn.ReplicateTensorToMesh(mesh)
        nblk = model.layers[0].attention.max_blocks
        pt = ttnn.from_torch(torch.arange(nblk, dtype=torch.int32).reshape(1, -1), dtype=ttnn.int32,
                             layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, mesh_mapper=rep)
        sh = ttnn.ShardTensorToMesh(mesh, dim=-1)
        x = ttnn.from_torch(torch.randn(1, 1, 1, H).to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                            device=mesh, mesh_mapper=sh)
        for t in range(3):
            cur = ttnn.from_torch(torch.tensor([t], dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT,
                                  device=mesh, mesh_mapper=rep)
            y = x
            for layer in model.layers:
                y = layer.forward(y, mode="decode", position_tensor=cur, page_table=pt)
            out = ttnn.to_torch(y, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=-1)).float()
            print(f"decode step {t}: out {tuple(out.shape)} finite {bool(torch.isfinite(out).all())} |x| {out.norm():.2f}")
        xr = ttnn.from_torch(torch.randn(1, 1, 8, H).to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                             device=mesh, mesh_mapper=sh)
        pos = ttnn.from_torch(torch.arange(3, 11, dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT,
                              device=mesh, mesh_mapper=rep)
        y = xr
        for layer in model.layers:
            y = layer.forward(y, mode="rows", position_tensor=pos, page_table=pt)
        out = ttnn.to_torch(y, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=-1)).float()
        print(f"rows group: out {tuple(out.shape)} finite {bool(torch.isfinite(out).all())}")
        xl = model._final_norm_decode(y)
        logits = model._lm_head(xl)
        print("logits", tuple(logits.shape))
        print("SMOKE OK")
    finally:
        MC.Qwen35MoeModelArgs.load_state_dict = orig
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()

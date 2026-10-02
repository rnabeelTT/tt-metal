# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Dense Qwen3.5-9B regression check for the shared gated-attention kernel changes (paged decode path).

Layer 3 (first full-attention layer) of Qwen/Qwen3.5-9B on one die: paged prefill (T=37) then paged decode at B=1 and at
B=4 with distinct positions, each compared with the HF ``Qwen3_5DecoderLayer``. Exercises the T==1 head-concat rewrite,
the Q permute and the one-user-per-core shard grid on the 9B shapes (16 q heads, 4 kv heads, head_dim 256, hidden 4096).

    HF_MODEL=Qwen/Qwen3.5-9B python models/autoports/qwen_qwen3_coder_next/tools/check_qwen36_9b_decode.py
"""
import glob
import os

import torch
from safetensors import safe_open
from transformers import AutoConfig
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5TextRotaryEmbedding

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.blackhole.qwen36.tt.layer import Qwen36DecoderLayer
from models.demos.blackhole.qwen36.tt.model_config import Qwen36ModelArgs
from models.demos.blackhole.qwen36.tt.rope import Qwen36RoPESetup
from models.demos.blackhole.qwen36.tt.weight_mapping import remap_qwen36_state_dict

os.environ.setdefault("HF_MODEL", "Qwen/Qwen3.5-9B")
LAYER = 3
BLOCK = 64


def main():
    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=24576, trace_region_size=0, num_command_queues=1)
    try:
        args = Qwen36ModelArgs(mesh_device=dev, max_batch_size=4, max_seq_len=1024)
        prefix = f"model.language_model.layers.{LAYER}."
        raw = {}
        for path in sorted(
            glob.glob(f"{args.CKPT_DIR}/model.safetensors*.safetensors")
            + glob.glob(f"{args.CKPT_DIR}/model*.safetensors")
        ):
            with safe_open(path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith(prefix) or key.startswith(f"model.layers.{LAYER}."):
                        raw[key] = f.get_tensor(key)
        assert raw, "no layer tensors found"
        sd = remap_qwen36_state_dict(raw)
        hf_state = {
            k[len(prefix) :] if k.startswith(prefix) else k[len(f"model.layers.{LAYER}.") :]: v for k, v in raw.items()
        }
        cfg = AutoConfig.from_pretrained(args.CKPT_DIR, trust_remote_code=True, local_files_only=True)
        text = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg
        text._attn_implementation = "eager"
        hf = Qwen3_5DecoderLayer(text, layer_idx=LAYER)
        missing, unexpected = hf.load_state_dict(hf_state, strict=False)
        assert not missing and not unexpected, (missing, unexpected)
        hf = hf.to(torch.bfloat16).eval()
        rot = Qwen3_5TextRotaryEmbedding(text)

        layer = Qwen36DecoderLayer(dev, args, sd, layer_num=LAYER)
        rope = Qwen36RoPESetup(dev, args)
        results = {}
        for B, lens in ((1, [37]), (4, [5, 37, 129, 300])):
            max_seq = 512
            bps = max_seq // BLOCK
            shape = (B * bps, args.n_kv_heads, BLOCK, args.head_dim)
            k = ttnn.zeros(shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
            v = ttnn.zeros(shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
            layer.attention.set_paged_kv_cache(k, v)
            pt = torch.randperm(B * bps).to(torch.int32).reshape(B, bps)
            pt_dev = ttnn.from_torch(pt, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
            caches, refs = [], []
            dec = (torch.randn(B, 1, args.dim) * 0.5).to(torch.bfloat16)
            for u, n in enumerate(lens):
                hidden = (torch.randn(1, n, args.dim) * 0.5).to(torch.bfloat16)
                cache = DynamicCache()
                pos = torch.arange(n).reshape(1, n)
                pe = rot(hidden, pos)
                mask = torch.triu(torch.full((1, 1, n, n), torch.finfo(torch.float32).min), diagonal=1).to(
                    torch.bfloat16
                )
                with torch.no_grad():
                    hf(hidden, position_embeddings=pe, attention_mask=mask, past_key_values=cache)
                    refs.append(
                        hf(
                            dec[u : u + 1],
                            position_embeddings=rot(dec[u : u + 1], torch.tensor([[n]])),
                            attention_mask=None,
                            past_key_values=cache,
                        )
                    )
                n_pad = -(-n // 128) * 128
                x = torch.nn.functional.pad(hidden, (0, 0, 0, n_pad - n))
                x_t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
                cos = ttnn.slice(rope.cos_device, [0, 0, 0], [1, n_pad, args.rope_head_dim])
                sin = ttnn.slice(rope.sin_device, [0, 0, 0], [1, n_pad, args.rope_head_dim])
                n_blocks = -(-n // BLOCK)
                row = ttnn.slice(pt_dev, [u, 0], [u + 1, bps])
                chunk_pt = ttnn.slice(row, [0, 0], [1, n_blocks])
                layer.forward(
                    x_t, cos=cos, sin=sin, mode="prefill", page_table=row, chunk_page_table=chunk_pt, chunk_start_idx=0
                )
            pos_t = torch.tensor(lens)
            cos_h = rope.cos_cpu[pos_t].reshape(B, 1, -1)
            sin_h = rope.sin_cpu[pos_t].reshape(B, 1, -1)
            out = layer.forward(
                ttnn.from_torch(dec, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev),
                ttnn.from_torch(cos_h, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev),
                ttnn.from_torch(sin_h, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev),
                mode="decode",
                position_tensor=ttnn.from_torch(
                    pos_t.to(torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev
                ),
                page_table=pt_dev,
            )
            got = ttnn.to_torch(out)
            for u in range(B):
                ok, pcc = comp_pcc(refs[u].float(), got[u : u + 1].float(), 0.99)
                print(f"9B layer3 paged decode B={B} user {u} pos={lens[u]}: PCC {pcc}")
                results[(B, u)] = float(str(pcc).split()[-1])
            layer.attention.use_paged_attention = False
            ttnn.deallocate(k)
            ttnn.deallocate(v)
        bad = {k: v for k, v in results.items() if v < 0.99}
        print("RESULT", "PASS" if not bad else f"FAIL {bad}")
        return 0 if not bad else 1
    finally:
        ttnn.close_mesh_device(dev)


if __name__ == "__main__":
    raise SystemExit(main())

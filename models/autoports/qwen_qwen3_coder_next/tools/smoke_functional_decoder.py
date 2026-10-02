# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Quick on-device smoke for FunctionalDecoder: real weights, one layer per kind, HF PCC. Not a gate."""
import argparse
import os
import time

import torch

import ttnn
from models.autoports.qwen_qwen3_coder_next.tests import reference as ref
from models.autoports.qwen_qwen3_coder_next.tt.functional_decoder import FunctionalDecoder
from models.common.utility_functions import comp_pcc

os.environ.setdefault("HF_MODEL", "Qwen/Qwen3-Coder-Next")


def run(device, layer_idx, seq_len, batch=1, block_size=64):
    cfg, text = ref.text_config()
    t0 = time.time()
    state = ref.real_layer_state(layer_idx)
    print(f"[L{layer_idx}] read weights {time.time()-t0:.1f}s")
    t0 = time.time()
    hf = ref.torch_layer(text, layer_idx, state)
    print(f"[L{layer_idx}] hf layer {time.time()-t0:.1f}s")
    t0 = time.time()
    tt = FunctionalDecoder.from_state_dict(
        state, hf_config=cfg, layer_idx=layer_idx, mesh_device=device, max_seq_len=8192
    )
    print(f"[L{layer_idx}] tt layer {time.time()-t0:.1f}s kind={tt.layer_type}")
    del state

    torch.manual_seed(7)
    hidden = (torch.randn(batch, seq_len, text.hidden_size) * 0.5).to(torch.bfloat16)
    dec_hidden = (torch.randn(batch, 1, text.hidden_size) * 0.5).to(torch.bfloat16)
    max_seq = ((seq_len + 1 + block_size - 1) // block_size) * block_size + block_size
    if tt.layer_type == "full_attention":
        cache = ref.DynamicCache()
        pos = torch.arange(seq_len).reshape(1, -1).expand(batch, -1)
        pe = ref.rotary(text, hidden, pos)
        with torch.no_grad():
            ref_pre = hf(
                hidden, position_embeddings=pe, attention_mask=ref.causal_mask(batch, seq_len), past_key_values=cache
            )
        tt.allocate_kv_cache(max_batch_size=batch, max_seq_len=max_seq, block_size=block_size)
        bps = tt.kv_cache["blocks_per_seq"]
        page_table = torch.randperm(bps * batch, dtype=torch.int32).reshape(batch, bps)
        pt_dev = ttnn.from_torch(page_table, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        x = ttnn.from_torch(hidden, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        t0 = time.time()
        out = tt.prefill_forward(x, page_table=pt_dev, user_id=0)
        print(f"[L{layer_idx}] prefill T={seq_len} {time.time()-t0:.2f}s")
        got = ttnn.to_torch(out)
        ok, pcc = comp_pcc(ref_pre, got, 0.995)
        print(f"[L{layer_idx}] PREFILL PCC {pcc}")
        # decode at position seq_len
        dpos = torch.full((batch, 1), seq_len, dtype=torch.long)
        pe_d = ref.rotary(text, dec_hidden, dpos)
        with torch.no_grad():
            ref_dec = hf(dec_hidden, position_embeddings=pe_d, attention_mask=None, past_key_values=cache)
        cos_h, sin_h = tt.rope_decode_host(dpos)
        cos = ttnn.from_torch(cos_h, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        sin = ttnn.from_torch(sin_h, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        cur = ttnn.from_torch(
            torch.full((batch,), seq_len, dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )
        xd = ttnn.from_torch(dec_hidden, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        t0 = time.time()
        outd = tt.decode_forward(xd, current_pos=cur, page_table=pt_dev, cos=cos, sin=sin)
        print(f"[L{layer_idx}] decode {time.time()-t0:.2f}s")
        ok, pcc = comp_pcc(ref_dec, ttnn.to_torch(outd), 0.995)
        print(f"[L{layer_idx}] DECODE PCC {pcc}")
    else:
        lc = ref.LinearCache(layer_idx)
        pos = torch.arange(seq_len).reshape(1, -1).expand(batch, -1)
        pe = ref.rotary(text, hidden, pos)
        with torch.no_grad():
            ref_pre = hf(hidden, position_embeddings=pe, attention_mask=None, past_key_values=lc)
        tt.allocate_gdn_state(batch_size=batch)
        x = ttnn.from_torch(hidden, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        t0 = time.time()
        out = tt.prefill_forward(x)
        print(f"[L{layer_idx}] prefill T={seq_len} {time.time()-t0:.2f}s")
        ok, pcc = comp_pcc(ref_pre, ttnn.to_torch(out), 0.995)
        print(f"[L{layer_idx}] PREFILL PCC {pcc}")
        with torch.no_grad():
            ref_dec = hf(dec_hidden, position_embeddings=pe, attention_mask=None, past_key_values=lc)
        xd = ttnn.from_torch(dec_hidden, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        cur = ttnn.from_torch(
            torch.full((batch,), seq_len, dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )
        t0 = time.time()
        outd = tt.decode_forward(xd, current_pos=cur)
        print(f"[L{layer_idx}] decode {time.time()-t0:.2f}s")
        ok, pcc = comp_pcc(ref_dec, ttnn.to_torch(outd), 0.995)
        print(f"[L{layer_idx}] DECODE PCC {pcc}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="0,3")
    ap.add_argument("--seq", type=int, default=37)
    ap.add_argument("--batch", type=int, default=1)
    a = ap.parse_args()
    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=24576, trace_region_size=0, num_command_queues=1)
    try:
        for l in [int(s) for s in a.layers.split(",")]:
            run(dev, l, a.seq, a.batch)
    finally:
        ttnn.close_mesh_device(dev)

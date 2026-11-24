# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""An image prompt through the vLLM generator, driven the way the TT plugin drives it (prefill_forward with the
plugin's mm_items, then one decode_forward per token, greedy on the host logits), without a server, vs the whole HF
model on the same prompt (reference/vision_ref.py --full): the image embeddings, the first token's logits (top-1,
pcc, KL over the reference's top-64), the greedy continuation (tokens equal until the first divergence) and, teacher-
forced on the reference's greedy continuation, how often TT's argmax is the reference's token.

usage: tools/run_dev.sh python -m models.autoports.thinkingmachines_inkling_2_mini.tests.test_vision_generator
env: MESH (1), FIX (/cache/fixtures/vision/full_cat.pt), NEW (128), SLOTS (2), CTX (4096)"""

import os

import torch

import ttnn
from models.autoports.thinkingmachines_inkling_2_mini.tt import generator_vllm as G


def pcc(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def main():
    N, NEW = int(os.environ.get("MESH", "1")), int(os.environ.get("NEW", "128"))
    if N > 1:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, N), trace_region_size=600 << 20)
    gen = G.Inkling2ForCausalLM.initialize_vllm_model(None, mesh, int(os.environ.get("SLOTS", "2")),
                                                      max_seq_len=int(os.environ.get("CTX", "4096")))
    gen.warmup_model_prefill()
    gen.warmup_model_decode()
    ref = torch.load(os.environ.get("FIX", "/cache/fixtures/vision/full_cat.pt"))["none"]
    ids, pv, grid = ref["ids"].long(), ref["pixel_values"], ref["image_grid_thw"]
    o = int((ids == 200054).nonzero()[0])
    n = int(grid.prod())
    emb = gen.vt(pv, grid).float()
    print(f"image embeddings ({n} tokens) vs HF (bf16 model): pcc {pcc(emb, ref['image_embeds']):.6f}", flush=True)
    item = {"modality": "image", "identifier": "cat", "offset": o, "length": n,
            "data": {"pixel_values": pv, "image_grid_thw": grid[0]}}

    def prefill(row, items):
        lg = gen.prefill_forward(ids.reshape(1, -1), None, None, [int(ids.numel())], empty_slots=[row],
                                 mm_items=[items])
        return lg[0, 0].float()

    lg = prefill(0, [item])
    r = ref["logits"].float()
    V = min(lg.numel(), r.numel())
    lg, r = lg[:V], r[:V]
    top = r.topk(64).indices
    p, q = torch.log_softmax(r[top], -1), torch.log_softmax(lg[top], -1)
    kl = float((p.exp() * (p - q)).sum())
    print(f"first token: TT {int(lg.argmax())} HF {int(r.argmax())}; logits pcc {pcc(lg, r):.6f}; KL(top-64) {kl:.2e}; "
          f"top-5 TT {lg.topk(5).indices.tolist()} HF {r.topk(5).indices.tolist()}", flush=True)
    lg_text = prefill(1, None)  # (the same prompt without its image: the placeholders' own embeddings)
    print(f"without the image: logits pcc vs HF {pcc(lg_text[:V], r):.4f} (the image must matter)", flush=True)

    out, p = [int(lg.argmax())], int(ids.numel())
    while len(out) < NEW:
        o_ = gen.decode_forward(tokens=torch.tensor([[out[-1]], [0]]), start_pos=torch.tensor([p, -1]))
        out.append(int(o_[0, 0].argmax()))
        p += 1
    hf = ref["greedy"].tolist()
    agree = next((j for j in range(min(len(out), len(hf))) if out[j] != hf[j]), min(len(out), len(hf)))
    # teacher-forced on HF's continuation: TT's argmax at each position vs HF's greedy token there
    prefill(0, [item])
    p, hit = int(ids.numel()), [int(lg.argmax()) == hf[0]]
    for j in range(len(hf) - 1):
        o_ = gen.decode_forward(tokens=torch.tensor([[hf[j]], [0]]), start_pos=torch.tensor([p, -1]))
        hit.append(int(o_[0, 0].argmax()) == hf[j + 1])
        p += 1
    print(f"teacher-forced on HF's greedy continuation: top-1 agreement {sum(hit)} / {len(hit)} "
          f"(misses at {[j for j, h in enumerate(hit) if not h]})", flush=True)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(G.HF_MODEL)
    print(f"greedy: {agree} / {min(len(out), len(hf))} tokens equal to HF's", flush=True)
    print(f"TT: {tok.decode(out)!r}", flush=True)
    print(f"HF: {tok.decode(hf)!r}", flush=True)
    ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Bit-exactness fingerprint of the served prefill: sha256 of the next-token logits (and the argmax) for fixed prompts
of several lengths, through ``prefill_serving`` (traced buckets, device slot writes). Run it on two builds / env
settings and compare the PREFILL_HASH lines.

    python -m models.autoports.ornith_ai_ornith_1_5_35b_a3b.tools.prefill_hash [T ...]

env: BATCHED=n also hashes n same-length prompts admitted in one call (the batched pass when it is enabled).
"""
import hashlib
import os
import sys

import torch

import ttnn
from models.autoports.ornith_ai_ornith_1_5_35b_a3b.tt.generator import build_generator


def _host(t, mesh):
    return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).float()


def snap(gen, mesh, slot, L):
    """Everything decode reads for request `slot` after its admission: per GDN layer the slot's recurrent state and
    conv taps, per attention layer the K / V rows [0, L) of its page-table blocks."""
    out = []
    for i, layer in enumerate(gen.model.layers):
        a = layer.attention
        if layer.is_full_attention:
            bs = int(a.paged_k.shape[2])
            nb = -(-L // bs)
            row = [int(b) for b in gen.page_table[slot, :nb]]
            for name, c in (("k", a.paged_k), ("v", a.paged_v)):
                blk = torch.cat([_host(ttnn.slice(c, (b, 0, 0, 0), (b + 1, c.shape[1], bs, c.shape[3])), mesh)
                                 for b in row], dim=2)  # [ndev, H, nb*bs, hd]
                out.append((f"L{i}.{name}", blk[:, :, :L]))
        else:
            r = a.rec_state
            out.append((f"L{i}.rec", _host(ttnn.slice(r, (slot, 0, 0, 0), (slot + 1, *[int(r.shape[d]) for d in range(1, len(r.shape))])), mesh)))
            for m, c in enumerate(a.conv_states):
                out.append((f"L{i}.conv{m}", _host(ttnn.slice(c, (0, slot, 0), (1, slot + 1, c.shape[-1])), mesh)))
    return out


def main():
    lens = [int(v) for v in sys.argv[1:]] or [20, 128, 500, 2048, 3000]
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), l1_small_size=24576, trace_region_size=768 << 20)
    try:
        gen = build_generator("models/autoports/ornith_ai_ornith_1_5_35b_a3b", mesh, max_seq_len=16384,
                              max_batch_size=int(os.environ.get("QWEN35_MAX_BATCH", 32)), warm_decode_trace=True)
        n = int(os.environ.get("BATCHED", 1))
        for T in lens:
            g = torch.Generator().manual_seed(T)
            toks = torch.randint(1000, 100000, (n, T), dtype=torch.int64, generator=g)
            # mixed lengths (served bursts are): user u's prompt is the first L_u tokens of its row
            L = [T, max(1, T // 2), max(1, T - 7), max(1, T // 3)][:n] if n > 1 else [T]
            if os.environ.get("LENS"):  # explicit per-user lengths (scaled to T: "a,b,c" are used as-is when T == 0)
                L = [min(T, int(v)) for v in os.environ["LENS"].split(",")][:n]
            slots = [int(v) for v in os.environ.get("SLOTS", ",".join(map(str, range(n)))).split(",")][:n]
            toks = toks.clone()
            for u in range(n):
                toks[u, L[u]:] = 0
            if os.environ.get("PRE") == "1" and n > 1:  # a different batch into other slots right before (served bursts)
                pslots = [s_ for s_ in range(gen.page_table.shape[0]) if s_ not in slots][:n]
                ptoks = torch.randint(1000, 100000, (n, T), dtype=torch.int64, generator=torch.Generator().manual_seed(T + 99))
                gen.prefill_serving(ptoks, gen.page_table[pslots], [T, max(1, T - 3), max(1, T // 2), max(1, T // 4)][:n], pslots)
            lg = gen.prefill_serving(toks, gen.page_table[slots], L, slots)
            lg = torch.as_tensor(lg).float().reshape(n, -1)
            sb = [snap(gen, mesh, slots[u], L[u]) for u in range(n)] if n > 1 else None
            for u in range(n):
                h = hashlib.sha256(lg[u].numpy().tobytes()).hexdigest()[:16]
                print(f"PREFILL_HASH T={T} n={n} u={u}: {h} argmax {int(lg[u].argmax())} max {float(lg[u].max()):.4f}",
                      flush=True)
            if n > 1:  # each prompt admitted alone into the same slot: must equal its row of the batched admission
                for u in range(n):
                    la = gen.prefill_serving(toks[u : u + 1, : L[u]], gen.page_table[slots[u] : slots[u] + 1], [L[u]], [slots[u]])
                    la = torch.as_tensor(la).float().reshape(-1)
                    d = float((la - lg[u]).abs().max())
                    print(f"PREFILL_ALONE T={T} u={u}: exact={bool(torch.equal(la, lg[u]))} max|d|={d:.4g} "
                          f"argmax {int(la.argmax())} vs batched {int(lg[u].argmax())}", flush=True)
                    bad = [(nm, float((x - y).abs().max())) for (nm, x), (_, y) in zip(sb[u], snap(gen, mesh, slots[u], L[u]))
                           if not torch.equal(x, y)]
                    print(f"STATE_ALONE T={T} u={u} (len {L[u]}): {len(sb[u]) - len(bad)}/{len(sb[u])} tensors exact; "
                          f"first diffs {bad[:6]}", flush=True)

    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Prefix cache on device: a prompt resumed from a snapshot (tt/prefix.py) + vLLM-style shared global KV blocks
(tt/attention.GlobalPages) must match the same prompt prefilled whole: the last prompt row's logits and 4 batched decode
steps after it. usage: DEV_ENV="-e MESH=4" MESH_DEVICE=P300x2 tools/run_dev.sh python .../tests/test_prefix_device.py
(env: CHUNK prefill chunk, Q snapshot position, P0 / P1 prompt lengths)"""

import json
import os

import torch

import ttnn
from models.autoports.thinkingmachines_inkling_2_mini.tt.batched import BatchedDecoder
from models.autoports.thinkingmachines_inkling_2_mini.tt.model import Inkling2Model
from models.autoports.thinkingmachines_inkling_2_mini.tt.prefix import PrefixStore


def pcc(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def main():
    N = int(os.environ.get("MESH", "1"))
    if N > 1:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, N))
    chunk = int(os.environ.get("CHUNK", "1024"))
    B = 4
    m = Inkling2Model(mesh, max_seq_len=4096, expert_dtype=ttnn.bfloat8_b, kv_dtype=ttnn.bfloat16, prefill_chunk=chunk,
                      slots=B)
    pg = m.pages
    mb = pg.mb
    pg.enable_vllm(B * mb - 8)
    store = PrefixStore(m, 2)
    dec = BatchedDecoder(m, B)
    V = m.V
    g = torch.Generator().manual_seed(0)
    Q = int(os.environ.get("Q", "1280"))
    PA = int(os.environ.get("P0", "1300"))
    PB = int(os.environ.get("P1", "1580"))
    tx = json.load(open(os.environ.get("TEXTS", "/cache/fixtures/texts.json")))  # real text (random ids are chaotic)
    A = torch.tensor(tx["doc_markdown"] + tx["chat_english"])[:PA]
    Bp = torch.cat([A[:Q], torch.tensor(tx["code_python_file"])[: PB - Q]])

    def row(blocks):
        r = torch.zeros(mb, dtype=torch.int32)
        r[: len(blocks)] = torch.tensor(blocks, dtype=torch.int32)
        return r

    nbA = (PA + 63) // 64 + 1
    nbB = (PB + 63) // 64 + 4
    rowA = row(list(range(1, 1 + nbA)))
    rowB_ref = row(list(range(200, 200 + nbB)))
    shared = Q // 64
    rowB_px = row(list(range(1, 1 + shared)) + list(range(400, 400 + nbB - shared)))

    def head(lg):
        return m.logits_host(lg, 1)[0]

    def prefill(ids, slot, rw, b0=0, snap=None):
        pg.set_row(slot, rw, pg.mb)
        m.reset()
        if b0:
            store.restore(snap[0], slot, b0)
        lg = None
        P = int(ids.numel())
        for p0 in range(b0, P, chunk):
            c = ids[p0 : p0 + chunk]
            n = int(c.numel())
            if n % 256:
                c = torch.cat([c, torch.zeros((-n) % 256, dtype=c.dtype)])
            arm = snap is not None and not b0 and p0 < snap[1] <= p0 + n
            if arm:
                store.arm(snap[1] - p0)
            lg = m.forward(c, p0, valid=n, slot=slot, last_only=True, last_row=n - 1)
            if arm:
                store.save(snap[0], slot, snap[1])
        dec.load_slot(slot)
        return head(lg)

    ref = prefill(Bp, 1, rowB_ref)  # B whole, slot 1
    prefill(A, 0, rowA, snap=(1, Q))  # A, snapshot 1 at Q, slot 0
    pg.free(0)  # A finished: its slot's row goes to scratch (its blocks stay cached for B)
    px = prefill(Bp, 2, rowB_px, b0=Q, snap=(1, Q))  # B resumed at Q, slot 2
    p = pcc(ref, px)
    ok = p > 0.999 and int(ref.argmax()) == int(px.argmax())
    print(f"prefill last row: pcc {p:.6f} max|d| {(ref - px).abs().max():.4f} argmax {int(ref.argmax())} / "
          f"{int(px.argmax())}", flush=True)
    # 4 decode steps of slots 1 and 2 on the same tokens (slot 0 idle: its writes land in scratch blocks)
    tok = int(ref.argmax())
    for i in range(4):
        pos = PB + i
        for s, rw in ((1, rowB_ref), (2, rowB_px)):
            pg.set_row(s, rw, pg.mb)
        pg.sync(B)
        dec.set_inputs(torch.tensor([0, tok, tok, 0]), torch.tensor([-1, pos, pos, -1]))
        out = dec.step()
        lg = m.logits_host(out, B)
        p = pcc(lg[1], lg[2])
        ok = ok and p > 0.999 and int(lg[1].argmax()) == int(lg[2].argmax())
        print(f"decode {i}: pcc {p:.6f} argmax {int(lg[1].argmax())} / {int(lg[2].argmax())}", flush=True)
        tok = int(lg[1].argmax())
    print("PASS" if ok else "FAIL", flush=True)
    from models.autoports.thinkingmachines_inkling_2_mini.tt.clean_exit import close

    close(mesh)


if __name__ == "__main__":
    main()

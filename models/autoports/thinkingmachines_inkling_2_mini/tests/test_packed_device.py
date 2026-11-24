# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Packed prefill + prefix cache through the vLLM adapter (tt/generator_vllm.py), vLLM-style block tables:

  1. three prompts in one prefill_forward call (packed chunks; one prompt spans two chunks at a non-chunk boundary)
     vs the same prompts one call each: first-token logits and 3 decode steps must agree;
  2. a prompt resumed from the prefix snapshot the packed call saved (restore directive, shared blocks) vs the same
     prompt prefilled whole.

usage: DEV_ENV="-e MESH=4" MESH_DEVICE=P300x2 tools/run_dev.sh python .../tests/test_packed_device.py"""

import json
import os

os.environ.setdefault("I2M_PREFILL_CHUNK", "1024")
os.environ.setdefault("I2M_TRACE", "0")
os.environ.setdefault("I2M_PREFIX_SLOTS", "2")

import torch  # noqa: E402

import ttnn  # noqa: E402
from models.autoports.thinkingmachines_inkling_2_mini.tt.generator_vllm import Inkling2ForCausalLM  # noqa: E402
from models.autoports.thinkingmachines_inkling_2_mini.tt.model import Inkling2Model  # noqa: E402

B, CTX, BS = 4, 4096, 64


def pcc(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def kl(a, b):
    """KL(softmax(a) || softmax(b)) over the vocab (the logits' meaningful part; full-vocab pcc is dominated by the
    noisy far tail)."""
    la, lb = torch.log_softmax(a.double(), -1), torch.log_softmax(b.double(), -1)
    return float((la.exp() * (la - lb)).sum())


class Blocks:
    def __init__(self, n):
        self.next = 1
        self.n = n

    def take(self, k):
        out = list(range(self.next, self.next + k))
        self.next += k
        assert self.next <= self.n
        return out


def main():
    N = int(os.environ.get("MESH", "1"))
    if N > 1:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, N))
    model = Inkling2Model(mesh, max_seq_len=CTX + 256, expert_dtype=ttnn.bfloat8_b, kv_dtype=ttnn.bfloat16,
                          prefill_chunk=1024, slots=B)
    ad = Inkling2ForCausalLM(model, mesh, B, CTX)
    nb = B * (CTX // BS)
    ad.allocate_kv_cache((nb, 1, BS, 128), None, 24)
    blocks = Blocks(nb)
    V = ad.V
    g = torch.Generator().manual_seed(0)
    lens = [300, 1500, 90]
    tx = json.load(open(os.environ.get("TEXTS", "/cache/fixtures/texts.json")))  # real text (random ids are chaotic)
    src = [tx["chat_english"], tx["doc_markdown"] + tx["code_python_file"], tx["math_gsm8k"]]
    prompts = [torch.tensor(t[:P]) for t, P in zip(src, lens)]
    W = CTX // BS

    def table(rows):
        t = torch.zeros(len(rows), W, dtype=torch.int32)
        for i, r in enumerate(rows):
            t[i, : len(r)] = torch.tensor(r, dtype=torch.int32)
        return t

    def pack(ps):
        t = torch.zeros(len(ps), max(int(p.numel()) for p in ps), dtype=torch.long)
        for i, p in enumerate(ps):
            t[i, : p.numel()] = p
        return t

    def decode(rows, toks, pos, rows_blocks, steps=3):
        """greedy decode of vLLM rows `rows` (all live) -> list of [len(rows), V] logits"""
        out = []
        R = max(rows) + 1
        for i in range(steps):
            tk = torch.zeros(R, dtype=torch.long)
            ps = torch.full((R,), -1, dtype=torch.long)
            for r, t, p in zip(rows, toks, pos):
                tk[r], ps[r] = t, p + i
            pt = table([rows_blocks.get(r, []) for r in range(R)])
            lg = ad.decode_forward(tokens=tk.reshape(-1, 1), start_pos=ps, page_table=pt)
            lg = lg.reshape(R, -1)[:, :V]
            out.append(torch.stack([lg[r] for r in rows]))
            toks = [int(lg[r].argmax()) for r in rows]
        return out

    def finish(R):
        ad.decode_forward(tokens=torch.zeros(R, 1, dtype=torch.long), start_pos=torch.full((R,), -1), page_table=None)

    ok = True
    # 1. packed: the three prompts in one call; prompt 0 saves a prefix snapshot at 256 (snapshot slot 0)
    rowsA = [blocks.take((P + 63) // 64 + 4) for P in lens]
    lgA = ad.prefill_forward(pack(prompts), table(rowsA), None, torch.tensor(lens), empty_slots=[0, 1, 2],
                             prefix_state=[{"req_id": "a", "restore": None, "saves": [(256, 0)]}, None, None])
    lgA = lgA.reshape(3, -1)[:, :V]
    decA = decode([0, 1, 2], [int(lgA[i].argmax()) for i in range(3)], lens, dict(enumerate(rowsA)))
    finish(3)
    # 1b. the same prompts one call each (fresh blocks)
    lgB, decB = [], []
    for i in range(3):
        rows = blocks.take((lens[i] + 63) // 64 + 4)
        lg = ad.prefill_forward(pack([prompts[i]]), table([rows]), None, torch.tensor([lens[i]]), empty_slots=[0])
        lgB.append(lg.reshape(-1)[:V])
        decB.append(decode([0], [int(lgB[-1].argmax())], [lens[i]], {0: rows}))
        finish(1)
    for i in range(3):
        p, d = pcc(lgA[i], lgB[i]), kl(lgA[i], lgB[i])
        good = d < 1e-4 and int(lgA[i].argmax()) == int(lgB[i].argmax())
        print(f"prompt {i} ({lens[i]} tokens): packed vs alone pcc {p:.6f} KL {d:.2e} argmax {int(lgA[i].argmax())} "
              f"/ {int(lgB[i].argmax())}", flush=True)
        for s in range(3):
            q, d = pcc(decA[s][i], decB[i][s][0]), kl(decA[s][i], decB[i][s][0])
            good = good and d < 0.05  # (decode widths 4 vs 1 run different programs: decode-path noise)
            print(f"   decode {s}: pcc {q:.6f} KL {d:.2e}", flush=True)
        ok = ok and good
    # 2. prefix: prompt 0's first 256 tokens + a new tail, resumed from snapshot 0 (prompt 0's first 4 blocks shared)
    tail = torch.tensor(tx["chat_french"][:200])
    pp = torch.cat([prompts[0][:256], tail])
    rows = rowsA[0][:4] + blocks.take(8)
    lgP = ad.prefill_forward(pack([pp]), table([rows]), None, torch.tensor([pp.numel()]), empty_slots=[0],
                             start_pos=torch.tensor([256]),
                             prefix_state=[{"req_id": "p", "restore": (0, 256), "saves": []}]).reshape(-1)[:V]
    decP = decode([0], [int(lgP.argmax())], [int(pp.numel())], {0: rows})
    finish(1)
    rows = blocks.take(12)
    lgR = ad.prefill_forward(pack([pp]), table([rows]), None, torch.tensor([pp.numel()]),
                             empty_slots=[0]).reshape(-1)[:V]
    decR = decode([0], [int(lgR.argmax())], [int(pp.numel())], {0: rows})
    finish(1)
    p, d = pcc(lgP, lgR), kl(lgP, lgR)
    # (different chunk shapes: rounding differences accumulate over the prompt; 1.7e-3 with the op-path prefill prep,
    # 1.1e-2 with tt/aprep.py's (exact sums, 1-2 bf16 ulps from the op path; 5e-4 / 7e-3 with it on the sliding /
    # global layers only); the exact check below is the resume test proper)
    good = d < 2e-2 and int(lgP.argmax()) == int(lgR.argmax())
    print(f"prefix resume vs whole: pcc {p:.6f} KL {d:.2e} argmax {int(lgP.argmax())} / {int(lgR.argmax())} "
          f"(hits {ad.px_stats})", flush=True)
    for s in range(3):
        q, d = pcc(decP[s][0], decR[s][0]), kl(decP[s][0], decR[s][0])
        good = good and d < 0.05
        print(f"   decode {s}: pcc {q:.6f} KL {d:.2e}", flush=True)
    ok = ok and good
    # 3. exactness: a snapshot at a chunk boundary (1024) of a request prefilled alone, resumed with the same chunk
    #    shapes as the whole prefill ([0, 1024) + the tail): the logits must be identical
    X = torch.tensor(tx["doc_markdown"] + tx["code_mbpp"])[:1100]
    rowsX = blocks.take(22)
    ad.prefill_forward(pack([X]), table([rowsX]), None, torch.tensor([1100]), empty_slots=[0],
                       prefix_state=[{"req_id": "x", "restore": None, "saves": [(1024, 1)]}])
    finish(1)
    pq = torch.cat([X[:1024], torch.tensor(tx["chat_chinese"][:150])])
    rows = rowsX[:16] + blocks.take(6)
    lgE = ad.prefill_forward(pack([pq]), table([rows]), None, torch.tensor([pq.numel()]), empty_slots=[0],
                             start_pos=torch.tensor([1024]),
                             prefix_state=[{"req_id": "e", "restore": (1, 1024), "saves": []}]).reshape(-1)[:V]
    finish(1)
    rows = blocks.take(22)
    lgW = ad.prefill_forward(pack([pq]), table([rows]), None, torch.tensor([pq.numel()]),
                             empty_slots=[0]).reshape(-1)[:V]
    finish(1)
    dmax = float((lgE - lgW).abs().max())
    print(f"exact resume at a chunk boundary: max|d| {dmax:.3g} (must be 0)", flush=True)
    ok = ok and dmax == 0.0
    print("PASS" if ok else "FAIL", flush=True)
    from models.autoports.thinkingmachines_inkling_2_mini.tt.clean_exit import close

    close(mesh)


if __name__ == "__main__":
    main()

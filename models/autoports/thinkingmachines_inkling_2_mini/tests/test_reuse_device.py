# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Served-sequence regression through the vLLM adapter with decode traces and vLLM-style block tables: request A on
slot 0 (prefill + greedy decode), finished; request B on slot 0 (its blocks reuse A's ids); then B again on slot 0 and
B on another slot. B's greedy tokens and first-step logits must be the same every time (stale state / races between
requests show up as differences). Repeats the whole sequence REPS times.
usage: DEV_ENV="-e MESH=4" MESH_DEVICE=P300x2 tools/run_dev.sh python .../tests/test_reuse_device.py"""

import json
import os

os.environ.setdefault("I2M_PREFILL_CHUNK", "1024")

import torch  # noqa: E402

import ttnn  # noqa: E402
from models.autoports.thinkingmachines_inkling_2_mini.tt.generator_vllm import Inkling2ForCausalLM  # noqa: E402
from models.autoports.thinkingmachines_inkling_2_mini.tt.model import Inkling2Model  # noqa: E402

B, CTX, BS = int(os.environ.get("B", "8")), 4096, 64
STEPS = int(os.environ.get("STEPS", "40"))
SAMPLE = os.environ.get("SAMPLE", "1") == "1"
STALE = os.environ.get("STALE", "1") == "1"  # emulate vLLM's uncleared block-table rows


def main():
    N = int(os.environ.get("MESH", "1"))
    if N > 1:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, N), trace_region_size=300 << 20)
    model = Inkling2Model(mesh, max_seq_len=CTX + 256, expert_dtype=ttnn.bfloat8_b, kv_dtype=ttnn.bfloat16,
                          prefill_chunk=1024, slots=B)
    ad = Inkling2ForCausalLM(model, mesh, B, CTX)
    nb = B * (CTX // BS)
    ad.allocate_kv_cache((nb, 1, BS, 128), None, 24)
    ad.warmup_model_prefill()
    ad.warmup_model_decode()
    tx = json.load(open("/cache/fixtures/texts.json"))
    A = torch.tensor(tx["chat_english"][:120])
    Bq = torch.tensor(tx["math_gsm8k"][:106])
    W = CTX // BS
    free = list(range(1, nb))

    def run(prompt, row, R):
        """prefill + STEPS greedy decode steps of one request in vLLM row `row` (rows < R live: only this one) ->
        (first-token logits, tokens)"""
        P = int(prompt.numel())
        blocks = [free.pop(0) for _ in range((P + STEPS + 63) // 64 + 1)]
        pt = torch.zeros(1, W, dtype=torch.int32)
        n_alloc = (P + 63) // 64  # vLLM's allocation for the prompt; later blocks appear as decode reaches them
        pt[0, : len(blocks)] = torch.tensor(blocks, dtype=torch.int32)
        if STALE:  # a reused vLLM row: entries past the allocation hold stale ids -- here this request's own blocks
            for j in range(n_alloc, W):
                pt[0, j] = blocks[j % n_alloc]
        lg = ad.prefill_forward(prompt.reshape(1, -1), pt, None, torch.tensor([P]), empty_slots=[row]).reshape(-1)
        toks = [int(lg[: ad.V].argmax())]
        for i in range(STEPS):
            tk = torch.zeros(R, dtype=torch.long)
            ps = torch.full((R,), -1, dtype=torch.long)
            tk[row], ps[row] = toks[-1], P + i
            ptab = torch.zeros(R, W, dtype=torch.int32)
            ptab[row] = pt[0]
            if STALE:  # the decode's real allocation: the blocks up to position P + i, stale ids after
                nb_i = (P + i) // 64 + 1
                ptab[row, :nb_i] = torch.tensor(blocks[:nb_i], dtype=torch.int32)
            if SAMPLE:  # the served path: device top-k + host greedy pick
                sp = __import__("types").SimpleNamespace(temperature=[0.0] * R, top_k=None, top_p=None, seed=None,
                                                         repetition_penalty=None, frequency_penalty=None,
                                                         presence_penalty=None)
                out = ad.decode_forward(tokens=tk.reshape(-1, 1), start_pos=ps, page_table=ptab, sampling_params=sp)
                toks.append(int(out.reshape(R, -1)[row, 0]))
            else:
                out = ad.decode_forward(tokens=tk.reshape(-1, 1), start_pos=ps, page_table=ptab)
                toks.append(int(out.reshape(R, -1)[row, : ad.V].argmax()))
        ad.decode_forward(tokens=torch.zeros(R, 1, dtype=torch.long), start_pos=torch.full((R,), -1), page_table=None)
        free.extend(blocks)
        return lg[: ad.V], toks

    ok = True
    ref = run(Bq, 0, 1)  # B as the first request after the warm-up: the reference
    print(f"B first: tokens {ref[1][:12]}", flush=True)
    for rep in range(int(os.environ.get("REPS", "3"))):
        run(A, 0, 1)
        for name, row, R in (("B after A, slot 0", 0, 1), ("B again, slot 0", 0, 1), ("B on row 3", 3, 4)):
            lg, toks = run(Bq, row, R)
            d = (lg - ref[0]).abs().max().item()
            same = toks == ref[1]
            first = next((i for i, (a, b) in enumerate(zip(toks, ref[1])) if a != b), None)
            print(f"rep {rep} {name}: first-token logits max|d| {d:.4g}, tokens same {same} (first diff {first})",
                  flush=True)
            ok = ok and d == 0 and same
    print("PASS" if ok else "FAIL", flush=True)
    from models.autoports.thinkingmachines_inkling_2_mini.tt.clean_exit import close

    close(mesh)


if __name__ == "__main__":
    main()

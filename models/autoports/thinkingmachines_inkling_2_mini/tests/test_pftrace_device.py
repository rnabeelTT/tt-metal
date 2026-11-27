# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Traced prefill chunks (I2M_PF_TRACE, tt/model.py PrefillTrace) as the server runs them: each prompt prefilled with
the prefill traces replayed and then eagerly (same slot, fresh blocks), with a prefix snapshot, 3 decode steps (the
decode traces) and the idle step that frees the slot; then the same prompt resumed from its snapshot both ways (a
chunk at p0 > 0). Prefill logits and every decode step's output must be bitwise equal.

usage: DEV_ENV="-e MESH=4 -e I2M_PREFILL_CHUNK=2048 -e LENS=100,300,450,2100" tools/run_dev.sh python .../tests/test_pftrace_device.py
(TRACE_MB: the trace region; TT_METAL_TRACE_ALLOC_TRACKING=1 checks the replays)"""

import json
import os
import types

os.environ.setdefault("I2M_PF_TRACE", "1")
os.environ.setdefault("I2M_PREFIX_SLOTS", "2")

import torch  # noqa: E402

import ttnn  # noqa: E402
from models.autoports.thinkingmachines_inkling_2_mini.tt.generator_vllm import Inkling2ForCausalLM  # noqa: E402
from models.autoports.thinkingmachines_inkling_2_mini.tt.model import Inkling2Model  # noqa: E402

B, CTX, BS = 4, 8192, 64


def main():
    N = int(os.environ.get("MESH", "1"))
    if N > 1:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, N), trace_region_size=int(os.environ.get("TRACE_MB", "256")) << 20)
    chunk = int(os.environ.get("I2M_PREFILL_CHUNK", "1024"))
    model = Inkling2Model(mesh, max_seq_len=CTX + 256, expert_dtype=ttnn.bfloat8_b, kv_dtype=ttnn.bfloat16,
                          prefill_chunk=chunk, slots=B, kv_pool_tokens=B * CTX)
    ad = Inkling2ForCausalLM(model, mesh, B, CTX)
    nb = B * (CTX // BS)
    ad.allocate_kv_cache((nb, 1, BS, 128), None, 24)
    ad.warmup_model_prefill()
    ad.warmup_model_decode()
    pft = model.pft
    assert pft is not None and pft.tid, "no prefill traces (I2M_PF_TRACE=1)"
    print(f"prefill traces: {pft.Ts}", flush=True)
    V = ad.V
    W = CTX // BS
    nxt = [1]

    def take(k):
        out = list(range(nxt[0], nxt[0] + k))
        nxt[0] += k
        if nxt[0] > nb:  # (blocks reused: every prompt writes its own before it reads them)
            nxt[0] = 1
            return take(k)
        return out

    def table(r):
        t = torch.zeros(1, W, dtype=torch.int32)
        t[0, : len(r)] = torch.tensor(r, dtype=torch.int32)
        return t

    def finish():
        ad.decode_forward(tokens=torch.zeros(1, 1, dtype=torch.long), start_pos=torch.full((1,), -1), page_table=None)

    spec = ad.dr is not None
    sp = types.SimpleNamespace(temperature=[0.0], top_k=[-1], top_p=[1.0], seed=[-1])

    def decode(tok, pos, rows, steps=3):
        """3 steps' logits; speculative (I2M_SPEC): 12 passes' emitted tokens (-1 padded: the drafts kept per pass
        show whether the drafter saw the same prefill)"""
        out = []
        for i in range(12 if spec else steps):
            kw = dict(sampling_params=sp) if spec else {}
            o = ad.decode_forward(tokens=torch.tensor([[tok]]), start_pos=torch.tensor([pos]), page_table=table(rows),
                                  **kw)
            o = torch.as_tensor(o).reshape(-1)
            out.append(o.clone())
            if spec:
                new = [int(x) for x in o if int(x) >= 0]
                tok, pos = new[-1], pos + len(new)
            else:
                tok, pos = int(o[:V].argmax()), pos + 1
        finish()
        return out

    def prefill(X, rows, name, q, k, resume=False):
        P = int(X.numel())
        kw = dict(prefix_state=[{"req_id": name, "restore": (k, q) if resume else None,
                                 "saves": [] if resume or q <= 0 else [(q, k)]}])
        if resume:
            kw["start_pos"] = torch.tensor([q])
        return ad.prefill_forward(X.reshape(1, -1), table(rows), None, torch.tensor([P]), empty_slots=[0],
                                  **kw).reshape(-1)[:V]

    tx = json.load(open(os.environ.get("TEXTS", "/cache/fixtures/texts.json")))
    allt = sum((tx[k] for k in ("code_python_file", "doc_markdown", "chat_english", "code_mbpp", "math_gsm8k",
                                "chat_french", "problems_livecodebench", "chat_chinese")), [])
    ok = True
    replays = [0]
    run0 = pft.run

    def counted(*a, **k):
        r = run0(*a, **k)
        replays[0] += r is not None
        return r

    pft.run = counted
    for ci, P in enumerate(int(v) for v in os.environ.get("LENS", "100,300,450,2100").split(",")):
        X = torch.tensor(allt[ci * 97 : ci * 97 + P])
        Q = (P - 1) // 64 * 64 if P >= 256 else 0
        res = {}
        for traced in (True, False):
            model.pft = pft if traced else None
            k = int(traced)
            rows = take((P + 63) // 64 + 4)
            r0 = replays[0]
            lg = prefill(X, rows, f"c{ci}{k}", Q, k)
            d = decode(int(lg.argmax()), P, rows)
            out = [lg] + d
            if Q:
                rows2 = rows[: Q // 64] + take((P + 63) // 64 - Q // 64 + 4)
                lgR = prefill(X, rows2, f"r{ci}{k}", Q, k, resume=True)
                out += [lgR] + decode(int(lgR.argmax()), P, rows2)
            res[traced] = (out, replays[0] - r0)
        (a, na), (b, _) = res[True], res[False]
        eq = [torch.equal(x, y) for x, y in zip(a, b)]
        kept = [sum(int((o >= 0).sum()) - 1 for o in r[0] if o.numel() < V) for r in (res[True], res[False])]
        print(f"P {P} Q {Q}: {na} replays; prefill / decode" + (" / resume / decode" if Q else "")
              + f" bitwise: {eq} (prefill max|d| {float((a[0] - b[0]).abs().max()):.3g})"
              + (f"; drafts kept traced / eager {kept[0]} / {kept[1]}" if spec else ""), flush=True)
        ok = ok and all(eq)
    model.pft = pft
    print("PASS" if ok else "FAIL", flush=True)
    from models.autoports.thinkingmachines_inkling_2_mini.tt.clean_exit import close

    close(mesh)


if __name__ == "__main__":
    main()

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Prefix-cache resume as the server does it (tt/generator_vllm.py): a prompt prefilled fresh with a snapshot at its last
64-token boundary before the last token (then 3 greedy decode steps and the idle step that frees its slot), then the
same prompt resumed from that snapshot (shared blocks): last-row logits and 3 decode steps vs the fresh ones. With the
same chunk shapes (the fresh run's last chunk holds Q and pads to the resumed chunk's size) they must be bitwise.

usage: DEV_ENV="-e I2M_PREFILL_CHUNK=1024 -e LENS=1100,1170,3484,2100" tools/run_dev.sh python .../tests/test_resume_device.py
(MESH=N for N dies; RESUMES=n: the same resume n more times, each must equal the first)"""

import json
import os

os.environ.setdefault("I2M_TRACE", "0")
os.environ.setdefault("I2M_PREFIX_SLOTS", "2")

import torch  # noqa: E402

import ttnn  # noqa: E402
from models.autoports.thinkingmachines_inkling_2_mini.tt.generator_vllm import Inkling2ForCausalLM  # noqa: E402
from models.autoports.thinkingmachines_inkling_2_mini.tt.model import Inkling2Model  # noqa: E402

B, CTX, BS = 4, 8192, 64


def kl(a, b):
    la, lb = torch.log_softmax(a.double(), -1), torch.log_softmax(b.double(), -1)
    return float((la.exp() * (la - lb)).sum())


def main():
    N = int(os.environ.get("MESH", "1"))
    if N > 1:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, N))
    chunk = int(os.environ.get("I2M_PREFILL_CHUNK", "1024"))
    pad = int(os.environ.get("I2M_PREFILL_PAD", "256"))
    model = Inkling2Model(mesh, max_seq_len=CTX + 256, expert_dtype=ttnn.bfloat8_b, kv_dtype=ttnn.bfloat16,
                          prefill_chunk=chunk, slots=B)
    ad = Inkling2ForCausalLM(model, mesh, B, CTX)
    nb = B * (CTX // BS)
    ad.allocate_kv_cache((nb, 1, BS, 128), None, 24)
    V = ad.V
    W = CTX // BS
    nxt = [1]

    def take(k):
        out = list(range(nxt[0], nxt[0] + k))
        nxt[0] += k
        assert nxt[0] <= nb
        return out

    def table(r):
        t = torch.zeros(1, W, dtype=torch.int32)
        t[0, : len(r)] = torch.tensor(r, dtype=torch.int32)
        return t

    def finish():  # (the request done: an idle decode step frees its slot, as vLLM's next step does)
        ad.decode_forward(tokens=torch.zeros(1, 1, dtype=torch.long), start_pos=torch.full((1,), -1), page_table=None)

    def decode(tok, pos, rows, steps=3):
        out = []
        for i in range(steps):
            lg = ad.decode_forward(tokens=torch.tensor([[tok]]), start_pos=torch.tensor([pos + i]),
                                   page_table=table(rows)).reshape(1, -1)[0, :V]
            out.append(lg)
            tok = int(lg.argmax())
        finish()
        return out

    def prefill(X, rows, name, q, k, resume=False):
        P = int(X.numel())
        kw = dict(prefix_state=[{"req_id": name, "restore": (k, q) if resume else None,
                                 "saves": [] if resume else [(q, k)]}])
        if resume:
            kw["start_pos"] = torch.tensor([q])
        return ad.prefill_forward(X.reshape(1, -1), table(rows), None, torch.tensor([P]), empty_slots=[0],
                                  **kw).reshape(-1)[:V]

    tx = json.load(open(os.environ.get("TEXTS", "/cache/fixtures/texts.json")))  # real text (random ids are chaotic)
    allt = sum((tx[k] for k in ("code_python_file", "doc_markdown", "chat_english", "code_mbpp", "math_gsm8k",
                                "chat_french", "problems_livecodebench", "chat_chinese")), [])
    ok = True
    for ci, P in enumerate(int(v) for v in os.environ.get("LENS", "1100,1170,3484,2100").split(",")):
        X = torch.tensor(allt[ci * 97 : ci * 97 + P])
        Q = (P - 1) // 64 * 64
        k = ci % 2
        rows = take((P + 63) // 64 + 4)
        lgF = prefill(X, rows, f"f{ci}", Q, k)
        dF = decode(int(lgF.argmax()), P, rows)

        def tail():  # (the prefix blocks shared, fresh blocks after them)
            return rows[: Q // 64] + take((P + 63) // 64 - Q // 64 + 4)

        rows2 = tail()
        lgR = prefill(X, rows2, f"r{ci}", Q, k, resume=True)
        dR = decode(int(lgR.argmax()), P, rows2)
        p0 = (P - 1) // chunk * chunk  # the fresh run's last chunk
        exact = p0 <= Q and -(-(P - p0) // pad) == -(-(P - Q) // pad)
        d, dm = kl(lgF, lgR), float((lgF - lgR).abs().max())
        good = int(lgF.argmax()) == int(lgR.argmax()) and (dm == 0.0 if exact else d < 2e-2)
        print(f"P {P} Q {Q} chunk {chunk}{' (exact)' if exact else ''}: resumed vs fresh KL {d:.2e} max|d| {dm:.3g} "
              f"argmax {int(lgF.argmax())} / {int(lgR.argmax())}", flush=True)
        for s in range(3):
            d = kl(dF[s], dR[s])
            good = good and (d == 0.0 if exact else d < 0.05) and int(dF[s].argmax()) == int(dR[s].argmax())
            print(f"   decode {s}: KL {d:.2e} argmax {int(dF[s].argmax())} / {int(dR[s].argmax())}", flush=True)
        for i in range(int(os.environ.get("RESUMES", "0"))):
            lg3 = prefill(X, tail(), f"r{ci}.{i}", Q, k, resume=True)
            finish()
            dm3 = float((lg3 - lgR).abs().max())
            good = good and dm3 == 0.0
            print(f"   resume again {i}: max|d| vs the first resume {dm3:.3g}", flush=True)
        ok = ok and good
    print(f"stats {ad.px_stats}", flush=True)
    print("PASS" if ok else "FAIL", flush=True)
    from models.autoports.thinkingmachines_inkling_2_mini.tt.clean_exit import close

    close(mesh)


if __name__ == "__main__":
    main()

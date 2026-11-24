# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The trace-safe decode path (tt/decode.Hy4Decoder) vs the eager forward, teacher forced on the same tokens.

    python -m models.autoports.tencent_hy4_flash.tools.check_decode --prompt-len 100 --steps 24 [--trace]

Prefills `prompt-len` random-text tokens, then feeds `steps` more tokens one at a time through (a) model.forward and
(b) the decoder (after a fresh prefill); prints per-step logits PCC / top-1 agreement and the step times.
"""

from __future__ import annotations

import argparse
import time

import torch
import ttnn
from loguru import logger


def pcc(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


def main():
    import faulthandler
    import signal

    faulthandler.register(signal.SIGUSR1, all_threads=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-len", type=int, default=100)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--layers", default="")
    ap.add_argument("--no-engram", action="store_true")
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--texts", default="", help="tf_texts.json: take the tokens from its first text instead")
    ap.add_argument("--rows", type=int, default=1, help="rows per decoder pass (verify width); checks all rows")
    args = ap.parse_args()
    from models.autoports.tencent_hy4_flash.tools.generate import build, open_mesh
    from models.autoports.tencent_hy4_flash.tt.decode import Hy4Decoder

    mesh = open_mesh()
    try:
        model = build(mesh, args)
        g = torch.Generator().manual_seed(0)
        if args.texts:
            import json

            seqs = json.load(open(args.texts))
            ids = torch.tensor(sum(seqs.values(), []))[: args.prompt_len + args.steps]
        else:
            ids = torch.randint(1000, 100000, (args.prompt_len + args.steps,), generator=g)
        P, S = args.prompt_len, args.steps

        def prefill():
            model.reset()
            lg = None
            for c0 in range(0, P, args.chunk):
                lg = model.forward(ids[c0 : min(P, c0 + args.chunk)])
            return lg

        prefill()
        ref, t_ref = [], []
        for i in range(S):
            t0 = time.time()
            ref.append(model.forward(ids[P + i : P + i + 1]))
            t_ref.append(time.time() - t0)
        prefill()
        dec = Hy4Decoder(model, trace=args.trace)
        dec.start(P)
        dec.set_history(ids[:P].tolist())
        got, t_got = [], []
        R = args.rows
        for i in range(0, S, R):
            t0 = time.time()
            n = min(R, S - i)
            out = dec.run([int(t) for t in ids[P + i : P + i + n]], logits_rows="all")
            got += [out[j] for j in range(n)]
            t_got += [(time.time() - t0)] * n
            tm = getattr(dec, "timing", None)
            logger.info(f"decoder pass {i // R} ({n} rows): {t_got[-1] * 1e3:.0f} ms" + (
                f" (host inputs {tm[0] * 1e3:.1f}, copies {tm[1] * 1e3:.1f}, device {tm[2] * 1e3:.1f}, read "
                f"{tm[3] * 1e3:.1f})" if tm else ""))
        for i in range(S):
            print(f"step {i:3d} pos {P + i:6d}: pcc {pcc(got[i], ref[i]):.5f} top1 {int(got[i].argmax())} vs "
                  f"{int(ref[i].argmax())}  ms eager {t_ref[i] * 1e3:.0f} decoder {t_got[i] * 1e3:.0f}")
        agree = sum(int(got[i].argmax()) == int(ref[i].argmax()) for i in range(S)) / S
        logger.info(f"top1 agreement {agree:.3f}; median ms eager {sorted(t_ref)[S // 2] * 1e3:.1f} "
                    f"decoder {sorted(t_got)[S // 2] * 1e3:.1f}")
        dec.release()
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()

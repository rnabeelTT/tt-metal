# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DFly2 speculative decoding vs plain greedy decode on chat prompts: tokens/s, tokens per verify pass, and whether the
speculative output equals the plain greedy output token for token.

    python -m models.autoports.tencent_hy4_flash.tools.spec_bench [--max-new 128] [--prompts N] [--no-plain]
"""

from __future__ import annotations

import argparse
import os
import time

import torch
import ttnn
from loguru import logger

PROMPTS = [
    "Write a Python function that checks whether a string is a palindrome, with a short docstring.",
    "Explain in four sentences why the sky is blue.",
    "List five practical tips for writing clear technical documentation.",
    "What is the derivative of x^3 * sin(x)? Show the steps briefly.",
    "Translate into French: 'The meeting has been moved to Thursday afternoon.'",
    "Summarize the plot of Romeo and Juliet in three sentences.",
    "Write a SQL query that returns the top 3 customers by total order value.",
    "Give a haiku about autumn rain.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--prompts", type=int, default=4)
    ap.add_argument("--no-plain", action="store_true")
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--layers", default="")
    ap.add_argument("--no-engram", action="store_true")
    args = ap.parse_args()
    from transformers import AutoTokenizer

    from models.autoports.tencent_hy4_flash.tools.generate import build, open_mesh
    from models.autoports.tencent_hy4_flash.tt.config import checkpoint_dir
    from models.autoports.tencent_hy4_flash.tt.decode import Hy4Decoder, Hy4SpecDecoder

    tok = AutoTokenizer.from_pretrained(checkpoint_dir())
    mesh = open_mesh()
    try:
        model = build(mesh, args)
        spec = Hy4SpecDecoder(model, trace=not args.eager)
        plain = None if args.no_plain else Hy4Decoder(model, trace=not args.eager)
        eos = model.cfg.eos
        tot = {"spec_tok": 0, "spec_t": 0.0, "passes": 0, "plain_tok": 0, "plain_t": 0.0, "match": 0, "n": 0}
        for pi, prompt in enumerate(PROMPTS[: args.prompts]):
            text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                           add_generation_prompt=True, reasoning_effort="no_think")
            ids = tok(text, add_special_tokens=False)["input_ids"]
            P = len(ids)

            def prefill():
                model.reset()
                if model.engram is not None:
                    model.engram.reset()
                lg = None
                for c0 in range(0, P, args.chunk):
                    lg = model.forward(torch.tensor(ids[c0 : c0 + args.chunk]))
                return int(torch.argmax(lg))

            # ---- plain greedy
            ref = None
            if plain is not None:
                b = prefill()
                plain.start(P)
                plain.set_history(ids)
                ref = [b]
                t0 = time.time()
                while len(ref) < args.max_new and ref[-1] != eos:
                    ref.append(int(torch.argmax(plain.run([ref[-1]]))))
                dt = time.time() - t0
                if pi > 0:
                    tot["plain_tok"] += len(ref) - 1
                    tot["plain_t"] += dt
            # ---- speculative
            b = prefill()
            spec.start(P)
            spec.set_history(ids)
            out = [b]
            t0 = time.time()
            lg = spec.run([b])  # row 0 = the bonus at position P
            nxt = int(torch.argmax(lg))
            out.append(nxt)
            anchor, n = P + 1, 0
            passes = 0
            while len(out) < args.max_new and out[-1] != eos:
                drafts = spec.draft(out[-1], anchor, n)
                rows = [out[-1]] + drafts
                lgs = spec.run(rows, logits_rows="all", p0=anchor)
                picks = lgs.argmax(-1).tolist()
                acc = 0
                while acc < len(drafts) and picks[acc] == drafts[acc]:
                    acc += 1
                emit = drafts[:acc] + [picks[acc]]
                if os.environ.get("HY4_SPEC_DEBUG") == "1" and passes < 6:
                    logger.info(f"anchor {anchor} n {n}: drafts {drafts} picks {picks} acc {acc}")
                passes += 1
                for t in emit:
                    out.append(t)
                    if t == eos or len(out) >= args.max_new:
                        break
                anchor += acc + 1
                n = acc
            dt = time.time() - t0
            if pi > 0:  # the first prompt compiles / captures
                tot["spec_tok"] += len(out) - 1
                tot["spec_t"] += dt
                tot["passes"] += passes
            same = ref is not None and out[: len(ref)] == ref[: len(out)]
            tot["match"] += int(same)
            tot["n"] += 1
            logger.info(f"prompt {pi}: {len(out)} tokens, {passes} verify passes "
                        f"({(len(out) - 2) / max(passes, 1):.2f} tok/pass), {len(out) / dt:.1f} tok/s"
                        + (f", plain match {same}" if ref is not None else ""))
            print(repr(tok.decode(out))[:300])
        if tot["spec_t"]:
            logger.info(f"SPEC: {tot['spec_tok'] / tot['spec_t']:.2f} tok/s, "
                        f"{tot['spec_tok'] / max(tot['passes'], 1):.2f} tok/pass")
        if tot["plain_t"]:
            logger.info(f"PLAIN: {tot['plain_tok'] / tot['plain_t']:.2f} tok/s")
        logger.info(f"spec == plain greedy on {tot['match']} / {tot['n']} prompts")
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()

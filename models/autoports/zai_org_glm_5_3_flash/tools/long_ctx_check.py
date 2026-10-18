# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 8 (datatype sweep) long-context check: next-token logits of one precision candidate on a long natural document.

The document is the repository's English documentation (every docs/**/*.md, *.rst and models/**/README.md file, sorted,
concatenated; ~1M characters), tokenized without special tokens. For each length L (--lengths): prefill the first
L - N tokens, then N traced decode steps teacher-forced with the document's own tokens (host logits, no on-device
sampling). Per position the top-5 token ids and logits are saved, plus the rank of the document's true next token.
tools/long_ctx_compare.py compares two candidates' files (top-1 agreement, top-5 overlap, true-token top-1 / top-5).

usage (one build): GLM53_PRECISION=<json or empty> python -m models.autoports.zai_org_glm_5_3_flash.tools.long_ctx_check \
    --lengths 32768,100000 --steps 200 --seq-len 102400 --out <file.json>
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

import ttnn

REPO = Path(os.environ.get("TT_METAL_HOME", Path(__file__).resolve().parents[4]))


def document_ids(tok, n: int) -> tuple[list[int], list[str]]:
    files = sorted(
        [p for p in (REPO / "docs").rglob("*") if p.suffix in (".md", ".rst")]
        + list((REPO / "models").rglob("README.md"))
    )
    ids, used = [], []
    for p in files:
        try:
            t = p.read_text(errors="ignore")
        except OSError:
            continue
        if len(t) < 2000:
            continue
        ids += tok.encode(t + "\n\n", add_special_tokens=False)
        used.append(str(p.relative_to(REPO)))
        if len(ids) >= n:
            break
    assert len(ids) >= n, f"document too short: {len(ids)} < {n}"
    return ids[:n], used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", default="32768,100000")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--seq-len", type=int, default=102400)
    ap.add_argument("--trace-region", type=int, default=64 << 20)
    ap.add_argument("--doc-ids", default=None, help="frozen document token ids (json {ids, files}); built if missing")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from models.autoports.zai_org_glm_5_3_flash.tools.readiness import _open_mesh
    from models.autoports.zai_org_glm_5_3_flash.tt.generator import Glm53Generator
    from models.autoports.zai_org_glm_5_3_flash.tt.precision import PrecisionOverrides

    lengths = [int(x) for x in a.lengths.split(",")]
    rec = {"precision_env": os.environ.get("GLM53_PRECISION", ""), "lengths": lengths, "steps": a.steps, "runs": {}}
    mesh = _open_mesh(a.trace_region)
    try:
        t0 = time.time()
        g = Glm53Generator(mesh, max_batch_size=1, max_seq_len=a.seq_len)
        rec["build_s"] = time.time() - t0
        rec["precision"] = PrecisionOverrides.from_env().record()
        dp = Path(a.doc_ids) if a.doc_ids else None
        if dp is not None and dp.exists():
            d = json.loads(dp.read_text())
            doc, files = d["ids"], d["files"]
        else:
            doc, files = document_ids(g.tokenizer, max(lengths) + 1)
            if dp is not None:
                dp.write_text(json.dumps({"ids": doc, "files": files}))
        rec["document_files"] = len(files)
        rec["document_first_files"] = files[:5]
        V = g.model.vocab
        for L in lengths:
            P = L - a.steps
            g.reset()
            t1 = time.time()
            logits = g.prefill_forward(torch.tensor(doc[:P]).reshape(1, -1), prompt_lens=[P])[0, -1, :V].float()
            t_pf = time.time() - t1
            rows = [logits]
            pos = P
            for i in range(a.steps - 1):
                out = g.decode_forward(
                    torch.tensor([[doc[P + i]]]),
                    torch.tensor([pos]),
                    enable_trace=True,
                    on_device_sampling=False,
                    reload_inputs=True,
                )
                rows.append(out[0, :V].float())
                pos += 1
            lg = torch.stack(rows)  # [steps, V]: row j predicts doc[P + j]
            top5 = torch.topk(lg, 5, dim=-1)
            true = torch.tensor(doc[P : P + a.steps])
            rank = (lg > lg.gather(1, true[:, None])).sum(-1)
            rec["runs"][str(L)] = {
                "prefill_tokens": P,
                "prefill_s": t_pf,
                "top5_ids": top5.indices.tolist(),
                "top5_logits": [[round(float(v), 4) for v in r] for r in top5.values],
                "true_ids": true.tolist(),
                "true_rank": rank.tolist(),
                "true_top1": float((rank == 0).float().mean()),
                "true_top5": float((rank < 5).float().mean()),
                "finite": bool(torch.isfinite(lg).all()),
            }
            print(
                f"LONGCTX L={L} prefill {t_pf:.0f}s true top-1 {rec['runs'][str(L)]['true_top1']:.3f} "
                f"top-5 {rec['runs'][str(L)]['true_top5']:.3f}",
                flush=True,
            )
        g.teardown()
    finally:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(rec) + "\n")
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()

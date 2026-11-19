# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Compare prefill_bench PF_DUMP logits: python tools/pf_cmp.py a.pt b.pt [...] (each against the first)."""
import sys

import torch

ref = torch.load(sys.argv[1]).float()
for p in sys.argv[2:]:
    x = torch.load(p).float()
    d = (x - ref).abs()
    print(f"{p}: identical {torch.equal(x, ref)} | max |d| {d.max().item():.4g} mean {d.mean().item():.4g} | "
          f"argmax {int(ref.argmax())} vs {int(x.argmax())} | top-10 overlap "
          f"{len(set(ref.topk(10).indices.tolist()) & set(x.topk(10).indices.tolist()))}/10")

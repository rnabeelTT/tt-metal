# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""On-device checks of the fused MoE decode ops (models/demos/blackhole/qwen36/tt/moe_decode) against torch.

    python test_moe_decode_ops.py            # all
    python test_moe_decode_ops.py topk
"""

import sys

import torch

import ttnn
from models.demos.blackhole.qwen36.tt.moe_decode.topk import RouterTopK

torch.manual_seed(0)


def check_topk(dev, E=256, k=8, trials=20):
    op = RouterTopK(dev, E, k)
    ids, scores = op.alloc_outputs()
    worst = 0.0
    for t in range(trials):
        logits = torch.randn(1, 1, 32, E) * (0.5 + 3 * torch.rand(1))
        if t % 5 == 4:  # exact bf16 ties inside and at the edge of the top-k
            logits[0, 0, 0, 7] = logits[0, 0, 0, 200] = logits[0, 0, 0, :].max()
        lg = logits.bfloat16()
        x = ttnn.from_torch(lg, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.L1_MEMORY_CONFIG)
        op(x, ids, scores)
        got_ids = ttnn.to_torch(ids).reshape(-1)[:k].long()
        got_w = ttnn.to_torch(scores).float().reshape(-1)[:k]
        row = lg[0, 0, 0].float()
        # reference: descending, ties to the lower id
        order = sorted(range(E), key=lambda i: (-row[i].item(), i))[:k]
        ref_ids = torch.tensor(order)
        ref_w = torch.softmax(row[ref_ids], dim=-1)
        assert torch.equal(got_ids, ref_ids), (t, got_ids.tolist(), ref_ids.tolist())
        err = ((got_w - ref_w).abs() / ref_w.clamp_min(1e-3)).max().item()
        worst = max(worst, err)
        assert err < 1e-2, (t, got_w.tolist(), ref_w.tolist())
        ttnn.deallocate(x)
    print(f"topk E={E} k={k}: {trials} trials, ids exact, max rel weight err {worst:.2e}")


def main():
    what = sys.argv[1:] or ["topk"]
    dev = ttnn.open_device(device_id=0)
    try:
        if "topk" in what:
            check_topk(dev)
            check_topk(dev, E=512, k=10)
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()

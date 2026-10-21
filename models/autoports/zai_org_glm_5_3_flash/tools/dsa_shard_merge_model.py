# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Torch model of the block-sharded DSA merge (tt/dsa_shard.py; doc/multichip_decoder/README.md, "Block-sharded DSA
cache", the "relative error 1.8e-3" line). Host only, no device.

The model uses the bf16 row format (544 columns: 512 latent + ONE (col 512), SINK (col 513), KILL (col 514)):
- the keys are spread over 4 dies at random; on each die the other dies' keys are NULL rows (-1e4 in col 514);
- each list has the 32-entry prefix [SINK, 31 x NULL]; the query is [q, q . c_self (bf16), 1] in cols 512..514;
- per die: softmax over its list, output o rounded to bf16 (the kernel output), u = o[512], w = o[513];
- merge: T = o / w with col 512 replaced by m (1 - w for w < 0.5, else u); o_lat = sum_d T[:512] / sum_d T[512].
Compared with the replicated softmax attention over all keys (fp32). The 2026-10-17 run (seed 0, 2051 keys, 64 heads)
gave relative error 1.836e-3 (Frobenius) and PCC 0.9999985: the bf16 rounding of the kernel output.

Usage: python models/autoports/zai_org_glm_5_3_flash/tools/dsa_shard_merge_model.py [--keys 2051] [--seed 0]
"""

import argparse

import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--keys", type=int, default=2051)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    H, R, K = 64, 512, 544
    tp = 4
    nkeys = args.keys
    lat = torch.randn(nkeys, R)
    q = torch.randn(H, R) * 0.3
    c_self = lat[-1]
    scale = 1 / 16
    # replicated reference
    s = (q @ lat.T) * scale
    ref = torch.softmax(s, -1) @ lat
    owner = torch.randint(0, tp, (nkeys,))

    def bf(x):
        return x.to(torch.bfloat16).float()

    sd = bf(q @ c_self)
    T = torch.zeros(H, K)
    for r in range(tp):
        sink = torch.zeros(K)
        sink[513] = 1
        null = torch.zeros(K)
        null[514] = -1e4
        keys = [sink] + [null] * 31
        for j in range(nkeys):
            if owner[j] == r:
                k = torch.zeros(K)
                k[:R] = lat[j]
                k[512] = 1
                keys.append(k)
            else:
                keys.append(null)
        Kt = torch.stack(keys)
        qx = torch.zeros(H, K)
        qx[:, :R] = q
        qx[:, 513] = sd
        qx[:, 514] = 1
        sc = (qx @ Kt.T) * scale
        p = torch.softmax(sc, -1)
        o = bf(p @ Kt)
        u = o[:, 512:513]
        w = o[:, 513:514].clamp_min(1e-30)
        mass = torch.where(w < 0.5, 1 - w, u)
        o2 = o.clone()
        o2[:, 512:513] = mass
        T += o2 / w
    out = T[:, :R] / T[:, 512:513]
    d = out - ref
    rel = (d.norm() / ref.norm()).item()
    pcc = torch.corrcoef(torch.stack([out.flatten(), ref.flatten()]))[0, 1].item()
    print(f"relative error (Frobenius) {rel:.6g} pcc {pcc:.9f}")


if __name__ == "__main__":
    main()

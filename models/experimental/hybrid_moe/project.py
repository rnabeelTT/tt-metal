# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Hybrid-MoE POC: project traced single-user decode tok/s with host offload, from measured inputs.

Measured on this QB2 (sources):
  bench1.json  per-layer host round trip in a split trace (us/layer)
  bench2.json  host int4 expert kernel GB/s (8 threads)
  bench3 log   host->device copies from DDR under CPU load: ~14 GB/s (see README)
  sim.json     122B decode hit rate vs device capacity (static and dynamic LRU)
  hybrid_poc.json  traced 122B baseline ms/token in the same build, in-situ hit rates / costs
Extrapolated (stated as ranges): device ms/token for models not ported, from the 122B's traced time.

Per token:  t = t_dev + L * t_sync + miss_bytes / (B_cpu + B_pcie)      (q* split, DDR-bound together)
            miss_bytes = L * k * (1 - h) * expert_bytes
PCIe load for a dynamic cache (every miss promoted): miss_bytes / B_pcie / t   (> 1 means promotion-limited)
"""

import json

import numpy as np

BFP4 = 0.5625  # bytes/param, bfp4_b
BFP8 = 1.0625
DEV_BYTES = 127 * 2**30
RESERVE = 15e9  # KV / GDN state / trace / scratch
B_PCIE = 14e9  # bench3: DDR-resident sources, under CPU load

MODELS = {
    # name: (layers_moe, experts, top_k, hidden, moe_inter, total_params)
    "Qwen3.5-122B-A10B (validation)": (48, 256, 8, 3072, 1024, 122e9),
    "Qwen3.5-397B-A17B": (60, 512, 10, 4096, 1024, 397e9),
    "Qwen3-235B-A22B-2507": (94, 128, 8, 4096, 1536, 235e9),
    "MiniMax-M2 (230B-A10B)": (62, 256, 8, 3072, 1536, 230e9),
}
ACTIVE = {
    "Qwen3.5-122B-A10B (validation)": 10e9,
    "Qwen3.5-397B-A17B": 17e9,
    "Qwen3-235B-A22B-2507": 22e9,
    "MiniMax-M2 (230B-A10B)": 10e9,
}


def main():
    b1 = json.load(open("bench1.json"))
    b2 = json.load(open("bench2.json"))
    sim = json.load(open("routing/sim.json"))["results"]
    try:
        poc = json.load(open("hybrid/hybrid_poc.json"))
        # the standalone generate() in this build measured ~30.6 ms (the model card's v1 number); the serving path is 7.6
        t_dev_122 = 7.6e-3
        src = f"model card, serving path (standalone generate in this build: {poc['traced_baseline']['ms_per_token']} ms)"
    except (FileNotFoundError, KeyError):
        poc, t_dev_122, src = None, 7.6e-3, "model card"
    import os

    t_sync = b1["roundtrip_us_per_layer"]["overhead_us_per_layer"] * 1e-6
    if os.environ.get("SYNC") == "asbuilt" and poc:
        m = poc["dyn_0.5"]
        t_sync = (m["ms_readback_per_layer"] + m["ms_upload_add_per_missing_layer"]) / 1e3
    b_cpu = b2["int4_8t"]["GB_per_s"] * 1e9
    caps = sorted(float(c) for c in sim)

    thr = json.load(open("routing/sim_throttle.json"))
    tcaps = sorted({float(k.split("_")[0]) for k in thr})
    tps = sorted({float(k.split("_")[1]) for k in thr})

    def hit(frac, p):
        """dynamic LRU hit rate at device capacity `frac` when a fraction `p` of misses can be promoted
        (122B routing; capacities above the simulated 0.7 are clamped -> conservative)"""
        if frac >= 1:
            return 1.0
        per_cap = [float(np.interp(p, tps, [thr[f"{c}_{q}"] for q in tps])) for c in tcaps]
        return float(np.interp(frac, tcaps, per_cap))

    print(f"inputs: t_dev(122B traced) {t_dev_122 * 1e3:.2f} ms ({src}); sync {t_sync * 1e6:.0f} us/layer; "
          f"CPU int4 {b_cpu / 1e9:.1f} GB/s; PCIe {B_PCIE / 1e9:.0f} GB/s\n")
    out = {}
    for name, (L, E, k, H, N, tot) in MODELS.items():
        eb = 3 * H * N * BFP4
        routed = L * E * 3 * H * N
        pool = routed * BFP4
        nonexp = (tot - routed) * BFP8
        budget = DEV_BYTES - nonexp - RESERVE
        frac = 0.5 if "validation" in name else min(1.0, budget / pool)
        L122 = 48
        if "122B" in name:
            tdev = (t_dev_122, t_dev_122)
        else:  # low: scales with layer count; high: with active params
            tdev = (t_dev_122 * L / L122, t_dev_122 * ACTIVE[name] / 10e9)
        res = []
        for td in tdev:
            # fixed point: promotion share p limited by PCIe time within one token
            p = 1.0
            for _ in range(20):
                h = hit(frac, p)
                miss_b = L * k * (1 - h) * eb
                t = td + L * t_sync + miss_b / (b_cpu + B_PCIE)
                p = min(1.0, B_PCIE * t / max(miss_b, 1.0))
            res.append((t, h, miss_b, p))
        rows = {
            "hit": [round(r[1], 3) for r in res],
            "promoted_share_of_misses": [round(r[3], 2) for r in res],
            "miss_MB_per_token": [round(r[2] / 1e6, 1) for r in res],
            "miss_ms_per_token": [round(r[2] / (b_cpu + B_PCIE) * 1e3, 2) for r in res],
            "tok_s": [round(1 / r[0], 1) for r in res],
        }
        all_dev = [1 / (td) for td in tdev]
        out[name] = {
            "expert_pool_GB": round(pool / 1e9, 1),
            "device_capacity_frac": round(frac, 2),
            "host_RAM_needed_GB": round((1 - frac) * pool / 1e9, 1) if frac < 1 else 0,
            "tok_s_if_all_on_device": [round(min(all_dev), 1), round(max(all_dev), 1)],
            "sync_ms_per_token": round(L * t_sync * 1e3, 2),
            "dynamic": rows,
        }
        print(name, json.dumps(out[name], indent=1))
    json.dump(out, open(f"projection_{os.environ.get('SYNC', 'engineered')}.json", "w"), indent=1)


if __name__ == "__main__":
    main()

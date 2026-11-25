# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Score device / emulated runs against the BF16 reference on the generated tokens only (or every position).

    python compare_gen_hy4.py TEXTS REF_DIR PLEN_JSON|- DIR TAG...

TEXTS {name: ids} (prompt + reference greedy continuation), PLEN_JSON {name: prompt length} ("-": score every position),
DIR holds dev_<TAG>_<name>.npz (tools/batch_gate.py or quant_emul_hy4.py output). Pooled over tokens: top-1 (device argmax
== reference argmax), top-5 / top-32 (device argmax in the reference top-k; top-32 implies top-100), KL over the
reference top 32, top-1 on confident positions (reference p > 0.5). The tt-metal autoport gate counts the same way.
"""
import json, sys, numpy as np, os
tx, rd, pl, d = sys.argv[1:5]; tags = sys.argv[5:]
seqs = json.load(open(tx)); plen = json.load(open(pl)) if pl != "-" else None
for tag in tags:
    tot = {"n": 0, "t1": 0, "t5": 0, "t32": 0, "kl": 0.0, "conf_n": 0, "conf_t1": 0}; per = []
    for n, ids in seqs.items():
        f = os.path.join(d, f"dev_{tag}_{n}.npz")
        if not os.path.exists(f):
            continue
        r = np.load(os.path.join(rd, f"ref_{n}.npz")); v = np.load(f)
        T = len(ids) - 1; s = (plen[n] - 1) if plen else 0
        rid = r["top_ids"][:T].astype(np.int64); lp = r["top_lp"][:T].astype(np.float32)
        top1 = v["top1"][:T]; lpd = v["lp_at_ref"][:T].astype(np.float32)
        sl = slice(s, T)
        a1 = top1[sl] == rid[sl, 0]
        a5 = np.array([top1[i] in rid[i, :5] for i in range(s, T)]); a32 = np.array([top1[i] in rid[i] for i in range(s, T)])
        p = np.exp(lp[sl]); kl = (p * (lp[sl] - lpd[sl])).sum(-1); conf = p[:, 0] > 0.5
        tot["n"] += len(a1); tot["t1"] += a1.sum(); tot["t5"] += a5.sum(); tot["t32"] += a32.sum(); tot["kl"] += kl.sum()
        tot["conf_n"] += conf.sum(); tot["conf_t1"] += a1[conf].sum()
        per.append(f"{n} {a1.mean():.3f}")
    N = tot["n"]
    if N:
        print(f"{tag:10s} tokens {N:4d}  top1 {tot['t1']/N:.3f}  top5 {tot['t5']/N:.3f}  top32 {tot['t32']/N:.3f}  KL {tot['kl']/N:.3f}  "
              f"confident top1 {tot['conf_t1']/max(1,tot['conf_n']):.3f} ({tot['conf_n']} tok) | " + ", ".join(per))

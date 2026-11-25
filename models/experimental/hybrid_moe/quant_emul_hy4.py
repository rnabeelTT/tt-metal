# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Hy4 CPU reference with the device's weight quantization emulated (2026-11-25).

python quant_emul_hy4.py <texts.json> <ref dir> <out dir> <tag> [names,...]
env QA_GU / QA_DN (routed experts gate_up / down), QA_ATTN (q_b, linear_gate, o_proj), QA_DENSE (layer-0 MLP):
    bf16 (default, untouched) | bfp8 | bfp4
Block-float emulation as the Blackhole host packer: 16 consecutive values along the matmul OUTPUT dim share the
largest exponent; bfp8 keeps 7 mantissa bits (q*2^(e-6), q<=127), bfp4 3 bits (q*2^(e-2), q<=7), round to nearest.
Writes dev_<tag>_<name>.npz {top1, lp_at_ref [T-1, 32], nll} in compare_hy4.py's device format, so
compare_hy4.py <texts> <out dir> <ref dir> scores it exactly like a device run.
Only weight rounding is emulated: activations, accumulation and kernel fidelity stay as the BF16 reference's.
"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ref_hy4 as R  # noqa: E402

M = R.M
MANT = {"bfp8": 7, "bfp4": 3}


def bfp_round(w, fmt):
    """w [out, in] (F.linear weight) -> bf16 with blocks of 16 consecutive OUTPUT rows sharing an exponent."""
    if fmt == "bf16":
        return w
    if fmt == "rtne4":  # M2.7's error-aware bfp4 grid choice (hybrid_moe/bfp4_round.py rtne)
        import bfp4_round
        return bfp4_round.rtne(w).to(torch.bfloat16)
    mb = MANT[fmt]
    out = w.shape[0]
    wb = w.to(torch.bfloat16).reshape(out // 16, 16, -1)
    eb = (wb.view(torch.int16).to(torch.int32) >> 7) & 0xFF  # biased exponent field (0 for zero / denormal)
    shared = eb.amax(dim=1, keepdim=True)
    scale = torch.exp2((shared - 127 - (mb - 1)).float())
    x = wb.float()
    q = torch.round(x.abs() / scale).clamp_(max=float(2**mb - 1))
    return (torch.sign(x) * q * scale).reshape(w.shape).to(torch.bfloat16)


GU, DN = os.environ.get("QA_GU", "bf16"), os.environ.get("QA_DN", "bf16")
ATTN, DENSE = os.environ.get("QA_ATTN", "bf16"), os.environ.get("QA_DENSE", "bf16")
F = torch.nn.functional


def experts_forward(self, hidden_states, top_k_index, top_k_weights):
    final = torch.zeros_like(hidden_states)
    with torch.no_grad():
        mask = F.one_hot(top_k_index, num_classes=self.num_experts + 1).permute(2, 1, 0)
        hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
    for expert_idx in hit:
        expert_idx = expert_idx[0]
        if expert_idx == self.num_experts:
            continue
        top_k_pos, token_idx = torch.where(mask[expert_idx])
        gu = bfp_round(self.gate_up_proj[expert_idx], GU)
        dn = bfp_round(self.down_proj[expert_idx], DN)
        current = self._apply_gate(F.linear(hidden_states[token_idx], gu))
        current = F.linear(current, dn) * top_k_weights[token_idx, top_k_pos, None]
        final.index_add_(0, token_idx, current.to(final.dtype))
    return final


M.HYV4FlashExperts.forward = experts_forward


def quantize_resident(tm):
    n = 0
    for li, layer in enumerate(tm.layers):
        if ATTN != "bf16":
            a = layer.self_attn
            for name in ("q_b_proj", "gate_proj", "o_proj"):
                lin = getattr(a, name)
                lin.weight = torch.nn.Parameter(bfp_round(lin.weight.data, ATTN), requires_grad=False)
                n += 1
        if DENSE != "bf16" and isinstance(layer.mlp, M.HYV4FlashMLP):
            for name in ("gate_proj", "up_proj", "down_proj"):
                lin = getattr(layer.mlp, name)
                lin.weight = torch.nn.Parameter(bfp_round(lin.weight.data, DENSE), requires_grad=False)
                n += 1
    return n


@torch.no_grad()
def main():
    texts, ref_dir, out_dir, tag = sys.argv[1:5]
    seqs = json.load(open(texts))
    if len(sys.argv) > 5:
        seqs = {k: seqs[k] for k in sys.argv[5].split(",")}
    os.makedirs(out_dir, exist_ok=True)
    ref = R.Ref()
    print(f"QA gu={GU} dn={DN} attn={ATTN} dense={DENSE}: {quantize_resident(ref.tm)} resident linears rounded", flush=True)
    # one batch: right-pad to the longest (causal: positions < len are unchanged); one expert rounding per layer
    names = list(seqs)
    T = max(len(v) for v in seqs.values())
    for _ in [0]:
        ids = torch.tensor([seqs[n] + [13] * (T - len(seqs[n])) for n in names])
        t0 = time.time()
        h = ref.forward_chunked(ids, int(os.environ.get("REF_CHUNK", "4096")))
        print(f"group T={T} B={len(names)}: {time.time() - t0:.0f}s", flush=True)
        for b, n in enumerate(names):
            L = len(seqs[n])
            lg = ref.logits(h[b, :L])
            lp = lg - torch.logsumexp(lg, -1)[:, None]
            r = np.load(os.path.join(ref_dir, f"ref_{n}.npz"))
            rid = torch.from_numpy(r["top_ids"].astype(np.int64))
            nxt = ids[b, 1:L]
            np.savez(os.path.join(out_dir, f"dev_{tag}_{n}.npz"),
                     top1=lp[:-1].argmax(-1).numpy().astype(np.int32),
                     lp_at_ref=lp[:-1].gather(1, rid[: L - 1]).numpy().astype(np.float32),
                     nll=(-lp[:-1].gather(1, nxt[:, None])[:, 0]).numpy())
            print(f"QA {tag} {n} top1 {float((lp[:-1].argmax(-1) == rid[: L - 1, 0]).float().mean()):.3f}", flush=True)


if __name__ == "__main__":
    main()

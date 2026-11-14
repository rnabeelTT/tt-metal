# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""CPU teacher-forced acceptance of RadixArk/Inkling-Small-DSpark-Preview against the BF16 reference, measured the
same way as mtp_inkling_ref.py (the drafter comparison of Phase 4).

DSpark (the checkpoint's dflash.py / dspark.py): context = the target's hidden states after layers [5, 11, 23, 29, 35]
(ref_inkling.py REF_TAPS -> taps.pt) for positions < s; block = [tok_s, mask x 14] at positions s .. s+14, attending
non-causally to [fc(context); block]; drafts = argmax over (block hidden -> target LM head) + the Markov bias of the
previous block token (sequential across the block). Accepted run at s = leading drafts equal to the target's argmax at
s, s+1, ... (spec_generate's rule). Sampled block starts, full teacher-forced context.

The conventions DSpark was trained with are not pinned by the checkpoint (SGLang-side), so every combination is
reported: block token embedding raw or backbone-normed (embed_norm), logits with or without the 1/16.

usage (host venv): python dspark_inkling_ref.py <texts.json> <ref dir> [block starts per text = 48] [block = 15]
"""

import glob
import importlib.util
import json
import os
import sys

import numpy as np
import torch
from safetensors.torch import load_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ref_inkling as R  # noqa: E402
from transformers.models.inkling import modeling_inkling as M  # noqa: E402

REPO = "RadixArk/Inkling-Small-DSpark-Preview"


def load_dspark():
    snap = sorted(glob.glob(f"{R.HUB}/models--{REPO.replace('/', '--')}/snapshots/*"))[0]
    # dspark.py imports DFlashDraftModel from the SpecForge package; the checkpoint ships that file (dflash.py):
    # register it under SpecForge's module name instead of installing SpecForge
    import types

    for name in ("specforge", "specforge.modeling", "specforge.modeling.draft"):
        sys.modules.setdefault(name, types.ModuleType(name))
    dspec = importlib.util.spec_from_file_location("specforge.modeling.draft.dflash", os.path.join(snap, "dflash.py"))
    dmod = importlib.util.module_from_spec(dspec)
    sys.modules["specforge.modeling.draft.dflash"] = dmod
    dspec.loader.exec_module(dmod)
    spec = importlib.util.spec_from_file_location("dspark", os.path.join(snap, "dspark.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfg = mod.DSparkConfig.from_pretrained(snap)
    cfg._attn_implementation = "sdpa"
    model = mod.DSparkDraftModel(cfg).to(torch.bfloat16).eval()
    sd = load_file(os.path.join(snap, "model.safetensors"))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"DSpark loaded: missing {missing[:6]} unexpected {unexpected[:6]}", flush=True)
    return model


@torch.no_grad()
def main():
    texts, ref_dir = sys.argv[1], sys.argv[2]
    n_starts = int(sys.argv[3]) if len(sys.argv) > 3 else 48
    BS = int(sys.argv[4]) if len(sys.argv) > 4 else 15
    torch.set_num_threads(int(os.environ.get("REF_THREADS", "16")))
    seqs = json.load(open(texts))
    taps = torch.load(os.path.join(ref_dir, "taps.pt"))
    names, tap = taps["names"], taps["taps"]  # [B, T, 5, H]
    model = load_dspark()
    BS = min(BS, model.block_size)
    mask_id = model.mask_token_id
    cfg = R.config()
    emb_w = R.get("model.llm.embed.weight")
    bnorm = M.InklingRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
    bnorm.weight.data = R.get("model.llm.embed_norm.weight").to(torch.bfloat16)
    head = R.get("model.llm.unembed.weight")[: cfg.unpadded_vocab_size]
    markov = getattr(model, "markov_head", None)
    variants = [(en, sc) for en in ("raw", "normed") for sc in (1.0, 1.0 / cfg.logits_mup_width_multiplier)]
    acc = {v: [] for v in variants}
    for b, name in enumerate(names):
        ids = torch.tensor(seqs[name])
        T = len(ids)
        target = torch.from_numpy(np.load(os.path.join(ref_dir, f"ref_{name}.npz"))["top_ids"][:, 0]).long()
        ctx_all = tap[b].reshape(T, -1)  # [T, 5H]
        starts = np.linspace(64, T - BS - 1, n_starts).astype(int)
        for s in starts:
            # tok_s: the target's accepted token at s (teacher forcing: the real token; the target argmax at s-1 when
            # it matches, which is the common case the drafter sees)
            block = torch.full((BS,), mask_id, dtype=torch.long)
            block[0] = ids[s]
            pos = torch.arange(0, s + BS).unsqueeze(0)
            ctx = ctx_all[:s].unsqueeze(0).to(torch.bfloat16)
            for en, sc in variants:
                e = torch.nn.functional.embedding(block, emb_w).unsqueeze(0)
                if en == "normed":
                    e = bnorm(e)
                h = model(target_hidden=ctx, noise_embedding=e, position_ids=pos, is_causal=False)  # [1, BS, H]
                base = (h[0, 1:].float() @ head.float().T) * sc  # [BS-1, V]
                drafts = []
                prev = block[0]
                for j in range(BS - 1):  # the Markov bias of the previous (drafted) token, sequential
                    lg = base[j]
                    if markov is not None:
                        lg = lg + markov.compute_step_bias(prev.view(1))[0, : lg.shape[0]].float()
                    prev = lg.argmax()
                    drafts.append(int(prev))
                tgt = target[s : s + BS - 1].tolist()  # target argmax at s.. (predicts s+1..)
                run = 0
                for d, t in zip(drafts, tgt):
                    if d != t:
                        break
                    run += 1
                acc[(en, sc)].append(run + 1)
        print(f"{name}: " + "  ".join(f"{en}/{'x1/16' if sc < 1 else 'x1'} {np.mean(acc[(en, sc)]):.2f}"
                                     for en, sc in variants), flush=True)
    print("MEAN acceptance length (block %d): " % BS + "  ".join(
        f"emb {en}, logits {'/16' if sc < 1 else 'x1'}: {np.mean(v):.3f}" for (en, sc), v in acc.items()))


if __name__ == "__main__":
    main()

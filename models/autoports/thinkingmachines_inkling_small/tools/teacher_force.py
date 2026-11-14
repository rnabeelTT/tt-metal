# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Teacher-forced quality of the Inkling TT model against the BF16 CPU reference (ref_inkling.py): the quality gate.

Runs each text of a texts.json through the device model (EVAL_PATH=prefill, the default: the chunked prefill path, one
attention launch per chunk; EVAL_PATH=rows: groups of R <= 8 consecutive tokens, the spec-verify code), reads the logits
of every position and compares them with the reference's
ref_<name>.npz (top-32 log-probs):
  top1      device argmax == reference argmax
  top1_conf the same on positions where the reference is confident (p > 0.5)
  kl32      KL(ref || device) truncated to the reference's top 32 (both log-probs over the full vocab; as compare_m27)
  nll       the device's NLL of the real next token (vs the reference's)

With the hybrid overlay: HYB_ENABLE=1 + the HYB_* env of the profile (HYB_PORT=thinkingmachines_inkling_small).
Without it, use EVAL_LAYERS to restrict to a model that fits (smoke test only: the logits are then meaningless).

usage (in the image): tools/run_dev.sh python models/autoports/thinkingmachines_inkling_small/tools/teacher_force.py \
        <texts.json> <ref dir> [out.json]       env: EVAL_TEXTS (comma list, default all), EVAL_T (tokens, default all),
        EVAL_ROWS (default 8), EVAL_LAYERS (e.g. 0-3, smoke only)
"""

import json
import os
import sys
import time

import numpy as np
import torch

if os.environ.get("HYB_ENABLE") == "1":
    sys.path.insert(0, os.environ.get("HYB_POC_DIR", "/opt/tt-metal/models/experimental/hybrid_moe"))
    os.environ.setdefault("HYB_PORT", "thinkingmachines_inkling_small")
    os.environ.setdefault("HYB_FUSED", "0")
    import hybrid

    hybrid.install()

import ttnn  # noqa: E402
from models.autoports.thinkingmachines_inkling_small.tt.model import create_model  # noqa: E402
from models.tt_transformers.tt.common import Mode  # noqa: E402


def main():
    texts_path, ref_dir = sys.argv[1], sys.argv[2]
    out_path = sys.argv[3] if len(sys.argv) > 3 else None
    seqs = json.load(open(texts_path))
    names = [n for n in (os.environ.get("EVAL_TEXTS") or ",".join(seqs)).split(",") if n]
    T_max = int(os.environ.get("EVAL_T", "100000"))
    RW = int(os.environ.get("EVAL_ROWS", "8"))
    layers = None
    if os.environ.get("EVAL_LAYERS"):
        a, b = os.environ["EVAL_LAYERS"].split("-")
        layers = list(range(int(a), int(b) + 1))
    T_all = max(len(seqs[n]) for n in names)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), l1_small_size=24576, trace_region_size=64_000_000)
    results = {}
    try:
        t0 = time.time()
        args, model = create_model(mesh, max_seq_len=max(4096, min(T_all, T_max) + 64), layer_indices=layers)
        print(f"model built in {time.time() - t0:.0f}s", flush=True)
        rep = ttnn.ReplicateTensorToMesh(mesh)
        nblk = model.layers[0].attention.max_blocks
        pt_host = torch.arange(nblk, dtype=torch.int32)
        pt = ttnn.from_torch(pt_host.reshape(1, -1), dtype=ttnn.int32,
                             layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, mesh_mapper=rep)
        V = int(args.unpadded_vocab_size)
        for name in names:
            ids = torch.tensor(seqs[name][:T_max])
            T = len(ids)
            ref = np.load(os.path.join(ref_dir, f"ref_{name}.npz"))
            model.reset_sequence_state()
            lps = []
            t1 = time.time()
            if os.environ.get("EVAL_PATH", "prefill") == "prefill":  # chunked prefill, logits of every row
                lps.append(torch.log_softmax(model.prefill_tokens(ids, pt_host, logits_rows="all"), -1))
            for g in range(0, T if os.environ.get("EVAL_PATH", "prefill") == "rows" else 0, RW):
                R = min(RW, T - g)
                tok = torch.zeros(1, 1, 1, 32, dtype=torch.int32)
                tok[0, 0, 0, :R] = ids[g : g + R].to(torch.int32)
                tok_tt = ttnn.from_torch(tok, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh,
                                         mesh_mapper=rep)
                x = model.embd(ttnn.slice(ttnn.reshape(tok_tt, (1, 32)), (0, 0), (1, R)))
                x = ttnn.reshape(x, (1, 1, R, x.shape[-1]))
                pos = ttnn.from_torch(torch.arange(g, g + R, dtype=torch.int32), dtype=ttnn.int32,
                                      layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, mesh_mapper=rep)
                for layer in model.layers:
                    x = layer.forward(x, mode="rows", position_tensor=pos, page_table=pt)
                x = model._final_norm_decode(x)
                logits = model._lm_head(x)
                lg = ttnn.to_torch(ttnn.get_device_tensors(logits)[0]).float().reshape(-1, logits.shape[-1])[:R, :V]
                lps.append(torch.log_softmax(lg, -1))
                for z in (tok_tt, x, logits, pos):
                    ttnn.deallocate(z)
            lp = torch.cat(lps)  # [T, V]
            dt = time.time() - t1
            top_ids, top_lp = torch.from_numpy(ref["top_ids"]).long(), torch.from_numpy(ref["top_lp"]).float()
            nxt = ids[1:]
            top1 = (lp[:-1].argmax(-1) == top_ids[:-1, 0]).float()
            conf = top_lp[:-1, 0].exp() > 0.5
            dev_lp32 = lp.gather(1, top_ids)  # device log-probs at the reference's top 32
            pref = top_lp.exp()
            kl = (pref * (top_lp - dev_lp32)).sum(-1)[:-1]
            nll = -lp[:-1].gather(1, nxt[:, None])[:, 0]
            r = {"T": T, "top1": top1.mean().item(), "top1_conf": top1[conf].mean().item(), "conf_frac": conf.float().mean().item(),
                 "kl32": kl.mean().item(), "nll": nll.mean().item(), "nll_ref": float(ref["nll"].mean()),
                 "tok_per_s": T / dt}
            results[name] = r
            print(f"{name}: top1 {r['top1']:.4f}  top1_conf {r['top1_conf']:.4f}  kl32 {r['kl32']:.4f}  nll {r['nll']:.3f} "
                  f"(ref {r['nll_ref']:.3f})  [{T} tok, {dt:.0f}s]", flush=True)
        if results:
            agg = {k: float(np.mean([v[k] for v in results.values()])) for k in ("top1", "top1_conf", "kl32", "nll", "nll_ref")}
            results["_mean"] = agg
            print("MEAN " + json.dumps(agg))
        if out_path:
            json.dump(results, open(out_path, "w"), indent=1)
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()

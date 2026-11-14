# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Where do verify rows 1.. leave plain decode? rows_consistency.py showed row 0 of a verify group bit-identical to plain
one-row decode, but rows 1.. far off (logits PCC down to 0.71). This runs, from the same committed prompt:

  plain: tokens x0, t1, .., t_{R-1} one row per step (forward_rows R=1, commit 1)
  group: the same R tokens as ONE verify group (forward_rows R rows, nothing committed)

and compares, for every layer and every row j, the attention output (attention.forward_rows) and the MoE output
(feed_forward.forward), shard 0 of the fractured tensors. The first (layer, component) where row j's PCC drops
localises the defect.

usage (in the image, hybrid env as tools/eval_hybrid.sh, e.g. TOOL=rows_layer_diff.py TOOL_ARGS="3"):
       .../tools/rows_layer_diff.py [R=3]   -> per-layer table, /cache/rows_layer_diff.json
"""

import json
import os
import sys

import torch

if os.environ.get("HYB_ENABLE") == "1":
    sys.path.insert(0, os.environ.get("HYB_POC_DIR", "/opt/tt-metal/models/experimental/hybrid_moe"))
    os.environ.setdefault("HYB_PORT", "thinkingmachines_inkling_small")
    os.environ.setdefault("HYB_FUSED", "0")
    import hybrid

    hybrid.install()

import ttnn  # noqa: E402
from models.autoports.thinkingmachines_inkling_small.tt.model import create_model  # noqa: E402

PROMPT = [{"role": "user", "content": "Explain in three short paragraphs how a hash map handles collisions."}]


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    if a.std() == 0 or b.std() == 0:
        return float(torch.equal(a, b))
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def main():
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), l1_small_size=24576, trace_region_size=64_000_000)
    try:
        args, model = create_model(mesh, max_seq_len=8192)
        tok = args.tokenizer
        rep = ttnn.ReplicateTensorToMesh(mesh)
        nblk = model.layers[0].attention.max_blocks
        pt_host = torch.arange(nblk, dtype=torch.int32)
        pt = ttnn.from_torch(pt_host.reshape(1, -1), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh,
                             mesh_mapper=rep)
        V = int(args.unpadded_vocab_size)

        cap = {}  # (layer, component) -> list of [rows, width] captures, in call order

        def host(t):
            return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float().reshape(-1, t.shape[-1]).clone()

        for i, layer in enumerate(model.layers):
            def wrap(obj, name, comp, i=i):
                f = getattr(obj, name)

                def g(*a, **k):
                    out = f(*a, **k)
                    cap.setdefault((i, comp), []).append(host(out))
                    return out

                setattr(obj, name, g)

            wrap(layer.attention, "forward_rows", "attn")
            wrap(layer.feed_forward, "forward", "moe")

        def dev_ids(ids):
            return ttnn.from_torch(torch.tensor(ids, dtype=torch.int32).reshape(1, len(ids)), dtype=ttnn.uint32,
                                   layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, mesh_mapper=rep)

        def dev_pos(p0, n):
            return ttnn.from_torch(torch.arange(p0, p0 + n, dtype=torch.int32), dtype=ttnn.int32,
                                   layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, mesh_mapper=rep)

        def rows(ids, p0):
            n = len(ids)
            x = model.embd(dev_ids(ids))
            x = ttnn.reshape(x, (1, 1, n, x.shape[-1]))
            pos = dev_pos(p0, n)
            for layer in model.layers:
                x = layer.forward_rows(x, pos, pt, commit_all=False)
            lg = model._lm_head(model._final_norm_decode(x))
            return host(lg)[:n, :V]

        def commit(n):
            for layer in model.layers:
                layer.commit_rows(n)

        def start(prompt):
            model.reset_sequence_state()
            P = len(prompt)
            pre = (P - 1) // 64 * 64
            if pre:
                model.prefill_tokens(torch.tensor(prompt[:pre]), pt_host, logits_rows="last")
            rest = prompt[pre:]
            for g in range(0, len(rest) - 1, 8):
                chunk = rest[g : min(g + 8, len(rest) - 1)]
                rows(chunk, pre + g)
                commit(len(chunk))
            return rest[-1], P - 1

        prompt = tok.encode(tok.apply_chat_template(PROMPT, tokenize=False, add_generation_prompt=True,
                                                    reasoning_effort="none"), add_special_tokens=False)
        # plain decode: R steps of one row; record the stream and each step's captures
        x, p0 = start(prompt)
        stream, plain_lg = [x], []
        cap.clear()
        for j in range(R):
            lg = rows([stream[j]], p0 + j)[0]
            commit(1)
            plain_lg.append(lg)
            stream.append(int(lg.argmax()))
        plain = {key: torch.cat(v) for key, v in cap.items()}  # R single-row captures stacked -> [R, w]
        # the same R tokens as one verify group
        start(prompt)
        cap.clear()
        group_lg = rows(stream[:R], p0)
        group = {key: v[0] for key, v in cap.items()}  # one R-row capture
        print(f"tokens {stream[:R]!r} -> {tok.decode(stream[1:R + 1])!r}")
        for j in range(R):
            print(f"logits row {j}: PCC {pcc(plain_lg[j], group_lg[j]):.6f} max|diff| "
                  f"{(plain_lg[j] - group_lg[j]).abs().max():.4f} argmax {int(plain_lg[j].argmax())} / "
                  f"{int(group_lg[j].argmax())}")
        table = []
        first = {}
        print("layer | " + " | ".join(f"attn r{j}  moe r{j}" for j in range(R)))
        for i in range(len(model.layers)):
            row = {"layer": i}
            cells = []
            for j in range(R):
                for comp in ("attn", "moe"):
                    a, b = plain[(i, comp)][j], group[(i, comp)][j]
                    v = pcc(a, b)
                    row[f"{comp}{j}"] = v
                    if v < 0.999 and (j, comp) not in first:
                        first[(j, comp)] = i
                cells.append(f"{row[f'attn{j}']:.5f} {row[f'moe{j}']:.5f}")
            table.append(row)
            print(f"{i:5d} | " + " | ".join(cells), flush=True)
        print("first layer with PCC < 0.999:", {f"row{j} {c}": L for (j, c), L in sorted(first.items())})
        json.dump({"R": R, "tokens": stream, "table": table,
                   "first_below_0.999": {f"row{j} {c}": L for (j, c), L in first.items()}},
                  open(os.environ.get("ROWS_DIFF_OUT", "/cache/rows_layer_diff.json"), "w"), indent=1)
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()

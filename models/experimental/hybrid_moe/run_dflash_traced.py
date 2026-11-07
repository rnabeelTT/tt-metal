# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""DFlash speculative decoding on the hybrid MiniMax-M2.7, traced: per pass two trace replays.

    drafter pass : the previous verify's accepted rows into the drafter's context cache, then the next block's drafts
                   (dflash_tt.DFlashDrafterPaged.pass_body; argmax on the device)
    verify       : the target over the 4 rows [t, d_1..d_3] (the port's multi-row decode, the hybrid MoE per layer with
                   its host steps between trace segments); the tap layers' outputs are all-gathered on the device straight
                   into the drafter's context input; argmax on the device

Prefill is eager (the tap layers' outputs are captured, read to the host and filled into the drafter's paged cache).
Everything persistent (drafter weights / cache / inputs) is allocated BEFORE the target is built, the verify inputs
before any capture: a buffer allocated after a capture can reuse the trace's freed intermediates.

env: as run_dflash.py (POC_PROMPTS, POC_PROMPT_START, POC_NEW_TOKENS, DFLASH_SNAPSHOT), DFLASH_CMAX (drafter context
cap, default 4096). usage (in the image): python run_dflash_traced.py <out dir>
"""

import glob
import json
import os
import sys
import time

sys.path.insert(0, "/poc")
import hybrid  # noqa: E402

HYB = hybrid.install()

import torch  # noqa: E402

import ttnn  # noqa: E402
from dflash_tt import DFlashDrafterPaged, argmax_max, read_argmax  # noqa: E402
from hybrid.patch import PORT, _port  # noqa: E402

rows = _port("mtp_rows")
K = 3  # drafts per pass: 4 verify rows (width 4)
R = K + 1
NEW = int(os.environ.get("POC_NEW_TOKENS", "128"))
DRAM = ttnn.DRAM_MEMORY_CONFIG


class Verify:
    """The target's 4-row verify pass with persistent inputs."""

    def __init__(self, gen, drafter):
        self.gen, self.model, self.dr = gen, gen.model, drafter
        m = self.model
        self.mesh = m.mesh_device
        self.BW = rows.rows_width(R)
        self.W = gen.page_table.shape[-1]
        tok, pos, rot, pt = m.prepare_decode_inputs_host(
            torch.zeros(self.BW, dtype=torch.int32),
            torch.zeros(self.BW, dtype=torch.int32),
            torch.zeros(self.BW, self.W, dtype=torch.int32),
        )
        dev = lambda t: ttnn.to_device(t, self.mesh, memory_config=DRAM)  # noqa: E731
        self.tok, self.pos, self.rot, self.pt = dev(tok), dev(pos), dev(rot), dev(pt)
        self.masks = dev(rows.rows_pos_masks_host([-1] * self.BW, self.BW))
        self.trace, self.outs = None, None

    def stage(self, tokens, positions, page_table_row):
        m = self.model
        tok, pos, rot, pt = m.prepare_decode_inputs_host(
            torch.tensor(tokens, dtype=torch.int32),
            torch.tensor(positions, dtype=torch.int32),
            page_table_row.repeat(self.BW, 1),
        )
        for h, d in ((tok, self.tok), (pos, self.pos), (rot, self.rot), (pt, self.pt)):
            ttnn.copy_host_to_device_tensor(h, d)
        ttnn.copy_host_to_device_tensor(rows.rows_pos_masks_host(list(positions), self.BW), self.masks)

    def body(self):
        m, BW = self.model, self.BW
        cos, sin = m._rope_from_idxs(self.rot)
        x = m.embd(m._decode_tokens_view(self.tok, BW))
        x = ttnn.to_memory_config(ttnn.reshape(x, (1, 1, BW, x.shape[-1])), DRAM)
        taps = []
        for li, layer in enumerate(m.layers):
            y = rows.layer_forward_rows(
                layer,
                x,
                R,
                cos=cos,
                sin=sin,
                position_tensor=self.pos,
                page_table=self.pt,
                rows_pos_masks=self.masks,
                gdn_index=0,
            )
            if li in self.dr.taps:
                taps.append(ttnn.all_gather(y, dim=3, topology=ttnn.Topology.Linear, memory_config=DRAM))
            ttnn.deallocate(x)
            x = y
        cat = ttnn.concat(taps, dim=-1)  # [1, 1, BW, n_taps * H], replicated
        ttnn.copy(cat, self.dr.c_taps[R])
        for t in taps + [cat, cos, sin]:
            ttnn.deallocate(t)
        h = m._final_norm_decode(x)
        logits = ttnn.linear(h, m.lm_head_weight, compute_kernel_config=m._lm_head_ckc())
        for t in (h, x):
            ttnn.deallocate(t)
        out = argmax_max(logits)
        ttnn.deallocate(logits)
        return out

    def run(self):
        if self.trace is not None:
            ttnn.execute_trace(self.mesh, self.trace, cq_id=0, blocking=False)
            outs = self.outs
        else:
            outs = self.body()
        return read_argmax(self.mesh, outs, int(ttnn.get_device_tensors(self.model.lm_head_weight)[0].shape[-1]))

    def release(self):
        if self.trace is not None:
            ttnn.release_trace(self.mesh, self.trace)
        self.trace, self.outs = None, None

    def capture(self):
        tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
        self.outs = self.body()
        ttnn.end_trace_capture(self.mesh, tid, cq_id=0)
        self.trace = tid


def main():
    out_dir = sys.argv[1]
    os.makedirs(out_dir, exist_ok=True)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), l1_small_size=24576, trace_region_size=96_000_000)
    res = {"runs": []}
    try:
        import ref_m27

        ref_m27.SNAP = glob.glob("/hf/hub/models--MiniMaxAI--MiniMax-M2.7/snapshots/*")[0]
        ref_m27.WMAP = json.load(open(f"{ref_m27.SNAP}/model.safetensors.index.json"))["weight_map"]
        embed = ref_m27.get("model.embed_tokens.weight")
        snap = (
            os.environ.get("DFLASH_SNAPSHOT") or glob.glob("/hf/hub/models--nvidia--MiniMax-M2.7-DFlash/snapshots/*")[0]
        )
        dr = DFlashDrafterPaged(
            mesh,
            snap,
            embed,
            None,
            None,
            cmax=int(os.environ.get("DFLASH_CMAX", "2048")),
            rows=R,
            tp=os.environ.get("DFLASH_TP", "1") == "1",
        )
        gen = _port("generator").build_generator(
            f"models/autoports/{PORT}",
            mesh,
            max_seq_len=int(os.environ.get("POC_MAX_SEQ", "4096")),
            warm_decode_trace=False,
            warm_prefill_trace=False,
        )
        model, tok = gen.model, gen.tokenizer
        H, V = gen.args.dim, gen.args.vocab_size
        dr.lm_head, dr.vocab = model.lm_head_weight, V
        page_table = gen.page_table[:1]
        t0 = time.time()  # the host library's RAM copies load in the background from the build on: wait for all of them
        for li in range(len(model.layers)):
            HYB.lib_layer(li)
        print(f"host library resident ({time.time() - t0:.1f} s wait)", flush=True)
        ver = Verify(gen, dr)
        # warm (compile + the hybrid's per-width buffers), then capture; positions far from any real prompt's rows
        ver.stage([0] * R, list(range(8, 8 + R)), page_table)
        ver.run()
        ver.run()
        dr.reset()
        dr.stage_pass([], 0)
        dr.run("pass", dr.pass_body)
        ttnn.synchronize_device(mesh)
        ttnn.synchronize_device(mesh)
        import prompts_v2

        start = int(os.environ.get("POC_PROMPT_START", "0"))
        prompts = prompts_v2.HELDOUT[start : start + int(os.environ.get("POC_PROMPTS", "2"))]
        long_n = int(os.environ.get("POC_LONG", "0") or 0)
        if long_n:  # a long prompt (TTFT / prefill at length): held-out texts repeated to ~long_n tokens
            # a realistic long input: source files of this repo (POC_LONG_FILES), cut to ~long_n tokens
            files = os.environ.get("POC_LONG_FILES", "/poc/hybrid/patch.py,/poc/run_dflash_traced.py,/poc/dflash_tt.py")
            body = "\n\n".join(f"# file: {os.path.basename(f)}\n" + open(f).read() for f in files.split(","))
            text = tok.decode(tok.encode(body)[:long_n])
            prompts.append("Here is some Python code:\n\n" + text + "\n\nExplain what this code does, file by file.")
        for p_text in prompts:
            chat = tok.apply_chat_template(
                [{"role": "user", "content": p_text}], add_generation_prompt=True, enable_thinking=False, tokenize=False
            )
            ids = tok.encode(chat, add_special_tokens=False)
            P = len(ids)
            gen.reset()
            dr.reset()
            captured = {li: [] for li in dr.taps}
            orig = {}
            for li in dr.taps:
                layer = model.layers[li]
                orig[li] = layer.forward

                def hook(*a, _f=orig[li], _li=li, **kw):
                    y = _f(*a, **kw)
                    captured[_li].append(ttnn.to_torch(y, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=-1)).float())
                    return y

                layer.forward = hook
            t_pf = time.time()
            try:
                logits0 = gen.prefill_forward(torch.tensor(ids).reshape(1, -1), page_table=page_table, prompt_lens=[P])
            finally:
                for li in dr.taps:
                    model.layers[li].forward = orig[li]
            print(f"TAPCHUNKS P={P} " + str([tuple(c.shape) for c in captured[dr.taps[0]]]), flush=True)
            # the prefill layer outputs are REPLICATED ([.., H] per die): the mesh concat holds tp copies side by side
            feats = torch.cat(
                [torch.cat([c.reshape(-1, c.shape[-1])[:, :H] for c in captured[li]])[:P] for li in dr.taps], dim=-1
            )
            # drafter context window (sliding): the drafter's acceptance falls with its context (window_oracle.py on an 8K
            # prompt: 2.03 tokens/pass at 1024, 1.47 at 4096, 1.10 with the whole context), so it sees the last W only
            W = int(os.environ.get("DFLASH_WINDOW", "1024"))
            hist = [feats[-W:].clone()]  # host copies of the context features (refills), concatenated only then
            n_hist = hist[0].shape[0]
            if P > W:
                dr.fill_prompt(feats[-W:], start=P - W)
            else:
                dr.fill_prompt(feats)
            n_refill = 0
            save_all = [feats.to(torch.float16)] if os.environ.get("DFLASH_SAVE_CTX") == "1" else None
            t_first = int(torch.as_tensor(logits0).reshape(-1, logits0.shape[-1])[-1, :V].argmax())
            if os.environ.get("DFLASH_LONGCHECK") == "1" and P > 1000:  # captured prefill rows vs a verify pass
                for k, c in enumerate(captured[dr.taps[2]]):
                    w = c.reshape(-1, c.shape[-1])
                    parts = [w[:, d * H : (d + 1) * H] for d in range(w.shape[-1] // H)]
                    print(
                        f"LONGCHECK chunk {k}: die copies max diff "
                        f"{max(float((parts[0] - q).abs().max()) for q in parts[1:]) if len(parts) > 1 else 0:.4f} "
                        f"row norm mean {float(parts[0].norm(dim=-1).mean()):.2f}",
                        flush=True,
                    )
                ver.stage(ids[P - R : P], list(range(P - R, P)), page_table)
                ver.run()  # eager (no trace yet): rewrites the same K / V of the last R prompt positions
                vt = ttnn.to_torch(ttnn.get_device_tensors(dr.c_taps[R])[0]).float().reshape(R, -1)
                for j in range(len(dr.taps)):
                    a_, b_ = vt[:, j * H : (j + 1) * H], feats[P - R :, j * H : (j + 1) * H]
                    pc = float(torch.corrcoef(torch.stack([a_.flatten(), b_.flatten()]))[0, 1])
                    print(
                        f"LONGCHECK tap {dr.taps[j]}: verify vs captured prefill rows pcc {pc:.4f} "
                        f"norms {float(a_.norm(dim=-1).mean()):.1f} / {float(b_.norm(dim=-1).mean()):.1f}",
                        flush=True,
                    )
            if os.environ.get("DFLASH_LONGCHECK") == "1" and P > 1000:  # eager drafter vs the CPU reference here
                dr.stage_pass([], t_first)
                d_eager = dr.pass_body()
                d_eager = read_argmax(mesh, d_eager, int(ttnn.get_device_tensors(dr.lm_head)[0].shape[-1]))[1:4]
                from safetensors.torch import load_file
                from transformers import Qwen3Config

                from dflash_ref import DFlashDraftModel

                cfg = Qwen3Config.from_pretrained(snap)
                cfg._attn_implementation = "eager"
                cpu = DFlashDraftModel(cfg).to(torch.bfloat16).eval()
                cpu.load_state_dict(load_file(f"{snap}/model.safetensors"), strict=False)
                head = ref_m27.get("lm_head.weight")
                with torch.no_grad():
                    blk = torch.tensor([t_first] + [cpu.mask_token_id] * (cpu.block_size - 1))
                    hh = cpu(
                        position_ids=torch.arange(P + cpu.block_size)[None],
                        noise_embedding=torch.nn.functional.embedding(blk, embed)[None],
                        target_hidden=feats.to(torch.bfloat16)[None],
                    )
                d_cpu = (hh[0, 1:].float() @ head.float().T).argmax(-1).tolist()[:3]
                print(f"LONGCHECK P={P} eager {d_eager} cpu {d_cpu}", flush=True)
                del cpu, head
                dr.reset()
                dr.fill_prompt(feats)
            # the traces are captured per prompt, after its eager prefill: eager device work after trace replays hung
            # the next replay (the second prompt's first drafter pass, 2026-11-06)
            ttnn.synchronize_device(mesh)
            t_cap = time.time()
            dr.capture("pass", dr.pass_body)
            ver.capture()
            ttnn.synchronize_device(mesh)
            t_cap = time.time() - t_cap
            if not res["runs"]:  # everything allocated: the footprint at this POC_MAX_SEQ
                print("MEMONLY " + json.dumps(gen._memory_report()), flush=True)
                if os.environ.get("DFLASH_MEMONLY") == "1":
                    return
            t_pf = time.time() - t_pf
            t = int(torch.as_tensor(logits0).reshape(-1, logits0.shape[-1])[-1, :V].argmax())
            out, passes, acc_hist = [], 0, [0] * R
            td = tv = th = 0.0
            HYB.reset_stats()
            pos, prev_acc = P, []
            t1 = time.time()
            while len(out) < NEW:
                a0 = time.perf_counter()
                if dr.need_refill():  # the window is full: refill from the last W features (traces re-captured)
                    ttnn.synchronize_device(mesh)
                    ver.release()
                    for name in list(dr.traces):
                        ttnn.release_trace(mesh, dr.traces.pop(name))
                    hist = [torch.cat(hist)[-W:]]
                    n_hist = hist[0].shape[0]
                    dr.fill_prompt(hist[0], start=pos - n_hist)
                    prev_acc = []
                    ttnn.synchronize_device(mesh)
                    dr.capture("pass", dr.pass_body)
                    ver.capture()
                    ttnn.synchronize_device(mesh)
                    n_refill += 1
                dr.stage_pass(prev_acc, t)  # the previous verify's accepted rows (c_taps[R] filled by the verify)
                drafts = dr.run("pass", None)[1 : 1 + K]
                a1 = time.perf_counter()
                ver.stage([t] + drafts, list(range(pos, pos + R)), page_table)
                top = ver.run()[:R]
                a2 = time.perf_counter()
                a = 0
                while a < K and drafts[a] == top[a]:
                    a += 1
                acc_hist[a] += 1
                if os.environ.get("DFLASH_LONGCHECK") == "1" and passes < 6:
                    print(f"PASS {passes} pos {pos} t {t} drafts {drafts} top {top} accept {a}", flush=True)
                rows_h = ttnn.to_torch(ttnn.get_device_tensors(dr.c_taps[R])[0]).float().reshape(R, -1)[: a + 1]
                hist.append(rows_h)
                n_hist += rows_h.shape[0]
                if save_all is not None:
                    save_all.append(rows_h.to(torch.float16))
                if n_hist > 2 * W:  # bound the host copy
                    hist = [torch.cat(hist)[-W:]]
                    n_hist = W
                out += [t] + drafts[:a]
                prev_acc = list(range(pos, pos + a + 1))
                t = top[a]
                pos += a + 1
                passes += 1
                td, tv = td + a1 - a0, tv + a2 - a1
                th += time.perf_counter() - a2
            dt = time.time() - t1
            ttnn.synchronize_device(mesh)
            ver.release()
            for name in list(dr.traces):
                ttnn.release_trace(mesh, dr.traces.pop(name))
            dr.outs = {}
            if save_all is not None:  # the whole context's features and tokens, for offline drafter studies
                torch.save(
                    {"feats": torch.cat(save_all), "tokens": list(ids) + out, "P": P},
                    os.path.join(out_dir, f"ctx_{len(res['runs'])}.pt"),
                )
            out = out[:NEW]
            r = {
                "prompt": p_text[:80],
                "prompt_tokens": P,
                "passes": passes,
                "tokens_per_pass": round(len(out) / passes, 3),
                "accept_hist": acc_hist,
                "ms_per_token": round(1e3 * dt / len(out), 1),
                "tok_s": round(len(out) / dt, 2),
                "ms_per_pass": {
                    "draft": round(1e3 * td / passes, 1),
                    "verify": round(1e3 * tv / passes, 1),
                    "host": round(1e3 * th / passes, 2),
                },
                "prefill_s": round(t_pf, 2),
                "capture_s": round(t_cap, 2),
                "drafter_refills": n_refill,
                "hyb_per_pass_ms_or_n": {
                    k: round(HYB.st[k] / passes * (1e3 if k.startswith("t_") else 1), 2)
                    for k in (
                        "dec_picks",
                        "dec_hits",
                        "host_experts",
                        "fetched",
                        "t_read",
                        "t_host",
                        "t_up",
                        "t_fetch",
                        "t_route",
                        "t_craft",
                    )
                },
                "text": tok.decode(out),
            }
            print("DFLASHT " + json.dumps({k: v for k, v in r.items() if k != "text"}), flush=True)
            print("TEXT " + json.dumps(r["text"][:300]), flush=True)
            res["runs"].append(r)
            json.dump(res, open(os.path.join(out_dir, "result.json"), "w"), indent=1)
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""DFlash speculative decoding on the hybrid MiniMax-M2.7, traced: the single-user step shared by the standalone runner
(run_dflash_traced.py) and vLLM serving (the port's generator_vllm loads `build` through QWEN35_EXT_SPEC).

Per pass two trace replays:
    drafter pass : the previous verify's accepted rows into the drafter's context cache, then the next block's drafts
                   (dflash_tt.DFlashDrafterPaged.pass_body; argmax on the device)
    verify       : the target over the R = k + 1 rows [t, d_1..d_k] (the port's multi-row decode, the hybrid MoE per
                   layer with its host steps between trace segments); the tap layers' outputs are all-gathered on the
                   device straight into the drafter's context input; argmax on the device

Prefill is eager: hooks on the tap layers capture their outputs, which are read to the host and filled into the
drafter's paged cache (`end_prefill`). The traces are captured per request after its prefill and released before the
next one: eager device work after trace replays hung the next replay (2026-11-06). Every persistent buffer (drafter,
verify inputs) exists before the first capture: a buffer allocated after a capture can reuse a trace's intermediates.

The drafter sees the last DFLASH_WINDOW (default 1024) context tokens: its acceptance falls with its context
(window_oracle.py on an 8K prompt: 2.03 tokens per pass at 1024, 1.47 at 4096, 1.10 with the whole context). When the
window is full, the drafter cache is refilled from host copies of the last window's features and the traces re-captured.

Greedy only: requests with temperature > 0 are decoded greedily (logged once per request).
env: DFLASH_SNAPSHOT, DFLASH_CMAX (drafter cache, default 2048), DFLASH_WINDOW, DFLASH_TP (drafter split over the mesh,
default 1), DFLASH_WARMUP (serving: tokens of the startup warm-up generation, default 64, 0 = off).
"""

from __future__ import annotations

import glob
import json
import os
import time

import torch
from loguru import logger

import ttnn
from dflash_tt import DFlashDrafterPaged, argmax_max, read_argmax
from hybrid.patch import HYB, _port

DRAM = ttnn.DRAM_MEMORY_CONFIG
WARMUP_PROMPT = "Write a haiku about the sea."


def load_drafter(mesh, rows):
    """The DFlash drafter on `mesh` with the target's host embedding table (LM head attached by DFlashServing)."""
    import ref_m27

    ref_m27.SNAP = glob.glob("/hf/hub/models--MiniMaxAI--MiniMax-M2.7/snapshots/*")[0]
    ref_m27.WMAP = json.load(open(f"{ref_m27.SNAP}/model.safetensors.index.json"))["weight_map"]
    snap = os.environ.get("DFLASH_SNAPSHOT") or glob.glob("/hf/hub/models--nvidia--MiniMax-M2.7-DFlash/snapshots/*")[0]
    return DFlashDrafterPaged(
        mesh,
        snap,
        ref_m27.get("model.embed_tokens.weight"),
        None,
        None,
        cmax=int(os.environ.get("DFLASH_CMAX", "2048")),
        rows=rows,
        tp=os.environ.get("DFLASH_TP", "1") == "1",
    )


class Verify:
    """The target's R-row verify pass with persistent inputs."""

    def __init__(self, gen, drafter, R):
        self.rows = _port("mtp_rows")
        self.gen, self.model, self.dr, self.R = gen, gen.model, drafter, R
        m = self.model
        self.mesh = m.mesh_device
        self.BW = self.rows.rows_width(R)
        self.W = int(gen.page_table.shape[-1])
        tok, pos, rot, pt = m.prepare_decode_inputs_host(
            torch.zeros(self.BW, dtype=torch.int32),
            torch.zeros(self.BW, dtype=torch.int32),
            torch.zeros(self.BW, self.W, dtype=torch.int32),
        )
        dev = lambda t: ttnn.to_device(t, self.mesh, memory_config=DRAM)  # noqa: E731
        self.tok, self.pos, self.rot, self.pt = dev(tok), dev(pos), dev(rot), dev(pt)
        self.masks = dev(self.rows.rows_pos_masks_host([-1] * self.BW, self.BW))
        self.trace, self.outs = None, None

    def stage(self, tokens, positions, page_table_row):
        row = torch.as_tensor(page_table_row).reshape(-1).to(torch.int32)
        if row.numel() < self.W:  # the plugin's row may be narrower than the bound width: the tail is never read
            row = torch.cat([row, torch.zeros(self.W - row.numel(), dtype=torch.int32)])
        tok, pos, rot, pt = self.model.prepare_decode_inputs_host(
            torch.tensor(tokens, dtype=torch.int32),
            torch.tensor(positions, dtype=torch.int32),
            row[: self.W].repeat(self.BW, 1),
        )
        for h, d in ((tok, self.tok), (pos, self.pos), (rot, self.rot), (pt, self.pt)):
            ttnn.copy_host_to_device_tensor(h, d)
        ttnn.copy_host_to_device_tensor(self.rows.rows_pos_masks_host(list(positions), self.BW), self.masks)

    def body(self):
        m, BW = self.model, self.BW
        cos, sin = m._rope_from_idxs(self.rot)
        x = m.embd(m._decode_tokens_view(self.tok, BW))
        x = ttnn.to_memory_config(ttnn.reshape(x, (1, 1, BW, x.shape[-1])), DRAM)
        taps = []
        # the per-row KV-write index tensors, once per pass for every layer (instead of slice + reshape per layer)
        row_idx = [ttnn.reshape(ttnn.slice(self.masks, (b, 0), (b + 1, BW)), (BW,)) for b in range(BW)]
        for layer in m.layers:
            layer.attention.rows_pos_rows = row_idx
        for li, layer in enumerate(m.layers):
            y = self.rows.layer_forward_rows(
                layer,
                x,
                self.R,
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
        for layer in m.layers:
            layer.attention.rows_pos_rows = None
        for t in row_idx:
            ttnn.deallocate(t)
        cat = ttnn.concat(taps, dim=-1)  # [1, 1, BW, n_taps * H], replicated
        ttnn.copy(cat, self.dr.c_taps[self.R])
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


class DFlashServing:
    """Single-user DFlash step with MtpServing's single-user interface (generator_vllm drives it as its `_mtp`)."""

    adaptive = multi = per_slot = in_trace = False

    def __init__(self, gen, k: int = 3, drafter=None):
        self.gen, self.k, self.R = gen, int(k), int(k) + 1
        self.model = gen.model
        self.mesh = self.model.mesh_device
        self.H, self.V = int(gen.args.dim), int(gen.args.vocab_size)
        self.dr = drafter if drafter is not None else load_drafter(self.mesh, self.R)
        if self.dr.rows != self.R:
            raise ValueError(f"drafter built for {self.dr.rows} verify rows, the step verifies {self.R}")
        self.dr.lm_head, self.dr.vocab = self.model.lm_head_weight, self.V
        self.window = int(os.environ.get("DFLASH_WINDOW", "1024"))
        self.ver = None
        self._hooks = None
        self._pos = self._tok = None
        self.stats = None
        self._warned_sampling = False

    # ------------------------------------------------------------------ startup (before any capture)
    def allocate_kv(self, kv_shape, dtype):
        pass  # the drafter's cache is allocated with the drafter; the target's KV belongs to vLLM

    def warm(self):
        """Host library resident and pinned, verify inputs allocated, every program compiled (verify + drafter pass)."""
        t0 = time.time()
        for li in range(len(self.model.layers)):  # the host library's RAM copies load in the background since build
            HYB.lib_layer(li)
        logger.info(f"DFlash: host library resident ({time.time() - t0:.1f} s wait)")
        HYB.fetch.finalize(HYB.loader, HYB.stream)  # HYB_PINX: pinned zero-copy expert library (before any capture)
        import threading

        logger.info(f"DFlash: python threads {[t.name for t in threading.enumerate()]}")
        self.ver = Verify(self.gen, self.dr, self.R)
        # positions far from any real prompt's rows, on the warm-up page table (blocks the warm-up owns)
        self.ver.stage([0] * self.R, list(range(8, 8 + self.R)), self.gen.page_table[0])
        self.ver.run()
        self.ver.run()
        self.dr.reset()
        self.dr.stage_pass([], 0)
        self.dr.run("pass", self.dr.pass_body)
        ttnn.synchronize_device(self.mesh)

    def warm_prefill_path(self):
        pass  # the served prefill is the generator's eager path (warmed by its own _warm_prefill)

    def capture(self):
        """The per-request traces are captured after each prefill; at startup only the warm-up generation runs (fills
        the fetch LRU slots and touches the pinned / locked host pages, so the first request runs at steady speed)."""
        n = int(os.environ.get("DFLASH_WARMUP", "64"))
        if n:
            r = self.generate(WARMUP_PROMPT, n, self.gen.page_table[:1])
            logger.info(f"DFlash warm-up: {r['tokens']} tokens at {r['tok_s']} tok/s")

    # ------------------------------------------------------------------ prefill
    def release(self):
        """Drop the per-request traces (before any eager device work)."""
        if self.ver is not None and self.ver.trace is not None:
            ttnn.synchronize_device(self.mesh)
        if self.ver is not None:
            self.ver.release()
        for name in list(self.dr.traces):
            ttnn.release_trace(self.mesh, self.dr.traces.pop(name))
        self.dr.outs = {}

    def begin_prefill(self, prompt_ids, page_table_row, slot: int = 0):
        if int(slot) != 0:
            raise ValueError(f"DFlash serving is single-user (--max-num-seqs 1); prefill into slot {slot}")
        self._log_stats("previous request")
        self.release()
        self.dr.reset()
        self._row = torch.as_tensor(page_table_row).reshape(-1).to(torch.int32).clone()
        self._P = len(prompt_ids)
        captured = {li: [] for li in self.dr.taps}
        orig = {}
        for li in self.dr.taps:
            layer = self.model.layers[li]
            orig[li] = layer.forward

            def hook(*a, _f=orig[li], _li=li, **kw):
                y = _f(*a, **kw)
                captured[_li].append(ttnn.to_torch(y, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh, dim=-1)).float())
                return y

            layer.forward = hook
        self._hooks = (orig, captured)

    def _unhook(self):
        if self._hooks is None:
            return None
        orig, captured = self._hooks
        for li, f in orig.items():
            self.model.layers[li].forward = f
        self._hooks = None
        return captured

    def abort_prefill(self):
        self._unhook()

    def end_prefill(self, tok0, P):
        captured = self._unhook()
        if captured is None:
            raise RuntimeError("end_prefill without begin_prefill")
        P = int(P)
        if P != self._P:
            raise ValueError(f"end_prefill P={P} but begin_prefill saw {self._P} prompt tokens")
        # the prefill layer outputs are REPLICATED ([.., H] per die): the mesh concat holds tp copies side by side
        feats = torch.cat(
            [torch.cat([c.reshape(-1, c.shape[-1])[:, : self.H] for c in captured[li]])[:P] for li in self.dr.taps],
            dim=-1,
        )
        if feats.shape[0] != P:
            raise RuntimeError(f"tap hooks captured {feats.shape[0]} rows for a {P}-token prompt")
        W = self.window
        self.hist = [feats[-W:].clone()]  # host copies of the context features (refills)
        self.n_hist = self.hist[0].shape[0]
        self.dr.fill_prompt(feats[-W:], start=max(0, P - W))
        self._capture()
        self._pos, self._tok, self._prev_acc = P, int(tok0), []
        self._warned_sampling = False
        self.stats = {"passes": 0, "tokens": 0, "hist": [0] * self.R, "draft": 0.0, "verify": 0.0, "wall": 0.0}
        self.stats["refills"] = 0
        self.stats["stage"] = 0.0

    def _capture(self):
        ttnn.synchronize_device(self.mesh)
        self.dr.capture("pass", self.dr.pass_body)
        self.ver.capture()
        ttnn.synchronize_device(self.mesh)

    # ------------------------------------------------------------------ decode
    def step(self, tok: int, pos: int, page_table_row=None, host_authoritative=False, sampling=None) -> list[int]:
        """Returns the accepted drafts followed by the verified next token (1..R ids). The step owns the continuation
        (as MtpServing.step: on the plugin's steady async path its host token can lag by a step); the plugin's tok / pos
        are checked when it flags the host inputs authoritative (the first decode after a prefill)."""
        if self._pos is None:
            raise RuntimeError("DFlash step before any prefill")
        if host_authoritative and (int(tok), int(pos)) != (self._tok, self._pos):
            raise RuntimeError(
                f"DFlash step input (tok {tok}, pos {pos}) differs from the step's continuation "
                f"(tok {self._tok}, pos {self._pos})"
            )
        if sampling is not None and sampling[0] > 0 and not self._warned_sampling:
            logger.warning(f"DFlash serving is greedy: request sampling {sampling} decoded with argmax")
            self._warned_sampling = True
        if page_table_row is not None:
            self._row = torch.as_tensor(page_table_row).reshape(-1).to(torch.int32)
        a0 = time.perf_counter()
        dr, ver, K, R, W = self.dr, self.ver, self.k, self.R, self.window
        t, p = self._tok, self._pos
        if dr.need_refill():  # the window is full: refill from the last W features (traces re-captured)
            self.release()
            self.hist = [torch.cat(self.hist)[-W:]]
            self.n_hist = self.hist[0].shape[0]
            dr.fill_prompt(self.hist[0], start=p - self.n_hist)
            self._prev_acc = []
            self._capture()
            self.stats["refills"] += 1
        s0 = time.perf_counter()
        dr.stage_pass(self._prev_acc, t)  # the previous verify's accepted rows (c_taps[R] filled by the verify)
        s1 = time.perf_counter()
        drafts = dr.run("pass", None)[1 : 1 + K]
        a1 = time.perf_counter()
        ver.stage([t] + drafts, list(range(p, p + R)), self._row)
        s2 = time.perf_counter()
        top = ver.run()[:R]
        a2 = time.perf_counter()
        a = 0
        while a < K and drafts[a] == top[a]:
            a += 1
        rows_h = ttnn.to_torch(ttnn.get_device_tensors(dr.c_taps[R])[0]).float().reshape(R, -1)[: a + 1]
        self.hist.append(rows_h)
        self.n_hist += rows_h.shape[0]
        if self.n_hist > 2 * W:  # bound the host copy
            self.hist = [torch.cat(self.hist)[-W:]]
            self.n_hist = W
        out = drafts[:a] + [top[a]]
        self._prev_acc = list(range(p, p + a + 1))
        self._tok, self._pos = top[a], p + a + 1
        st = self.stats
        st["passes"] += 1
        st["tokens"] += len(out)
        st["hist"][a] += 1
        st["draft"] += a1 - a0
        st["verify"] += a2 - a1
        st["stage"] += (s1 - s0) + (s2 - a1)
        st["wall"] += time.perf_counter() - a0
        if st["passes"] % 64 == 0:
            self._log_stats("running")
        return out

    def _log_stats(self, why):
        st = self.stats
        if st and st["passes"]:
            n = st["passes"]
            logger.info(
                f"DFLASH_STATS ({why}): {n} passes, {st['tokens'] / n:.2f} tokens per pass, acceptance hist "
                f"{st['hist']}, draft {1e3 * st['draft'] / n:.1f} + verify {1e3 * st['verify'] / n:.1f} ms per pass, "
                f"{st['refills']} refills -> {st['tokens'] / st['wall']:.1f} tok/s (decode steps only)"
            )

    # ------------------------------------------------------------------ standalone generation
    def generate(self, text, n_new, page_table, chat=True):
        """Eager prefill + traced DFlash decode of one prompt on `page_table` [1, W] (the runner; the serving warm-up).
        Returns the generated ids and timings; the traces are released at the end."""
        tok = self.gen.tokenizer
        if chat:
            text = tok.apply_chat_template(
                [{"role": "user", "content": text}], add_generation_prompt=True, enable_thinking=False, tokenize=False
            )
        ids = tok.encode(text, add_special_tokens=False)
        P = len(ids)
        self.gen.reset()
        t_pf = time.time()
        self.begin_prefill(ids, page_table[0])
        try:
            logits0 = self.gen.prefill_forward(torch.tensor(ids).reshape(1, -1), page_table=page_table, prompt_lens=[P])
        except Exception:
            self.abort_prefill()
            raise
        t0 = int(torch.as_tensor(logits0).reshape(-1, logits0.shape[-1])[-1, : self.V].argmax())
        self.end_prefill(t0, P)
        t_pf = time.time() - t_pf
        HYB.reset_stats()
        out, pos = [t0], P
        t1 = time.time()
        while len(out) < n_new:
            new = self.step(out[-1], pos)
            out += new
            pos += len(new)
        dt = time.time() - t1
        self.release()
        st, n = self.stats, max(1, self.stats["passes"])
        return {
            "prompt_tokens": P,
            "ids": out[:n_new],
            "tokens": len(out),
            "passes": st["passes"],
            "tokens_per_pass": round((len(out) - 1) / n, 3),
            "accept_hist": st["hist"],
            "ms_per_token": round(1e3 * dt / (len(out) - 1), 1),
            "tok_s": round((len(out) - 1) / dt, 2),
            "ms_per_pass": {
                "draft": round(1e3 * st["draft"] / n, 1),
                "verify": round(1e3 * st["verify"] / n, 1),
                "stage": round(1e3 * st["stage"] / n, 2),  # host input uploads of both (inside draft + verify)
            },
            "prefill_s": round(t_pf, 2),
            "drafter_refills": st["refills"],
        }


def build(gen, k):
    """generator_vllm's QWEN35_EXT_SPEC factory."""
    return DFlashServing(gen, k)

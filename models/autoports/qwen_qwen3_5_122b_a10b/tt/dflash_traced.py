# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Traced DFlash 2 step for ONE user: the MTP step's verify pass / commit (tt/mtp_traced.py) with the DFlash drafter
(tt/dflash_device.py) in place of the MTP head.

    pass   : MtpTraced's pass over BW = bs rows [t, d_1 .. d_{bs-1}] (rows mode), plus the context layers' outputs of
             every row concatenated into `feats` [1,1,BW,8H] (persistent).
    commit : MtpTraced's commit (GDN replay of the accepted a + 1 rows), skipped when every draft was accepted.
    ctx    : the drafter's K / V of the accepted rows (feats rows 0..a) into its rings.
    draft  : the drafter over [anchor, mask...] -> per-shard top-32 candidates of every row + the selector projection,
             read back; the selector path runs on the host (tt/dflash_ref.DFlash2Ref.select's math).

Per round the host does: write the drafts -> pass (read the per-row argmax, or the candidates when sampling) -> accept
-> commit (when a < bs - 1) -> ctx -> draft (read the candidates). Greedy accept = exact match; sampled requests use the
exact speculative accept (tt/mtp_sampling.SpecSampler) with the selector's distribution as q.
"""
from __future__ import annotations

import os
import time

import torch

import ttnn
from models.autoports.qwen_qwen3_5_122b_a10b.tt.mtp_rows import layer_forward_rows
from models.autoports.qwen_qwen3_5_122b_a10b.tt.mtp_traced import MtpTraced

_DRAM = ttnn.DRAM_MEMORY_CONFIG


class DFlashTraced(MtpTraced):
    def __init__(self, gen, drafter, bs: int | None = None):
        self.dr = drafter
        bs = int(bs or drafter.bs)
        super().__init__(gen, None, bs - 1, bw=bs)
        self.bs = bs
        self.lids = set(drafter.layer_ids)
        # [1,1,BW,8H] persistent: the pass's context rows (the ctx program's input; tools also write host rows here)
        self.feats = ttnn.from_torch(
            torch.zeros(1, 1, self.BW, len(self.lids) * self.H, dtype=torch.bfloat16), dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT, device=self.mesh, memory_config=_DRAM,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )
        from safetensors import safe_open

        with safe_open(os.path.join(drafter.ckpt_dir, "model.safetensors"), "pt") as f:
            self.pre_cb = f.get_tensor("candidate_selector.predecessor_codebook").float()
            self.suc_cb = f.get_tensor("candidate_selector.successor_codebook").float()
        self.top_k = 16
        self.timing.update({"ctx": [], "draft_dev": [], "select": []})

    # ------------------------------------------------------------------------------------------------ programs
    def _pass_body(self):
        m = self.model
        BW = self.BW
        for dn, rec, taps in zip(self.dn_layers, self.snap_rec, self.snap_conv):
            ttnn.copy(dn.rec_state, rec)
            for c, t in zip(dn.conv_states, taps):
                ttnn.copy(c, t)
        cos, sin = m._rope_from_idxs(self.p_rot)
        tok = m._decode_tokens_view(self.p_tok, BW)
        x = m.embd(tok)
        x = ttnn.reshape(x, (1, 1, BW, x.shape[-1]))
        x = ttnn.to_memory_config(x, _DRAM)
        gi = 0
        caps = []
        for li, layer in enumerate(m.layers):
            y = layer_forward_rows(
                layer, x, self.plen, cos=cos, sin=sin, position_tensor=self.p_pos, page_table=self.p_pt,
                saved=self.saved, rows_pos_masks=self.p_masks, gdn_index=gi,
            )
            if not layer.is_full_attention:
                gi += 1
            ttnn.deallocate(x)
            x = y
            if li in self.lids:
                c = ttnn.clone(x, memory_config=_DRAM)
                caps.append(self.dr._full(c))
        feats = ttnn.concat(caps, dim=3, memory_config=_DRAM)  # [1,1,BW,8H]
        for c in caps:
            ttnn.deallocate(c)
        ttnn.copy(feats, self.feats)
        ttnn.deallocate(feats)
        h = m._final_norm_decode(x)
        ttnn.deallocate(x)
        logits = ttnn.linear(h, m.lm_head_weight, compute_kernel_config=m._lm_head_ckc())
        ttnn.deallocate(h)
        am, mx, lg_rm = self._argmax_max(logits)
        for t in (cos, sin):
            ttnn.deallocate(t)
        return am, mx, lg_rm, logits

    def _ctx_body(self):
        return self.dr.ctx_body(self.feats)

    def _draft_body(self):
        logits, hp = self.dr.draft_body()
        return (logits, hp)

    _BODIES = {"pass": "_pass_body", "commit": "_commit_body", "ctx": "_ctx_body", "draft": "_draft_body"}

    # ------------------------------------------------------------------------------------------------ warm / capture
    def warm(self, flush=None):
        self.plen.stage(self.bs)
        self.plen_commit.stage(1)
        self.dr.stage_ctx([0], self.BW)
        self.dr.stage_draft(0, 1, self.bs)
        bodies = (("pass", self._pass_body), ("pass", self._pass_body), ("commit", self._commit_body),
                  ("ctx", self._ctx_body), ("draft", self._draft_body), ("draft", self._draft_body))
        for name, body in bodies:
            outs = body()
            if name in ("pass", "draft"):
                for t in self._cands_body(outs[3] if name == "pass" else outs[0]):
                    ttnn.deallocate(t)
            for t in outs:
                ttnn.deallocate(t)
            if flush:
                flush()
        self.dr.reset()
        ttnn.synchronize_device(self.mesh)

    def capture(self, flush=None):
        for name in ("pass", "commit", "ctx", "draft"):
            tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
            outs = getattr(self, self._BODIES[name])()
            ttnn.end_trace_capture(self.mesh, tid, cq_id=0)
            self.traces[name] = tid
            self.outs[name] = outs
        for name, src in (("pass", 3), ("draft", 0)):
            tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
            outs = self._cands_body(self.outs[name][src])
            ttnn.end_trace_capture(self.mesh, tid, cq_id=0)
            self.traces["cands_" + name] = tid
            self.outs["cands_" + name] = outs
        ttnn.synchronize_device(self.mesh)

    def _exec(self, name):
        """Replay (or run eagerly before capture, freeing the outputs) without reading anything back."""
        if name in self.traces:
            ttnn.execute_trace(self.mesh, self.traces[name], cq_id=0, blocking=False)
            return self.outs[name], False
        return getattr(self, self._BODIES[name])(), True

    # ------------------------------------------------------------------------------------------------ host side
    def ctx(self, positions):
        """The drafter's K / V of feats rows 0..len(positions)-1 at `positions` (no read-back)."""
        t0 = time.perf_counter()
        self.dr.stage_ctx(positions, self.BW)
        outs, eager = self._exec("ctx")
        self.timing["ctx"].append(time.perf_counter() - t0)

    def draft(self, anchor: int, start: int, sampler=None):
        """bs - 1 drafts after `anchor` at position start - 1 (the block starts at `start`). Returns (drafts, q rows or
        None) - q_i the selector's distribution over the vocab (sampled requests)."""
        t0 = time.perf_counter()
        self.dr.stage_draft(anchor, start, self.bs)
        outs, eager = self._exec("draft")
        if eager:
            cand_outs = self._cands_body(outs[0])
        else:
            ttnn.execute_trace(self.mesh, self.traces["cands_draft"], cq_id=0, blocking=False)
            cand_outs = self.outs["cands_draft"]
        vals, ids = self._read_cands(cand_outs)  # [ROWS, n] each
        hp = ttnn.to_torch(ttnn.get_device_tensors(outs[1])[0]).float().reshape(-1, outs[1].shape[-1])
        if eager:
            for t in tuple(outs) + tuple(cand_outs):
                ttnn.deallocate(t)
        t1 = time.perf_counter()
        self.timing["draft_dev"].append(t1 - t0)
        n = self.bs - 1
        v, j = torch.topk(vals[1 : 1 + n], self.top_k, dim=-1)  # rows 1..bs-1 = draft positions
        cands = torch.gather(ids[1 : 1 + n], 1, j)
        sampling = sampler is not None and not sampler.greedy
        prev = int(anchor)
        drafts, qs = [], []
        suc = self.suc_cb[cands]  # [n, k, r]
        hpn = hp[1 : 1 + n]
        for i in range(n):
            sc = v[i] + suc[i] @ (self.pre_cb[prev] * hpn[i])
            if sampling:
                q = torch.softmax(sc / sampler.temperature, dim=-1)
                u = float(torch.rand(1, generator=sampler.gen))
                c = int(torch.searchsorted(torch.cumsum(q, 0), torch.tensor([u])).clamp(max=self.top_k - 1))
                dense = torch.zeros(self.V)
                dense[cands[i]] = q
                qs.append(dense)
            else:
                c = int(sc.argmax())
            prev = int(cands[i, c])
            drafts.append(prev)
        self.timing["select"].append(time.perf_counter() - t1)
        return drafts, (qs if sampling else None)

    def verify(self, rows, pos: int, want_logits=False, sampler=None):
        assert len(rows) == self.BW
        return super().verify(rows, pos, want_logits=want_logits, sampler=sampler)

    def round(self, tok: int, pos: int, sampler=None):
        """One DFlash round from the last emitted token `tok` at position `pos` (its K/V not yet written). The drafter's
        context must hold every verified position < pos. Returns the emitted ids (accepted drafts + next token)."""
        sampling = sampler is not None and not sampler.greedy
        drafts, qs = self.draft(tok, pos, sampler)
        rows = [tok] + drafts
        if sampling:
            _preds, target = self.verify(rows, pos, want_logits=True, sampler=sampler)
            a, nxt = sampler.accept(target, drafts, qs)
        else:
            preds = self.verify(rows, pos)
            a = 0
            for d, p in zip(drafts, preds):
                if d != p:
                    break
                a += 1
            nxt = preds[a]
        if a < len(drafts):
            self.commit(a + 1)
        self.ctx([pos + i for i in range(a + 1)])
        return rows[1 : a + 1] + [nxt]

    def generate(self, tok0: int, pos: int, max_new_tokens: int, sampler=None, log=None):
        out = [tok0]
        lens = []
        while len(out) < max_new_tokens:
            em = self.round(out[-1], pos, sampler)
            out.extend(em)
            pos += len(em)
            lens.append(len(em))
            if log:
                log(f"round {len(lens)}: +{len(em)} (total {len(out)})")
        return out[:max_new_tokens], lens

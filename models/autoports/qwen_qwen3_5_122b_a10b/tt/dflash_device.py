# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DFlash 2 drafter on the device (one user), the programs of the traced DFlash step (tt/dflash_traced.py).

    Same math as the host reference tt/dflash_ref.py. Weights are replicated on every die (bfp8 matmul weights); the
    target's embedding and vocab-sharded LM head are reused. Fixed shapes throughout (32 block rows, 16 real), so both
    programs are captured once and replayed:

    ctx   : feats [1,1,32,8H] (the verify pass's context-layer rows, row i at position pos_i) -> fc -> hidden_norm ->
            per layer k / v (k_norm, RoPE) scattered into the layer's K / V ring [1, nkv, R, hd] at slot pos_i mod R
            (one-hot [R, 32] built on device from the host's slot row; rows marked -1 are not written).
    draft : the block [anchor, mask x 15 (+16 pad rows)] -> 6 layers (non-causal attention over [ring; block] with a
            visibility row written by the host) -> final norm -> LM head logits (vocab-sharded, every row) and the
            selector's hidden projection [1,1,32,256] (read back with the logits' per-shard top-k candidates).

    Sliding window: one visibility row for the whole block (positions in (start - window, start)); rows 1..15 of a
    block therefore see up to 15 positions their own window would drop - this only touches draft quality (the verify
    pass decides every emitted token), never the output.
"""
from __future__ import annotations

import json
import os

import torch

import ttnn

_DRAM = ttnn.DRAM_MEMORY_CONFIG
TILE = 32
ROWS = 32  # block rows on device (bs <= 32 real)


def _rot_matrix(hd):
    """x @ M == rotate_half(x) for [.., hd] rows."""
    m = torch.zeros(hd, hd)
    h = hd // 2
    for i in range(h):
        m[i + h, i] = -1.0  # out[i] = -x[i + h]
        m[i, i + h] = 1.0  # out[i + h] = x[i]
    return m


class DFlashDevice:
    def __init__(self, model, ckpt_dir, *, ring=2272, wdtype=ttnn.bfloat8_b, cache_dir=None):
        from safetensors.torch import load_file

        self.model = model
        self.mesh = model.mesh_device
        cfg = json.load(open(os.path.join(ckpt_dir, "config.json")))
        dc = cfg["dflash_config"]
        self.H = cfg["hidden_size"]
        self.nh, self.nkv, self.hd = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
        self.g = self.nh // self.nkv
        self.eps = cfg["rms_norm_eps"]
        self.window = cfg["sliding_window"]
        self.theta = cfg["rope_parameters"]["rope_theta"]
        self.bs = dc["block_size"]
        self.mask_id = dc["mask_token_id"]
        self.layer_ids = list(dc["target_layer_ids"])
        self.L = cfg["num_hidden_layers"]
        self.K = dc["conv_kernel_size"]
        self.gs = dc["conv_group_size"]
        self.I = cfg["intermediate_size"]
        self.R = int(ring)
        assert self.R % TILE == 0 and self.R >= self.window + 2 * ROWS
        assert self.K == 2, "the device conv is written for two taps"
        self.ckpt_dir = ckpt_dir
        H, nh, nkv, hd = self.H, self.nh, self.nkv, self.hd
        G = H // self.gs
        cache_dir = cache_dir or os.path.join(os.environ.get("TT_CACHE_PATH", "/tmp"), "dflash2_o122")
        os.makedirs(cache_dir, exist_ok=True)
        w = load_file(os.path.join(ckpt_dir, "model.safetensors"))
        rep = ttnn.ReplicateTensorToMesh(self.mesh)

        def put(name, t, dtype=wdtype, layout=ttnn.TILE_LAYOUT):
            return ttnn.as_tensor(
                t.contiguous(), dtype=dtype, layout=layout, device=self.mesh, memory_config=_DRAM, mesh_mapper=rep,
                cache_file_name=os.path.join(cache_dir, f"{name}_{str(dtype).split('.')[-1]}"),
            )

        # tensor parallel over the mesh (DFLASH_TP=1, default on several dies): q / kv heads and the MLP's intermediate
        # columns split per die, o-proj / down-proj partial sums all-reduced, fc column-split + all-gathered; the
        # dynamic convs, norms and the selector projection stay replicated (they act on the full hidden)
        D = getattr(self.mesh, "get_num_devices", lambda: 1)()
        self.tp = D > 1 and os.environ.get("DFLASH_TP", "1") == "1"
        self.D = D if self.tp else 1
        assert nh % self.D == 0 and nkv % self.D == 0 and self.I % self.D == 0 and H % self.D == 0
        self.nh_l, self.nkv_l, self.I_l = nh // self.D, nkv // self.D, self.I // self.D
        shard = ttnn.ShardTensorToMesh(self.mesh, dim=0) if self.tp else rep

        def put_tp(name, per_die, dtype=wdtype):
            """per_die: list of D [K, N] weights -> each die its own (replicated as one when not TP)."""
            if not self.tp:
                return put(name, per_die[0])
            t = torch.stack([x.contiguous() for x in per_die]).unsqueeze(1)  # [D, 1, K, N]
            return ttnn.as_tensor(
                t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=self.mesh, memory_config=_DRAM, mesh_mapper=shard,
                cache_file_name=os.path.join(cache_dir, f"{name}_tp{self.D}_{str(dtype).split('.')[-1]}"),
            )

        def cols(t, d, n):  # [K, N] column block d of n
            c = t.shape[1] // n
            return t[:, d * c : (d + 1) * c]

        def rows(t, d, n):
            r = t.shape[0] // n
            return t[d * r : (d + 1) * r]

        def vec(name, t):
            return put(name, t.float().reshape(1, 1, 1, -1), dtype=ttnn.bfloat16)

        def norm_w(name, t):
            # ttnn.rms_norm takes the weight as a ROW_MAJOR [hd / 32, 32] stick view
            return put(name, t.float().reshape(1, 1, -1, TILE), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

        fcT = w["fc.weight"].T  # [8H, H]
        self.fc = put_tp("fc", [cols(fcT, d, self.D) for d in range(self.D)])  # column split + all-gather
        self.hidden_norm = norm_w("hidden_norm", w["hidden_norm.weight"])
        self.final_norm = norm_w("norm", w["norm.weight"])
        self.sel_proj = put("sel_proj", w["candidate_selector.hidden_projection.weight"].T, dtype=ttnn.bfloat16)
        # one-hot group expansion [G, H] (dynamic kernel of group j -> channels j*gs .. (j+1)*gs - 1)
        E = torch.zeros(G, H)
        for j in range(G):
            E[j, j * self.gs : (j + 1) * self.gs] = 1.0
        self.expand = put("expand", E, dtype=ttnn.bfloat16)
        # block-diagonal expansion of the 2 x K dynamic kernels at once ([2K G, 2K H]; 0 / 1 entries, exact in bfp8)
        E4 = torch.block_diag(*([E] * (2 * self.K)))
        self.expand_all = put("expand_all", E4, dtype=ttnn.bfloat8_b)
        sh = torch.zeros(ROWS, ROWS)
        for i in range(1, ROWS):
            sh[i, i - 1] = 1.0  # row i <- row i - 1 (row 0 <- 0)
        self.shift = put("shift", sh.reshape(1, 1, ROWS, ROWS), dtype=ttnn.bfloat16)
        self.rot = put("rot", _rot_matrix(hd), dtype=ttnn.bfloat16)
        self.layers = []
        for i in range(self.L):
            p = f"layers.{i}"
            a = f"{p}.self_attn"
            Dn = self.D
            wq, wk, wv = w[f"{a}.q_proj.weight"], w[f"{a}.k_proj.weight"], w[f"{a}.v_proj.weight"]  # [heads*hd, H]
            wgate, wup = w[f"{p}.mlp.gate_proj.weight"], w[f"{p}.mlp.up_proj.weight"]  # [I, H]
            lw = {
                "in_ln": norm_w(f"{p}.in_ln", w[f"{p}.input_layernorm.weight"]),
                "post_ln": norm_w(f"{p}.post_ln", w[f"{p}.post_attention_layernorm.weight"]),
                # die d: q heads [d nh_l, (d+1) nh_l), kv heads [d nkv_l, (d+1) nkv_l) (GQA groups stay on one die)
                "wqkv": put_tp(f"{p}.wqkv", [torch.cat([rows(wq, d, Dn), rows(wk, d, Dn), rows(wv, d, Dn)], 0).T
                                             for d in range(Dn)]),
                "wkv": put_tp(f"{p}.wkv", [torch.cat([rows(wk, d, Dn), rows(wv, d, Dn)], 0).T for d in range(Dn)]),
                "q_norm": norm_w(f"{p}.q_norm", w[f"{a}.q_norm.weight"]),
                "k_norm": norm_w(f"{p}.k_norm", w[f"{a}.k_norm.weight"]),
                "wo": put_tp(f"{p}.wo", [rows(w[f"{a}.o_proj.weight"].T, d, Dn) for d in range(Dn)]),
                "wgu": put_tp(f"{p}.wgu", [torch.cat([rows(wgate, d, Dn), rows(wup, d, Dn)], 0).T for d in range(Dn)]),
                "wdown": put_tp(f"{p}.wdown", [rows(w[f"{p}.mlp.down_proj.weight"].T, d, Dn) for d in range(Dn)]),
            }
            for conv in ("attention_conv", "mlp_conv"):
                proj = w[f"{p}.{conv}.kernel_projection.weight"]  # [2*K*G, H], rows (half, tap, group)
                base = w[f"{p}.{conv}.base_kernel"]  # [2, K, H]
                lw[f"{conv}.proj_all"] = put(f"{p}.{conv}.proj_all", proj.T, dtype=ttnn.bfloat16)
                lw[f"{conv}.base_all"] = vec(f"{p}.{conv}.base_all", base.reshape(-1))
                if os.environ.get("DFLASH_CONV_EXPAND", "all") == "all":
                    continue  # the per-part kernels are only read by the older expansion paths
                for half in range(2):
                    for tap in range(self.K):
                        r0 = (half * self.K + tap) * G
                        lw[f"{conv}.proj{half}{tap}"] = put(f"{p}.{conv}.proj{half}{tap}", proj[r0 : r0 + G].T,
                                                            dtype=ttnn.bfloat16)
                        lw[f"{conv}.base{half}{tap}"] = vec(f"{p}.{conv}.base{half}{tap}", base[half, tap])
            # K / V ring of this layer (replicated), zero-initialised; visibility comes from the host row
            # slots [0, R): context positions (slot = position mod R); [R, R + ROWS): the current block's own rows
            lw["ring_k"] = ttnn.from_torch(torch.zeros(1, self.nkv_l, self.R + ROWS, hd), dtype=ttnn.bfloat16,
                                           layout=ttnn.TILE_LAYOUT, device=self.mesh, memory_config=_DRAM,
                                           mesh_mapper=rep)
            lw["ring_v"] = ttnn.clone(lw["ring_k"])
            self.layers.append(lw)
        del w
        # ---- persistent per-round inputs (host-written)
        def zeros(shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(torch.zeros(*shape), dtype=dtype, layout=layout, device=self.mesh,
                                   memory_config=_DRAM, mesh_mapper=rep)

        self._zeros = zeros
        self._ones = {}
        self._cstage = {}  # ctx rows CR -> (slot [1,1,1,CR] fp32: target slot per row, -1 = skip; cos; sin)
        self.d_cos = zeros((1, 1, ROWS, hd))
        self.d_sin = zeros((1, 1, ROWS, hd))
        self.d_vis = zeros((1, 1, 1, self.R + ROWS))  # 0 visible / -inf hidden, ring slots then block rows
        self.d_mask = zeros((1, 1, ROWS, self.R + ROWS))
        kc = 256 if (self.R + ROWS) % 256 == 0 else 128 if (self.R + ROWS) % 128 == 0 else 32
        self.sdpa_cfg = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.mesh.compute_with_storage_grid_size(), q_chunk_size=ROWS,
            k_chunk_size=kc, exp_approx_mode=False,
        )
        self.conv_expand = os.environ.get("DFLASH_CONV_EXPAND", "all")
        self.d_tok = None  # built lazily from the model's decode-token layout (first draft_inputs call)
        self.feats = None  # [1,1,ROWS,8H] persistent ctx input (the verify pass copies its captures in)
        self.scale = hd ** -0.5
        self.ckc = ttnn.init_device_compute_kernel_config(
            self.mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self.ckc_hi = ttnn.init_device_compute_kernel_config(
            self.mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        # host-side state
        self.slot_pos = torch.full((self.R,), -(1 << 30), dtype=torch.long)  # position held by each ring slot

    # ------------------------------------------------------------------------------------------------ host helpers
    def _rope_rows(self, positions, n=ROWS):
        inv = 1.0 / (self.theta ** (torch.arange(0, self.hd, 2, dtype=torch.float32) / self.hd))
        p = torch.zeros(n)
        p[: len(positions)] = torch.tensor(positions, dtype=torch.float32)
        f = p[:, None] * inv[None]
        emb = torch.cat([f, f], dim=-1)
        return emb.cos().reshape(1, 1, n, self.hd), emb.sin().reshape(1, 1, n, self.hd)

    def _host(self, t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(t, dtype=dtype, layout=layout)

    def _write(self, dev_t, host_t):
        ttnn.copy_host_to_device_tensor(host_t, dev_t)

    def reset(self):
        self.slot_pos[:] = -(1 << 30)

    def ctx_stage(self, CR):
        if CR not in self._cstage:
            z = self._zeros
            iota = ttnn.from_torch(
                torch.arange(self.R + ROWS, dtype=torch.float32).reshape(1, 1, -1, 1).expand(1, 1, self.R + ROWS, CR)
                .contiguous(),
                dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.mesh, memory_config=_DRAM,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
            )  # [1,1,R+ROWS,CR]: slot index down the rows (compared with the per-row target slot, broadcast over rows)
            self._cstage[CR] = (z((1, 1, 1, CR), dtype=ttnn.float32), z((1, 1, CR, self.hd)), z((1, 1, CR, self.hd)),
                                iota)
            self._ones_hd(CR)  # persistent before any capture
        return self._cstage[CR]

    def _ones_hd(self, CR):
        if CR not in self._ones:
            self._ones[CR] = ttnn.from_torch(
                torch.ones(1, 1, CR, self.hd), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh,
                memory_config=_DRAM, mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh))
        return self._ones[CR]

    def stage_ctx(self, positions, CR):
        """Rows 0..n-1 of the next CR-row ctx run land at `positions` (rows past n are skipped). Only the last
        window + bs positions can ever be visible: older rows are skipped (a long prefill segment)."""
        slot_t, cos_t, sin_t, _iota = self.ctx_stage(CR)
        slots = torch.full((1, 1, 1, CR), -1.0)
        last = positions[-1] if len(positions) else 0
        for i, p in enumerate(positions):
            if p > last - self.R + 2 * ROWS:
                slots[0, 0, 0, i] = float(p % self.R)
                self.slot_pos[p % self.R] = int(p)
        self._write(slot_t, self._host(slots, dtype=ttnn.float32))
        c, s = self._rope_rows(positions, CR)
        self._write(cos_t, self._host(c))
        self._write(sin_t, self._host(s))

    def stage_draft(self, anchor, start, bs=None):
        bs = bs or self.bs
        c, s = self._rope_rows(list(range(start, start + ROWS)))
        self._write(self.d_cos, self._host(c))
        self._write(self.d_sin, self._host(s))
        vis = torch.full((1, 1, 1, self.R + ROWS), -1e30)  # finite: a key chunk may be fully hidden for a row
        ok = (self.slot_pos < start) & (self.slot_pos > start - self.window)
        vis[0, 0, 0, : self.R][ok] = 0.0
        vis[0, 0, 0, self.R : self.R + bs] = 0.0
        self._write(self.d_vis, self._host(vis))
        toks = torch.full((ROWS,), self.mask_id, dtype=torch.int32)
        toks[0] = int(anchor)
        self._write_tokens(toks)

    # ------------------------------------------------------------------------------------------------ device pieces
    def _tok_host(self, toks):
        tok, _p, _r, _pt = self.model.prepare_decode_inputs_host(
            toks, torch.zeros(ROWS, dtype=torch.int32), torch.zeros(ROWS, 1, dtype=torch.int32)
        )
        return tok

    def _write_tokens(self, toks):
        if self.d_tok is None:
            self.d_tok = ttnn.to_device(self._tok_host(toks), self.mesh, memory_config=_DRAM)
        else:
            self._write(self.d_tok, self._tok_host(toks))

    def _mm(self, a, b, hi=False):
        return ttnn.matmul(a, b, memory_config=_DRAM, compute_kernel_config=self.ckc_hi if hi else self.ckc)

    def _rms(self, x, w):
        return ttnn.rms_norm(x, epsilon=self.eps, weight=w, memory_config=_DRAM, compute_kernel_config=self.ckc_hi)

    def _rope(self, x, cos, sin):
        """x [1, n, ROWS, hd] (cos / sin [1,1,ROWS,hd] broadcast over n)."""
        xr = ttnn.matmul(x, self.rot, memory_config=_DRAM, compute_kernel_config=self.ckc_hi)
        a = ttnn.multiply(x, cos, memory_config=_DRAM)
        b = ttnn.multiply(xr, sin, memory_config=_DRAM)
        ttnn.deallocate(xr)
        out = ttnn.add(a, b, memory_config=_DRAM)
        ttnn.deallocate(a)
        ttnn.deallocate(b)
        return out

    def _heads(self, x, n, rows=ROWS):
        """[1,1,rows,n*hd] -> [1,n,rows,hd]."""
        x = ttnn.reshape(x, (1, rows, n, self.hd))
        return ttnn.permute(x, (0, 2, 1, 3))

    def _conv(self, lw, conv, half, h, dyn):
        """One grouped dynamic causal conv over the block rows: sum over taps of (base[tap] + dyn[tap]) * shift^tap(h),
        dyn[tap] the [1,1,ROWS,G] dynamic kernel (expanded to the channels here)."""
        out = None
        x = h
        for tap in range(self.K):
            if tap > 0:
                x = ttnn.matmul(self.shift, x, memory_config=_DRAM, compute_kernel_config=self.ckc_hi)
            de = self._mm(dyn[tap], self.expand, hi=True)  # [1,1,ROWS,H]
            k = ttnn.add(de, lw[f"{conv}.base{half}{tap}"], memory_config=_DRAM)
            ttnn.deallocate(de)
            term = ttnn.multiply(k, x, memory_config=_DRAM)
            ttnn.deallocate(k)
            if tap > 0:
                ttnn.deallocate(x)
            out = term if out is None else ttnn.add(out, term, memory_config=_DRAM)
        return out

    def ctx_body(self, feats):
        """feats [1,1,CR,8H] -> the K / V rows of every layer written into the rings at the staged slots."""
        CR = int(feats.shape[-2])
        slot_t, cos_t, sin_t, iota = self.ctx_stage(CR)
        x = self._mm(feats, self.fc)  # [1,1,CR,H / D]
        if self.tp:
            x = self._gather(x)
        x = self._rms(x, self.hidden_norm)
        # one-hot [1,1,R,CR]: slot j <- row i
        sel = ttnn.eq(iota, slot_t, memory_config=_DRAM)  # [1,1,R,CR] fp32
        sel = ttnn.typecast(sel, ttnn.bfloat16)
        sel_h = ttnn.repeat(sel, (1, self.nkv_l, 1, 1))
        # written slots as a full [1, nkv, R+ROWS, hd] predicate (one-hot rows x ones): the ring update is then one
        # in-place select per tensor
        written = ttnn.matmul(sel_h, self._ones_hd(CR), memory_config=_DRAM, compute_kernel_config=self.ckc_hi)
        for lw in self.layers:
            kv = self._mm(x, lw["wkv"])  # [1,1,ROWS,2*nkv*hd]
            k = ttnn.slice(kv, (0, 0, 0, 0), (1, 1, CR, self.nkv_l * self.hd))
            v = ttnn.slice(kv, (0, 0, 0, self.nkv_l * self.hd), (1, 1, CR, 2 * self.nkv_l * self.hd))
            ttnn.deallocate(kv)
            k = self._heads(k, self.nkv_l, CR)
            k = self._rms(k, lw["k_norm"])
            k = self._rope(k, cos_t, sin_t)
            v = self._heads(v, self.nkv_l, CR)
            for ring, rows in ((lw["ring_k"], k), (lw["ring_v"], v)):
                new = ttnn.matmul(sel_h, rows, memory_config=_DRAM, compute_kernel_config=self.ckc_hi)  # [1,nkv,R,hd]
                ttnn.where(written, new, ring, output_tensor=ring)
                ttnn.deallocate(new)
                ttnn.deallocate(rows)
        for t in (x, sel, sel_h, written):
            ttnn.deallocate(t)
        return ()

    def draft_body(self):
        """The block through the drafter -> (logits [1,1,ROWS,V/TP] vocab-sharded TILE, hp [1,1,ROWS,256])."""
        self._make_mask()
        x = self.embed_body()
        for lw in self.layers:
            x = self.layer_body(lw, x)
        return self.head_body(x)

    def _make_mask(self):
        """The [1,1,ROWS,R+ROWS] attention mask (every block row gets the host's visibility row)."""
        m = ttnn.repeat(self.d_vis, (1, 1, ROWS, 1))
        ttnn.copy(m, self.d_mask)
        ttnn.deallocate(m)

    def embed_body(self):
        m = self.model
        x = m.embd(m._decode_tokens_view(self.d_tok, ROWS))
        x = ttnn.to_memory_config(ttnn.reshape(x, (1, 1, ROWS, x.shape[-1])), _DRAM)
        return self._full(x)

    def _dyn(self, lw, conv, h):
        """The conv's 4 dynamic kernels from its input h, expanded to the channels with the base kernel added:
        {(half, tap): [1,1,ROWS,H]} (one matmul per part + a column repeat; DFLASH_CONV_EXPAND=mm: expansion matmul)."""
        out = {}
        if self.conv_expand == "all":
            d = self._mm(h, lw[f"{conv}.proj_all"], hi=True)  # [1,1,ROWS,2K G]
            de = self._mm(d, self.expand_all, hi=True)  # [1,1,ROWS,2K H]
            ttnn.deallocate(d)
            k = ttnn.add(de, lw[f"{conv}.base_all"], memory_config=_DRAM)
            ttnn.deallocate(de)
            for half in range(2):
                for tap in range(self.K):
                    c0 = (half * self.K + tap) * self.H
                    out[(half, tap)] = ttnn.slice(k, (0, 0, 0, c0), (1, 1, ROWS, c0 + self.H))
            ttnn.deallocate(k)
            return out
        for half in range(2):
            for tap in range(self.K):
                d = self._mm(h, lw[f"{conv}.proj{half}{tap}"], hi=True)  # [1,1,ROWS,G]
                if self.conv_expand == "mm":
                    de = self._mm(d, self.expand, hi=True)
                else:
                    de = ttnn.repeat_interleave(d, self.gs, dim=3)
                ttnn.deallocate(d)
                out[(half, tap)] = ttnn.add(de, lw[f"{conv}.base{half}{tap}"], memory_config=_DRAM)
                ttnn.deallocate(de)
        return out

    def _conv_apply(self, kern, half, x):
        """sum over taps of kern[(half, tap)] * shift^tap(x) (causal along the block rows)."""
        out = ttnn.multiply(kern[(half, 0)], x, memory_config=_DRAM)
        xs = x
        for tap in range(1, self.K):
            xs2 = ttnn.matmul(self.shift, xs, memory_config=_DRAM, compute_kernel_config=self.ckc_hi)
            if xs is not x:
                ttnn.deallocate(xs)
            xs = xs2
            out2 = ttnn.addcmul(out, kern[(half, tap)], xs, value=1.0, memory_config=_DRAM)  # out + k * shift(x)
            ttnn.deallocate(out)
            out = out2
        if xs is not x:
            ttnn.deallocate(xs)
        return out

    def layer_body(self, lw, x):
        nh, nkv = self.nh_l, self.nkv_l
        res = x
        h = self._rms(x, lw["in_ln"])
        kern = self._dyn(lw, "attention_conv", h)
        hc = self._conv_apply(kern, 0, h)
        ttnn.deallocate(h)
        qkv = self._mm(hc, lw["wqkv"])  # [1,1,ROWS,(nh+2nkv)*hd]
        ttnn.deallocate(hc)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv, num_heads=nh, num_kv_heads=nkv, transpose_k_heads=False, memory_config=_DRAM
        )  # [1,nh,ROWS,hd], [1,nkv,ROWS,hd] x 2
        ttnn.deallocate(qkv)
        q2 = ttnn.experimental.rotary_embedding(self._rms(q, lw["q_norm"]), self.d_cos, self.d_sin, memory_config=_DRAM)
        k2 = ttnn.experimental.rotary_embedding(self._rms(k, lw["k_norm"]), self.d_cos, self.d_sin, memory_config=_DRAM)
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        # the block's own K / V into the ring's last ROWS slots (in place, tile aligned), then flash attention over the
        # ring with the visibility mask (non-causal)
        ttnn.fill_cache(lw["ring_k"], k2, 0, update_idx=self.R)
        ttnn.fill_cache(lw["ring_v"], v, 0, update_idx=self.R)
        ttnn.deallocate(k2)
        ttnn.deallocate(v)
        o = ttnn.transformer.scaled_dot_product_attention(
            q2, lw["ring_k"], lw["ring_v"], attn_mask=self.d_mask, is_causal=False, scale=self.scale,
            program_config=self.sdpa_cfg, compute_kernel_config=self.ckc_hi, memory_config=_DRAM,
        )  # [1,nh,ROWS,hd]
        ttnn.deallocate(q2)
        o = ttnn.experimental.nlp_concat_heads(o, memory_config=_DRAM)  # [1,1,ROWS,nh*hd]
        o2 = self._reduce(self._mm(o, lw["wo"]))
        ttnn.deallocate(o)
        o3 = self._conv_apply(kern, 1, o2)
        ttnn.deallocate(o2)
        for t in kern.values():
            ttnn.deallocate(t)
        x = ttnn.add(res, o3, memory_config=_DRAM)
        ttnn.deallocate(o3)
        ttnn.deallocate(res)
        res = x
        h = self._rms(x, lw["post_ln"])
        kern = self._dyn(lw, "mlp_conv", h)
        hc = self._conv_apply(kern, 0, h)
        ttnn.deallocate(h)
        gu = self._mm(hc, lw["wgu"])
        ttnn.deallocate(hc)
        gate = ttnn.slice(gu, (0, 0, 0, 0), (1, 1, ROWS, self.I_l))
        up = ttnn.slice(gu, (0, 0, 0, self.I_l), (1, 1, ROWS, 2 * self.I_l))
        ttnn.deallocate(gu)
        a = ttnn.multiply(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], memory_config=_DRAM)
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        mo = self._reduce(self._mm(a, lw["wdown"]))
        ttnn.deallocate(a)
        mo2 = self._conv_apply(kern, 1, mo)
        ttnn.deallocate(mo)
        for t in kern.values():
            ttnn.deallocate(t)
        x = ttnn.add(res, mo2, memory_config=_DRAM)
        ttnn.deallocate(mo2)
        ttnn.deallocate(res)
        return x

    def head_body(self, x):
        m = self.model
        h = self._rms(x, self.final_norm)
        ttnn.deallocate(x)
        logits = ttnn.linear(h, m.lm_head_weight, compute_kernel_config=m._lm_head_ckc())  # [1,1,ROWS,V/TP]
        hp = self._mm(h, self.sel_proj, hi=True)  # [1,1,ROWS,256]
        ttnn.deallocate(h)
        return logits, hp

    def _reduce(self, part):
        """Sum of the dies' [1,1,ROWS,H] partials (TP), replicated; identity when not TP."""
        if not self.tp:
            return part
        from models.tt_transformers.tt.ccl import tt_all_reduce

        # on a (1, N) mesh tt_all_reduce is a reduce-scatter (each die gets H / N columns of the sum): gather after it
        out = tt_all_reduce(part, self.mesh, self.model.tt_ccl, cluster_axis=0, dim=3,
                            topology=self.model.args.ccl_topology(), memory_config=_DRAM)
        if out.shape[-1] != self.H:
            out = self._gather(out)
        if out.is_sharded():
            out = ttnn.sharded_to_interleaved(out, _DRAM)
        if tuple(out.shape) != (1, 1, ROWS, self.H):
            out = ttnn.reshape(out, (1, 1, ROWS, self.H))
        return out

    def _gather(self, x):
        from models.tt_transformers.tt.ccl import tt_all_gather

        return tt_all_gather(x, self.mesh, self.model.tt_ccl, cluster_axis=None, dim=3,
                             topology=self.model.args.ccl_topology())

    def _full(self, x):
        """A [1,1,ROWS,H/TP] fractured row block -> replicated [1,1,ROWS,H] (no-op when already full)."""
        if x.shape[-1] == self.H:
            return x
        from models.tt_transformers.tt.ccl import tt_all_gather

        return tt_all_gather(x, self.mesh, self.model.tt_ccl, cluster_axis=None, dim=3,
                             topology=self.model.args.ccl_topology())

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Decode sampling from the device's top-32 candidates (the vLLM plugin's sample_on_device_mode, decode steps).

On device, inside the decode trace: the logits [1, 1, W, Vp] fp32 -> bf16 -> qwen36 decode_sampling.RowsTopK (one fused
scan + merge program, <= 8 live rows per call, UINT32 indices above 65536 columns) twice, the second on the rows shifted
by the first pass's max (see DeviceTopK.device). Only [W, 32] indices + values cross PCIe (vs [W, Vp] fp32 logits), and vLLM's CPU sampler over the
full vocab is skipped. On the host: vLLM's semantics over the candidates (greedy at temperature 0; temperature, top-k,
top-p; per-request seeds; repetition / frequency / presence penalties applied to the candidates). Truncating the
distribution to the top 32 is what models/common/sampling (max_top_k 32) does for the other ports.
"""

from __future__ import annotations

import torch

import ttnn

K = 32
GROUP = 8  # RowsTopK's live-row limit
# I2M_TOPK_GATHER (default 1): a sharded head's per-die candidates all-gathered on device (one die read back)
GATHER = __import__("os").environ.get("I2M_TOPK_GATHER", "1") == "1"


class DeviceTopK:
    def __init__(self, mesh, Vp: int, V: int, sharded: bool = False):
        """Vp: one die's logits columns. sharded: the LM head is split over the vocab (die d holds columns
        [d * Vp, (d + 1) * Vp)): each die's candidates are its slice's top K, shifted by the max over ALL dies (one
        small all-gather) so the host can merge them by value."""
        from models.demos.blackhole.qwen36.tt.decode_sampling.ops import RowsTopK

        self.mesh, self.Vp, self.V = mesh, Vp, V
        self.N = mesh.get_num_devices()
        self.sharded = sharded and self.N > 1
        self.rtk = RowsTopK(mesh, Vp, k=K)

    def device(self, lg, W: int):
        """Device half (traceable): lg [1, 1, W, Vp] fp32 TILE -> [(indices [1, 1, rows, K], logits - row max
        [1, 1, rows, K] bf16, rows)] per group of <= 8 rows. Two fused top-k passes: the first's best bf16 value (within
        one bf16 ulp of the row max) shifts the fp32 row so the candidates sit near 0, where bf16 resolves them to
        ~1e-3 (softmax is shift-invariant; the arg max stays exact); a gather of the fp32 values would cost 3.5 ms."""
        lb = ttnn.typecast(lg, ttnn.bfloat16)
        parts = []
        for g0 in range(0, W, GROUP):
            rows = min(GROUP, W - g0)
            part_b = lb if W <= GROUP else ttnn.slice(lb, [0, 0, g0, 0], [1, 1, g0 + rows, self.Vp])
            part_f = lg if W <= GROUP else ttnn.slice(lg, [0, 0, g0, 0], [1, 1, g0 + rows, self.Vp])
            v1, _ = self.rtk(part_b, rows)
            top = ttnn.typecast(ttnn.slice(v1, [0, 0, 0, 0], [1, 1, rows, 1]), ttnn.float32)
            if self.sharded:  # every die shifts by the same value: the max over the dies' maxima
                tops = ttnn.all_gather(top, dim=3, cluster_axis=1, topology=ttnn.Topology.Linear)  # [1, 1, rows, N]
                top = ttnn.max(tops, dim=-1, keepdim=True)
            if int(part_f.shape[2]) != rows:
                part_f = ttnn.slice(part_f, [0, 0, 0, 0], [1, 1, rows, self.Vp])
            shifted = ttnn.typecast(ttnn.subtract(part_f, top), ttnn.bfloat16)
            v2, i2 = self.rtk(shifted, rows)
            i2, v2 = ttnn.slice(i2, [0, 0, 0, 0], [1, 1, rows, K]), ttnn.slice(v2, [0, 0, 0, 0], [1, 1, rows, K])
            if self.sharded and GATHER:  # every die's candidates on every die: the host reads one die (2 reads, not 2 N)
                i2 = ttnn.all_gather(i2, dim=3, cluster_axis=1, topology=ttnn.Topology.Linear)  # [1, 1, rows, N K]
                v2 = ttnn.all_gather(v2, dim=3, cluster_axis=1, topology=ttnn.Topology.Linear)
            parts.append((i2, v2, rows))
        return parts

    def host(self, parts):
        """-> host (indices [W, K] int64, logits [W, K] fp32) by row, sorted descending (sharded head: the dies'
        candidates merged by value, ties to the lower index)."""
        devs = (lambda t: ttnn.get_device_tensors(t)) if self.N > 1 else (lambda t: [t])
        nd = self.N if self.sharded else 1
        idx, val = [], []
        for i, v, r in parts:
            if self.sharded and GATHER:  # die 0 holds every die's K candidates, die by die
                ii = ttnn.to_torch(devs(i)[0]).reshape(r, nd, K).to(torch.int64)
                ii = (ii + self.Vp * torch.arange(nd).reshape(1, nd, 1)).reshape(r, nd * K)
                vv = ttnn.to_torch(devs(v)[0]).float().reshape(r, nd * K)
            else:
                ii = [ttnn.to_torch(t).reshape(r, K).to(torch.int64) + d * self.Vp for d, t in enumerate(devs(i)[:nd])]
                vv = [ttnn.to_torch(t).float().reshape(r, K) for t in devs(v)[:nd]]
                ii, vv = torch.cat(ii, -1), torch.cat(vv, -1)
            if nd > 1:
                # by value descending, then by index ascending (stable sort on the index order first)
                o = torch.argsort(ii, dim=-1, stable=True)
                ii, vv = ii.gather(-1, o), vv.gather(-1, o)
                o = torch.argsort(vv, dim=-1, descending=True, stable=True)[:, :K]
                ii, vv = ii.gather(-1, o), vv.gather(-1, o)
            idx.append(ii)
            val.append(vv)
        return torch.cat(idx), torch.cat(val)

    def __call__(self, lg, W: int):
        return self.host(self.device(lg, W))


class DeviceSampler:
    """The token on device from DeviceTopK.device's candidates (tt/kernels/tsample.cpp, one core per row group): argmax_j
    (logit_j + tg_j) over the candidates with id < V, tg = T * Gumbel noise from the host (zeros: greedy, the host
    sampler's first-max tie rule). Exactly the host sampler's softmax(logits / T) over the same candidates when the
    request has no penalties / top-k < 32 / top-p (eligible). Written straight into the decode's token input."""

    def __init__(self, topk: DeviceTopK):
        self.tk = topk
        self.nt = topk.N if (topk.sharded and GATHER) else 1
        mesh = topk.mesh
        rep = ttnn.ReplicateTensorToMesh(mesh) if topk.N > 1 else None
        self.rep = rep
        self.tg = ttnn.from_torch(torch.zeros(32, K), dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh,
                                  memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=rep)
        core = ttnn.CoreCoord(0, 0)
        self.crs = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])

    def device(self, parts, tokens, W: int):
        """parts: DeviceTopK.device(lg, W); tokens: uint32 [1, W] ROW_MAJOR (traceable)."""
        import hashlib
        from pathlib import Path

        g0 = 0
        for i2, v2, rows in parts:
            io = [i2, v2, self.tg, tokens]
            ct = [0, rows, g0, W, self.nt, self.tk.Vp if self.nt > 1 else 0, self.tk.V]
            for t in io:
                ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
            rt = ttnn.RuntimeArgs()
            rt[0][0] = [t.buffer_address() for t in io]
            k = ttnn.KernelDescriptor(
                kernel_source=str(Path(__file__).parent / "kernels" / "tsample.cpp"), core_ranges=self.crs,
                compile_time_args=ct, runtime_args=rt,
                config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0,
                                                         noc=ttnn.NOC.NOC_0))
            nb = self.nt * 6144 + 8 * 128 + 128 + self.nt * K * 8
            fmt = ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.float32, page_size=nb)
            prog = ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=[
                ttnn.CBDescriptor(total_size=nb, core_ranges=self.crs, format_descriptors=[fmt])])
            key = ("i2m_tsample", tuple(ct))
            prog.custom_program_hash = int.from_bytes(hashlib.sha1(repr(key).encode()).digest()[:8], "little") & (
                (1 << 63) - 1)
            ttnn.generic_op(io, prog)
            g0 += rows
        return tokens

    def set_noise(self, tg):
        """tg: torch [W <= 32, K] (T * Gumbel; zeros = greedy), by physical slot."""
        x = torch.zeros(32, K)
        x[: tg.shape[0]] = tg.float()
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT,
                                                        mesh_mapper=self.rep), self.tg)


def device_ok(sp, r) -> bool:
    """Row r's sampling is exactly DeviceSampler's: no penalties, no top-k truncation below the candidates (top-k 1 =
    greedy), no top-p."""
    if float(_param(sp, "repetition_penalty", r, 1.0)) != 1.0 or float(_param(sp, "frequency_penalty", r, 0.0)) != 0.0 \
            or float(_param(sp, "presence_penalty", r, 0.0)) != 0.0:
        return False
    tk = int(_param(sp, "top_k", r, -1))
    tp = float(_param(sp, "top_p", r, 1.0))
    if tk == 1:
        return True
    return (tk <= 0 or tk >= K) and not (0.0 < tp < 1.0)


def _param(sp, name, r, default):
    v = getattr(sp, name, None)
    if v is None:
        return default
    if isinstance(v, torch.Tensor):
        return v.reshape(-1)[r].item() if v.numel() > r else default
    if isinstance(v, (list, tuple)):
        return v[r] if len(v) > r else default
    return v


class HostSampler:
    """vLLM-semantics sampling over each row's top-K candidates."""

    def __init__(self, V: int):
        self.V = V
        self.gens: dict[int, tuple[int, torch.Generator]] = {}
        # sampled drafts: the drafter's temperature = qt x the request's (any q keeps the acceptance exact; qt tunes
        # how well q matches the target's p: I2M_SPEC_QT)
        self.qt = float(__import__("os").environ.get("I2M_SPEC_QT", "1.0"))

    def forget(self, row: int):
        self.gens.pop(row, None)

    def _gen(self, row, seed):
        if seed is None or int(seed) < 0:
            return None
        g = self.gens.get(row)
        if g is None or g[0] != int(seed):
            gen = torch.Generator()
            gen.manual_seed(int(seed))
            self.gens[row] = g = (int(seed), gen)
        return g[1]

    def dist(self, ids, vals, r, sp, outs=None, prompt=None):
        """One row's candidates (ids, logits [K]) under vLLM's semantics -> (ids, probs) after penalties, temperature,
        top-k, top-p; probs None at temperature 0 (ids then holds the greedy token alone). outs / prompt: the
        request's output / prompt tokens (penalties)."""
        ids, lg = ids.clone(), vals.double().clone()
        ok = ids < self.V  # the LM head's zero-padded vocab columns are never sampled
        ids, lg = ids[ok], lg[ok]
        rep = float(_param(sp, "repetition_penalty", r, 1.0))
        fre = float(_param(sp, "frequency_penalty", r, 0.0))
        pre = float(_param(sp, "presence_penalty", r, 0.0))
        if (rep != 1.0 or fre != 0.0 or pre != 0.0) and outs is not None:
            outs = outs[outs >= 0].to(torch.int64)
            cnt = (ids[:, None] == outs[None, :]).sum(-1).double()
            if rep != 1.0:
                seen = cnt > 0
                if prompt is not None:
                    pr = prompt[prompt >= 0].to(torch.int64)
                    seen = seen | (ids[:, None] == pr[None, :]).any(-1)
                lg = torch.where(seen, torch.where(lg > 0, lg / rep, lg * rep), lg)
            lg = lg - fre * cnt - pre * (cnt > 0).double()
        temp = float(_param(sp, "temperature", r, 0.0))
        if temp <= 0.0:
            i = int(torch.argmax(lg))  # ties: the lower index (candidates sorted descending)
            return ids[i : i + 1], None
        lg = lg / temp
        order = torch.argsort(lg, descending=True)
        ids, lg = ids[order], lg[order]
        tk = int(_param(sp, "top_k", r, -1))
        if 0 < tk < lg.numel():
            ids, lg = ids[:tk], lg[:tk]
        p = torch.softmax(lg, -1)
        tp = float(_param(sp, "top_p", r, 1.0))
        if 0.0 < tp < 1.0:
            keep = torch.cumsum(p, -1) - p < tp
            keep[0] = True
            ids, p = ids[keep], p[keep] / p[keep].sum()
        return ids, p

    def sample(self, idx, vals, rows, sp, output_tokens=None, prompt_tokens=None):
        """idx / vals [W, K] by slot; rows: list of (vLLM row r, slot s). Returns {r: token}."""
        out = {}
        for r, s in rows:
            ids, p = self.dist(idx[s], vals[s], r, sp, None if output_tokens is None else output_tokens[r],
                               None if prompt_tokens is None else prompt_tokens[r])
            if p is None:
                out[r] = int(ids[0])
                continue
            gen = self._gen(r, _param(sp, "seed", r, None))
            out[r] = int(ids[int(torch.multinomial(p, 1, generator=gen))])
        return out

    def gumbel(self, r, sp, d, k):
        """[d, k] T * Gumbel noise for a sampled draft (zeros at temperature 0: the drafter's argmax)."""
        temp = float(_param(sp, "temperature", r, 0.0))
        if temp <= 0.0:
            return torch.zeros(d, k)
        u = torch.rand(d, k, generator=self._gen(r, _param(sp, "seed", r, None)), dtype=torch.float64)
        return (self.qt * temp * -torch.log(-torch.log(u.clamp(1e-12, 1 - 1e-12)))).float()

    def spec(self, idx, vals, r, drafts, sp, outs=None, prompt=None, q=None):
        """Speculative acceptance. idx / vals [d + 1, K]: verify row j scores the token after drafts[:j]. q: per draft,
        the drafter's (ids, logits) candidates it was sampled from (q_j = softmax(logits / T)); None: a point-mass
        (greedy) proposal. Draft j is kept with probability min(1, p_j / q_j) (greedy target: iff it is the argmax); the
        first rejection draws from max(p_j - q_j, 0) renormalised; all kept: a draw from row d. The emitted tokens follow
        the target's distribution exactly. -> (tokens emitted, drafts kept)."""
        gen = self._gen(r, _param(sp, "seed", r, None))
        temp = float(_param(sp, "temperature", r, 0.0))
        hist = outs[outs >= 0].to(torch.int64) if outs is not None else torch.zeros(0, dtype=torch.int64)
        emitted = []
        for j in range(len(drafts) + 1):
            ids, p = self.dist(idx[j], vals[j], r, sp, torch.cat([hist, torch.tensor(emitted, dtype=torch.int64)]),
                               prompt)
            last = j == len(drafts)
            if p is None:  # greedy
                if not last and int(ids[0]) == drafts[j]:
                    emitted.append(drafts[j])
                    continue
                return emitted + [int(ids[0])], j
            if not last:
                pq = torch.zeros_like(p)  # q_j on p's support
                qd = 1.0
                if q is not None:
                    qi, ql = q[j]
                    qp = torch.softmax(ql.double() / (self.qt * temp), -1)
                    qd = float(qp[qi == drafts[j]].sum())
                    m = ids[:, None] == qi[None, :]
                    pq = (m.double() * qp[None, :]).sum(-1)
                else:
                    pq = (ids == drafts[j]).double()
                hit = ids == drafts[j]
                pd = float(p[hit].sum())
                if pd > 0.0 and qd > 0.0 and float(torch.rand(1, generator=gen, dtype=torch.float64)) < min(1.0, pd / qd):
                    emitted.append(drafts[j])
                    continue
                res = (p - pq).clamp(min=0.0)
                if float(res.sum()) > 0.0:
                    p = res / res.sum()
            return emitted + [int(ids[int(torch.multinomial(p, 1, generator=gen))])], j
        raise AssertionError("unreachable")

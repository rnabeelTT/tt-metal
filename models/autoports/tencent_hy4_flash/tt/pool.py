# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Shared context pool (HY4_POOL_TOKENS / tt/generator_vllm.py pool_tokens): every slot's context-proportional caches
(the 12 sparse layers' latent rows, the indexers' group keys) live in ONE pool of P tokens; a slot holds an extent
[base, base + size) of it (base a multiple of 128 tokens: its group keys at base / 4, a whole 32-group tile row).
One user can take nearly the whole pool; up to `slots` users share it.

PoolCopy: device row copies within / between the pools (gather / scatter slab_write, src-offset mode): extent moves
(overlap-safe chunk order), work-buffer copy-in / out, zeroing. Fixed programs (CH slabs per op, the offsets through one
persistent uint32 buffer written by the host before each op, ordered behind the queue): no program is compiled at
serve time after warm-up.
PoolAlloc: host bookkeeping of the extents (Hy4Model.ext: slot -> (base, size)): first fit, in-place growth,
relocation, full compaction (every extent slid to the lowest free base, in base order)."""

from __future__ import annotations

import os

import numpy as np
import torch
import ttnn

CH = 256  # slabs (32 rows / one tile row of 32 groups) per copy program
ALIGN = 128  # extent granularity (tokens): group base a multiple of 32


def rup(n, a=ALIGN):
    return -(-int(n) // a) * a


class PoolCopy:
    _OBJ = {}

    @classmethod
    def get(cls, mesh):
        if id(mesh) not in cls._OBJ:
            cls._OBJ[id(mesh)] = cls(mesh)
        return cls._OBJ[id(mesh)]

    def __init__(self, mesh):
        self.mesh = mesh
        rep = ttnn.ReplicateTensorToMesh(mesh)
        self.offs = ttnn.from_torch(torch.zeros(1, 1, 1, 2 * CH, dtype=torch.int32), dtype=ttnn.uint32,
                                    layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, mesh_mapper=rep)
        self.zsrc = ttnn.zeros((1, 1, 32 * CH, 128), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG)
        self.ops = 0

    def _run(self, src, dst, pairs):
        """One program: dst rows d = src rows s for (s, d) in pairs (<= CH; padded with the last pair)."""
        from models.autoports.tencent_hy4_flash.tt.slab_write import slab_write

        pairs = list(pairs) + [pairs[-1]] * (CH - len(pairs))
        o = np.zeros(2 * CH, dtype=np.int32)
        o[:CH] = [d for _, d in pairs]
        o[CH:] = [s for s, _ in pairs]
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(torch.from_numpy(o).reshape(1, 1, 1, -1), dtype=ttnn.uint32,
                                                        layout=ttnn.ROW_MAJOR_LAYOUT), self.offs)
        slab_write(src, dst, self.offs, CH, slab_rows=32, src_offs=True)
        self.ops += 1

    @staticmethod
    def plan(s, d, n):
        """Ops [[(src row, dst row) x <= CH]] moving rows [s, s + n) to [d, d + n) (n a multiple of 32) such that no op
        reads rows another slab of it writes and every row is read before any later op overwrites it."""
        if n <= 0 or s == d:
            return []
        k = n // 32
        lim = CH
        if abs(s - d) < n:
            lim = min(CH, abs(s - d) // 32)
            assert lim >= 1, (s, d)
        idx = range(k) if d < s else range(k - 1, -1, -1)
        ops, cur = [], []
        for j in idx:
            cur.append((s + 32 * j, d + 32 * j))
            if len(cur) == lim:
                ops.append(cur)
                cur = []
        if cur:
            ops.append(cur)
        return ops

    def move(self, tensors, s, d, n):
        """Rows [s, s + n) -> [d, d + n) within each tensor (same row map for all; overlap-safe)."""
        for op in self.plan(s, d, n):
            for t in tensors:
                self._run(t, t, op)

    def copy(self, src, s, dst, d, n):
        """dst rows [d, d + n) = src rows [s, s + n) (different tensors; n a multiple of 32)."""
        k = n // 32
        assert n % 32 == 0, n
        for j0 in range(0, k, CH):
            self._run(src, dst, [(s + 32 * j, d + 32 * j) for j in range(j0, min(k, j0 + CH))])

    def zero(self, dst, d, n):
        """TILE dst rows [d, d + n) = 0 (n a multiple of 32)."""
        assert n % 32 == 0, n
        k = n // 32
        for j0 in range(0, k, CH):
            self._run(self.zsrc, dst, [(32 * (j - j0), d + 32 * j) for j in range(j0, min(k, j0 + CH))])

    def warm(self, latent, gpool):
        """Compile the programs (eager, before any capture): RM rows and TILE groups, a spare-row self copy each."""
        nl = int(latent.shape[-2])
        self._run(latent, latent, [(nl - 32, nl - 32)])
        ng = int(gpool.shape[-2])
        self._run(gpool, gpool, [(ng - 32, ng - 32)])
        self.zero(gpool, ng - 32, 32)


class PoolFull(RuntimeError):
    pass


class PoolAlloc:
    """Extents of the pool's P tokens (ext: slot -> (base, size), shared with the model's layers)."""

    def __init__(self, P, ext):
        self.P = int(P)
        self.ext = ext

    def used(self):
        return sum(int(v[1]) for v in self.ext.values())

    def gaps(self, exclude=None):
        """Free [(base, size)] in base order (exclude: a slot whose extent counts as free)."""
        ex = sorted((int(b), int(n)) for u, (b, n) in self.ext.items() if u != exclude)
        out, cur = [], 0
        for b, n in ex:
            if b > cur:
                out.append((cur, b - cur))
            cur = max(cur, b + n)
        if cur < self.P:
            out.append((cur, self.P - cur))
        return out

    def room_after(self, slot):
        """Tokens free right after slot's extent."""
        b, n = self.ext[slot]
        nxt = min([int(v[0]) for u, v in self.ext.items() if u != slot and int(v[0]) >= b + n] + [self.P])
        return nxt - (b + n)

    def fit(self, size):
        """Base of the best-fitting free gap of >= size tokens (smallest such gap, lowest base), or None."""
        g = [(n, b) for b, n in self.gaps() if n >= size]
        return min(g)[1] if g else None

    def compaction(self, around=None):
        """[(slot, src base, dst base, size)]: every extent slid to the lowest free base, in base order; around=slot:
        the extents up to it slid down, the ones after it slid up to the pool's end (last first), so all the free space
        follows it (its growth in place)."""
        items = sorted(self.ext.items(), key=lambda kv: kv[1][0])
        k = next((i for i, (u, _) in enumerate(items) if u == around), len(items) - 1)
        cur, moves = 0, []
        for u, (b, n) in items[: k + 1]:
            if b != cur:
                moves.append((u, int(b), cur, int(n)))
            cur += int(n)
        end = self.P
        for u, (b, n) in reversed(items[k + 1 :]):
            d = end - int(n)
            if d != b:
                moves.append((u, int(b), d, int(n)))
            end = d
        return moves


class PoolOps:
    """Device side of extent changes for a model (+ its batched decoder): moves (latent rows + group keys, the lanes'
    bases, the active slot's layers re-addressed), compaction, zeroing grown group keys."""

    def __init__(self, model, dec=None):
        self.model, self.dec = model, dec
        self.lat = [l.attn.cache for l in model.layers if l.attn.kind == "sparse"]
        self.gp = [l.attn.indexer.gpool for l in model.layers if l.attn.indexer is not None
                   and l.attn.indexer.gpool is not None]
        self.pc = PoolCopy.get(model.mesh) if self.gp else None
        self.pa = PoolAlloc(model.pool_tokens, model.ext)
        self.stats = {"moves": 0, "moved_tokens": 0, "compactions": 0}

    def warm(self):
        if self.pc is not None:
            self.pc.warm(self.lat[0], self.gp[0])

    def zero_tail(self, slot, n0):
        """Group keys of slot's extent past its first n0 tokens zeroed (as a reset slot's)."""
        b, n = self.model.ext[slot]
        g0, g1 = (b + n0) // 4, (b + n) // 4
        if g1 > g0 and self.pc is not None:
            for gp in self.gp:
                self.pc.zero(gp, g0, g1 - g0)

    def move(self, slot, nb):
        """Slot's extent rows (latent rows, group keys) to base nb: exact copies, overlap-safe. The caller has no pass
        in flight."""
        m = self.model
        b, n = m.ext[slot]
        if nb == b:
            return
        assert nb % ALIGN == 0 and 0 <= nb and nb + n <= self.pa.P, (nb, n)
        self.pc.move(self.lat, b, nb, n)
        self.pc.move(self.gp, b // 4, nb // 4, n // 4)
        m.ext[slot] = (nb, n)
        if self.dec is not None:
            for ln in self.dec.lanes:
                if ln is not None and ln["slot"] == slot:
                    ln["base"]["sparse"], ln["gbase"] = nb, nb // 4
            self.dec._pre = None
        if m.slot == slot:
            m.refresh_bases()
        self.stats["moves"] += 1
        self.stats["moved_tokens"] += n

    def compact(self, around=None) -> bool:
        mv = self.pa.compaction(around)
        for u, s, d, n in mv:
            self.move(u, d)
        if mv:
            self.stats["compactions"] += 1
        return bool(mv)


def pool_env():
    """HY4_POOL_TOKENS (> 0): the shared pool's tokens (a multiple of 2048); 0 / unset: fixed per-slot extents."""
    v = int(os.environ.get("HY4_POOL_TOKENS", "0") or 0)
    return v if v > 0 else None

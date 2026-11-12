# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Elastic full layers (HYB_ELASTIC_DEMOTE): device memory moves between full MoE layers and the KV cache with the
context a request needs, instead of a fixed profile.

The model is built with HYB_FULL_LAYERS = the short profile's set (e.g. 44 layers) and a KV pool for the short context
(QWEN35_KV_TOKENS, e.g. 32K) while max_seq_len (RoPE tables, page-table width) is the long profile's (128K). A request
that needs more context than the pool holds first DEMOTES the layers listed in HYB_ELASTIC_DEMOTE (the long profile's
hybrid layers that the short profile keeps full) and grows the pool to max_seq_len; a later request that fits the
short pool PROMOTES them back and shrinks the pool. Both happen in begin_prefill, before the request's prefill, with
no trace live (DFlashServing.release()); the next capture records the new layer set.

A layer changes kind by being rebuilt the way the model build made it: its feed-forward module is dropped (its expert
tensors deallocated) and the decoder layer's own _build_feed_forward runs again with the layer moved between
HYB.full_layers and the hybrid set, so the patched init produces exactly the module a static build of that profile
produces (the routed experts come from the HYB_WCACHE device images: no checkpoint conversion, the same bytes). The
host side follows: the int4 library rows, the pinned device-format library (fetch.finalize_more), the stream / fetch
registrations. The KV pool is reallocated (generator.resize_kv) empty, or with its blocks kept when a request grows past it mid-decode;
the page table's width follows the pool (the paged K / V update needs the table no wider than the cache).

env: HYB_ELASTIC_DEMOTE (comma list of layers), HYB_ELASTIC_MARGIN (tokens a request may generate past its prompt,
default 4096: a prompt of P tokens needs P + margin), HYB_ELASTIC_FORCE (long: switch to the long profile at the first
request, for exactness A/B tests)."""

from __future__ import annotations

import gc
import os
import time

import torch
from loguru import logger

import ttnn


class Elastic:
    def __init__(self, hyb, gen, srv, demote):
        self.hyb, self.gen, self.srv = hyb, gen, srv
        self.D = sorted(int(L) for L in demote)
        missing = [L for L in self.D if L not in hyb.ff_build]
        if missing:
            raise RuntimeError(f"HYB_ELASTIC_DEMOTE layers {missing} were not recorded at build (not MoE layers?)")
        not_full = [L for L in self.D if L not in hyb.full_layers]
        if not_full:
            raise RuntimeError(f"HYB_ELASTIC_DEMOTE layers {not_full} are not in HYB_FULL_LAYERS")
        self.short_tokens = int(gen.kv_tokens)
        self.long_tokens = int(gen.max_seq_len)
        self.margin = int(os.environ.get("HYB_ELASTIC_MARGIN", "4096"))
        self.force = os.environ.get("HYB_ELASTIC_FORCE", "")
        self.long = False
        self.switches = 0
        self.bmap = None  # serving: vLLM block ids -> pool blocks (BlockMap, made by map_row's first call)
        self._last = None  # map_row's last (map epoch, vLLM row, pool row, mapped blocks): decode extends it
        logger.info(
            f"HYB_ELASTIC: {len(self.D)} demotable layers, KV pool {self.short_tokens} tokens (short) / "
            f"{self.long_tokens} (long), margin {self.margin}"
        )

    @property
    def capacity(self):
        return int(self.gen.kv_tokens)

    def prepare(self, prompt_len):
        """Before a request's prefill (traces released): the profile its context needs. Returns True if it changed."""
        need = int(prompt_len) + self.margin
        want_long = need > self.short_tokens or self.force == "long"
        if need > self.long_tokens:
            raise ValueError(f"request needs {need} tokens of context, max_seq_len is {self.long_tokens}")
        if want_long == self.long:
            return False
        self._switch(want_long)
        return True

    def grow_now(self):
        """In the middle of a request whose context reached the short pool (decode past the prompt + margin estimate):
        the long profile, KEEPING the pool's blocks (ids and contents). The caller recaptures its traces."""
        if self.long:
            raise RuntimeError(f"elastic grow_now in the long profile (pool {self.capacity} tokens)")
        self._switch(True, keep=True)

    # ------------------------------------------------------------------ serving: vLLM block ids
    def map_row(self, row, n_tokens, start=0):
        """vLLM's page-table row (block ids of its 128K-wide table) -> the pool's row for a request whose first
        n_tokens positions are used (the blocks holding them mapped, the rest -> the scratch block), and the prefill
        start: a prefix-cache hit (start > 0) whose blocks lost their contents (evicted, or a profile switch dropped
        the pool) restarts at 0. Returns (pool row, start)."""
        gen = self.gen
        if self.bmap is None:
            self.bmap = BlockMap(int(gen.kv_scratch_block))
        bs = int(gen.block_size)
        row = torch.as_tensor(row).reshape(-1).to(torch.int32)
        nb = -(-int(n_tokens) // bs)
        if nb > row.numel():
            raise ValueError(f"elastic map_row: {n_tokens} tokens need {nb} blocks, the row has {row.numel()}")
        if start and not self.bmap.mapped(row[: -(-int(start) // bs)].tolist()):
            start = 0
        width = int(gen.page_table.shape[-1])  # the pool's table width (narrower than vLLM's in the short profile)
        if nb > width:
            raise RuntimeError(f"elastic map_row: {nb} blocks for {n_tokens} tokens, the pool's table has {width}")
        last = self._last
        if (
            last is not None
            and last[0] == self.bmap.epoch
            and last[2].numel() == width
            and nb >= last[3]
            and torch.equal(last[1][: last[3]], row[: last[3]])
        ):  # the same request's row, a few positions on: only the newly reached blocks
            out, n0 = last[2], last[3]
            if nb > n0:
                out[n0:nb] = torch.tensor(self.bmap.map(row[n0:nb].tolist(), pin=row[:n0].tolist()), dtype=torch.int32)
        else:
            out = torch.full((width,), int(gen.kv_scratch_block), dtype=torch.int32)
            out[:nb] = torch.tensor(self.bmap.map(row[:nb].tolist()), dtype=torch.int32)
        self._last = (self.bmap.epoch, row.clone(), out, nb)
        return out.clone(), int(start)

    # ------------------------------------------------------------------ switching
    def _switch(self, to_long, keep=False):
        t0 = time.perf_counter()
        hyb, gen, mesh = self.hyb, self.gen, self.gen.model.mesh_device
        self.srv.release()
        if hyb.fetch.pending:
            hyb.fetch.commit(hyb.slot_of)
        ttnn.synchronize_device(mesh)
        check = keep and os.environ.get("HYB_ELASTIC_CHECK") == "1"
        if check:  # the kept pool's bytes: layer 0's K blocks read back before and after (bfp8 -> fp32 is exact)
            n_old = int(gen.kv_cache_shape[0])
            before = ttnn.to_torch(ttnn.get_device_tensors(gen.kv_cache[0][0])[0]).float()[:n_old]
        if not keep:  # first: the pool's buffers are freed before the experts are reshuffled
            gen.model.free_kv_caches()
            gen.kv_cache = None
        self.t = {"build": 0.0, "fetch_load": 0.0, "lib": 0.0}
        t1 = time.perf_counter()
        for L in self.D:
            self._rebuild(L, hybrid=to_long)
        t2 = time.perf_counter()
        if to_long:
            hyb.fetch.finalize_more(self.D, hyb.loader)
        t3 = time.perf_counter()
        gen.resize_kv(self.long_tokens if to_long else self.short_tokens, keep=keep)
        t4 = time.perf_counter()
        if self.bmap is not None:
            self.bmap.reset(int(gen.kv_scratch_block), keep=keep)
        if check:
            after = ttnn.to_torch(ttnn.get_device_tensors(gen.kv_cache[0][0])[0]).float()[:n_old]
            if not torch.equal(before, after):
                raise RuntimeError(
                    f"HYB_ELASTIC_CHECK: kept KV blocks changed ({int((before != after).sum())} values differ)"
                )
            logger.info(f"HYB_ELASTIC_CHECK: layer 0 K blocks 0..{n_old} identical after the pool grew")
        self.long = to_long
        self.switches += 1
        t5 = time.perf_counter()
        self.srv.warm_after_switch()
        t6 = time.perf_counter()
        mem = gen._memory_report() if hasattr(gen, "_memory_report") else {}
        logger.info(
            f"HYB_ELASTIC: switch breakdown: release {t1 - t0:.1f} s, layers {t2 - t1:.1f} s (module builds "
            f"{self.t['build']:.1f}, host library loads {self.t['fetch_load']:.1f}, int4 rows {self.t['lib']:.1f}), pin "
            f"{t3 - t2:.1f} s, KV {t4 - t3:.1f} s, warm {t6 - t5:.1f} s"
        )
        logger.info(
            f"HYB_ELASTIC: -> {'long' if to_long else 'short'} profile{' (pool kept)' if keep else ''} in {time.perf_counter() - t0:.1f} s "
            f"(KV pool {gen.kv_tokens} tokens, {len(hyb.full_layers)} full layers, DRAM "
            f"{mem.get('dram_allocated_gib_per_device', '?')} GiB per die)"
        )

    def _rebuild(self, L, hybrid):
        hyb = self.hyb
        layer, mesh, args, cache, tt_ccl = hyb.ff_build[L]
        root = hyb.sd  # the streaming checkpoint the model was built from: a fresh, lazy view of the layer
        sd = root.layer(L) if hasattr(root, "layer") else root
        old = layer.feed_forward
        layer.feed_forward = None
        for name in ("up_gate", "down"):
            t = getattr(old, name, None)
            if t is not None:
                ttnn.deallocate(t)
        if hybrid:
            hyb.full_layers.discard(L)
        else:
            hyb.full_layers.add(L)
            hyb.fetch.drop_layer(L)
            if L in hyb.stream.layers:
                hyb.stream.layers = [v for v in hyb.stream.layers if v != L]
            for d in (hyb.sel_bias, hyb.gate_wT):
                d.pop(L, None)
        for d in (hyb.resident, hyb.slot_of, hyb.dyn_of, hyb.dyn_lru, hyb.dyn_free, hyb.lib):
            d.pop(L, None)
        del old
        gc.collect()
        a = time.perf_counter()
        layer.feed_forward = layer._build_feed_forward(mesh, args, sd, L, cache, tt_ccl)
        b = time.perf_counter()
        if hybrid:
            fl = getattr(layer.feed_forward, "_fetch_load", None)
            if fl is not None:
                fl.result()
        c = time.perf_counter()
        hyb.lib_layer(L)  # the int4 rows of the layer's new cold set (a stub for a full layer)
        d = time.perf_counter()
        self.t["build"] += b - a
        self.t["fetch_load"] += c - b
        self.t["lib"] += d - c


class BlockMap:
    """vLLM block ids -> physical KV blocks of the elastic pool (serving: vLLM sizes its block table for the long
    profile, the pool holds fewer blocks in the short one). Single user. A request's blocks are mapped in order of
    use; a block of an earlier request keeps its mapping (and its K / V: a prefix-cache hit reuses it) until the
    least-recently-used eviction needs its physical block or a profile switch drops the pool's contents (reset)."""

    def __init__(self, pool_blocks):
        from collections import OrderedDict

        self._od = OrderedDict
        self.reset(pool_blocks)

    epoch = 0  # bumped whenever mappings may have been dropped (a reused physical block, a pool reset)

    def reset(self, pool_blocks, keep=False):
        """A new pool of pool_blocks physical blocks; keep=True (the pool grew with its contents): mappings stay."""
        self.epoch += 1
        if keep:
            used = set(self.v2p.values())
            self.free += [b for b in range(self.pool, int(pool_blocks)) if b not in used]
        else:
            self.v2p = self._od()
            self.free = list(range(int(pool_blocks)))
        self.pool = int(pool_blocks)

    def mapped(self, vids):
        return all(int(v) in self.v2p for v in vids)

    def map(self, vids, pin=()):
        """Physical ids for vLLM ids vids (allocating, evicting the least recently used block not in vids / pin)."""
        keep = {int(v) for v in vids} | {int(v) for v in pin}
        out = []
        for v in vids:
            v = int(v)
            p = self.v2p.get(v)
            if p is None:
                if not self.free:
                    victim = next((u for u in self.v2p if u not in keep), None)
                    if victim is None:
                        raise RuntimeError(f"BlockMap: {len(keep)} blocks needed, the pool has {self.pool}")
                    self.free.append(self.v2p.pop(victim))
                    self.epoch += 1
                p = self.v2p[v] = self.free.pop()
            else:
                self.v2p.move_to_end(v)
            out.append(p)
        return out

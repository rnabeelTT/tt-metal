# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-die DRAM budget of a serving-shaped Hy4 build: every live device tensor reachable from the model / engine (and
the port's class-level caches), bytes per die by category, at each stage (model built, warm-up, traces captured), plus
the peak DRAM of a long prefill (sampled at every ttnn.deallocate).

    python -m models.autoports.tencent_hy4_flash.tools.dram_budget --users 8 --ctx 12288 [--long 12000] [--no-engram]

`account(roots)` is importable (tt/generator_vllm.py logs it with HY4_DRAM_ACCOUNT=1).
"""

from __future__ import annotations

import argparse
import os
import re
import time
from collections import defaultdict

import ttnn
from loguru import logger

_CATS = [  # (category, regex over the tensor's path), first match wins
    ("experts (bfp4)", r"\.mlp\.(up_gate|down)$"),
    ("fused-MoE decode bufs", r"\.mlp\.fused\."),
    ("MoE decode copies", r"\.mlp\.(dec|dec_router)\b"),
    ("MoE latent/router/shared", r"\.mlp\."),
    ("KV cache (sparse)", r"attn\.cache$#sparse"),
    ("KV cache (SWA ring)", r"attn\.cache$#swa"),
    ("indexer group keys", r"indexer\.(gkeys|gpool|_slots)"),
    ("indexer lane bufs", r"indexer\._lane|indexer\.lt\b|\._lt\b"),
    ("indexer other", r"indexer\."),
    ("attn decode copies", r"attn\.(dec|dec_ukv)\b"),
    ("attn weights", r"attn\.(wqa|wqb|wkva|wuk|wuvT|wo|g|wg|\w*norm|qs|o_)"),
    ("attn other", r"attn\."),
    ("engram device", r"\.engram\."),
    ("HC / norms", r"_hc\.|norm\."),
    ("embedding", r"\.emb$"),
    ("lm head", r"\.lm$"),
    ("lm head decode copy", r"\.lm_dec\."),
    ("drafter", r"dfly|draft|spec"),
    ("vision", r"vit|vision"),
    ("decoder persistent", r"^engine\.(dec|single|decoder)|^decoder"),
    ("class caches", r"^class:"),
]


def _bytes(t):
    try:
        return int(t.buffer_num_pages()) * int(t.buffer_aligned_page_size())
    except Exception:  # noqa: BLE001
        try:
            d0 = ttnn.get_device_tensors(t)[0]
            return int(d0.buffer_num_pages()) * int(d0.buffer_aligned_page_size())
        except Exception:  # noqa: BLE001
            return 0


def _is_dram(t):
    try:
        return t.memory_config().buffer_type == ttnn.BufferType.DRAM
    except Exception:  # noqa: BLE001
        return False


def collect(roots: dict, class_roots=()):
    """roots {name: obj} -> {buffer key: (path, bytes per die, shape, dtype)} of the live DRAM device tensors."""
    seen, out, classes = set(), {}, set()

    def rec(o, path, depth):
        if depth > 14:
            return
        if isinstance(o, ttnn.Tensor):
            try:
                if o.storage_type() != ttnn.StorageType.DEVICE or not o.is_allocated() or not _is_dram(o):
                    return
                key = int(o.buffer_address())
            except Exception:  # noqa: BLE001
                return
            if key not in out:
                kind = ""
                if path.endswith("attn.cache"):
                    kind = "#" + ("sparse" if "_sparse_" in path else "swa")
                out[key] = (path + kind, _bytes(o), tuple(o.shape), str(o.dtype).split(".")[-1])
            return
        oid = id(o)
        if oid in seen or isinstance(o, (str, bytes, int, float, bool, type(None))):
            return
        seen.add(oid)
        if isinstance(o, dict):
            for k, v in list(o.items()):
                rec(v, f"{path}[{k!r}]" if not isinstance(k, str) else f"{path}.{k}", depth + 1)
        elif isinstance(o, (list, tuple, set, frozenset)):
            for i, v in enumerate(list(o)):
                rec(v, f"{path}[{i}]", depth + 1)
        elif type(o).__module__.startswith(("models.", "__main__")) or hasattr(o, "__dict__") and type(o).__module__.startswith("models"):
            cls = type(o)
            if cls not in classes:
                classes.add(cls)
                for c in cls.__mro__:
                    for k, v in vars(c).items():
                        if isinstance(v, (dict, list, ttnn.Tensor)):
                            rec(v, f"class:{c.__name__}.{k}", depth + 1)
            d = getattr(o, "__dict__", None)
            if d:
                for k, v in list(d.items()):
                    rec(v, f"{path}.{k}", depth + 1)
            for k in getattr(cls, "__slots__", ()):
                rec(getattr(o, k, None), f"{path}.{k}", depth + 1)

    for name, o in roots.items():
        rec(o, name, 0)
    for c in class_roots:
        for k, v in vars(c).items():
            if isinstance(v, (dict, list, ttnn.Tensor)):
                rec(v, f"class:{c.__name__}.{k}", 0)
    return out


def categorize(items, model=None):
    """-> {category: bytes} (paths of the attention caches tagged by the layer kind)."""
    kinds = {}
    if model is not None:
        for i, layer in enumerate(model.layers):
            kinds[i] = layer.attn.kind
    agg = defaultdict(int)
    for _, (path, b, _, _) in items.items():
        m = re.match(r"model\.layers\[(\d+)\]\.attn\.cache#", path)
        if m:
            path = re.sub(r"#\w+$", "#" + kinds.get(int(m.group(1)), "swa"), path)
        for cat, rx in _CATS:
            core, _, tag = rx.partition("#")
            if re.search(core, path.split("#")[0]) and (not tag or path.endswith("#" + tag)):
                agg[cat] += b
                break
        else:
            agg["other: " + re.sub(r"\[\d+\]", "[*]", path)[:60]] += b
    return dict(agg)


def report(model, extra_roots=None, title="", top=12):
    """Log the categorized live-tensor bytes per die vs the allocator's used bytes; returns (cats, items)."""
    from models.autoports.tencent_hy4_flash.tt import attention, indexer, moe, rope

    roots = {"model": model}
    roots.update(extra_roots or {})
    cls = [getattr(mod, n) for mod in (attention, indexer, moe, rope) for n in dir(mod)
           if isinstance(getattr(mod, n), type) and getattr(mod, n).__module__ == mod.__name__]
    items = collect(roots, cls)
    cats = categorize(items, model)
    mv = ttnn.get_memory_view(model.mesh, ttnn.BufferType.DRAM)
    used = mv.total_bytes_allocated_per_bank * mv.num_banks
    free = mv.total_bytes_free_per_bank * mv.num_banks
    tot = sum(cats.values())
    G = 2**30
    lines = [f"DRAM BUDGET {title}: allocator used {used / G:.3f} GiB, free {free / G:.3f} GiB (largest "
             f"{mv.largest_contiguous_bytes_free_per_bank / 2**20:.0f} MiB/bank x {mv.num_banks}); tensors found "
             f"{tot / G:.3f} GiB; unaccounted {(used - tot) / G:.3f} GiB"]
    try:
        tv = ttnn.get_memory_view(model.mesh, ttnn.BufferType.TRACE)
        lines.append(f"  trace region: used {tv.total_bytes_allocated_per_bank * tv.num_banks / 2**20:.1f} MiB, free "
                     f"{tv.total_bytes_free_per_bank * tv.num_banks / 2**20:.1f} MiB")
    except Exception as e:  # noqa: BLE001
        lines.append(f"  trace region: n/a ({e})")
    for k, v in sorted(cats.items(), key=lambda kv: -kv[1]):
        lines.append(f"  {k:32s} {v / 2**20:10.1f} MiB")
    agg = defaultdict(lambda: [0, 0])  # by path with layer indices folded
    for p, b, _, _ in items.values():
        k = re.sub(r"\[\d+\]", "[*]", p.split("#")[0])
        agg[k][0] += b
        agg[k][1] += 1
    lines.append("  by path (layers folded), >= 8 MiB:")
    for k, (b, n) in sorted(agg.items(), key=lambda kv: -kv[1][0]):
        if b >= 8 << 20:
            lines.append(f"    {b / 2**20:9.1f} MiB x{n:<4d} {k}")
    big = sorted(items.values(), key=lambda x: -x[1])[:top]
    lines.append("  largest tensors:")
    for p, b, s, dt in big:
        lines.append(f"    {b / 2**20:9.1f} MiB {dt:10s} {s} {p}")
    msg = "\n".join(lines)
    logger.info(msg)
    return cats, items


class PeakSampler:
    """Track the allocator's peak used DRAM (sampled before every ttnn.deallocate, i.e. at each op's output peak)."""

    def __init__(self, mesh, every=1):
        self.mesh, self.every, self.n = mesh, every, 0
        self.peak, self.where = 0, ""
        self._orig = None

    def sample(self, tag=""):
        mv = ttnn.get_memory_view(self.mesh, ttnn.BufferType.DRAM)
        u = mv.total_bytes_allocated_per_bank * mv.num_banks
        if u > self.peak:
            self.peak, self.where = u, tag
        return u

    def __enter__(self):
        self._orig = ttnn.deallocate
        orig = self._orig

        def dealloc(t, *a, **k):
            self.n += 1
            if self.n % self.every == 0:
                self.sample(f"dealloc #{self.n}")
            return orig(t, *a, **k)

        ttnn.deallocate = dealloc
        return self

    def __exit__(self, *a):
        ttnn.deallocate = self._orig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=8)
    ap.add_argument("--ctx", type=int, default=12288, help="max_model_len per user (as vLLM's)")
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--no-engram", action="store_true")
    ap.add_argument("--no-warm", action="store_true")
    ap.add_argument("--vision", action="store_true")
    ap.add_argument("--long", type=str, default="", help="comma list of prompt lengths to prefill (peak sampled)")
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--decode", type=int, default=0, help="decode steps after each long prefill (tok/s)")
    args = ap.parse_args()
    from models.autoports.tencent_hy4_flash.tools.generate import open_mesh
    from models.autoports.tencent_hy4_flash.tt.config import Hy4Config, checkpoint_dir
    from models.autoports.tencent_hy4_flash.tt.engine import Hy4Engine
    from models.autoports.tencent_hy4_flash.tt.engine_batch import Hy4BatchEngine

    SPEC_K = int(os.environ.get("HY4_SPEC_K", "0") or 0)  # (as tt/generator_vllm.py; importing it needs vLLM with k > 0)
    from models.autoports.tencent_hy4_flash.tt.model import Hy4Model

    B = args.users
    mesh = open_mesh()
    try:
        mv = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
        logger.info(f"DRAM BUDGET empty device: free {mv.total_bytes_free_per_bank * mv.num_banks / 2**30:.3f} GiB "
                    f"({mv.num_banks} banks)")
        cfg = Hy4Config.load()
        engram = None
        if not args.no_engram:
            from models.experimental.hybrid_moe.engram_store import EngramRows

            engram = EngramRows(os.environ.get("HY4_ENGRAM_DIR", os.path.expanduser("~/.cache/hybrid-moe/hy4_engram")),
                                checkpoint_dir())
        cache_dir = os.environ.get("HY4_TENSOR_CACHE", os.path.expanduser("~/.cache/hybrid-moe/hy4_tt"))
        model = Hy4Model(mesh, cfg, max_ctx=args.ctx + 64, chunk=args.chunk, engram=engram, cache_dir=cache_dir,
                         slots=B)
        engine = Hy4BatchEngine(model, k=SPEC_K) if B > 1 else Hy4Engine(model, SPEC_K)
        roots = {"engine": engine}
        vit = None
        if args.vision:
            from models.autoports.tencent_hy4_flash.tt.vision import Hy4Vision

            vit = roots["vision"] = Hy4Vision(mesh)
        report(model, roots, f"model built ({B} x {args.ctx}, spec k {SPEC_K})")
        if args.no_warm:
            return
        if engram is not None and hasattr(engram.store, "wait_ram"):
            engram.store.wait_ram()
        t0 = time.time()
        if vit is not None:  # (as the vLLM adapter's warm-up: the ViT first, then the engine, then the captures)
            vit.warm(model)
        engine.warm()
        engine.capture()
        logger.info(f"warm-up {time.time() - t0:.0f} s")
        report(model, roots, "after warm-up + capture")
        import torch

        g = torch.Generator().manual_seed(1)
        for L in [int(x) for x in args.long.split(",") if x]:
            ids = torch.randint(1000, 100000, (L,), generator=g).tolist()
            ps = PeakSampler(mesh, args.every)
            base = ps.sample("start")
            with ps:
                t0 = time.time()
                if B > 1:
                    s = engine.alloc()
                    engine.prefill(s, ids)
                else:
                    engine.prefill(ids)
                dt = time.time() - t0
            logger.info(f"DRAM BUDGET long prefill {L}: TTFT {dt:.2f} s; used before {base / 2**30:.3f} GiB, peak "
                        f"{ps.peak / 2**30:.3f} GiB (+{(ps.peak - base) / 2**20:.0f} MiB at {ps.where}); {model.dram_report()}")
            if args.decode:
                t0 = time.time()
                for _ in range(args.decode):
                    engine.step({s: (None, None, False)}) if B > 1 else engine.step()
                logger.info(f"DRAM BUDGET decode at {L}: {args.decode / (time.time() - t0):.2f} tok/s")
            if B > 1:
                engine.finish(s)
            report(model, roots, f"after long prefill {L}", top=6)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()

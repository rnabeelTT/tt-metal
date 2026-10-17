# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""On-disk cache of the final device weights of GLM-5.3-Flash (stage 6: weight load time).

A cold build dequantizes the FP8 checkpoint on host and converts every weight to its device format (the bfp4 routed
experts dominate: ~40 s per MoE layer on the (1, 4) mesh). Each weight is built exactly as without the cache and the
resulting DEVICE tensor is dumped (ttnn.dump_tensor: the per-device shards, dtype, layout and memory config); a warm
build loads that file straight to the mesh (ttnn.load_tensor) and never reads the checkpoint tensor (the layer state
dicts are lazy, see ckpt.LazyMap). Policy retypes (KDA projections, DSA W_UK / W_UV) happen inside the build, so the
cached tensor is the final one and a warm build converts nothing. Warm weights are bit-identical to cold ones by
construction (the file holds the device bytes).

Layout: $GLM53_CACHE_DIR/<key>/{top,layer_<i>}/<component>-<digest>/<name>.tensorbin, key = cache format, mesh shape,
precision policy name plus a digest of its weight dtypes, and a digest of the checkpoint (config + tensor index): a
different mesh, policy or checkpoint gets a different directory, never a stale file; each component (top / kda / dsa /
mlp / moe) is keyed by a digest of the source that builds its weights. Files are written to a temporary name and
renamed, so a killed build leaves no truncated tensor. GLM53_CACHE_DIR=<dir> enables the cache (DEFAULT_DIR when unset;
off / none / 0 disable it). The ROW_MAJOR embedding table is not cached (its load_tensor stalled, tt/model.py).
"""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Callable

from loguru import logger

import ttnn

# Bump when the build of any cached weight changes (preprocessing, packing, layout): old directories are then ignored.
CACHE_FORMAT = 1
DEFAULT_DIR = "off"  # e.g. ~/.cache/glm53_flash once verified on device (GLM53_CACHE_DIR sets it per run)


def _digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:10]


def policy_weight_dtypes(opt) -> dict:
    """The fields of an OptConfig that change a weight's device bytes."""
    m = opt.moe.moe_precision()
    return {
        "kda_in": str(opt.kda_in_dtype),
        "kda_out": str(opt.kda_out_dtype),
        "dsa_bmm": str(opt.dsa_bmm_dtype),
        "routed_gate_up": str(m.routed_gate_up),
        "routed_down": str(m.routed_down),
        "shared": str(m.shared),
        "router": str(m.router),
        "gate_up_tile": str(opt.moe.gather_gate_up_output_tile),
    }


def checkpoint_digest(rd) -> str:
    """Config + tensor index (+ the E192 mask) of the checkpoint the reader serves."""
    mask = rd.path / "e192_mask.json"
    return _digest(
        {
            "config": rd.config,
            "files": sorted(rd._files.items()),
            "mask": mask.read_text() if mask.exists() else None,
        }
    )


@lru_cache(maxsize=None)
def component_digest(kind: str) -> str:
    """Digest of the source that turns checkpoint tensors into the cached device tensors of one component (the
    checkpoint reader / dequantization is common to all)."""
    from . import ckpt

    common = (ckpt.dequant_fp8, ckpt.CheckpointReader)
    if kind == "top":
        from .model import Glm53Model

        objs = (Glm53Model.__init__,)
    elif kind == "kda":
        import models.demos.deepseek_v3_d_p.tt.kda.weights as kda_weights

        from .kda import kda_state_dict

        objs = (load_kda_weights_cached, kda_state_dict, kda_weights)
    elif kind == "dsa":
        from .dsa import GlmDSA

        objs = (GlmDSA.__init__,)
    elif kind == "mlp":
        from .mlp import DenseMLP

        objs = (DenseMLP.__init__,)
    elif kind == "moe":
        from .functional_decoder import moe_state_dict
        from .moe import Qwen35MoE
        from .moe_opt import Qwen35MoEFused

        objs = (moe_state_dict, Qwen35MoE.__init__, Qwen35MoEFused.__init__)
    else:
        raise ValueError(f"unknown weight-cache component {kind!r}")
    return _digest([inspect.getsource(o) for o in common + objs])[:8]


class WeightCache:
    """get(name, build): the cached device tensor `name`, else build() (a device tensor), dumped for the next run."""

    def __init__(self, root: Path, mesh, scope: str = "top", stats: dict | None = None):
        self.root = Path(root)
        self.mesh = mesh
        self.scope = scope
        self.stats = stats if stats is not None else {"hits": 0, "misses": 0}  # shared by every scope of one cache

    @property
    def hits(self) -> int:
        return self.stats["hits"]

    @property
    def misses(self) -> int:
        return self.stats["misses"]

    def sub(self, scope: str) -> "WeightCache":
        return WeightCache(self.root, self.mesh, scope, self.stats)

    def component(self, kind: str) -> "WeightCache":
        """Scope of one weight-building component (top / kda / dsa / mlp / moe), keyed by a digest of the source that
        builds its weights: an edit of that code gets a fresh directory instead of a stale tensor."""
        return WeightCache(self.root, self.mesh, f"{self.scope}/{kind}-{component_digest(kind)}", self.stats)

    def file(self, name: str) -> Path:
        return self.root / self.scope / f"{name}.tensorbin"

    def has(self, *names: str) -> bool:
        return all(self.file(n).is_file() for n in names)

    def load(self, name: str) -> ttnn.Tensor:
        t0 = time.time()
        t = ttnn.load_tensor(self.file(name), device=self.mesh)
        dt = time.time() - t0
        self.stats["hits"] += 1
        if dt > 10:
            logger.warning(f"weight cache: slow load {self.file(name)} ({dt:.1f}s)")
        return t

    def store(self, name: str, t: ttnn.Tensor) -> ttnn.Tensor:
        self.stats["misses"] += 1
        path = self.file(name)
        tmp = path.with_name(f".{path.stem}.{os.getpid()}.tmp.tensorbin")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            ttnn.dump_tensor(tmp, t)
            os.replace(tmp, path)
        except (OSError, RuntimeError) as e:  # read-only or full disk: serve uncached rather than fail the load
            logger.warning(f"weight cache: cannot write {path} ({e})")
            tmp.unlink(missing_ok=True)
        return t

    def get(self, name: str, build: Callable[[], ttnn.Tensor]) -> ttnn.Tensor:
        if self.has(name):
            try:
                return self.load(name)
            except RuntimeError as e:  # corrupt file: rebuild it
                logger.warning(f"weight cache: {self.file(name)} unreadable ({e}); rebuilding")
        return self.store(name, build())


def cached(cache: WeightCache | None, name: str, build: Callable[[], ttnn.Tensor]) -> ttnn.Tensor:
    return build() if cache is None else cache.get(name, build)


def weight_cache_from_env(mesh, rd, opt) -> WeightCache | None:
    """GLM53_CACHE_DIR (unset = DEFAULT_DIR; off / none / 0 disables) -> the WeightCache of this mesh,
    precision policy and checkpoint."""
    base = os.environ.get("GLM53_CACHE_DIR", DEFAULT_DIR)
    if base.strip().lower() in ("", "0", "off", "none"):
        return None
    shape = tuple(mesh.shape)
    dtypes = policy_weight_dtypes(opt)
    ckpt = checkpoint_digest(rd)
    key = f"v{CACHE_FORMAT}-mesh{shape[0]}x{shape[1]}-{opt.name}-{_digest(dtypes)}-ckpt{ckpt}"
    root = Path(os.path.expanduser(base)) / key
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.warning(f"weight cache disabled: cannot create {root} ({e})")
        return None
    manifest = root / "manifest.json"
    if not manifest.exists() and os.access(root, os.W_OK):
        tmp = root / f".manifest.{os.getpid()}.tmp"
        tmp.write_text(
            json.dumps(
                {
                    "format": CACHE_FORMAT,
                    "mesh_shape": list(shape),
                    "policy": opt.name,
                    "policy_weight_dtypes": dtypes,
                    "checkpoint": str(rd.path),
                    "checkpoint_digest": ckpt,
                    "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                },
                indent=1,
            )
        )
        os.replace(tmp, manifest)
    logger.info(f"weight cache: {root}")
    return WeightCache(root, mesh)


# ------------------------------------------------------------------------------------------------------------- KDA
_KDA_FIELDS = (
    "input_projection",
    "decay_output_projection",
    "output_projection",
    "decay_scale_flat",
    "decay_bias_flat",
    "norm",
)


def load_kda_weights_cached(mesh, cfg, layer_sd, cache: WeightCache | None, in_dtype, out_dtype):
    """GlmKDA's weights (load_kda_weights + the on-device decay-scale fold) with the input / output projections retyped
    to the policy dtypes; all fields cached as one set (they are built together)."""
    from models.demos.deepseek_v3_d_p.tt.kda.weights import KDAWeights, load_kda_weights

    from .kda import kda_state_dict

    names = list(_KDA_FIELDS) + [f"conv_tap_{j}" for j in range(cfg.conv_kernel_size)]
    names = [f"kda.{n}" for n in names]
    if cache is not None and cache.has(*names):
        t = {}
        try:
            for n in names:
                t[n[len("kda.") :]] = cache.load(n)
        except RuntimeError as e:  # corrupt file: rebuild the set
            logger.warning(f"weight cache: {cache.scope} KDA weights unreadable ({e}); rebuilding")
            for v in t.values():
                ttnn.deallocate(v)
            t = None
        if t is not None:
            taps = tuple(t.pop(f"conv_tap_{j}") for j in range(cfg.conv_kernel_size))
            return KDAWeights(
                **t, convolution_taps=taps, tensor_parallel_size=tuple(mesh.shape)[1], tensor_parallel_axis=1
            )
    w = load_kda_weights(mesh, cfg, kda_state_dict(layer_sd), None, tensor_parallel_axis=1)
    new = {}
    for name, dt in (("input_projection", in_dtype), ("output_projection", out_dtype)):
        old = getattr(w, name)
        if dt is not None and old.dtype != dt:  # the optimized decoder's retype, done once before caching
            new[name] = ttnn.typecast(old, dt)
            ttnn.deallocate(old)
    w = dataclasses.replace(w, **new) if new else w
    if cache is not None:
        for n in _KDA_FIELDS:
            cache.store(f"kda.{n}", getattr(w, n))
        for j, tap in enumerate(w.convolution_taps):
            cache.store(f"kda.conv_tap_{j}", tap)
    return w

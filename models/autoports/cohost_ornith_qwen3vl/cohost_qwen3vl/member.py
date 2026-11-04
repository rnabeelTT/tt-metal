# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Co-host app members ``cohost_qwen3vl.member:embed`` and ``cohost_qwen3vl.member:rerank``.

Both members share one ``_Group`` (one chip-sharded scorer for both checkpoints, see the package docstring). The first
member to load builds it; the other's ``load`` / ``warmup`` / ``release_traces`` are no-ops, so the group's DRAM shows
up under the first member in the co-host status. Each member serves the stock Qwen3-VL server API on its own port
(``/v1/embeddings`` on the embedder; ``/v1/rerank`` and ``/v1/score`` on the reranker; ``/health`` and ``/v1/models``
on both), from its own copy of the server module bound to its own checkpoint spec.

Environment (set on the profile):
  QWEN3VL_COHOST_LANES        task per chip, default ``embed,embed,rerank,rerank``
  QWEN3VL_COHOST_PRECISION    precision policy for both (one program for every chip), default ``p0_bfp8_hifi2``
  QWEN3VL_COHOST_MAX_TOKENS   longest pair served (8192)
  QWEN3VL_COHOST_TRACE_MAX_T  longest traced bucket (4096; longer buckets run eagerly)
  QWEN3VL_COHOST_VISION       1: build both vision towers (experimental: tensor-parallel over the whole mesh)
  QWEN3VL_COHOST_LAYERS       build only the first N decoder layers (bring-up / mock checks only)
"""
from __future__ import annotations

import importlib.util
import logging
import os
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from tt_kernel.cohost.apps import AppMember

log = logging.getLogger("qwen3vl_cohost")

RERANK = "models.autoports.qwen_qwen3_vl_reranker_8b.tt.spec:QWEN3_VL_RERANKER_8B"
EMBED = "models.autoports.qwen_qwen3_vl_embedding_8b.spec:SPEC"


def _spec(ref: str):
    mod, attr = ref.split(":")
    return getattr(importlib.import_module(mod), attr)


@contextmanager
def use_spec(spec):
    """Make ``spec.current()`` return ``spec`` (cache namespaces, the server module's SPEC) for a build step."""
    from models.autoports.qwen_qwen3_vl_reranker_8b.tt import spec as specmod

    prev = specmod._CURRENT
    specmod._CURRENT = spec
    try:
        yield
    finally:
        specmod._CURRENT = prev


def _snapshot(spec) -> str:
    from models.autoports.qwen_qwen3_vl_reranker_8b.tt.spec import model_path

    return model_path(spec)  # explicit spec: never reads HF_MODEL (the LLM member's)


class _Group:
    _lock = threading.Lock()
    _inst = None

    @classmethod
    def get(cls, host):
        with cls._lock:
            if cls._inst is None:
                cls._inst = cls(host)
            return cls._inst

    def __init__(self, host):
        self.host = host
        self.specs = {"embed": _spec(os.environ.get("QWEN3VL_COHOST_EMBED_SPEC", EMBED)),
                      "rerank": _spec(os.environ.get("QWEN3VL_COHOST_RERANK_SPEC", RERANK))}
        self.lanes = [t.strip() for t in os.environ.get("QWEN3VL_COHOST_LANES", "embed,embed,rerank,rerank").split(",")]
        self.members = {}         # task -> member name
        self.owner = None         # the member that drives the device phases
        self.engine = None
        self.combiner = None
        self.rounds = 0

    def join(self, task: str, name: str):
        self.members[task] = name
        if self.owner is None:
            self.owner = name

    # ---- device phases (on the device thread) ------------------------------------------------------------------
    def load(self, mesh):
        from transformers import AutoConfig

        from models.autoports.qwen_qwen3_vl_reranker_8b.tt.text_model import Precision
        from models.autoports.qwen_qwen3_vl_reranker_8b.tt.weights import Checkpoint

        from .engine import SplitEngine
        from .split import SplitScorer

        t0 = time.time()
        paths = {t: _snapshot(s) for t, s in self.specs.items()}
        ckpts = {t: Checkpoint(p) for t, p in paths.items()}
        revisions = {t: s.revision for t, s in self.specs.items()}
        cfg = AutoConfig.from_pretrained(paths["rerank"])
        precision = Precision.load(os.environ.get("QWEN3VL_COHOST_PRECISION", "p0_bfp8_hifi2"))
        layers = int(os.environ.get("QWEN3VL_COHOST_LAYERS", "0")) or None
        scorer = SplitScorer(mesh, ckpts, revisions, self.lanes, cfg, precision, num_layers=layers)
        towers = {}
        if os.environ.get("QWEN3VL_COHOST_VISION", "0") == "1":
            from models.autoports.qwen_qwen3_vl_reranker_8b.tt.vision_tower import VisionTower

            for task in dict.fromkeys(self.lanes):
                with use_spec(self.specs[task]):  # its own weight-cache namespace
                    towers[task] = VisionTower(mesh, ckpts[task], cfg, paths[task])
        self.engine = SplitEngine(scorer, towers)
        self.paths = paths
        log.info("qwen3vl-cohost: built lanes %s (precision %s, vision %s) in %.1f s", self.lanes, precision.tag,
                 sorted(towers) or "off", time.time() - t0)

    def warmup(self):
        self.rounds += 1
        if self.rounds == 1:
            self.engine.compile_all()
        else:
            self.engine.capture_all()

    def release_traces(self):
        self.engine.release()

    # ---- serving ---------------------------------------------------------------------------------------------------
    def start_combiner(self):
        with self._lock:
            if self.combiner is None:
                from .engine import Combiner

                step = self.host.step if self.host is not None else (lambda who, fn: fn())
                self.combiner = Combiner(self.engine, step, self.members)
        return self.combiner


def _load_server_module(spec, task: str):
    """A private copy of the stock server module, bound to ``spec`` (its module-level SPEC, names and endpoints)."""
    from models.autoports.qwen_qwen3_vl_reranker_8b.server import app as stock

    path = Path(stock.__file__)
    mspec = importlib.util.spec_from_file_location(f"cohost_qwen3vl._server_{task}", path)
    mod = importlib.util.module_from_spec(mspec)
    sys.modules[mspec.name] = mod  # pydantic resolves the module's postponed annotations through sys.modules
    with use_spec(spec):
        mspec.loader.exec_module(mod)
    return mod


def _text_only(formatter_cls):
    """A subclass of the formatter that rejects image / video inputs with a 400 before they reach the device (the
    vision towers are off). A subclass, not a wrapper: the service clones formatters with ``copy.copy`` plus a deep
    copy of ``.processor``, one per preprocessing thread."""

    def __call__(self, *a, **kw):
        from models.autoports.qwen_qwen3_vl_reranker_8b.server.media import RequestError

        p = formatter_cls.__call__(self, *a, **kw)
        if p.has_vision:
            raise RequestError("this server is text-only: image and video inputs are not served (start the co-host "
                               "profile with QWEN3VL_COHOST_VISION=1, or use the single-model package)",
                               code="unsupported_input")
        return p

    return type(f"TextOnly{formatter_cls.__name__}", (formatter_cls,), {"__call__": __call__})


class Qwen3VLMember(AppMember):
    def __init__(self, host, member, task: str):
        self.host, self.member, self.task = host, member, task
        self.group = _Group.get(host)
        self.group.join(task, member.name)
        if task not in self.group.lanes:
            raise ValueError(f"member {member.name}: no chip in QWEN3VL_COHOST_LANES={self.group.lanes} runs {task}")

    @property
    def _owner(self) -> bool:
        return self.group.owner == self.member.name

    def load(self, mesh):
        if self._owner:
            self.group.load(mesh)

    def warmup(self):
        if self._owner:
            self.group.warmup()

    def release_traces(self):
        if self._owner:
            self.group.release_traces()

    def asgi_app(self):
        from models.autoports.qwen_qwen3_vl_reranker_8b.server.service import RerankService, env_int

        from .engine import Lane

        spec = self.group.specs[self.task]
        mod = _load_server_module(spec, self.task)
        combiner = self.group.start_combiner()
        chips = [i for i, t in enumerate(self.group.lanes) if t == self.task]
        cls = spec.formatter_class()
        if not self.group.engine.towers.get(self.task):
            cls = _text_only(cls)
        with use_spec(spec):
            formatter = cls(self.group.paths[self.task])
        task = self.task

        def build_service():
            return RerankService(
                [Lane(combiner, c, task) for c in chips],
                formatter,
                preprocess_threads=env_int("QWEN3VL_RERANKER_PREPROCESS_THREADS", 4),
                max_pairs_per_call=env_int("QWEN3VL_RERANKER_MAX_PAIRS_PER_CALL", 32),
                mesh_shape=(1, len(self.group.lanes)),
            )

        mod.build_service = build_service
        mod._FORMATTER = formatter
        return mod.app


def embed(host, member):
    return Qwen3VLMember(host, member, "embed")


def rerank(host, member):
    return Qwen3VLMember(host, member, "rerank")

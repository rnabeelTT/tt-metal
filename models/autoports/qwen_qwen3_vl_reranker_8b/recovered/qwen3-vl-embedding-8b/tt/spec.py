# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Which Qwen3 reranker checkpoint this process serves.

The scorer, engine and server are shared by the Qwen3 yes/no rerankers (a Qwen3 decoder whose score is
``logit("yes") - logit("no")`` at the end of a fixed prompt). A ``ModelSpec`` names what differs: the checkpoint, the
formatter that builds a pair's tokens, and whether images / video are accepted. ``QWEN3_RERANKER_SPEC`` selects one
(``"module.path:ATTRIBUTE"``); the default is Qwen3-VL-Reranker-8B.
"""
from __future__ import annotations

import importlib
import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelSpec:
    hf_model: str
    revision: str
    slug: str  # cache namespace and short name
    aliases: tuple = ()
    formatter: str = ""  # "module:Class", constructed as Class(model_path)
    multimodal: bool = True
    max_tokens: int = 32768  # longest pair the engine serves (largest text bucket)
    owned_by: str = "Qwen"
    precision: str | None = None  # policy in config/precision.json; None = that file's "selected"
    task: str = "rerank"  # "rerank": yes/no score per pair; "embed": the normed last-token hidden state per input
    extra: dict = field(default_factory=dict)

    @property
    def served_name(self) -> str:
        return self.hf_model

    def all_names(self) -> tuple:
        return (self.hf_model, *self.aliases)

    def formatter_class(self):
        mod, attr = self.formatter.split(":")
        return getattr(importlib.import_module(mod), attr)


QWEN3_VL_RERANKER_8B = ModelSpec(
    hf_model="Qwen/Qwen3-VL-Reranker-8B",
    revision="b212dc8c91a8164aef1ea2de9c1a867611e75c04",
    slug="qwen3-vl-reranker-8b",
    aliases=("Qwen3-VL-Reranker-8B", "qwen3-vl-reranker-8b"),
    formatter="models.autoports.qwen_qwen3_vl_reranker_8b.tt.inputs:PairFormatter",
    multimodal=True,
    max_tokens=32768,
)

_CURRENT = None


def current() -> ModelSpec:
    global _CURRENT
    if _CURRENT is None:
        ref = os.environ.get("QWEN3_RERANKER_SPEC")
        if ref:
            mod, attr = ref.split(":")
            _CURRENT = getattr(importlib.import_module(mod), attr)
        else:
            _CURRENT = QWEN3_VL_RERANKER_8B
    return _CURRENT


def model_path(spec: ModelSpec | None = None) -> str:
    """Local snapshot of the spec's checkpoint at its pinned revision (offline once downloaded)."""
    from huggingface_hub import snapshot_download

    if spec is not None:  # an explicit spec: its own checkpoint, whatever the environment serves
        return snapshot_download(spec.hf_model, revision=spec.revision)
    spec = current()
    return os.environ.get("QWEN3VL_RERANKER_PATH") or snapshot_download(
        os.environ.get("HF_MODEL", spec.hf_model), revision=os.environ.get("HF_REVISION", spec.revision)
    )

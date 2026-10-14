# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Run tests/hf_ref/hf_layer_ref.py in the HF venv (transformers >= 5.16) and cache its results by request hash.

GLM53_HF_PYTHON: interpreter of the HF venv (default ~/glm53-flash-bringup/.venv_hf/bin/python).
GLM53_REF_CACHE: result cache directory (default ~/glm53-flash-bringup/refs/layer_cache).
"""

from __future__ import annotations

import hashlib
import io
import os
import subprocess
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "hf_layer_ref.py"


def _hf_python() -> str:
    return os.environ.get("GLM53_HF_PYTHON", str(Path.home() / "glm53-flash-bringup/.venv_hf/bin/python"))


def _cache_dir() -> Path:
    d = Path(os.environ.get("GLM53_REF_CACHE", str(Path.home() / "glm53-flash-bringup/refs/layer_cache")))
    d.mkdir(parents=True, exist_ok=True)
    return d


# Cache key = the request + this version. Bump it when the HF reference semantics change (hf_layer_ref.py, the
# synthetic generators in tt/synthetic.py): file contents are not hashed, so formatting edits keep the cache.
REF_VERSION = "glm53-ref-1"


def _digest(req: dict) -> str:
    buf = io.BytesIO()
    torch.save({k: req[k] for k in sorted(req)}, buf)
    h = hashlib.sha256(buf.getvalue())
    h.update(REF_VERSION.encode())
    return h.hexdigest()[:24]


def cache_path(req: dict) -> Path:
    return _cache_dir() / f"{_digest(req)}.pt"


def hf_layer_reference(req: dict, timeout: int = 7200) -> dict:
    """req: see hf_layer_ref.py. Returns the result dict (cached)."""
    key = _digest(req)
    out = cache_path(req)
    if not out.exists():
        req_path = _cache_dir() / f"{key}.req.pt"
        torch.save(req, req_path)
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)  # the HF venv must not import tt-metal / ttnn packages
        cmd = [_hf_python(), "-I", str(SCRIPT), "--request", str(req_path), "--out", str(out)]
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)
        if proc.returncode != 0 or not out.exists():
            raise RuntimeError(
                f"HF reference failed ({proc.returncode}):\n{proc.stdout[-4000:]}\n{proc.stderr[-8000:]}"
            )
        req_path.unlink(missing_ok=True)
    return torch.load(out, weights_only=False)

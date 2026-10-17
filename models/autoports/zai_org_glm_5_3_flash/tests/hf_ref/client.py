# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Run tests/hf_ref/hf_layer_ref.py in the HF venv (transformers >= 5.16) and cache its results by request hash.

GLM53_HF_PYTHON: interpreter of the HF venv (default ~/glm53-flash-bringup/.venv_hf/bin/python).
GLM53_REF_CACHE: result cache directory (default ~/glm53-flash-bringup/refs/layer_cache).
GLM53_REF_DRYRUN: a directory: record every missing request there (<key>.req.pt) and skip the test instead of
    computing it; tools/compute_refs.py then computes the recorded requests in parallel (CPU) without the device.
"""

from __future__ import annotations

import ast
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


# Cache key = the request + REF_VERSION + the code of the reference (hf_layer_ref.py and the modules it runs:
# tt/ckpt.py, tt/synthetic.py) as an AST dump without docstrings: formatting / comment edits keep the cache, any
# semantic edit of the reference code invalidates it (no stale reference can be served after a code change).
REF_VERSION = "glm53-ref-2"
REF_CODE = [SCRIPT, HERE.parents[1] / "tt" / "ckpt.py", HERE.parents[1] / "tt" / "synthetic.py"]
# Router near-tie counterfactuals (request key "flip", hf_flip_ref.py) have their own code digest, so adding them left
# every base reference cached.
FLIP_SCRIPT = HERE / "hf_flip_ref.py"


def _code_digest(paths=REF_CODE) -> str:
    h = hashlib.sha256()
    for path in paths:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            body = getattr(node, "body", None)
            if isinstance(body, list) and body and isinstance(body[0], ast.Expr):
                if isinstance(getattr(body[0], "value", None), ast.Constant) and isinstance(body[0].value.value, str):
                    node.body = body[1:] or [ast.Pass()]
        h.update(ast.dump(tree, annotate_fields=False).encode())
    return h.hexdigest()


_CODE_DIGEST = _code_digest()
_FLIP_DIGEST = _code_digest([FLIP_SCRIPT] + REF_CODE)


def _digest(req: dict) -> str:
    buf = io.BytesIO()
    torch.save({k: req[k] for k in sorted(req)}, buf)
    h = hashlib.sha256(buf.getvalue())
    h.update(REF_VERSION.encode())
    h.update((_FLIP_DIGEST if "flip" in req else _CODE_DIGEST).encode())
    return h.hexdigest()[:24]


class DryRunResult(dict):
    """Placeholder of a recorded (not computed) reference: any use skips the test, so a test that requests several
    references records all of them before skipping."""

    def __init__(self, key):
        super().__init__()
        self.key = key

    def _skip(self, *a, **k):
        import pytest

        pytest.skip(f"reference {self.key} recorded for tools/compute_refs.py (GLM53_REF_DRYRUN)")

    __getitem__ = get = _skip


def cache_path(req: dict) -> Path:
    return _cache_dir() / f"{_digest(req)}.pt"


def hf_layer_reference(req: dict, timeout: int = 7200) -> dict:
    """req: see hf_layer_ref.py. Returns the result dict (cached)."""
    key = _digest(req)
    out = cache_path(req)
    dry = os.environ.get("GLM53_REF_DRYRUN")
    if not out.exists() and dry:
        Path(dry).mkdir(parents=True, exist_ok=True)
        torch.save(req, Path(dry) / f"{key}.req.pt")
        return DryRunResult(key)
    if not out.exists():
        req_path = _cache_dir() / f"{key}.req.pt"
        torch.save(req, req_path)
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)  # the HF venv must not import tt-metal / ttnn packages
        cmd = [_hf_python(), "-I", str(SCRIPT), "--request", str(req_path), "--out", str(out)]
        if "flip" in req:  # the base reference result is the input
            cmd[2] = str(FLIP_SCRIPT)
            cmd += ["--base", str(_cache_dir() / f"{req['flip']}.pt")]
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)
        if proc.returncode != 0 or not out.exists():
            raise RuntimeError(
                f"HF reference failed ({proc.returncode}):\n{proc.stdout[-4000:]}\n{proc.stderr[-8000:]}"
            )
        req_path.unlink(missing_ok=True)
    return torch.load(out, weights_only=False)


def flip_reference(req: dict, rows) -> dict | None:
    """Router near-tie counterfactual of a cached real-weight reference `req` (hf_flip_ref.py): the layer output of
    `rows` with the router's 8th expert replaced by its 9th. Returns {row: y_swap row} or None (no rows)."""
    rows = sorted(int(r) for r in rows)
    if not rows:
        return None
    assert req.get("weights") == "real", "near-tie counterfactuals are computed for real weights only"
    base = _digest(req)
    assert (_cache_dir() / f"{base}.pt").exists(), "base reference not cached"
    r = hf_layer_reference(
        {"flip": base, "layer": req["layer"], "weights": "real", "segments": list(req["segments"]), "rows": rows}
    )
    if isinstance(r, DryRunResult):
        return r
    return {int(i): r["y_swap"][j] for j, i in enumerate(r["rows"].tolist())}

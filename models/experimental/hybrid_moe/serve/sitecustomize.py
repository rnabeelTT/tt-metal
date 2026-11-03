# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Loaded by every Python process started with this directory on PYTHONPATH (vLLM API server and its EngineCore
subprocess). With HYB_ENABLE=1 it installs the hybrid-MoE overlay before any autoport module is imported."""

import os
import sys

if os.environ.get("HYB_ENABLE") == "1":
    sys.path.insert(0, os.environ.get("HYB_POC_DIR", "/poc"))
    try:
        import hybrid

        hybrid.install()
        sys.stderr.write(f"[hybrid] installed in pid {os.getpid()}\n")
    except Exception as e:  # noqa: BLE001 - surface loudly, then re-raise
        sys.stderr.write(f"[hybrid] install FAILED in pid {os.getpid()}: {e!r}\n")
        raise

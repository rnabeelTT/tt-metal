# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
import os

os.environ.setdefault("HF_MODEL", "Qwen/Qwen3.5-122B-A10B")


def pytest_configure(config):
    config.addinivalue_line("markers", "perf_single: single-die baseline perf cases (separate process)")
    config.addinivalue_line("markers", "perf_probe: hardware probes that may hang the board (run alone, under timeout)")
    for marker in (
        "real_weights: uses the local HF checkpoint",
        "long: full advertised context",
        "perf: Tracy signposted perf runs",
    ):
        config.addinivalue_line("markers", marker)

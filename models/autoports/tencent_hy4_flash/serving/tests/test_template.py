# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The chat template's facts the parsers / engine rely on (render_chat_template.py's assertions), as a test."""

import io
import os
import sys
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render_chat_template  # noqa: E402


def test_chat_template():
    buf = io.StringIO()
    sys.argv = ["render", os.environ.get("HY4_TOKENIZER", "tencent/Hy4-flash")]
    with redirect_stdout(buf):
        render_chat_template.main()
    assert "all template checks passed" in buf.getvalue()

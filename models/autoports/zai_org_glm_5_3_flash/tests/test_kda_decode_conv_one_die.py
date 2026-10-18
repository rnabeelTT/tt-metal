# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage-3 review #2: ttnn.experimental.deepseek_prefill.kda_decode_conv at the one-die KDA width (64 heads: conv width
W = 64 x (128 + 128 + 128) = 24576, input projection P = 32960) - the width tests/test_kda_decode_conv.py (TP4 per-die
W = 6144) does not cover. Same checks (qkv vs an fp64 torch reference, history update bit-exact, idle slots' NaN rows
kept out of qkv), at the decode batches the single-die decoder runs the op for (<= 8 slots) and at 32."""

import pytest

from models.autoports.zai_org_glm_5_3_flash.tests import test_kda_decode_conv as T


@pytest.mark.parametrize("B", [1, 2, 8, 32])
def test_kda_decode_conv_one_die(device, B, monkeypatch):
    monkeypatch.setattr(T, "W", 24576)
    monkeypatch.setattr(T, "P", 32960)
    T.test_kda_decode_conv(device, B)

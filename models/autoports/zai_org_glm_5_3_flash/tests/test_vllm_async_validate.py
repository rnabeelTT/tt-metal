# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device test: async decode through the vLLM adapter against solo controls (tools/vllm_async_validate.py, DESIGN.md
8.4). Reduced stack (GLM53_LAYERS default 0,3,4) on the (1, 4) mesh, real weights; about 3 minutes with a weight cache.
"""

import os

import pytest

os.environ.setdefault("GLM53_LAYERS", "0,3,4")

from models.autoports.zai_org_glm_5_3_flash.tests.harness import close_mesh  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tools import vllm_async_validate as vav  # noqa: E402

pytestmark = pytest.mark.skipif(not os.environ.get("GLM53_CKPT"), reason="needs the E192 checkpoint (GLM53_CKPT)")


def test_async_decode_matches_solo_controls(tmp_path):
    mesh = vav.open_mesh()
    try:
        res = vav.run(mesh, tmp_path / "async_validate.json")
    finally:
        close_mesh(mesh)
    bad = {k: v for k, v in res["checks"].items() if not v["equal_to_solo"]}
    assert not bad, bad
    assert res["program_guard"]["base"] == res["program_guard"]["end"], res["program_guard"]
    assert res["result"] == "PASS"

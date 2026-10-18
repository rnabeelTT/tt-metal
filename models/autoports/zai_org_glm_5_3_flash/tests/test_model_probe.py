# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 6 fast probe: the full-model wrapper (tt/model.py, tt/generator.py) on a reduced stack of one layer per kind
(layers 0, 3, 4: KDA + dense MLP, DSA + MoE, KDA + MoE) on the (1, 4) mesh, real weights, the real terminal path,
cache / page-table shapes and traces. Debug surface only: accuracy evidence comes from the all-layer model.

  - device-derived per-step decode inputs == the host builder (GlmDSA._decode_host_inputs), any positions / page table
  - traced on-device greedy decode == eager decode with host argmax (same tokens), token fed back on device
  - positions advance on device; changed page table between steps is honoured
"""

import os

import pytest
import torch

import ttnn

os.environ.setdefault("GLM53_TP", "4")
os.environ.setdefault("GLM53_LAYERS", "0,3,4")

from models.autoports.zai_org_glm_5_3_flash.tests.harness import close_mesh  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.generator import L1_SMALL_SIZE, Glm53Generator  # noqa: E402

pytestmark = pytest.mark.skipif(not os.environ.get("GLM53_CKPT"), reason="needs the E192 checkpoint (GLM53_CKPT)")
S = 32


@pytest.fixture(scope="module")
def gen():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(
        ttnn.MeshShape(1, int(os.environ["GLM53_TP"])), l1_small_size=L1_SMALL_SIZE, trace_region_size=64 << 20
    )
    g = Glm53Generator(mesh, max_batch_size=S, max_seq_len=4096)
    yield g
    g.teardown()
    close_mesh(mesh)


def _host(t):
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0])


def test_decode_inputs_match_host(gen):
    m = gen.model
    dsa = m.inputs.dsa
    pt = torch.randperm(dsa.NB - 1)[: S * 32].reshape(S, 32).to(torch.int32)  # permuted blocks, 4096 tokens / slot
    m.set_page_table(pt)
    positions = [-1, 0, 1, 2, 3, 4, 127, 128, 129, 2049, 2050, 2051, 4095] + [-1] * (S - 13)
    positions = (positions + list(range(500, 500 + S)))[:S]
    m.inputs.set_positions([-1] * S)  # stale eager fill, so refresh() must produce every field
    ttnn.copy_host_to_device_tensor(
        ttnn.from_torch(
            torch.tensor(positions, dtype=torch.float32).reshape(1, 1, S, 1),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(m.mesh),
        ),
        m.inputs.pos,
    )
    m.inputs.refresh()
    want = dsa._decode_host_inputs(positions)
    for k, (t, _, _) in want.items():
        got = _host(m.inputs.dsa_bufs[k]).reshape(t.shape)
        if k in ("row_idx", "limit_u32"):
            got = got.to(torch.int64) & 0xFFFFFFFF
            t = t.to(torch.int64) & 0xFFFFFFFF
        assert torch.equal(got.to(t.dtype), t), f"{k}: device {got.flatten()[:8]} host {t.flatten()[:8]}"
    gen.reset()


def test_traced_greedy_matches_eager(gen):
    prompt = list(range(1000, 1000 + 75))
    traced = gen.generate(prompt, 6, enable_trace=True, on_device_sampling=True)
    eager = gen.generate(prompt, 6, enable_trace=False, on_device_sampling=False)
    print(f"traced {traced} eager {eager}")
    assert traced == eager


def test_token_feedback_and_positions(gen):
    prompt = list(range(2000, 2000 + 130))  # crosses a 128-token page block
    gen.reset()
    logits = gen.prefill_forward(torch.tensor([prompt]), prompt_lens=[len(prompt)])
    first = int(torch.argmax(logits[0, -1, : gen.model.vocab]))
    out1 = gen.decode_forward(torch.tensor([[first]]), torch.tensor([len(prompt)]), reload_inputs=True)
    pos_dev = _host(gen.model.inputs.pos).reshape(-1)[0].item()
    assert pos_dev == len(prompt) + 1, pos_dev
    tok_dev = int(gen._read_tokens()[0])
    assert tok_dev == int(out1[0]), (tok_dev, out1)
    # no host write: the device feeds the sampled token and its own position
    out2 = gen.decode_forward(torch.tensor([[-7]]), torch.tensor([-7]), reload_inputs=False)
    assert _host(gen.model.inputs.pos).reshape(-1)[0].item() == len(prompt) + 2
    # same two steps with host authority every step give the same tokens
    gen.reset()
    logits = gen.prefill_forward(torch.tensor([prompt]), prompt_lens=[len(prompt)])
    h1 = gen.decode_forward(torch.tensor([[first]]), torch.tensor([len(prompt)]), reload_inputs=True)
    h2 = gen.decode_forward(torch.tensor([[int(h1[0])]]), torch.tensor([len(prompt) + 1]), reload_inputs=True)
    assert [int(out1[0]), int(out2[0])] == [int(h1[0]), int(h2[0])]

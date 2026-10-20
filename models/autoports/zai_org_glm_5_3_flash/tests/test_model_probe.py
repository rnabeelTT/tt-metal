# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 6 fast probe: the full-model wrapper (tt/model.py, tt/generator.py) on a reduced stack of one layer per kind
(layers 0, 3, 4: KDA + dense MLP, DSA + MoE, KDA + MoE) on the (1, 4) mesh, real weights, the real terminal path,
cache / page-table shapes and traces. Debug surface only: accuracy evidence comes from the all-layer model.

  - device-derived per-step decode inputs == the host builder (GlmDSA._decode_host_inputs), any positions / page table;
    with the block-sharded DSA cache (GLM53_DSA_SHARD=1) every die's row_idx == ShardedDSAMixin._decode_local_rows;
    a position at max_seq_len is inactive on device (decode-overrun guard)
  - traced on-device greedy decode == eager decode with host argmax (same tokens), token fed back on device
  - positions advance on device
  - a page table changed between two TRACED steps (new permuted block at a 128-token boundary) is honoured: the cache
    write lands in the new block, the old one is untouched, logits == eager on the same tables; an unchanged table
    passed every step costs no device copy (tools/s67r_checks.run_ptchange)
  - batch > 1 (S >= 2): prompts in non-adjacent slots decoded together == their solo runs, same prompt in two slots ==,
    a mid-decode prefill leaves the others unchanged (tools/s67r_checks.run_batch)
  - traced split sampler with alternating greedy / sampled / penalties params, dies in lockstep, seeded reproducibility
    (tools/s67r_checks.run_sampling)

GLM53_PROBE_SLOTS (default 32; 8 runs the pairs MoE path with the inactive-row skip) sets the slot count.
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
S = int(os.environ.get("GLM53_PROBE_SLOTS", "32"))


@pytest.fixture(scope="module")
def gen():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    kw = {"worker_l1_size": int(os.environ["GLM53_WORKER_L1_SIZE"])} if os.environ.get("GLM53_WORKER_L1_SIZE") else {}
    mesh = ttnn.open_mesh_device(
        ttnn.MeshShape(1, int(os.environ["GLM53_TP"])), l1_small_size=L1_SMALL_SIZE, trace_region_size=64 << 20, **kw
    )
    g = Glm53Generator(mesh, max_batch_size=S, max_seq_len=4096)
    print(f"probe generator plan {g.plan} skip_inactive {g.model.skip_inactive}")
    yield g
    g.teardown()
    close_mesh(mesh)


def _host(t):
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0])


def test_decode_inputs_match_host(gen, expect_error):
    m = gen.model
    dsa = m.inputs.dsa
    pt = torch.randperm(dsa.NB - 1)[: S * 32].reshape(S, 32).to(torch.int32)  # permuted blocks, 4096 tokens / slot
    m.set_page_table(pt)
    if S >= 13:
        positions = [-1, 0, 1, 2, 3, 4, 127, 128, 129, 2049, 2050, 2051, 4095] + [-1] * (S - 13)
        positions = (positions + list(range(500, 500 + S)))[:S]
    else:
        positions = ([-1, 3, 128, 129, 2050, 2051, 4095, 127] + list(range(500, 500 + S)))[:S]
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
    sharded = getattr(dsa, "dsa_shard", False)
    for k, (t, _, _) in want.items():
        if k == "row_idx" and sharded:  # per-die local rows: every die against the host builder
            local = dsa._decode_local_rows(positions) & 0xFFFFFFFF  # [tp, S]
            for r, dt in enumerate(ttnn.get_device_tensors(m.inputs.dsa_bufs[k])):
                got = ttnn.to_torch(dt).reshape(-1).to(torch.int64) & 0xFFFFFFFF
                assert torch.equal(got, local[r]), f"row_idx die {r}: device {got[:8]} host {local[r][:8]}"
            continue
        got = _host(m.inputs.dsa_bufs[k]).reshape(t.shape)
        if k in ("row_idx", "limit_u32"):
            got = got.to(torch.int64) & 0xFFFFFFFF
            t = t.to(torch.int64) & 0xFFFFFFFF
        assert torch.equal(got.to(t.dtype), t), f"{k}: device {got.flatten()[:8]} host {t.flatten()[:8]}"
    # decode-overrun guard: a slot at position max_seq_len writes no cache row and counts as inactive
    over = list(positions)
    over[0] = dsa.T
    ttnn.copy_host_to_device_tensor(
        ttnn.from_torch(
            torch.tensor(over, dtype=torch.float32).reshape(1, 1, S, 1),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(m.mesh),
        ),
        m.inputs.pos,
    )
    m.inputs.refresh()
    for r, dt in enumerate(ttnn.get_device_tensors(m.inputs.dsa_bufs["row_idx"])):
        got = ttnn.to_torch(dt).reshape(-1).to(torch.int64) & 0xFFFFFFFF
        assert int(got[0]) == 0xFFFFFFFF, f"die {r}: overrun slot writes row {int(got[0])}"
    assert float(_host(m.inputs.dsa_bufs["active"]).reshape(-1)[0]) == 0.0
    with expect_error(ValueError, "cache is full"):
        gen.decode_forward(torch.tensor([[1]]), torch.tensor([gen.max_seq_len]), slots=[0])
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


def test_page_table_change_traced(gen):
    from models.autoports.zai_org_glm_5_3_flash.tools.s67r_checks import run_ptchange

    r = run_ptchange(gen, slot=(1 if S > 1 else 0), L=250, steps=12)
    assert r["pass"], r


@pytest.mark.skipif(S < 2, reason="batch > 1")
def test_batch_slots(gen):
    from models.autoports.zai_org_glm_5_3_flash.tools.s67r_checks import run_batch

    slots = sorted({0, S // 2 + 1, S - 1})
    r = run_batch(gen, slots, [161, 300, 517][: len(slots)], steps=8)
    for s, v in r["joint_vs_solo"].items():
        assert v["tokens_equal"] and v["min_pcc"] > 0.999, (s, v)
    assert r["same_prompt_two_slots"]["bit_identical_steps"] == r["same_prompt_two_slots"]["steps_compared"], r
    for s, v in r["mid_prefill_others_vs_solo"].items():
        assert v["tokens_equal"] and v["min_pcc"] > 0.999, (s, v)


def test_sampling_alternation(gen):
    from models.autoports.zai_org_glm_5_3_flash.tools.s67r_checks import run_sampling

    r = run_sampling(gen, [0] if S == 1 else [0, S - 1], L=97, steps=12)
    assert r["pass"], r


def test_no_shared_buffer_frees(gen):
    """With GLM53_DEALLOC_CHECK=1 (set by the probe runs): no autoport tt/ module force-frees a buffer that a view still
    uses (the decode-logits pad bug). Sites in other modules are reported, not failed (owned elsewhere)."""
    from models.autoports.zai_org_glm_5_3_flash.tt import generator as G

    if os.environ.get("GLM53_DEALLOC_CHECK") != "1":
        pytest.skip("GLM53_DEALLOC_CHECK=1 not set")
    gen.generate(list(range(3000, 3161)), 4)
    print(f"shared-buffer frees: {G.SHARED_FREES}")
    mine = {
        k: v
        for k, v in G.SHARED_FREES.items()
        if k.split(":")[0] in ("model.py", "generator.py", "decode_inputs.py", "dsa.py", "dsa_shard.py")
    }
    assert not mine, mine

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Chunked serving prefill (turn 7): a prompt prefilled in scheduler-style chunks (intermediate chunks run to a 2048-token
segment boundary with the GDN state carried per request; the final chunk runs to the end) must give the same logits and
the same decode-slot GDN state as the whole-prompt prefill. Reduced 2-layer generator, 2 slots (the bound-scratch path)."""
import os

import pytest
import torch

import ttnn
from models.autoports.qwen_qwen3_5_122b_a10b.tests.test_mtp_verify import MODEL_DIR
from models.autoports.qwen_qwen3_5_122b_a10b.tt.generator import build_generator
from models.autoports.qwen_qwen3_5_122b_a10b.tt.mtp_verify import snapshot_gdn_state
from models.common.utility_functions import comp_pcc


def _state_host(model, snap):
    comp0 = ttnn.ConcatMeshToTensor(model.mesh_device, dim=0)
    recs, convs = snap
    return [ttnn.to_torch(r, mesh_composer=comp0).float() for r in recs], [
        [ttnn.to_torch(t, mesh_composer=comp0).float() for t in taps] for taps in convs
    ]


@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 24576, "trace_region_size": 128 * 1024 * 1024}],
    indirect=True,
)
@pytest.mark.parametrize(
    "chunks",
    [[2048, 2500], [1500, 2500], [2100, 2500], [700, 2200, 2500]],
    ids=["aligned", "short_first", "ragged", "three"],
)
def test_chunked_prefill_matches_whole(mesh_device, chunks):
    os.environ.setdefault("MESH_DEVICE", "P300x2")
    P = chunks[-1]
    gen = build_generator(
        MODEL_DIR,
        mesh_device,
        max_batch_size=2,
        max_seq_len=4096,
        layer_indices=[0, 3],
        warm_decode_trace=False,
        warm_prefill_trace=False,
    )
    model = gen.model
    torch.manual_seed(7)
    ids = torch.randint(1000, 100000, (1, P), dtype=torch.int64)
    pt = gen.page_table[:1]
    pt1 = gen.page_table[1:2]
    gen.reset()
    # chunked prefill into slot 1 (its own page-table row -> its own first block id key), with ANOTHER request's whole
    # prefill (slot 0, the reference) interleaved after the first chunk: the carried state must survive the scratch
    # being used by someone else between chunks
    start = 0
    lg_whole = None
    for i, end in enumerate(chunks):
        final = i == len(chunks) - 1
        lg = gen.prefill_serving(ids[:, :end], pt1, [end], [1], start_pos=[start], final=[final])
        if not final:
            assert float(lg.abs().max()) == 0.0, "an intermediate chunk must return zero logits"
        if i == 0:
            lg_whole = gen.prefill_serving(ids, pt, [P], [0])  # whole prefill into slot 0 between the chunks
        start = end
    snap_all = snapshot_gdn_state(model)
    rec_w, conv_w = _state_host(model, snap_all)
    rec_c, conv_c = rec_w, conv_w  # same snapshot: slot 0 = whole, slot 1 = chunked
    _, pcc = comp_pcc(lg_whole.reshape(-1), lg.reshape(-1), 0.99)
    print(f"CHUNKED {chunks}: logits PCC {pcc:.6f} argmax whole {int(lg_whole.argmax())} chunked {int(lg.argmax())}")
    assert pcc >= 0.999, pcc
    # decode-slot GDN state: slot 0 (whole) vs slot 1 (chunked)
    for li, (rw, rc) in enumerate(zip(rec_w, rec_c)):
        _, p_rec = comp_pcc(
            rw[0], rc[1], 0.99
        )  # rec_state [B, Nv, Dk, Dv]: row per slot (concat over devices on dim 0)
        taps = [comp_pcc(cw[0, 0], cc[0, 1], 0.99)[1] for cw, cc in zip(conv_w[li], conv_c[li])]
        print(f"CHUNKED {chunks} layer {li}: rec_state PCC {p_rec:.6f} conv taps PCC {min(taps):.6f}")
        assert p_rec >= 0.999 and min(taps) >= 0.999, (p_rec, taps)
    assert not getattr(gen, "_pending_prefill", {}), "the final chunk must release the carried state"

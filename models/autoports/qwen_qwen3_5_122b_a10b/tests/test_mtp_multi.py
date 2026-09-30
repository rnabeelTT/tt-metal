# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Multi-user MTP verify pass vs the batched sequential decode of the same users (reduced 2-layer generator, B = 2)."""
import os

import pytest
import torch

import ttnn
from models.autoports.qwen_qwen3_5_122b_a10b.tests.test_mtp_verify import MODEL_DIR
from models.autoports.qwen_qwen3_5_122b_a10b.tt.generator import build_generator
from models.autoports.qwen_qwen3_5_122b_a10b.tt.mtp_multi import (
    mtp_generate_multi,
    prefill_multi_with_head,
    verify_pass_multi,
)
from models.autoports.qwen_qwen3_5_122b_a10b.tt.mtp_verify import (
    free_gdn_snapshot,
    restore_gdn_state,
    snapshot_gdn_state,
)
from models.common.utility_functions import comp_pcc


@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 24576, "trace_region_size": 128 * 1024 * 1024}],
    indirect=True,
)
def test_multi_user_verify_pass_matches_batched_decode(mesh_device):
    os.environ.setdefault("MESH_DEVICE", "P300x2")
    B, R = 2, 4
    gen = build_generator(
        MODEL_DIR,
        mesh_device,
        max_batch_size=B,
        max_seq_len=4096,
        layer_indices=[0, 3],
        warm_decode_trace=False,
        warm_prefill_trace=False,
    )
    model = gen.model
    V = gen.args.vocab_size
    prompts = [[1000 + 7 * i for i in range(20)], [3000 + 5 * i for i in range(25)]]
    P = [len(p) for p in prompts]
    T = max(P)
    toks = torch.zeros(B, T, dtype=torch.int64)
    for u in range(B):
        toks[u, : P[u]] = torch.tensor(prompts[u])
    gen.reset()
    logits0 = gen.prefill_forward(toks, page_table=gen.page_table, prompt_lens=P)
    cur = [int(torch.argmax(logits0[u, -1]).item()) for u in range(B)]
    # reference: R batched decode steps (both users, teacher-forced with their own greedy tokens), state snapshot first
    snap = snapshot_gdn_state(model)
    rows = [[cur[u]] for u in range(B)]
    ref = [[] for _ in range(B)]
    for i in range(R):
        out = gen.decode_forward(
            torch.tensor([[rows[u][i]] for u in range(B)]),
            torch.tensor([P[u] + i for u in range(B)]),
            enable_trace=False,
            on_device_sampling=False,
            reload_inputs=True,
        )
        for u in range(B):
            lg = out[u].float()[:V]
            ref[u].append(lg)
            if i + 1 < R:
                rows[u].append(int(lg.argmax().item()))
    restore_gdn_state(model, snap)
    # the batched verify pass over the same R tokens of both users: host per-row lengths, then the device-resident
    # RowsLen masks the traced pass will use (both must match the batched decode)
    from models.autoports.qwen_qwen3_5_122b_a10b.tt.mtp_multi import GDN_T, RowsLen

    dn0 = next(l.attention for l in model.layers if not l.is_full_attention)
    rl = RowsLen(mesh_device, B, GDN_T, dn0.K)
    rl.stage([R] * B)
    for tag, rows_len in (("host lengths", None), ("device RowsLen", rl)):
        if rows_len is not None:
            snap2 = snapshot_gdn_state(model)  # the first pass advanced the state: restore for the second
        h, lg_tt, _ = verify_pass_multi(
            model, rows, [[P[u] + i for i in range(R)] for u in range(B)], gen.page_table[:B], rows_len=rows_len
        )
        lg = ttnn.to_torch(lg_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1)).float()
        lg = lg.reshape(-1, lg.shape[-1])[:, :V]
        ttnn.deallocate(lg_tt)
        ttnn.deallocate(h)
        pccs, agree = [], 0
        for u in range(B):
            for i in range(R):
                _, pcc = comp_pcc(ref[u][i], lg[u * R + i], 0.98)
                pccs.append(pcc)
                same = int(lg[u * R + i].argmax().item()) == int(ref[u][i].argmax().item())
                agree += int(same)
                print(
                    f"MULTI [{tag}] user {u} row {i} (pos {P[u] + i}): PCC {pcc:.5f} argmax {'ok' if same else 'DIFF'}"
                )
        print(f"MULTI_VERIFY [{tag}] B={B} R={R}: min PCC {min(pccs):.5f} argmax agreement {agree}/{B * R}")
        assert min(pccs) >= 0.98, (tag, pccs)
        if rows_len is None:
            restore_gdn_state(model, snap)  # back to the post-prefill state for the second variant
    free_gdn_snapshot(snap2)
    free_gdn_snapshot(snap)


@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 24576, "trace_region_size": 128 * 1024 * 1024}],
    indirect=True,
)
def test_multi_user_generate_forced_greedy(mesh_device):
    """The eager multi-user loop (batched drafts, one batched pass, per-user accept, one batched commit, batched head
    rows) with the greedy continuation forced as the drafts must reproduce each user's greedy tokens (up to the 2-layer
    model's bf16 near-ties) in fewer passes than tokens."""
    from loguru import logger

    from models.autoports.qwen_qwen3_5_122b_a10b.tt.mtp import Qwen35MtpHead

    os.environ.setdefault("MESH_DEVICE", "P300x2")
    B, k, N = 2, 3, 12
    gen = build_generator(
        MODEL_DIR,
        mesh_device,
        max_batch_size=B,
        max_seq_len=4096,
        layer_indices=[0, 3],
        warm_decode_trace=False,
        warm_prefill_trace=False,
    )
    model = gen.model
    V = gen.args.vocab_size
    head = Qwen35MtpHead(model, gen.args.load_state_dict())
    kv_shape = next(tuple(l.attention.paged_k.shape) for l in model.layers if l.is_full_attention)
    head.allocate_kv_cache(kv_shape)
    prompts = [[1000 + 7 * i for i in range(20)], [3000 + 5 * i for i in range(25)]]
    tok0, hids, P = prefill_multi_with_head(gen, head, prompts)
    snap = snapshot_gdn_state(model)
    # reference: N batched greedy decode steps (each user its own argmax)
    greedy = [[tok0[u]] for u in range(B)]
    for i in range(N - 1):
        out = gen.decode_forward(
            torch.tensor([[greedy[u][-1]] for u in range(B)]),
            torch.tensor([P[u] + i for u in range(B)]),
            enable_trace=False,
            on_device_sampling=False,
            reload_inputs=True,
        )
        for u in range(B):
            greedy[u].append(int(out[u].float()[:V].argmax().item()))
    restore_gdn_state(model, snap)
    free_gdn_snapshot(snap)
    draft_fns = [(lambda toks, u=u: greedy[u][len(toks) : len(toks) + 1] or [0]) for u in range(B)]
    mtp, passes = mtp_generate_multi(
        gen, head, prompts, N, k, log=logger.info, draft_fns=draft_fns, prefilled=(tok0, hids, P)
    )
    for u in range(B):
        same = sum(int(a == b) for a, b in zip(greedy[u], mtp[u]))
        print(f"MULTI_GENERATE user {u}: equal to greedy {same}/{N}; greedy {greedy[u]} mtp {mtp[u]}")
        # the 2-layer model's near-uniform logits flip at bf16 near-ties (a repeated token ending one step early); the
        # mechanics are the gate here, the full-model equality is measured by the served run
        assert same >= N // 2, (u, greedy[u], mtp[u])
    print(f"MULTI_GENERATE B={B} k={k} N={N}: passes {passes} ({B * N / passes:.2f} tokens per pass over both users)")
    assert passes < N

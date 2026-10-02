# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Batched short-prompt prefill (tt/prefill_batched.py) vs the production per-user masked-bucket prefill: next-token
logits, the GDN decode-slot states and the paged K/V cache must match (reduced 2-layer generator: GDN layer 0 +
attention layer 3, B = 4 slots)."""
import os

import pytest
import torch

import ttnn
from models.autoports.qwen_qwen3_coder_next.tests.test_mtp_verify import MODEL_DIR
from models.autoports.qwen_qwen3_coder_next.tt.generator import build_generator
from models.autoports.qwen_qwen3_coder_next.tt.prefill_batched import BatchedPrefill
from models.common.utility_functions import comp_pcc


def _slot_states(model):
    comp = ttnn.ConcatMeshToTensor(model.mesh_device, dim=0)
    out = []
    for layer in model.layers:
        if layer.is_full_attention:
            continue
        dn = layer.attention
        out.append(
            (
                ttnn.to_torch(dn.rec_state, mesh_composer=comp).float(),
                [ttnn.to_torch(c, mesh_composer=comp).float() for c in dn.conv_states],
            )
        )
    return out


def _kv(model):
    out = []
    for layer in model.layers:
        if layer.is_full_attention:
            a = layer.attention
            out.append(
                (
                    ttnn.to_torch(ttnn.get_device_tensors(a.paged_k)[0]).float().clone(),
                    ttnn.to_torch(ttnn.get_device_tensors(a.paged_v)[0]).float().clone(),
                )
            )
    return out


@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 24576, "trace_region_size": 128 * 1024 * 1024}],
    indirect=True,
)
def test_batched_prefill_matches_per_user(mesh_device):
    os.environ.setdefault("MESH_DEVICE", "P300x2")
    B = 4
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
    bp = BatchedPrefill(model, B=B)
    eager = {}  # lens tuple -> (logits, slot states) of the eager batched pass, for the traced replay check below
    # two groups: a full bucket-128 group with awkward lengths and a 3-request bucket-512 group (one idle row)
    for lens in ([20, 57, 128, 100], [300, 129, 512]):
        n = len(lens)
        T = max(lens)
        toks = torch.zeros(B, T, dtype=torch.int64)
        g = torch.Generator().manual_seed(sum(lens))
        for u in range(n):
            toks[u, : lens[u]] = torch.randint(1000, 120000, (lens[u],), generator=g)
        slots = list(range(n))
        # batched pass first (so a missing K/V write cannot hide behind the reference's)
        gen.reset()
        got, got_sh, hid = bp.prefill(toks[:n], lens, gen.page_table[:n], slots)
        assert tuple(hid.shape) == (n, gen.args.dim) and hid.float().abs().sum() > 0, hid.shape
        st_b = _slot_states(model)
        kv_b = _kv(model)
        eager[tuple(lens)] = (toks.clone(), got.clone(), st_b)
        sh_host = ttnn.to_torch(got_sh, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3)).float()
        # reference: the production per-user path (same page-table rows, same slots)
        gen.reset()
        ref = gen.prefill_forward(toks[:n], page_table=gen.page_table[:n], prompt_lens=lens)
        st_r = _slot_states(model)
        kv_r = _kv(model)
        worst = 1.0
        for u in range(n):
            _, pcc = comp_pcc(ref[u, 0], got[u, 0], 0.99)
            worst = min(worst, pcc)
            # the two paths differ by the batched GDN chunk form's accumulation order (PCC >= 0.999); on a 2-layer model
            # over random tokens the top logits are near-ties, so the batched argmax is required inside the reference's
            # top-5 (the teacher-forcing gate's criterion), exact agreement is reported
            same = torch.argmax(ref[u, 0]).item() == torch.argmax(got[u, 0]).item()
            in_top5 = torch.argmax(got[u, 0]).item() in torch.topk(ref[u, 0], 5).indices.tolist()
            _, pcc_sh = comp_pcc(got[u, 0], sh_host[0, 0, u], 0.999)  # the sharded row == the gathered row
            print(
                f"BATCHED_PREFILL lens={lens} user {u}: logits PCC {pcc:.5f} argmax {'ok' if same else 'in top-5' if in_top5 else 'DIFF'} sharded-row PCC {pcc_sh:.5f}"
            )
            assert pcc >= 0.999 and in_top5 and pcc_sh >= 0.999, (lens, u, pcc, same, in_top5, pcc_sh)
        for li, ((rec_b, conv_b), (rec_r, conv_r)) in enumerate(zip(st_b, st_r)):
            for u in range(n):
                _, p = comp_pcc(rec_r[u], rec_b[u], 0.999)
                d = (rec_r[u] - rec_b[u]).abs().max().item()
                assert p >= 0.999 and d <= 2e-2 * max(1.0, rec_r[u].abs().max().item()), ("rec", li, u, p, d)
                for m in range(len(conv_b)):
                    dm = (conv_r[m][:, u] - conv_b[m][:, u]).abs().max().item()
                    assert dm <= 2e-2 * max(1.0, conv_r[m][:, u].abs().max().item()), ("conv", li, u, m, dm)
            print(f"BATCHED_PREFILL lens={lens} gdn layer {li}: slot states match ({n} slots)")
        # the users' own blocks must be identical; the reserved scratch block (pad rows of both paths, an idle row's
        # tokens in a short group) is never read and is not compared
        blocks = gen.page_table[:n].reshape(-1).long()
        for (k_b, v_b), (k_r, v_r) in zip(kv_b, kv_r):
            assert torch.equal(k_b[blocks], k_r[blocks]) and torch.equal(v_b[blocks], v_r[blocks]), "paged K/V differ"
        print(f"BATCHED_PREFILL lens={lens}: min logits PCC {worst:.5f}, K/V identical")
    # traced replay (the served form): one trace per batched bucket, then the same groups through the traces must
    # reproduce the eager pass, compiling no program (compile-after-capture guard)
    bp.warm(gen.page_table)  # every bucket's programs (128 / 256 / 512) before any capture, as the served warm-up does
    captured = bp.capture_traces(gen.page_table)
    assert set(captured) >= {128, 512}, captured
    n_prog = mesh_device.num_program_cache_entries()
    for lens, (toks, got_e, st_e) in eager.items():
        n = len(lens)
        gen.reset()
        got_t, _, _ = bp.prefill(toks[:n], list(lens), gen.page_table[:n], list(range(n)))
        st_t = _slot_states(model)
        for u in range(n):
            _, p = comp_pcc(got_e[u, 0], got_t[u, 0], 0.9999)
            assert p >= 0.9999 and torch.argmax(got_e[u, 0]) == torch.argmax(got_t[u, 0]), ("traced logits", lens, u, p)
        for li, ((rec_e, conv_e), (rec_t, conv_t)) in enumerate(zip(st_e, st_t)):
            for u in range(n):
                assert torch.allclose(rec_e[u], rec_t[u], atol=1e-2, rtol=1e-2), ("traced rec", lens, li, u)
        print(f"BATCHED_PREFILL traced lens={list(lens)}: logits / states match the eager pass")
    assert mesh_device.num_program_cache_entries() == n_prog, "the traced batched prefill compiled a program"
    bp.release_traces()
    print(f"BATCHED_PREFILL traces {captured}: no program compiled after capture")

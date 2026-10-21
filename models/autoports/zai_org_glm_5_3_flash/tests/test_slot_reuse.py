# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Slot reuse: a request's output must not depend on the requests served before it (vLLM reuses slots request after
request and never calls reset_state; the prefill must overwrite every per-slot state it reads).

Reduced stack (layers 0, 3, 4: KDA + dense MLP, DSA + MoE, KDA + MoE) on the (1, 4) mesh, real weights. Round 1 serves
prompts B_s in every slot (prefill + traced decode, host logits): the reference, as in a fresh process. Round 2 serves
LONGER prompts A_s in the same slots (their cache rows / pool keys reach past B_s's prompts, open pools differ).
Round 3 serves B_s again WITHOUT a generator reset: prefill and decode logits must be bit-identical to round 1.
Prompt lengths cover the dense (<= 2050) and the sparse (indexer top-512) regimes, and lengths % 4 != 0 (open-pool
hand-off) and == 0 (a decode that starts a new pool over a stale accumulator row).

GLM53_REUSE_SLOTS (default 8) sets the slot count; 1 runs the single-slot case.
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
S = int(os.environ.get("GLM53_REUSE_SLOTS", "8"))
T = 16384
STEPS = 6
LEN_B = [12000, 9001, 300, 2500, 15002, 8500, 4003, 11000][:S]
LEN_A = [16000, 12003, 3000, 5001, 16200, 11000, 9002, 14000][:S]


@pytest.fixture(scope="module")
def gen():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(
        ttnn.MeshShape(1, int(os.environ["GLM53_TP"])), l1_small_size=L1_SMALL_SIZE, trace_region_size=64 << 20
    )
    g = Glm53Generator(mesh, max_batch_size=S, max_seq_len=T)
    yield g
    g.teardown()
    close_mesh(mesh)


def _serve(g, prompts):
    """Prefill every prompt into its slot (no reset), then STEPS traced decode steps of all slots (own argmax fed
    back). Returns [prefill logits per slot] + [decode logits per step]."""
    V = g.model.vocab
    out = []
    for s, ids in enumerate(prompts):
        lg = g.prefill_forward(ids.reshape(1, -1), prompt_lens=[len(ids)], slots=[s])
        out.append(lg.reshape(-1)[:V].clone())
    tok = torch.tensor([int(torch.argmax(p)) for p in out])
    pos = torch.tensor([len(p) for p in prompts])
    for _ in range(STEPS):
        lg = g.decode_forward(
            tok.reshape(-1, 1), pos, slots=list(range(S)), on_device_sampling=False, enable_trace=True
        )[:, :V].clone()
        out.append(lg)
        tok = torch.argmax(lg.float(), dim=-1)
        pos = pos + 1
    return out


def test_slot_reuse_bit_identical(gen):
    V = gen.model.vocab
    rnd = lambda L, seed: torch.randint(100, V - 100, (L,), generator=torch.Generator().manual_seed(seed))
    B = [rnd(L, 1000 + s) for s, L in enumerate(LEN_B)]
    A = [rnd(L, 2000 + s) for s, L in enumerate(LEN_A)]
    gen.reset()
    ref = _serve(gen, B)
    _serve(gen, A)
    again = _serve(gen, B)  # no reset: the serving path
    bad = [
        f"{'prefill slot' if i < S else 'decode step'} {i if i < S else i - S}: max |d| "
        f"{float((x.float() - y.float()).abs().max())}"
        for i, (x, y) in enumerate(zip(ref, again))
        if not torch.equal(x, y)
    ]
    assert not bad, f"slot reuse changed the output: {bad}"

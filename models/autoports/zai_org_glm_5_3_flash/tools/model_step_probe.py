# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 6 debug probe: one prefill and decode steps of the full-model wrapper, synchronizing after every phase so a
hang is located by the last "ok" line. Phases: prefill, eager decode (active slot), trace replay, eager sampler,
traced sampler, a second replay with device feedback.

usage: GLM53_TP=4 GLM53_LAYERS=0,3,4 python -m models.autoports.zai_org_glm_5_3_flash.tools.model_step_probe
"""

import os
import time

import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tests.harness import close_mesh
from models.autoports.zai_org_glm_5_3_flash.tt.generator import L1_SMALL_SIZE, Glm53Generator


def ok(mesh, what, t0):
    ttnn.synchronize_device(mesh)
    print(f"ok {what} ({time.time() - t0:.1f}s)", flush=True)


def main():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(
        ttnn.MeshShape(1, int(os.environ.get("GLM53_TP", "4"))), l1_small_size=L1_SMALL_SIZE, trace_region_size=64 << 20
    )
    t0 = time.time()
    try:
        S = int(os.environ.get("PROBE_S", "32"))
        g = Glm53Generator(mesh, max_batch_size=S, max_seq_len=4096)
        ok(mesh, "build + warm + model trace capture", t0)
        m = g.model
        prompt = torch.arange(1000, 1075).reshape(1, -1)
        pre = os.environ.get("PROBE_PRE", "")
        if "pt" in pre:  # test_decode_inputs_match_host: permuted page table, eager refresh with active slots
            dsa = m.inputs.dsa
            m.set_page_table(torch.randperm(dsa.NB - 1)[: 32 * 32].reshape(32, 32).to(torch.int32))
            ok(mesh, "permuted page table", t0)
            m.inputs.set_positions(list(range(500, 532)))
            m.inputs.refresh()
            ok(mesh, "eager refresh, all slots active", t0)
        if "reset" in pre:
            g.reset()
            ok(mesh, "reset", t0)
        if "pf" in pre:  # a prefill before the reset (the test order: two generate() calls)
            g.prefill_forward(prompt, prompt_lens=[75])
            ok(mesh, "prefill before reset", t0)
            g.reset()
            ok(mesh, "reset after prefill", t0)
        logits = g.prefill_forward(prompt, prompt_lens=[75])
        first = int(torch.argmax(logits[0, -1, : m.vocab]))
        ok(mesh, f"prefill (first token {first})", t0)
        pos = [-1] * S
        pos[0] = 75
        m.inputs.set_positions(pos)
        g._write_tokens([first])
        lg = m.decode_step(g._tok_view())
        ok(mesh, "eager decode step, slot 0 active", t0)
        host = m.logits_to_host(lg)[0, 0, 0, : m.vocab]
        print(f"eager argmax {int(torch.argmax(host))}", flush=True)
        ttnn.deallocate(lg)
        m.inputs.set_positions(pos)  # the eager step advanced the device positions
        g._write_tokens([first])
        ttnn.execute_trace(mesh, g.trace_id, cq_id=0, blocking=True)
        ok(mesh, "model trace replay", t0)
        host = m.logits_to_host(g.logits_out)[0, 0, 0, : m.vocab]
        print(f"traced argmax {int(torch.argmax(host))}", flush=True)
        m.sampling.sample(g.logits_out, tt_out_tok=g.tok_buf, enable_trace=False)
        ok(mesh, f"eager sampler -> {int(g._read_tokens()[0])}", t0)
        m.sampling.sample(g.logits_out, tt_out_tok=g.tok_buf, enable_trace=True)
        ok(mesh, f"sampler trace capture + replay -> {int(g._read_tokens()[0])}", t0)
        ttnn.execute_trace(mesh, g.trace_id, cq_id=0, blocking=False)
        m.sampling.sample(g.logits_out, tt_out_tok=g.tok_buf, enable_trace=True)
        ok(mesh, f"replay with device feedback -> {int(g._read_tokens()[0])}", t0)
        g.teardown()
    finally:
        close_mesh(mesh)


if __name__ == "__main__":
    main()

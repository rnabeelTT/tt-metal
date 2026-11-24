# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Persistent L1 buffers of the served adapter by build stage (init, prefill warm-up, decode warm-up + captures), then
one prefill of PROMPT tokens (the CB / L1 clash check). env: MESH, B (32), CTX (1048576), PROMPT (2048), LOW (L1
addresses listed from here, default 1000000)"""

import os

import torch

import ttnn
from models.autoports.thinkingmachines_inkling_2_mini.tt import generator_vllm as G


def dump(mesh, tag, seen):
    low = int(os.environ.get("LOW", "1000000"))
    bufs = [b for b in ttnn._ttnn.reports.get_buffers(mesh) if b.buffer_type == ttnn.BufferType.L1 and b.device_id == 0]
    new = sorted((b.address, b.max_size_per_bank, str(b.buffer_layout)) for b in bufs
                 if (b.address, b.max_size_per_bank) not in seen)
    for b in bufs:
        seen.add((b.address, b.max_size_per_bank))
    lowest = min((b.address for b in bufs), default=None)
    print(f"== {tag}: {len(bufs)} L1 buffers on die 0, lowest at {lowest}; new at >= {low}:", flush=True)
    for a, s, lay in new:
        if a >= low:
            print(f"   {a:8d} .. {a + s:8d}  {s:7d} B  {lay}", flush=True)


def main():
    N = int(os.environ.get("MESH", "1"))
    if N > 1:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, N), trace_region_size=300 << 20, l1_small_size=24576)
    seen = set()
    dump(mesh, "open", seen)
    gen = G.Inkling2ForCausalLM.initialize_vllm_model(None, mesh, int(os.environ.get("B", "32")),
                                                      max_seq_len=int(os.environ.get("CTX", "1048576")))
    dump(mesh, "init", seen)
    gen.warmup_model_prefill()
    dump(mesh, "prefill warm-up", seen)
    gen.warmup_model_decode()
    dump(mesh, "decode warm-up + captures", seen)
    P = int(os.environ.get("PROMPT", "2048"))
    try:
        gen.prefill_forward(torch.arange(1000, 1000 + P).reshape(1, -1), None, None, [P], empty_slots=[0])
        ttnn.synchronize_device(mesh)
        print(f"prefill of {P} tokens OK", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"prefill of {P} tokens FAILED: {str(e).splitlines()[0][:300]}", flush=True)
        print([ln for ln in str(e).splitlines() if "clash" in ln][:1], flush=True)
    print("== l1_probe done", flush=True)
    from models.autoports.thinkingmachines_inkling_2_mini.tt.clean_exit import close

    close(mesh)


if __name__ == "__main__":
    main()

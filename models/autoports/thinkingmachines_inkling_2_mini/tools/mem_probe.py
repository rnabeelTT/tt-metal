# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DRAM per die after building the model with SLOTS x CTX (bfp8 experts, bf16 KV) and one decoder + trace per width:
prints the allocator statistics (die 0). env: MESH (1), SLOTS (8), CTX (8192), POOL (the shared global-KV pool in
tokens; default SLOTS x CTX): with a small POOL the free DRAM at the end is what a bigger pool may take; VISION_TOKENS
(image sizes run through the vision tower after the trace captures)"""

import os

import ttnn
from models.autoports.thinkingmachines_inkling_2_mini.tt.model import Inkling2Model


def stats(mesh, tag):
    d = mesh.get_devices()[0] if hasattr(mesh, "get_devices") else mesh
    try:
        s = ttnn.get_memory_view(d, ttnn.BufferType.DRAM)
        print(f"{tag}: DRAM allocated {s.total_bytes_allocated_per_bank * s.num_banks / 2**30:.2f} GiB, free "
              f"{s.total_bytes_free_per_bank * s.num_banks / 2**30:.2f} GiB, largest free block "
              f"{s.largest_contiguous_bytes_free_per_bank * s.num_banks / 2**30:.2f} GiB", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"{tag}: memory view unavailable ({e})", flush=True)


def main():
    N = int(os.environ.get("MESH", "1"))
    if N > 1:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, N), trace_region_size=300 << 20)
    stats(mesh, "empty")
    S, C = int(os.environ.get("SLOTS", "8")), int(os.environ.get("CTX", "8192"))
    pool = int(os.environ.get("POOL", "0")) or S * C
    m = Inkling2Model(mesh, max_seq_len=C + 256, expert_dtype=ttnn.bfloat8_b, kv_dtype=ttnn.bfloat16,
                      prefill_chunk=int(os.environ.get("I2M_PREFILL_CHUNK", "256")), slots=S, kv_pool_tokens=pool)
    stats(mesh, f"model {S} slots x {C}")
    if os.environ.get("ADAPTER", "1") == "1":  # + the vLLM adapter: decoders (MixPlan), prefix pools, warm-ups, traces
        from models.autoports.thinkingmachines_inkling_2_mini.tt.generator_vllm import Inkling2ForCausalLM

        ad = Inkling2ForCausalLM(m, mesh, S, C)
        stats(mesh, "adapter (decoders, prefix pools)")
        ad.allocate_kv_cache((pool // 64, 1, 64, 128), None, 24)
        ad.warmup_model_prefill()
        stats(mesh, "after the prefill warm-up")
        ad.warmup_model_decode()
        stats(mesh, "after the decode trace captures")
        if ad.vt is not None:  # VISION_TOKENS: images of these token counts after the captures (fits or not)
            import torch

            for n in [int(v) for v in os.environ.get("VISION_TOKENS", "169,1024,2048,4032").split(",")]:
                try:
                    ad.vt(torch.zeros(n, 40, 40, 3), torch.tensor([[1, 1, n]]))
                    ttnn.synchronize_device(mesh)
                    print(f"vision {n} tokens: ok", flush=True)
                except Exception as e:  # noqa: BLE001
                    print(f"vision {n} tokens: FAILED ({str(e).splitlines()[0][:160]})", flush=True)
                    break
            stats(mesh, "after the vision forwards")
    ttnn.close_mesh_device(mesh)
    print("== probe done")


if __name__ == "__main__":
    main()

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Persistent L1 buffers of the served adapter by build stage (init, prefill warm-up, decode warm-up, each trace
capture), with the python object holding each new one, then one prefill of PROMPT tokens (the CB / L1 clash check).
env: MESH, B (32), CTX (1048576), PROMPT (2048), LOW (L1 addresses listed from here, default 1000000)"""

import sys
import os

import torch

import ttnn
from models.autoports.thinkingmachines_inkling_2_mini.tt import generator_vllm as G


def holders(roots):
    """L1 address -> the attribute path of a live python tensor holding it (walked from roots)"""
    out, seen, stack = {}, set(), [(r, n) for n, r in roots.items()]
    while stack:
        o, path = stack.pop()
        if id(o) in seen or len(path) > 200:
            continue
        seen.add(id(o))
        if isinstance(o, ttnn.Tensor):
            try:
                if o.is_allocated() and o.memory_config().buffer_type == ttnn.BufferType.L1:
                    out.setdefault(o.buffer_address(), f"{path} {tuple(o.shape)} {o.dtype}")
            except Exception:  # noqa: BLE001
                pass
            continue
        if isinstance(o, dict):
            stack += [(v, f"{path}[{k!r}]") for k, v in o.items()]
        elif isinstance(o, (list, tuple, set)):
            stack += [(v, f"{path}[{i}]") for i, v in enumerate(o)]
        elif hasattr(o, "__dict__") and not isinstance(o, type) and type(o).__module__.startswith("models."):
            stack += [(v, f"{path}.{k}") for k, v in vars(o).items()]
        elif type(o).__name__ == "module" and o.__name__.startswith("models."):
            stack += [(v, f"{path}.{k}") for k, v in vars(o).items() if not k.startswith("__")]
    return out


def dump(mesh, tag, seen, roots=None):
    low = int(os.environ.get("LOW", "1000000"))
    bufs = [b for b in ttnn._ttnn.reports.get_buffers(mesh) if b.buffer_type == ttnn.BufferType.L1 and b.device_id == 0]
    new = sorted((b.address, b.max_size_per_bank, str(b.buffer_layout).split(".")[-1]) for b in bufs
                 if (b.address, b.max_size_per_bank) not in seen)
    for b in bufs:
        seen.add((b.address, b.max_size_per_bank))
    lowest = min((b.address for b in bufs), default=None)
    print(f"== {tag}: {len(bufs)} L1 buffers on die 0, lowest at {lowest}; new at >= {low}:", flush=True)
    h = holders(dict(roots or {}, **{k: v for k, v in sys.modules.items() if k.startswith("models.")})) \
        if any(a >= low for a, _, _ in new) else {}
    for a, s, lay in new:
        if a >= low:
            print(f"   {a:8d} .. {a + s:8d}  {s:7d} B  {lay:14s} {h.get(a, '-')}", flush=True)


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

    def wrap(obj, name, tag):
        f = getattr(obj, name)

        def g(*a, **k):
            r = f(*a, **k)
            dump(mesh, tag, seen, {"gen": gen})
            return r

        setattr(obj, name, g)

    for w, tr in (gen.trs or {}).items():
        wrap(tr, "capture", f"capture W={w}")
    if gen.dr is not None:
        wrap(gen, "_spec_warm", "eager widths + spec warm-up")
        wrap(gen.vtr, "capture", "capture verify")
        wrap(gen.dr, "capture", "capture drafter")
    gen.warmup_model_decode()
    dump(mesh, "decode warm-up + captures", seen, {"gen": gen})
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

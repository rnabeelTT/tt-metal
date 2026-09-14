#!/usr/bin/env python3
"""Measure prefill and decode throughput of the ported model on a given mesh.

Reports the two numbers a deployment decision actually turns on -- time to first token
and per-token decode rate -- plus the arithmetic that explains them, because on this
model the explanation is the interesting part: the unfused expert path touches every
expert a device owns rather than only the 8 a token selected, so decode is bounded by
reading experts_per_device weights per layer, not by the 8 that were routed to.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import time

import torch

import ttnn
from models.autoports.ornith_ai_ornith_1_5_35b_a3b.tt.model import OrnithModel
from models.autoports.ornith_ai_ornith_1_5_35b_a3b.tt.model_config import OrnithModelArgs
from models.autoports.ornith_ai_ornith_1_5_35b_a3b.tt.weight_mapping import load_ornith_state_dict
from models.demos.blackhole.qwen36.tt.model_config import GDN_CONV1D_L1_SMALL_SIZE

MESH_SHAPES = {"P150": (1, 1), "P300": (1, 2), "P150x4": (1, 4)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True, choices=sorted(MESH_SHAPES))
    ap.add_argument("--weights", required=True)
    ap.add_argument("--isl", type=int, nargs="+", default=[128, 512])
    ap.add_argument("--decode-steps", type=int, default=16)
    ap.add_argument(
        "--batch",
        type=int,
        default=1,
        help="decode batch (users). A 256-expert top-8 MoE without sparse "
        "dispatch reads every local expert per STEP, not per token, so "
        "that cost amortizes across the batch.",
    )
    ap.add_argument("--no-trace", action="store_true", help="skip the traced-decode measurement")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    rows, cols = MESH_SHAPES[a.mesh]
    # trace_region_size is required on EVERY mesh, not just multi-device: without a
    # trace region the capture cannot allocate and traced decode silently falls back.
    params = {
        "l1_small_size": GDN_CONV1D_L1_SMALL_SIZE,
        # Decode is dispatch-bound (marginal 8.41 us per program in the traced model),
        # so how programs are ISSUED matters more than what they compute. Overridable.
        "num_command_queues": int(os.environ.get("ORNITH_CQS", "2")),
        "trace_region_size": 1024 * 1024 * 1024,
    }
    # Fabric ONLY under expert parallelism. With replicated experts (the default) the
    # model runs no collectives on any mesh, and requesting an unused fabric costs a
    # fragile cold ethernet-link training on P300 plus an "Unsupported fabric config"
    # throw from ccl_common if anything does reach a collective.
    # A fabric is needed by anything that runs a collective: expert parallelism, or the
    # vocab-sharded LM head's logit all-gather.
    if rows * cols > 1 and any(
        os.environ.get(v) == "1" for v in ("ORNITH_MOE_EP", "ORNITH_SHARD_LM_HEAD", "ORNITH_TP")
    ):
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        # Cold multi-chip opens fail ethernet router sync once per process; absorb it.
        try:
            ttnn.close_mesh_device(ttnn.open_mesh_device(ttnn.MeshShape(rows, cols)))
        except Exception:
            pass
        # close_mesh_device RESETS the fabric config to DISABLED, so the warm-up above
        # silently undoes the line above it and the real mesh then opens with no fabric
        # -- which surfaces much later as "Unsupported fabric config" from ccl_common
        # the first time a collective actually runs. Re-assert it after the warm-up.
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(rows, cols), **params)

    weights = pathlib.Path(a.weights)
    cfg = json.loads((weights / "config.json").read_text())["text_config"]
    args = OrnithModelArgs(mesh, max_batch_size=a.batch, max_seq_len=max(a.isl) + a.decode_steps + 8)

    t0 = time.perf_counter()
    state = load_ornith_state_dict(weights, cfg)
    cache = pathlib.Path(args.weight_cache_path(args.routed_expert_dtype))
    cache.mkdir(parents=True, exist_ok=True)
    model = OrnithModel(mesh, args, state, tensor_cache_path=cache)
    load_s = time.perf_counter() - t0

    epd = args.num_experts // max(mesh.get_num_devices(), 1)
    report = {
        "mesh": a.mesh,
        "devices": mesh.get_num_devices(),
        "batch": a.batch,
        "routed_expert_dtype": args.routed_expert_dtype,
        "experts_per_device": epd,
        "load_seconds": round(load_s, 1),
        "runs": [],
    }
    print(
        f"{a.mesh}: {mesh.get_num_devices()} device(s), {epd} experts/device, "
        f"routed={args.routed_expert_dtype}, load {load_s:.0f}s"
    )

    # Paged KV cache + identity page table.
    #
    # Not optional for decode: the attention module only takes its decode branch when
    # `use_paged_attention and T == 1`. Without a page table it falls into the CONCAT
    # PREFILL branch even for a single token, which is both wrong for decode and
    # untraceable -- trace capture dies with "!is_capturing_trace" deep inside the
    # prefill path rather than saying the cache was never allocated.
    BLOCK_SIZE = 64
    longest = max(a.isl) + a.decode_steps + 8
    num_blocks = max(64, (longest + BLOCK_SIZE - 1) // BLOCK_SIZE)
    num_blocks = ((num_blocks + 7) // 8) * 8  # SDPA page-table stick alignment
    num_blocks *= a.batch
    model.allocate_kv_caches(
        [num_blocks, args.n_local_kv_heads, BLOCK_SIZE, args.head_dim],
        ttnn.bfloat16,
        batch_size=a.batch,
    )
    bpu = num_blocks // a.batch  # blocks per user
    page_table = torch.stack([torch.arange(u * bpu, (u + 1) * bpu, dtype=torch.int32) for u in range(a.batch)])
    report["num_blocks"] = num_blocks

    for isl in a.isl:
        ids = torch.randint(1000, 40000, (1, isl))
        if os.environ.get("ORNITH_DUMP_L1"):
            # What is RESIDENT in L1 before this prefill. The GDN prefill kernel wants
            # ~1.36 MB of circular buffers in a 1.5 MB L1, so anything left sitting in
            # L1 by an earlier stage shows up as "statically allocated circular buffers
            # clash with L1 buffers" at whatever prompt length first grows the CBs past
            # the gap -- several ops away from whatever actually holds the memory.
            print(f"@@@ L1 state before prefill isl={isl}", flush=True)
            ttnn.dump_device_memory_state(mesh, prefix=f"isl{isl}_")
        model.prefill(ids)  # warm the program cache; the first call compiles
        ttnn.synchronize_device(mesh)

        t0 = time.perf_counter()
        model.prefill(ids)
        ttnn.synchronize_device(mesh)
        ttft = time.perf_counter() - t0

        if a.batch > 1:
            # Prefill here runs ONE sequence, and the GDN state handoff replaces the
            # externally allocated [B, ...] buffers with the [1, ...] ones it just
            # produced -- after which a batch-B decode dies in a copy with
            # "Input tensor shape [4, 1, 8192] does not match output [1, 1, 8192]".
            # Real serving prefills each user into its own slot; this bench does not,
            # so reinstall zeroed batch-B state and measure decode alone. TTFT above is
            # still the single-sequence number.
            # Free and reallocate EVERYTHING at batch B: the GDN state and the paged KV
            # caches. Resetting only the GDN state leaves user 0 holding the KV this
            # prefill just wrote while users 1..B-1 hold zeros, which makes the
            # cross-user check below compare users that are not actually in the same
            # state -- it reads as a batching bug and is not one.
            model.free_kv_caches()
            model.allocate_kv_caches(
                [num_blocks, args.n_local_kv_heads, BLOCK_SIZE, args.head_dim],
                ttnn.bfloat16,
                batch_size=a.batch,
            )

        # Decode, eager and traced.
        #
        # Tracing is the second half of the dispatch story: batching the experts cut the
        # op COUNT, tracing removes the per-op host dispatch from the steady state
        # entirely by replaying a captured command stream. Both are measured because
        # they attack the same bottleneck from different sides.
        tok = torch.full((a.batch, 1), 1234, dtype=torch.long)
        pos = lambda p: torch.full((a.batch,), p, dtype=torch.long)

        if a.batch > 1:
            # Correctness invariant for batched decode: every user starts from the same
            # (zeroed) state and is fed the same token, so every user's logits must be
            # IDENTICAL. This catches a batch axis that is being read as time, or users
            # bleeding into each other -- neither of which changes any tensor's shape.
            _lg = model.decode(tok, pos(isl))
            ttnn.synchronize_device(mesh)
            _lt = ttnn.to_torch(_lg).float().reshape(a.batch, -1)
            _dev = max((_lt[u] - _lt[0]).abs().max().item() for u in range(1, a.batch))
            _amax = [int(_lt[u].argmax()) for u in range(a.batch)]
            print(
                f"  batch {a.batch} cross-user check: max|logit diff|={_dev:.3e} "
                f"argmax={'same' if len(set(_amax)) == 1 else f'DIFFER {_amax}'}",
                flush=True,
            )
        dec = dec_traced = None
        try:
            model.decode(tok, pos(isl))  # warm / compile
            ttnn.synchronize_device(mesh)
            t0 = time.perf_counter()
            for i in range(a.decode_steps):
                model.decode(tok, pos(isl + 1 + i))
            ttnn.synchronize_device(mesh)
            dec = (time.perf_counter() - t0) / a.decode_steps
        except Exception as e:
            print(f"  eager decode unavailable: {str(e).splitlines()[0][:90]}")

        if not a.no_trace:
            try:
                from models.demos.blackhole.qwen36.tt.generator_interface import prime_decode_trace
                from models.tt_transformers.tt.generator import Generator

                gen = Generator([model], [model.args], mesh)
                # The stock capture would double-advance the GDN recurrent state, so the
                # qwen36 helper snapshots and restores it around the capture.
                prime_decode_trace(gen, model, tok, pos(isl), page_table)
                for _ in range(2):  # warm the replay
                    gen.decode_forward(
                        tok, pos(isl), page_table=page_table, kv_cache=None, enable_trace=True, read_from_device=True
                    )
                ttnn.synchronize_device(mesh)
                t0 = time.perf_counter()
                for i in range(a.decode_steps):
                    gen.decode_forward(
                        tok,
                        pos(isl + 1 + i),
                        page_table=page_table,
                        kv_cache=None,
                        enable_trace=True,
                        read_from_device=True,
                    )
                ttnn.synchronize_device(mesh)
                dec_traced = (time.perf_counter() - t0) / a.decode_steps
                if a.batch > 1:
                    print(
                        f"  batch {a.batch}: {1/dec_traced:.2f} steps/s -> "
                        f"{a.batch/dec_traced:.2f} tok/s aggregate, "
                        f"{1/dec_traced:.2f} tok/s per user",
                        flush=True,
                    )

                # Same replay, WITHOUT pulling logits back to host. read_from_device
                # ships the full [batch, 1, 248320] logits over PCIe and syncs every
                # step; a server that samples on device returns a token id instead.
                # The gap between these two numbers is what on-device sampling is worth.
                t0 = time.perf_counter()
                for i in range(a.decode_steps):
                    gen.decode_forward(
                        tok,
                        pos(isl + 1 + i),
                        page_table=page_table,
                        kv_cache=None,
                        enable_trace=True,
                        read_from_device=False,
                    )
                ttnn.synchronize_device(mesh)
                dec_noread = (time.perf_counter() - t0) / a.decode_steps
                print(
                    f"  [traced, no host readback] {1/dec_noread:.2f} tok/s " f"({dec_noread*1e3:.2f} ms/token)",
                    flush=True,
                )

                # PURE device trace execution: no input refresh, no readback, no host
                # work of any kind -- just replaying the captured command stream. This
                # separates "the device takes this long" from "the host takes this long
                # getting there", which the two numbers above cannot.
                try:
                    tid = list(gen.trace_ids_decode[False].values())[0]
                    ttnn.synchronize_device(mesh)
                    t0 = time.perf_counter()
                    for _ in range(a.decode_steps):
                        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
                    ttnn.synchronize_device(mesh)
                    raw = (time.perf_counter() - t0) / a.decode_steps
                    print(f"  [pure execute_trace] {1/raw:.2f} tok/s " f"({raw*1e3:.2f} ms/token)", flush=True)
                except Exception as e:
                    print(f"  pure execute_trace unavailable: {type(e).__name__}: {e}", flush=True)
            except Exception as e:
                import traceback

                print(
                    f"  traced decode unavailable: {type(e).__name__}: " f"{str(e).splitlines()[0][:160]}", flush=True
                )
                traceback.print_exc()

        run = {
            "isl": isl,
            "ttft_ms": round(ttft * 1e3, 1),
            "prefill_tok_s": round(isl / ttft, 1),
            "decode_ms_per_token": round(dec * 1e3, 2) if dec else None,
            "decode_tok_s": round(1 / dec, 2) if dec else None,
            "decode_traced_ms_per_token": round(dec_traced * 1e3, 2) if dec_traced else None,
            "decode_traced_tok_s": round(1 / dec_traced, 2) if dec_traced else None,
            # What a server sees: every user advances one token per step.
            "decode_aggregate_tok_s": round(a.batch / dec, 1) if dec else None,
        }
        report["runs"].append(run)
        print(
            f"  ISL {isl:>6}: TTFT {run['ttft_ms']:>9.1f} ms "
            f"({run['prefill_tok_s']:>8.1f} tok/s prefill)"
            + (f"   decode {run['decode_tok_s']:.2f} tok/s" if dec else "   decode n/a")
            + (f"   traced {run['decode_traced_tok_s']:.2f} tok/s" if dec_traced else "")
        )

    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(report, indent=2))
        print(f"wrote {a.out}")
    ttnn.close_mesh_device(mesh)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

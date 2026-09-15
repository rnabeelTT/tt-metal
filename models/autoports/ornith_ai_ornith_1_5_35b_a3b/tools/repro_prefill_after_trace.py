"""Does a prefill into a second slot deadlock once the decode trace is captured?

The batch-isolation test prefills all slots BEFORE any decode and passes. Serving does
prefill(A) -> capture trace -> decode -> prefill(B) while A is live, and that deadlocks.
This reproduces that ORDER without vLLM, so the loop is minutes instead of twenty.
"""
import json
import os
import pathlib
import time

import torch

import ttnn
from models.autoports.ornith_ai_ornith_1_5_35b_a3b.tt.model import OrnithModel
from models.autoports.ornith_ai_ornith_1_5_35b_a3b.tt.model_config import OrnithModelArgs
from models.autoports.ornith_ai_ornith_1_5_35b_a3b.tt.weight_mapping import load_ornith_state_dict
from models.demos.blackhole.qwen36.tt.generator_interface import prime_decode_trace
from models.demos.blackhole.qwen36.tt.model_config import GDN_CONV1D_L1_SMALL_SIZE
from models.tt_transformers.tt.generator import Generator

W = pathlib.Path(os.environ["ORNITH_WEIGHTS"])
B, BLOCK = 4, 64
mesh = ttnn.open_mesh_device(
    ttnn.MeshShape(1, 1),
    l1_small_size=GDN_CONV1D_L1_SMALL_SIZE,
    num_command_queues=1,
    trace_region_size=1024 * 1024 * 1024,
)
cfg = json.loads((W / "config.json").read_text())["text_config"]
args = OrnithModelArgs(mesh, max_batch_size=B, max_seq_len=512)
model = OrnithModel(
    mesh,
    args,
    load_ornith_state_dict(W, cfg),
    tensor_cache_path=pathlib.Path(args.weight_cache_path(args.routed_expert_dtype)),
)
bpu = 8
NB = bpu * B
model.allocate_kv_caches([NB, args.n_local_kv_heads, BLOCK, args.head_dim], ttnn.bfloat16, batch_size=B)
page_table = torch.stack([torch.arange(u * bpu, (u + 1) * bpu, dtype=torch.int32) for u in range(B)])

torch.manual_seed(0)
p0 = torch.randint(1000, 40000, (1, 12), dtype=torch.int32)
p1 = torch.randint(1000, 40000, (1, 12), dtype=torch.int32)


def step(msg):
    print(f"@@@ {msg}", flush=True)


step("prefill slot 0")
model.prefill_paged_slots_replicated([p0], page_table, [0], valid_lens=[12])
ttnn.synchronize_device(mesh)
step("  ok")

step("capture decode trace")
gen = Generator([model], [model.args], mesh)
tok = torch.full((B, 1), 1234, dtype=torch.long)
pos = lambda p: torch.full((B,), p, dtype=torch.long)
prime_decode_trace(gen, model, tok, pos(12), page_table)
ttnn.synchronize_device(mesh)
step("  ok")

step("decode 3 traced steps")
for i in range(3):
    gen.decode_forward(tok, pos(12 + i), page_table=page_table, kv_cache=None, enable_trace=True, read_from_device=True)
ttnn.synchronize_device(mesh)
step("  ok")

step("condense remap (vLLM does this when a request finishes)")
model._remap_gdn_slots_replicated(list(range(B)))  # identity first
model._remap_gdn_slots_replicated([1, 0] + list(range(2, B)))  # a real swap
ttnn.synchronize_device(mesh)
step("  ok")

step("decode 2 steps after remap")
for i in range(2):
    gen.decode_forward(tok, pos(15 + i), page_table=page_table, kv_cache=None, enable_trace=True, read_from_device=True)
ttnn.synchronize_device(mesh)
step("  ok")

p2 = torch.randint(1000, 40000, (1, 12), dtype=torch.int32)
step("PREFILL TWO SLOTS IN ONE CALL (vLLM concurrency-2 shape)")
t0 = time.perf_counter()
model.prefill_paged_slots_replicated([p1, p2], page_table, [1, 2], valid_lens=[12, 12])
ttnn.synchronize_device(mesh)
step(f"  ok in {time.perf_counter()-t0:.2f}s  <-- NO DEADLOCK")

step("decode 3 more steps")
for i in range(3):
    gen.decode_forward(tok, pos(20 + i), page_table=page_table, kv_cache=None, enable_trace=True, read_from_device=True)
ttnn.synchronize_device(mesh)
step("  ok")
ttnn.close_mesh_device(mesh)
print("PATDONE")

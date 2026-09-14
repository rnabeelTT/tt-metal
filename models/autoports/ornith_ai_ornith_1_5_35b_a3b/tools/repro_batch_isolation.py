"""Four different prompts into four slots: does each user get its OWN continuation?

This is the check that matters for per-slot prefill. Bitwise comparison is useless here
(decode is not reproducible -- see NONDETERMINISM.md), but cross-user CONTAMINATION is
grossly visible: if slot writes leak, a user continues someone else's prompt.
"""
import json
import os
import pathlib

import torch
from transformers import AutoTokenizer

import ttnn
from models.autoports.ornith_ai_ornith_1_5_35b_a3b.tt.model import OrnithModel
from models.autoports.ornith_ai_ornith_1_5_35b_a3b.tt.model_config import OrnithModelArgs
from models.autoports.ornith_ai_ornith_1_5_35b_a3b.tt.weight_mapping import load_ornith_state_dict
from models.demos.blackhole.qwen36.tt.model_config import GDN_CONV1D_L1_SMALL_SIZE

W = pathlib.Path(os.environ["ORNITH_WEIGHTS"])
B, STEPS, BLOCK = 4, 12, 64
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
tok = AutoTokenizer.from_pretrained(str(W))

PROMPTS = [
    "Count upward: 1, 2, 3, 4, 5, 6,",
    "The letters in order are: A, B, C, D, E, F,",
    "Days of the week: Monday, Tuesday, Wednesday,",
    "Colours of the rainbow: red, orange, yellow,",
]
enc = [torch.tensor([tok(p, add_special_tokens=True)["input_ids"]], dtype=torch.int32) for p in PROMPTS]
lens = [int(e.shape[1]) for e in enc]
print(f"@@@ prompt lengths {lens}", flush=True)

bpu = 8  # blocks per user, 8*64 = 512 positions
NB = bpu * B
model.allocate_kv_caches([NB, args.n_local_kv_heads, BLOCK, args.head_dim], ttnn.bfloat16, batch_size=B)
page_table = torch.stack([torch.arange(u * bpu, (u + 1) * bpu, dtype=torch.int32) for u in range(B)])

slots = list(range(B))
logits = model.prefill_paged_slots_replicated(enc, page_table, slots, valid_lens=lens)
print(f"@@@ prefilled {len(logits)} requests into slots {slots}", flush=True)

nxt = torch.tensor([[int(torch.tensor(l).reshape(-1).argmax())] for l in logits], dtype=torch.long)
pos = torch.tensor(lens, dtype=torch.long)
out = [[int(nxt[u, 0])] for u in range(B)]
for step in range(STEPS):
    lg = model.decode_paged(nxt, pos, page_table)
    ttnn.synchronize_device(mesh)
    t = ttnn.to_torch(lg).float().reshape(B, -1)
    nxt = t.argmax(dim=-1).reshape(B, 1).to(torch.long)
    for u in range(B):
        out[u].append(int(nxt[u, 0]))
    pos = pos + 1

print("\n@@@ BATCH-4 CONTINUATIONS (each user should continue ITS OWN prompt)")
for u in range(B):
    print(f"@@@  user {u}: {PROMPTS[u]!r}")
    print(f"@@@        -> {tok.decode(out[u])!r}")
ttnn.close_mesh_device(mesh)
print("ISOLATIONDONE")

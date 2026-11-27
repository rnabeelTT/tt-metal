# DrafterSnapshots save / restore with the tensor-parallel drafter (per-die caches): caches after restore == before
import os
import torch
import ttnn
from models.autoports.thinkingmachines_inkling_2_mini.tt import drafter as D

N = int(os.environ.get("MESH", "4"))
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, N), trace_region_size=64 << 20)
dr = D.DSparkDrafter(mesh, None, max_len=2048, d_max=3)
sn = D.DrafterSnapshots(dr, 2)
sn.warm()
g = torch.Generator().manual_seed(0)
dr.commit(torch.randn(1200, 4, dr.H, generator=g).to(torch.bfloat16), 0)
caches = dr.kc + dr.vc
dump = lambda: [[ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(c)] for c in caches]  # noqa: E731
q = 1100
before = dump()
sn.save(list(range(q)), q)
dr.commit(torch.randn(1200, 4, dr.H, generator=g).to(torch.bfloat16), 0)
mid = dump()
sn.restore(sn.find(list(range(q)), q), q)
after = dump()
for li, (b, m, a) in enumerate(zip(before, mid, after)):
    for d in range(N):
        rows = slice(None)
        same = torch.equal(b[d], a[d]) if (li % dr.L) in dr.local else None
        # global layer: rows [0, q) per head restored
        S = dr.S[li % dr.L]
        bb, aa = b[d].reshape(dr.NKV, S, dr.D)[:, :q], a[d].reshape(dr.NKV, S, dr.D)[:, :q]
        print(f"cache {li} die {d}: restored rows equal {torch.equal(bb, aa)}, changed by 2nd commit "
              f"{not torch.equal(bb, m[d].reshape(dr.NKV, S, dr.D)[:, :q])}, dies differ {not torch.equal(b[d], b[0]) if d else '-'}"
              + (f", whole ring equal {same}" if same is not None else ""), flush=True)
ttnn.close_mesh_device(mesh)

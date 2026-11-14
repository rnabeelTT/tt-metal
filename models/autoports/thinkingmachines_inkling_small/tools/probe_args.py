# probe: does the shared Qwen36ModelArgs construct with Inkling's config? (no device)
import glob, os, traceback
os.environ["HF_MODEL"] = glob.glob("/hf/hub/models--thinkingmachines--Inkling-Small/snapshots/*")[0]
from models.autoports.thinkingmachines_inkling_small.tt.hf_config_inkling import register
register()
from models.demos.blackhole.qwen36.tt.model_config import Qwen36ModelArgs
try:
    a = Qwen36ModelArgs(None, max_batch_size=1, max_seq_len=4096)
    print("OK", a.dim, a.n_layers, a.n_heads, a.n_kv_heads, a.head_dim, getattr(a, "rope_theta", None))
except Exception:
    traceback.print_exc()

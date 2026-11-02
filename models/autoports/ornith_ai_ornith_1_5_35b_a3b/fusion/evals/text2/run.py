"""python run.py <bench> <label> <base_url> <model_id> <concurrency>  -> benches/<bench>.py main(cfg)"""
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.core import RunCfg, check_server  # noqa: E402

bench, label, base_url, model, conc = sys.argv[1:6]
mod = importlib.import_module(f"benches.{bench.replace('-', '_')}")
cfg = RunCfg.from_env(getattr(mod, "BENCH_NAME", bench), label, base_url, model, int(conc))
if not getattr(mod, "NO_SERVER_CHECK", False):
    check_server(cfg)
mod.main(cfg)

"""python judge_ingest.py <run_dir> <judgments.jsonl>: compute the official metric from judge verdicts.
Dispatches to benches/<bench>.ingest(cfg_like, verdicts, missing) using run_dir/config.json."""
import importlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.core import RunCfg  # noqa: E402
from common.judge import load_verdicts  # noqa: E402

run_dir, judg = Path(sys.argv[1]), Path(sys.argv[2])
c = json.loads((run_dir / "config.json").read_text())
mod = importlib.import_module(f"benches.{c.get('module', c['bench'])}")
cfg = RunCfg(**{k: c[k] for k in RunCfg.__dataclass_fields__ if k in c})
cfg.out = run_dir
verdicts, missing = load_verdicts(run_dir, judg)
if missing:
    print(f"WARNING: {len(missing)} judge inputs have no judgment (counted as incorrect), e.g. {missing[:5]}")
mod.ingest(cfg, verdicts, missing)

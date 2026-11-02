"""Runs inside the text2-ojbench container: official ojbench.judge_jsonl on /work/<in> -> /work/<out>."""
import sys
from pathlib import Path

import ojbench

inp, outp, workers = sys.argv[1], sys.argv[2], int(sys.argv[3])
ojbench.init(problem_dirs=[Path("/data/NOI"), Path("/data/ICPC")])
ojbench.judge_jsonl(input_path=inp, output_path=outp, num_workers=workers, worker_log_path="/work/judge_logs",
                    identifier=Path(inp).name)

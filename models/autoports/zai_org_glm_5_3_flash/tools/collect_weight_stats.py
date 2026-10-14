# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Write tests/weight_stats.json from the real E192 checkpoint (shape, dtype, mean, std; small tensors verbatim).

Usage: python -m models.autoports.zai_org_glm_5_3_flash.tools.collect_weight_stats [--layers 0,3,4]
Merges into an existing file, so layers can be added as their shards become available.
"""

import argparse
import json
from pathlib import Path

from models.autoports.zai_org_glm_5_3_flash.tt.ckpt import CheckpointReader
from models.autoports.zai_org_glm_5_3_flash.tt.synthetic import STATS_LAYERS, collect_stats

OUT = Path(__file__).resolve().parents[1] / "tests" / "weight_stats.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default=",".join(str(x) for x in STATS_LAYERS))
    args = ap.parse_args()
    layers = [int(x) for x in args.layers.split(",")]
    rd = CheckpointReader()
    stats = collect_stats(rd, layers)
    if OUT.exists():
        old = json.loads(OUT.read_text())
        old["layers"].update(stats["layers"])
        stats = old
    OUT.write_text(json.dumps(stats))
    print("wrote", OUT, sorted(stats["layers"]))


if __name__ == "__main__":
    main()

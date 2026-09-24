# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Join the single-chip (PERF_SINGLE) and mesh (PERF ... mesh(1x4)) host-wall lines of the stage-4 perf logs into one table.

usage: python -m models.autoports.qwen_qwen3_5_122b_a10b.tools.perf_join <single.log> <mesh.log> [--md out.md]
"""

import argparse
import re
import sys

SINGLE = re.compile(r"^PERF_SINGLE (\S+) (\S+): ([\d.]+) ms")
MESH = re.compile(r"^PERF (\S+) (\S+): .*mesh\(1x(\d)\) ([\d.]+) ms")


def parse(path, rx, group):
    out = {}
    with open(path, errors="replace") as f:
        for line in f:
            m = rx.match(line.strip())
            if m:
                out[(m.group(1), m.group(2))] = float(m.group(group))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("single")
    ap.add_argument("mesh")
    ap.add_argument("--md")
    ap.add_argument("--tp", type=int, default=4)
    a = ap.parse_args(argv)
    single = parse(a.single, SINGLE, 3)
    mesh = parse(a.mesh, MESH, 4)
    rows = [
        "| layer | case | single-chip ms | mesh (1x%d) ms | speedup | efficiency |" % a.tp,
        "|---|---|---|---|---|---|",
    ]
    for key in sorted(set(single) | set(mesh)):
        s, m = single.get(key), mesh.get(key)
        sp = s / m if s and m else None
        rows.append(
            "| %s | %s | %s | %s | %s | %s |"
            % (
                key[0],
                key[1],
                f"{s:.3f}" if s else "n/a",
                f"{m:.3f}" if m else "n/a",
                f"{sp:.2f}x" if sp else "n/a",
                f"{100 * sp / a.tp:.0f}%" if sp else "n/a",
            )
        )
    text = "\n".join(rows)
    print(text)
    if a.md:
        with open(a.md, "w") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    sys.exit(main())

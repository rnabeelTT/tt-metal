"""Freeze a bench's item list (and the sampled ids) under data/<bench>/.
  python -m qt.prepare <bench>|all [--force]
Reruns reuse the frozen file, so every run scores the identical sample."""
import json
import os
import sys
from collections import Counter

from . import registry
from .common import DATA, write_jsonl


def prepare(bench, force=False):
    d = os.path.join(DATA, bench)
    if os.path.exists(os.path.join(d, "items.jsonl")) and not force:
        print(f"{bench}: frozen items exist ({d}); --force to rebuild")
        return
    mod = registry.module(bench)
    items, meta = mod.build(bench)
    ids = [it["id"] for it in items]
    assert len(ids) == len(set(ids)), f"{bench}: duplicate item ids"
    os.makedirs(d, exist_ok=True)
    write_jsonl(os.path.join(d, "items.jsonl"), items)
    with open(os.path.join(d, "sample_ids.txt"), "w") as f:
        f.write("\n".join(it["id"] for it in items) + "\n")
    meta = {**meta, "n_items": len(items), "groups": dict(sorted(Counter(it["group"] for it in items).items()))}
    json.dump(meta, open(os.path.join(d, "meta.json"), "w"), indent=1, ensure_ascii=False)
    print(f"{bench}: {len(items)} items (full set {meta.get('full_size')}, sample={meta.get('sample')}), "
          f"{len(meta['groups'])} groups -> {d}")


if __name__ == "__main__":
    force = "--force" in sys.argv
    names = [a for a in sys.argv[1:] if not a.startswith("--")]
    if names == ["all"]:
        names = registry.BENCHES
    for b in names:
        prepare(b, force)

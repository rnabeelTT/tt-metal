"""Offline toolchain check: for every exercise, drop the reference example
solution in place of the stub and run the harness's own run_unit_tests().
Usage (in container): python verify_examples.py <exercises_root> <work_root> <procs> [lang,...]"""
import json, shutil, sys, subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, "/aider/benchmark")
import benchmark  # noqa: E402

src_root, work_root, procs = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
langs = sys.argv[4].split(",") if len(sys.argv) > 4 else None
dirs = sorted(d for d in src_root.glob("*/exercises/practice/*") if d.is_dir()
              and (langs is None or d.parts[-4] in langs))


def one(d):
    rel = d.relative_to(src_root)
    t = work_root / rel
    shutil.copytree(d, t)
    cfg = json.loads((t / ".meta/config.json").read_text())["files"]
    sols = [s for s in cfg["solution"] if not s.endswith("Cargo.toml")]
    for ex in cfg["example"]:
        ext = Path(ex).suffix
        cand = [s for s in sols if Path(s).suffix == ext]
        if len(cand) > 1:
            cand = [s for s in cand if Path(s).stem == Path(ex).stem] or cand
        shutil.copy(t / ex, t / cand[0])
    # multi-file references (java) and example-only crates (rust), checker only
    ref = t / ".meta/src/reference/java"
    if ref.is_dir():
        for f in ref.glob("*.java"):
            shutil.copy(f, t / "src/main/java" / f.name)
    if (t / ".meta/Cargo-example.toml").exists():
        shutil.copy(t / ".meta/Cargo-example.toml", t / "Cargo.toml")
    try:
        err = benchmark.run_unit_tests(src_root, t, t / ".hist.md", cfg["test"])
    except subprocess.TimeoutExpired:
        err = "TIMEOUT"
    return str(rel), err


fails = []
with ThreadPoolExecutor(procs) as ex:
    for rel, err in ex.map(one, dirs):
        print(("FAIL " if err else "ok   ") + rel, flush=True)
        if err:
            fails.append((rel, err[-600:]))
print(f"\nVERIFY total={len(dirs)} pass={len(dirs)-len(fails)} fail={len(fails)}")
for rel, err in fails:
    print("----", rel, "\n", err)

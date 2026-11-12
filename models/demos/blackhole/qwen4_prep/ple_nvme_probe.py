# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Can the PLE n-gram table be read per token from NVMe instead of the host page cache?

One decode token reads (ngram_size - 1) * heads_per_ngram = 16 rows of the table (160 bf16 = 320 B each for
Qwen3.8-Flash-Next); an MTP verify pass of k drafts reads 16 * (k + 1). This probe times exactly that access pattern
with O_DIRECT (the page cache is bypassed, so the numbers hold on a host with no RAM to spare for the table) against
any large file, e.g. a 100+ GB safetensors shard or blob, one 4 KiB aligned page per row.

    python -m models.demos.blackhole.qwen4_prep.ple_nvme_probe FILE [--rows 16 80] [--tokens 2000] [--threads 16]
"""

from __future__ import annotations

import argparse
import mmap
import os
import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

PAGE = 4096


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("file")
    p.add_argument("--rows", type=int, nargs="+", default=[16, 80], help="rows per token (16 = 1 row, 80 = MTP k=4)")
    p.add_argument("--tokens", type=int, default=2000)
    p.add_argument("--threads", type=int, default=16)
    p.add_argument(
        "--mode",
        choices=("direct", "willneed"),
        default="direct",
        help="direct: O_DIRECT reads on a thread pool. willneed: the upstream port's pattern (qwen38_flash_next "
        "Qwen38ResidentPLELookup): POSIX_FADV_WILLNEED on every row, then one buffered pread per row, on a file "
        "whose pages were dropped from the page cache first (cold, as on a host without RAM for the table)",
    )
    args = p.parse_args()

    direct = args.mode == "direct"
    fd = os.open(args.file, os.O_RDONLY | (os.O_DIRECT if direct else 0))
    size = os.fstat(fd).st_size
    pages = size // PAGE
    if not direct:
        os.posix_fadvise(fd, 0, size, os.POSIX_FADV_RANDOM)
    print(f"{args.file}: {size / 1e9:.1f} GB, mode {args.mode}" + (f", {args.threads} threads" if direct else ""))
    local = __import__("threading").local()

    def read(page: int) -> None:
        buf = getattr(local, "buf", None)
        if buf is None:
            buf = local.buf = mmap.mmap(-1, PAGE)  # page-aligned, as O_DIRECT requires
        os.preadv(fd, [buf], page * PAGE)

    def willneed_token(batch: list[int]) -> None:
        for page in batch:
            os.posix_fadvise(fd, page * PAGE, 320, os.POSIX_FADV_WILLNEED)
        for page in batch:
            os.pread(fd, 320, page * PAGE)

    rng = random.Random(0)
    with ThreadPoolExecutor(args.threads) as pool:
        for rows in args.rows:
            if not direct:
                os.posix_fadvise(fd, 0, size, os.POSIX_FADV_DONTNEED)  # evict this file's clean pages
            for _ in range(50):  # warm the threads
                batch = [rng.randrange(pages) for _ in range(rows)]
                list(pool.map(read, batch)) if direct else willneed_token(batch)
            lat = []
            for _ in range(args.tokens):
                batch = [rng.randrange(pages) for _ in range(rows)]
                t0 = time.perf_counter()
                list(pool.map(read, batch)) if direct else willneed_token(batch)
                lat.append((time.perf_counter() - t0) * 1e3)
            lat.sort()
            q = lambda f: lat[min(len(lat) - 1, int(f * len(lat)))]
            print(
                f"  {rows:3d} rows/token: mean {statistics.fmean(lat):.3f} ms, p50 {q(0.5):.3f}, p99 {q(0.99):.3f}, "
                f"max {lat[-1]:.3f} ms over {args.tokens} tokens"
            )
    os.close(fd)


if __name__ == "__main__":
    main()

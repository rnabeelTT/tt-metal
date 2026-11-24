# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Weight upload with an on-disk cache, avoiding host -> device writes from FILE-BACKED memory.

On this host (kernel 7.0, page cache in large folios: FileHugePages tens of GB) a host -> device write whose source
buffer is a file mapping -- a safetensors mmap view, or a ttnn .tensorbin loaded by ttnn.as_tensor -- spends its time
pinning pages (get_user_pages splitting huge PMDs, TLB flushes): a 0.2 s upload of a 0.4-1.4 GB tensor took tens of
minutes, intermittently. Uploads from anonymous memory are fast. So:
  * Checkpoint tensors are cloned before use (weights.Checkpoint.get);
  * cached tensors are staged through /dev/shm (shmem, no THP) before loading, and the staged copy is removed after
    the upload;
  * a cache miss converts from an anonymous torch tensor, uploads, then dumps the host tensor for next time.
"""

from __future__ import annotations

import os
import shutil
import tempfile

import ttnn

DRAM = ttnn.DRAM_MEMORY_CONFIG
SHM = os.environ.get("I2M_SHM", "/dev/shm")


def cached(make, dtype, layout, mesh, path: str | None, mesh_mapper=None, memory_config=DRAM):
    """make() -> torch tensor (called on a cache miss only). path: the cache file stem (None: no cache)."""
    f = None if path is None else f"{path}_{dtype.name}_{layout.name}.tensorbin"
    if f is not None and os.path.isfile(f):
        fd, tmp = tempfile.mkstemp(dir=SHM, suffix=".tensorbin")
        os.close(fd)
        try:
            shutil.copyfile(f, tmp)
            host = ttnn._ttnn.tensor.load_tensor_flatbuffer(tmp, device=None)
            dev = host.to(mesh, memory_config)
            ttnn.synchronize_device(mesh)
        finally:
            os.unlink(tmp)
        return dev
    host = ttnn.from_torch(make().contiguous(), dtype=dtype, layout=layout, mesh_mapper=mesh_mapper)
    dev = host.to(mesh, memory_config)
    if f is not None:
        os.makedirs(os.path.dirname(f), exist_ok=True)
        tmp = f + ".tmp"
        ttnn._ttnn.tensor.dump_tensor_flatbuffer(tmp, host)
        os.replace(tmp, f)  # atomic: a killed run never leaves a truncated cache file
    return dev

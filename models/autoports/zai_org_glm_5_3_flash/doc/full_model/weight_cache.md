# Weight cache (stage 6: weight load time)

`tt/weight_cache.py` keeps the **final device tensors** of the model on disk, so a second build skips the FP8
dequantization and the bfp4 / bfp8 conversion. Before the cache, a full 45-layer build spent about 34 s on each of
the 42 MoE layers.

## Design

- **What is cached.** Each weight is built exactly as without the cache. The resulting device tensor is then dumped
  (`ttnn.dump_tensor`: the per-device shards, dtype, layout and memory config). A warm build loads that file straight
  to the mesh, so warm weights are byte-identical to cold ones by construction.
- **Policy retypes happen before the dump.** The optimized decoder's retypes (KDA input / output projections,
  DSA W_UK / W_UV) are applied inside the build (`OptimizedDecoder.from_state_dict` passes `kda_dtypes` and
  `dsa_bmm_dtype`). The cached tensor is therefore the final one, and `_retype_kda` / `_set_dsa` find nothing to do.
- **Lazy checkpoint reads.** `CheckpointReader.layer_state(i, lazy=True)` returns a `ckpt.LazyMap`, and
  `moe_state_dict` is lazy too. A tensor is read and dequantized only on a cache miss. On a hit, the build reads only
  small tensors: mHC, norms, router weight shape and DSA `ape`.
- **Coverage.** Cached: the MoE experts (`up_gate`, `down`), the packed shared expert, `shared_w2`, the router and
  the bias, every KDA weight (set of 10), every DSA projection, the dense MLP and `lm_head`.
  - The ROW_MAJOR embedding table is **not** cached; it is read from the checkpoint, which takes seconds.
  - The un-packed `shared_w1` / `shared_w3` are no longer built when the fused MoE packs them.
- **Keying.** The cache key guards against stale reuse at three levels:
  - Directory: `$GLM53_CACHE_DIR/v<CACHE_FORMAT>-mesh<r>x<c>-<OptConfig.name>-<digest of the policy's weight dtypes>-ckpt<digest of config + tensor index + E192 mask>/`.
  - Component subdirectory: `{top,layer_<i>}/<component>-<digest of the source that builds it>/<name>.tensorbin`, for
    the components top / kda / dsa / mlp / moe.
  - Effect: changing the mesh, the policy (name or any weight dtype) or the checkpoint selects a new directory, and
    editing a component's weight-building code selects a new subdirectory. A stale file is never loaded.
  - Caveat: old directories are left on disk; prune them by hand.
- **Key gaps closed (stage-6 review #1 P2-6, rework 2026-10-18):**
  - Per-layer precision exceptions. A layer whose effective routed-expert policy (`MoEPrecision.from_args(args, layer)`,
    i.e. `args.precision_layer_exceptions`) differs from the policy of the key gets its own moe subdirectory
    `moe-<digest>-p<digest of the effective MoEPrecision>` (`WeightCache.component(kind, policy)`,
    `tt/functional_decoder.py`). Layers without an exception keep the old subdirectory, so the existing cache stays
    valid. The stage-8 `GLM53_PRECISION` candidates already bypass the cache for layers with a MoE weight-dtype exception
    (`PrecisionOverrides.layer_cache`); this covers the args path too.
  - Checkpoint files. `manifest.json` now records `checkpoint_files` = {shard: [size, mtime_ns]} for every file the
    tensor index names (plus `config.json` and `e192_mask.json`). An older manifest adopts the current fingerprint on
    first use. If the shards later differ (same config / index, rewritten files), the build logs a warning and uses
    a fresh directory `<key>-f<digest of the fingerprint>`.
  - The top component digest now also covers `Glm53Model._build_lm_head_ds` (the `lm_ds` LM-head build).
- **Writes** go to a temporary name followed by `os.replace`, so a killed build leaves no truncated file. A read-only
  or full disk logs a warning and the model is served uncached.
- **Loads** (`load_file_tensor`) copy the file into an anonymous memfd with `sendfile`, then call
  `load_tensor_flatbuffer(memfd, mesh)`. See "Load hang" below for why.
  - A watchdog runs in a separate process, because a Python thread cannot run while the stuck call holds the GIL.
  - If one load exceeds `GLM53_CACHE_LOAD_TIMEOUT` seconds (default 120; the largest file, 1.8 GB of bfp4 experts,
    takes about 1 s), the watchdog renames the file to `<name>.stalled` and SIGKILLs the process.
  - The next start rebuilds that tensor from the checkpoint.

Environment variables:

| Variable | Meaning |
|---|---|
| `GLM53_CACHE_DIR` | Cache root. Unset or `off` / `none` / `0` disables the cache. The default (`DEFAULT_DIR`) is currently **off**. |
| `GLM53_CACHE_LOAD_TIMEOUT` | Per-tensor load watchdog in seconds (default 120). |

`Glm53Model(..., weight_cache=None | WeightCache | "env")` overrides the environment.

## Load hang (root cause and fix)

**Symptom.** The first version loaded with `ttnn.load_tensor(path, device=mesh)`. It intermittently stalled forever:

- Cases seen: the embedding file once, a layer-0 KDA file once, and a layer-19 DSA file after 216 good loads in a
  repro script. Each of those files also loads in under 0.1 s in other runs.
- State of the stuck main thread:
  - running at 100 % CPU, with about 2 M minor page faults per second (1.7 billion in total);
  - RSS flat at 0.8 GB, and no cgroup memory limit;
  - holding the GIL, so the in-process watchdog thread never ran;
  - deaf to SIGTERM (`timeout` could not stop it); ptrace (gdb / py-spy) could not attach.

**Mechanism.**

- `ttnn.load_tensor` mmaps the file `MAP_PRIVATE, PROT_READ`. `to_device` then pins that host buffer read-only
  through the KMD (`tt_metal/distributed/pinned_memory_cache.cpp` `try_pin`, whose comments name this "mmap(PROT_READ)
  of a file" case).
- Host: kernel 7.0.0-34, ext4 page cache in large folios (FileHugePages about 0.9 GB), KMD 2.10.0, no movable zone or
  CMA.
- A fault storm with flat RSS means the kernel keeps faulting the same pages without ever finishing the long-term pin.

**Control.** A host-only read of all 113 GB of cache files through the same `MAP_PRIVATE` mmaps never stalled: 3
passes with the page cache evicted between them, 24-31 s per pass (`scripts/wc_debug/mmap_read.py` in the bring-up
workspace). So the mmap reads are fine and the pin is the trigger.

We did not trace this into the kernel; the mechanism is inferred from this signature and from the `try_pin` source.

**Fix.** Load from a memfd copy instead of mapping the ext4 file:

- `sendfile` reads through the page-cache read path, not mmap, at about 3 GB/s.
- shmem has huge pages disabled on this host, so the pinned pages are small anonymous-shmem folios, never ext4
  page-cache folios.
- No stall has occurred since. The external watchdog is the backstop.

**Related fix.** Calling Python `ttnn.to_device(host, mesh, memory_config=...)` on the loaded host tensor is avoided.
The C++ `load_tensor_flatbuffer(path, mesh)` path is used instead, the same `to_device` that `ttnn.load_tensor` uses.
A segfault seen in that run turned out to be the check tool dumping a weight that a later stage had deallocated; the
tool now skips tensors that are no longer allocated.

## Verification

`tools/weight_cache_check.py` builds the model, times the load, dumps every listed device weight and hashes its data
and header, and saves the logits of one prefill (prompt `arange(1000, 1075)`) plus one eager decode step.
`--compare` checks the runs for byte / bit identity.

**Reduced model** (layers 0, 3, 4, TP4, 32 slots; all phases under one device lock, with the same md5 of `tt/*.py`
before and after; log `logs/s6_wc_r_all.log`, results `artifacts/weight_cache_check/reduced/`):

| Run | Weight load | Generator build | Cache |
|---|---|---|---|
| no cache | 66.2 s | 69.5 s | none |
| cold (fill) | 68.4 s | 71.7 s | 48 stored |
| warm (page cache evicted) | 9.3 s | 15.6 s | 48 hits |
| mixed (top/ and layer-4 MoE removed) | 35.5 s | 42.5 s | 41 hits, 7 stored |

Result: 39/39 device weights byte-identical (data and header) in all four runs, and prefill + decode logits
bit-identical in all four (argmax 62273 / 67404). Cache size: 6.4 GB.

Earlier attempts whose phases were queued hours apart showed logits differences. Those came from the shared tree
changing between phases, not from the cache: weights were still byte-identical, and a no-load cold build under the
newer code reproduced the warm logits exactly. For that reason, verification now runs all phases under one lock
acquisition.

**Full model** (45 layers, TP4, 8 slots x 4096; `logs/s6_wc_f_all.log`, `artifacts/weight_cache_check/full/`): see
the results section below.

## Results: full model

Run glm53-wc-full8, 2026-10-17 20:32-21:01 UTC, one device-lock acquisition. Policy `opt_s5`; the cache directory
started empty.

| Run | Weight load | Generator build | Cache |
|---|---|---|---|
| cold (fill) | 1432.0 s (23.9 min) | 1447.7 s | 734 stored |
| warm (page cache evicted, disk reads) | 110.1 s | 132.9 s | 734 hits |
| warm hot (page cache warm) | 73.5 s | 86.3 s | 734 hits |

- Weights: 54/54 digested device weights byte-identical across the three runs. They cover `lm_head` and every weight
  of layers 0, 3, 4 and 44, including the bfp4 experts.
- Logits: prefill and decode logits bit-identical (argmax 1004 / 1005).
- Cache size: 116 GB at `~/.cache/glm53_flash/v1-mesh1x4-opt_s5-56b50afe6f-ckpt5bbd7a14bd`.
- Stalls: none; the watchdog never fired.
- Caveat: the md5 of `tt/*.py` differed between the start and the end of the run (another agent edited the tree
  meanwhile). The three phases still produced byte-identical weights and bit-identical logits.

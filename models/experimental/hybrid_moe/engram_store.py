# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Host side of Hy4-flash's Engram memories: an FP8 copy of the four hash-embedding tables and the lookup that feeds
the device.

The model reads every looked-up 96-wide row through a per-row FP8 e4m3 round trip (`_fp8_round_trip` in
modeling_hy_v4_flash: scale = amax / 448 in bf16, q = fp8(x / scale) * scale). This file stores exactly that fp8 value
and the bf16 scale per row, so `lookup()` returns rows bit-identical to the reference's
`_fp8_round_trip(multi_head_embedding(hash_ids))` (x + (q - x) == q exactly in bf16: q is within a factor 2 of x or
0, so q - x is exact, Sterbenz) at half the RAM: 4 x ~15.8 GB fp8 + 4 x ~0.33 GB scales instead of 4 x 31.7 GB.

The hash ids depend only on token ids (n-grams of orders 2 and 3 ending at each position, pad id before the document
start), so the host computes them ahead of a pass: decode, verify rows and prefill alike.

  python engram_store.py convert [layers]     # build ~/.cache/hybrid-moe/hy4_engram/ (HY4_ENGRAM_DIR) from the BF16 shards
  python engram_store.py test                 # bit-exactness vs the HF reference code + lookup timing

Layers are decoder layer indices (1, 11, 21, 31); the Engram layer id of the config (`engram_layer_ids` 2, 12, 22, 32)
is layer + 1. `engram_mask`: positions whose token is the pad id (or an image / video token) get no Engram output at
all (the whole residual is zeroed, not just the rows), so `lookup()` returns that mask with the rows.
"""

import glob
import json
import os
import struct
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

REPO = os.environ.get("HY4_REPO", "tencent/Hy4-flash")
HUB = os.environ.get("HF_HUB") or os.environ.get("HF_HUB_CACHE") or os.path.join(
    os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub"
)
OUT = os.environ.get("HY4_ENGRAM_DIR", os.path.expanduser("~/.cache/hybrid-moe/hy4_engram"))
LAYERS = (1, 11, 21, 31)
FP8_MAX = 448.0  # torch.finfo(torch.float8_e4m3fn).max
FP32_EPS = 1.1920928955078125e-07  # nn.RMSNorm(eps=None) on bf16 input
IMAGE_TOKEN_ID, VIDEO_TOKEN_ID = 120120, 120124  # HYV4FlashConfig (top level), kept out of the n-gram memories


_SNAP = os.environ.get("HY4_CKPT_DIR")  # a snapshot dir; overrides the HF cache lookup


def snapshot():
    if _SNAP:
        return _SNAP
    found = sorted(glob.glob(f"{HUB}/models--{REPO.replace('/', '--')}/snapshots/*"))
    if not found:
        raise FileNotFoundError(f"{REPO} is not in {HUB}")
    return found[0]


def text_config():
    raw = json.load(open(os.path.join(snapshot(), "config.json")))
    return raw.get("text_config", raw)


def weight_map():
    snap = snapshot()
    p = os.path.join(snap, "model.safetensors.index.json")
    if not os.path.exists(p):
        p = os.environ.get("HY4_INDEX", os.path.expanduser("~/m27-work/hy4/index.json"))
    return json.load(open(p))["weight_map"]


def tensor_memmap(key, wmap=None):
    """np.memmap view of one checkpoint tensor (no copy; pages come from the page cache on access)."""
    wmap = wmap or weight_map()
    path = os.path.join(snapshot(), wmap[key])
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    meta = hdr[key]
    dt = {"BF16": np.uint16, "F32": np.float32, "I32": np.int32, "I64": np.int64}[meta["dtype"]]
    start, end = meta["data_offsets"]
    return np.memmap(path, dtype=dt, mode="r", offset=8 + n + start, shape=tuple(meta["shape"]))


def table_key(layer):
    return f"model.layers.{layer}.engram_layer.multi_head_embedding.embedding.weight"


class EngramHasher:
    """Exactly `HYV4FlashEngramLayer.hash_ngrams` + `HYV4FlashEngramEmbedding` offsets, in numpy int64."""

    def __init__(self, layer, cfg=None, wmap=None):
        cfg = cfg or text_config()
        self.layer = layer
        layer_id = layer + 1
        self.max_ngram = cfg["engram_max_ngram_size"]  # 3
        self.n_head = cfg["engram_n_head_per_ngram"]  # 8 per order
        self.pad_id = cfg["engram_pad_id"]
        sizes = [s for order in cfg["engram_layer_head_vocab_sizes"][str(layer_id)] for s in order]
        self.moduli = np.array(sizes, dtype=np.int64)  # [16], n-gram-major (order 2 heads, then order 3)
        self.offsets = np.concatenate([[0], np.cumsum(self.moduli)[:-1]]).astype(np.int64)
        self.lookup_table = np.asarray(tensor_memmap(f"model.layers.{layer}.engram_layer.lookup_table", wmap)).astype(
            np.int64
        )
        # `_init_weights` redraws the multipliers from the loaded map: bounded by max(map) + 1
        compressed_vocab = int(self.lookup_table.max()) + 1
        half_bound = max(1, np.iinfo(np.int64).max // compressed_vocab // 2)
        gen = np.random.default_rng(cfg.get("engram_seed", 0) + 10007 * layer_id)
        drawn = gen.integers(0, half_bound, size=self.max_ngram, dtype=np.int64)
        self.multipliers = drawn * 2 + 1

    def ngram_tokens(self, ids, start=0, history=None):
        """[T, max_ngram] token ids of the n-gram ending at each of `ids` (positions start .. start + T - 1 of one
        document): column 0 is the token itself, column k the token k back, the pad id before the document start.
        `history`: the tokens before `ids` (at least the last max_ngram - 1 of them when start > 0)."""
        ids = np.asarray(ids, dtype=np.int64)
        T = len(ids)
        k = self.max_ngram - 1
        if start > 0:
            prev = np.asarray(history, dtype=np.int64)[-k:] if history is not None else None
            if prev is None or len(prev) < min(k, start):
                raise ValueError("ngram_tokens: need the previous tokens when start > 0")
            ctx = np.concatenate([np.full(k - len(prev), self.pad_id, np.int64), prev, ids])
        else:
            ctx = np.concatenate([np.full(k, self.pad_id, np.int64), ids])
        out = np.stack([ctx[k - o : k - o + T] for o in range(self.max_ngram)], -1)
        pos = start + np.arange(T)
        for o in range(1, self.max_ngram):  # positions reaching past the document start read the pad id
            out[pos < o, o] = self.pad_id
        return out

    def hash_ids(self, ngram):
        """[T, max_ngram] n-gram token ids -> [T, 16] hash ids (< each head's modulus, never negative for text)."""
        size = self.lookup_table.shape[0]
        comp = self.lookup_table[np.minimum(ngram, size - 1)]
        comp = np.where(ngram < size, comp, self.lookup_table[self.pad_id])
        mixed = comp * self.multipliers  # bounded by the draw: no int64 overflow
        out = []
        for n in range(2, self.max_ngram + 1):
            mix = mixed[..., 0]
            for o in range(1, n):
                mix = np.bitwise_xor(mix, mixed[..., o])
            s = (n - 2) * self.n_head
            out.append(mix[..., None] % self.moduli[s : s + self.n_head])
        return np.concatenate(out, -1)

    def rows(self, ngram):
        """Flat table rows [T, 16] (hash id + head offset)."""
        return self.hash_ids(ngram) + self.offsets


def fp8_round_trip_parts(x):
    """x bf16 [..., 96] -> (fp8 e4m3 bytes uint8, bf16 scale) exactly as `_fp8_round_trip` computes them."""
    scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / FP8_MAX  # bf16 arithmetic, as the model
    q8 = (x / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q8.view(torch.uint8), scale


def dequant(q8_u8, scale_bf16):
    # + 0.0: the reference returns x + (q - x), which is +0 wherever q is +-0 (x - x rounds to +0); q itself would
    # keep the sign of a negative zero
    return q8_u8.view(torch.float8_e4m3fn).to(torch.bfloat16) * scale_bf16 + 0.0


def convert(layers=LAYERS, chunk=1 << 21):
    torch.set_num_threads(int(os.environ.get("ENGRAM_THREADS", "4")))
    os.makedirs(OUT, exist_ok=True)
    wmap = weight_map()
    for li in layers:
        done = os.path.join(OUT, f"L{li}.done")
        if os.path.exists(done):
            print(f"layer {li}: done already", flush=True)
            continue
        src = tensor_memmap(table_key(li), wmap)
        R, D = src.shape
        q = np.lib.format.open_memmap(os.path.join(OUT, f"L{li}_fp8.npy"), "w+", np.uint8, (R, D))
        s = np.lib.format.open_memmap(os.path.join(OUT, f"L{li}_scale.npy"), "w+", np.uint16, (R,))
        t0 = time.time()
        for r in range(0, R, chunk):
            x = torch.from_numpy(np.array(src[r : r + chunk])).view(torch.bfloat16)
            q8, sc = fp8_round_trip_parts(x)
            q[r : r + chunk] = q8.numpy()
            s[r : r + chunk] = sc[:, 0].view(torch.uint16).numpy()
            if (r // chunk) % 16 == 0:
                print(f"layer {li}: {r / R:6.1%} {time.time() - t0:.0f}s", flush=True)
        q.flush()
        s.flush()
        del q, s
        open(done, "w").write(f"{R} {D}\n")
        print(f"layer {li}: {R} rows in {time.time() - t0:.0f}s", flush=True)


def _psi_avg10():
    try:
        return float(open("/proc/pressure/memory").readline().split()[1].split("=")[1])
    except Exception:  # noqa: BLE001
        return None


def _mem_available():
    try:
        return int(next(l for l in open("/proc/meminfo") if l.startswith("MemAvailable")).split()[1]) * 1024
    except Exception:  # noqa: BLE001
        return None


def _psi_wait(limit=None, max_wait=30.0, floor=0):
    """Throttle a bulk host-memory load on the kernel's memory pressure (PSI some avg10): systemd-oomd kills the user
    scope above 50 % sustained for 30 s; wait while avg10 > limit (default HY4_ENGRAM_PSI_MAX, 15 %) or MemAvailable <
    floor bytes, at most max_wait s per call."""
    limit = float(os.environ.get("HY4_ENGRAM_PSI_MAX", "15")) if limit is None else limit
    t0 = time.time()
    while time.time() - t0 < max_wait:
        avg10 = _psi_avg10()
        avail = _mem_available()
        if (avg10 is None or avg10 <= limit) and (avail is None or avail >= floor):
            return
        time.sleep(0.5)


def _mlock(mm):
    """mlock an mmap object's pages (returns the bytes locked, 0 when the lock fails: RLIMIT_MEMLOCK, ...)."""
    import ctypes

    try:
        libc = ctypes.CDLL(None, use_errno=True)
        addr = ctypes.addressof(ctypes.c_char.from_buffer(mm))
        if libc.mlock(ctypes.c_void_p(addr), ctypes.c_size_t(len(mm))) == 0:
            return len(mm)
        print(f"engram: mlock failed (errno {ctypes.get_errno()}): the copy may be swapped out", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"engram: mlock failed ({e})", flush=True)
    return 0


class EngramStore:
    """The four FP8 tables (memory-mapped; `in_ram=True` reads them into RAM) and their hashers."""

    def __init__(self, layers=LAYERS, in_ram=None):
        cfg = text_config()
        wmap = weight_map()
        self.layers = tuple(layers)
        self.pad_id = cfg["engram_pad_id"]
        self.hashers = {li: EngramHasher(li, cfg, wmap) for li in self.layers}
        self.q, self.s = {}, {}
        if in_ram is None:
            # HY4_ENGRAM_RAM: anon (default) = one copy in process memory, loaded in the background without passing
            # through the page cache; warm = page cache only (this host's page cache was dropped repeatedly while the
            # device model loaded: lookups went back to NVMe); 0 = cold memory map
            in_ram = {"1": "anon", "anon": "anon", "warm": "warm"}.get(os.environ.get("HY4_ENGRAM_RAM", "anon"), False)
        for li in self.layers:
            mm = "r" if in_ram in (False, "warm", "anon") else None
            self.q[li] = np.load(os.path.join(OUT, f"L{li}_fp8.npy"), mmap_mode=mm)
            self.s[li] = np.load(os.path.join(OUT, f"L{li}_scale.npy"), mmap_mode=mm)
        self.loaders = []
        self.mode = in_ram
        if in_ram == "anon":
            import threading

            th = threading.Thread(target=self._load_anon, daemon=True, name="engram-anon")
            th.start()
            self.loaders.append(th)
        if in_ram == "warm":
            # A cold memory-mapped lookup is 64 random NVMe reads per token (~7 ms per Engram layer); from the page
            # cache it is ~0.05 ms. The tables (~65 GB) are read ONCE, sequentially, by one background thread into the
            # page cache (evictable; no anonymous copy). Never copy them into process memory: page cache + copy is
            # ~2x the size, and the reclaim pressure made systemd-oomd kill the whole user scope (2026-11-19).
            import threading

            th = threading.Thread(target=self._warm, daemon=True, name="engram-warm")
            th.start()
            self.loaders.append(th)

    def _warm(self, chunk=1 << 26):
        files = [os.path.join(OUT, f"L{li}_{n}.npy") for li in self.layers for n in ("scale", "fp8")]
        need = sum(os.path.getsize(f) for f in files)
        margin = int(float(os.environ.get("HY4_ENGRAM_WARM_MARGIN_GB", "48")) * 2**30)
        try:
            avail = int(next(l for l in open("/proc/meminfo") if l.startswith("MemAvailable")).split()[1]) * 1024
        except Exception:  # noqa: BLE001
            avail = 0
        if avail < need + margin:
            print(f"engram: page-cache warm skipped (MemAvailable {avail / 2**30:.0f} GiB < tables "
                  f"{need / 2**30:.0f} + margin {margin / 2**30:.0f} GiB): lookups read the NVMe", flush=True)
            return
        t0 = time.time()
        buf = bytearray(chunk)
        for f in files:
            with open(f, "rb", buffering=0) as fh:
                while fh.readinto(buf):
                    pass
        print(f"engram tables in the page cache ({need / 2**30:.0f} GiB, {time.time() - t0:.0f} s)", flush=True)

    def _load_anon(self, chunk=1 << 26):
        """Each table into an anonymous page-aligned buffer with O_DIRECT reads by parallel readers (tt/fastload.py
        read_direct; HY4_ENGRAM_THREADS, default 4): no page cache at all, so memory holds one copy (~61 GB), never page
        cache + copy (that doubled footprint made systemd-oomd kill the user scope, 2026-11-19), and the NVMe's ~4 GB/s
        instead of this host's ~0.2-1 GB/s buffered reads. Lookups use the memory map until a table's copy is complete.
        Throttle (per 64 MiB piece): memory pressure (PSI some avg10) above max(HY4_ENGRAM_PSI_MAX (15), the avg10
        measured at the start + HY4_ENGRAM_PSI_DELTA (15)), capped at HY4_ENGRAM_PSI_CAP (40, below oomd's 50; at least start + 5), or
        MemAvailable under HY4_ENGRAM_MIN_FREE_GB (24). Skipped (memory map only) without MemAvailable >= tables +
        HY4_ENGRAM_WARM_MARGIN_GB (48). HY4_ENGRAM_DIRECT=0: the old buffered single-thread copy."""
        files = [(li, n) for li in self.layers for n in ("scale", "fp8")]
        need = sum(os.path.getsize(os.path.join(OUT, f"L{li}_{n}.npy")) for li, n in files)
        margin = int(float(os.environ.get("HY4_ENGRAM_WARM_MARGIN_GB", "48")) * 2**30)
        avail = _mem_available() or 0
        if avail < need + margin:
            print(f"engram: RAM copy skipped (MemAvailable {avail / 2**30:.0f} GiB < tables {need / 2**30:.0f} + margin "
                  f"{margin / 2**30:.0f} GiB): lookups read the memory map", flush=True)
            return
        base = _psi_avg10() or 0.0
        limit = min(max(float(os.environ.get("HY4_ENGRAM_PSI_MAX", "15")), base + float(os.environ.get("HY4_ENGRAM_PSI_DELTA", "15"))),
                    float(os.environ.get("HY4_ENGRAM_PSI_CAP", "40")))
        limit = max(limit, base + 5)  # a baseline already at the cap is the host's, not this load's: don't stall on it
        floor = int(float(os.environ.get("HY4_ENGRAM_MIN_FREE_GB", "24")) * 2**30)
        waited = [0.0]

        def wait():
            t = time.time()
            _psi_wait(limit, floor=floor)
            waited[0] += time.time() - t

        direct = os.environ.get("HY4_ENGRAM_DIRECT", "1") == "1"
        # HY4_ENGRAM_MLOCK=1 (default): the copy locked in RAM. Table rows are touched at random and rarely, so the
        # kernel swapped them out on this host (swap in use, kswapd busy): served lookups took ~1.3 ms per decode step
        # instead of ~0.1 ms (major faults on the row gathers). Locking keeps the same footprint, unswappable.
        lock = direct and os.environ.get("HY4_ENGRAM_MLOCK", "1") == "1"
        huge = direct and os.environ.get("HY4_ENGRAM_HUGEPAGE", "0") == "1"
        locked = [0]
        rd = None
        if direct:
            import mmap
            from concurrent.futures import ThreadPoolExecutor

            try:
                from models.autoports.tencent_hy4_flash.tt import fastload
            except ImportError:  # run from this directory
                sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../autoports/tencent_hy4_flash/tt"))
                import fastload
            rd = ThreadPoolExecutor(int(os.environ.get("HY4_ENGRAM_THREADS", "4")), thread_name_prefix="engram-direct")
        t0 = time.time()
        for li, n in files:
            d = self.s if n == "scale" else self.q
            src = d[li]
            path = os.path.join(OUT, f"L{li}_{n}.npy")
            if direct:
                # the whole file (its .npy header too) into one page-aligned anonymous mapping; the array is a view
                # at the header offset (no second copy)
                size = os.path.getsize(path)
                mm = mmap.mmap(-1, -(-size // fastload.ALIGN) * fastload.ALIGN)
                if huge and hasattr(mmap, "MADV_HUGEPAGE"):
                    mm.madvise(mmap.MADV_HUGEPAGE)
                fastload.read_direct(path, mm, 0, size, chunk=chunk, wait=wait, pool=rd)
                if lock:
                    locked[0] += _mlock(mm)
                dst = np.frombuffer(mm, src.dtype, count=src.size, offset=src.offset).reshape(src.shape)
            else:
                dst = np.empty(src.shape, src.dtype)
                buf = memoryview(dst.reshape(-1).view(np.uint8))
                with open(path, "rb", buffering=0) as fh:
                    fd = fh.fileno()
                    off = src.offset
                    fh.seek(off)
                    done = 0
                    while done < len(buf):
                        wait()
                        k = fh.readinto(buf[done : done + (1 << 28)])
                        if not k:
                            raise IOError(f"short read {path}")
                        os.posix_fadvise(fd, off + done, k, os.POSIX_FADV_DONTNEED)
                        done += k
            d[li] = dst
        if rd is not None:
            rd.shutdown()
        self.load_s = time.time() - t0
        print(f"engram tables in RAM ({need / 2**30:.0f} GiB, {self.load_s:.0f} s; throttled {waited[0]:.0f} s, PSI limit "
              f"{limit:.0f} (start {base:.0f}){', O_DIRECT' if direct else ''}"
              f"{f', mlocked {locked[0] / 2**30:.0f} GiB' if lock else ''}{', MADV_HUGEPAGE' if huge else ''})", flush=True)

    def rewarm(self):
        if self.mode != "warm":
            return
        """Warm the page cache again (after a large load, e.g. the device model's tensor cache, evicted the tables)."""
        import threading

        th = threading.Thread(target=self._warm, daemon=True, name="engram-rewarm")
        th.start()
        self.loaders.append(th)

    def wait_ram(self):
        for th in self.loaders:
            th.join()

    def mask(self, ids):
        ids = np.asarray(ids)
        return (ids != IMAGE_TOKEN_ID) & (ids != VIDEO_TOKEN_ID) & (ids != self.pad_id)

    def lookup(self, ids, start=0, history=None, layers=None):
        """Rows for `ids` (positions start ..): {layer: bf16 [T, 16, 96]} and the engram mask [T] (bool)."""
        out = {}
        ng = None
        for li in layers or self.layers:
            h = self.hashers[li]
            if ng is None:
                ng = h.ngram_tokens(ids, start, history)  # the same for every layer
            rows = h.rows(ng)  # [T, 16]
            q8 = torch.from_numpy(self.q[li][rows.reshape(-1)])  # [T*16, 96] uint8
            sc = torch.from_numpy(self.s[li][rows.reshape(-1)].astype(np.uint16)).view(torch.bfloat16)[:, None]
            out[li] = dequant(q8, sc).view(len(ids), h.moduli.shape[0], -1)
        return out, self.mask(ids)


class EngramRows:
    """The device model's view (tt/model.py): rows(li, ids, p0) -> the Engram layer's `embeddings` after the per-head
    `lookup_per_head_norms` and the flatten, i.e. the input of value_proj / key_projs: bf16 [T, 16 x 96 = 1536],
    bit-exact with the HF layer (fp8 round trip, then nn.RMSNorm(96) with eps=None: fp32 math, eps = finfo(fp32).eps).

    li: 0-based decoder layer (1, 11, 21, 31). ids: the next T tokens of ONE sequence at positions p0 .. p0+T-1; calls
    for a sequence are in order and contiguous per layer; the store keeps the last max_ngram_size - 1 tokens itself;
    p0 == 0 or reset() starts a new sequence.
    Engram mask: the reference zeroes the WHOLE Engram output (not just these rows) at positions whose token is the pad
    id (120002) or an image / video token; `mask(ids)` returns it (all True for ordinary text).
    store_dir: the converted tables (default HY4_ENGRAM_DIR); ckpt_dir: the checkpoint snapshot dir (config.json,
    model.safetensors.index.json and the shards with the lookup tables / norm weights; default the HF cache)."""

    def __init__(self, store_dir=None, ckpt_dir=None, in_ram=None, layers=LAYERS):
        global OUT, _SNAP
        if store_dir:
            OUT = store_dir
        if ckpt_dir:
            _SNAP = ckpt_dir
        self.store = EngramStore(layers, in_ram=in_ram)
        cfg = text_config()
        wmap = weight_map()
        self.n_head = 2 * cfg["engram_n_head_per_ngram"]
        self.keep = cfg["engram_max_ngram_size"] - 1
        self.norm_w = {
            li: torch.stack(
                [
                    torch.from_numpy(
                        np.array(tensor_memmap(f"model.layers.{li}.engram_layer.lookup_per_head_norms.{h}.weight", wmap))
                    ).view(torch.bfloat16)
                    for h in range(self.n_head)
                ]
            )
            for li in layers
        }  # [16, 96] bf16
        self.reset()

    def reset(self):
        self.hist = {li: np.zeros(0, np.int64) for li in self.store.layers}
        self.pos = {li: 0 for li in self.store.layers}

    def mask(self, ids):
        return torch.from_numpy(self.store.mask(np.asarray(ids)))

    def rows(self, li, ids, p0):
        ids = np.asarray(ids.cpu() if torch.is_tensor(ids) else ids, dtype=np.int64).reshape(-1)
        if p0 == 0:
            self.hist[li], self.pos[li] = np.zeros(0, np.int64), 0
        if p0 != self.pos[li]:
            raise ValueError(f"engram rows: layer {li} expected position {self.pos[li]}, got {p0} (call reset())")
        out, _ = self.store.lookup(ids, start=p0, history=self.hist[li] if p0 else None, layers=[li])
        x = out[li]  # [T, 16, 96] bf16, after the fp8 round trip
        # nn.RMSNorm(96, eps=None) on bf16: computed in fp32 with eps = finfo(float32).eps (NOT bf16's), the weight
        # applied in fp32, one rounding to bf16; here for the 16 heads at once (bit-identical, checked by test())
        xf = x.float()
        y = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + FP32_EPS) * self.norm_w[li].float()).to(torch.bfloat16)
        self.hist[li] = np.concatenate([self.hist[li], ids])[-self.keep :]
        self.pos[li] = p0 + len(ids)
        return y.flatten(1)


# ------------------------------------------------------------------------------------------------- self-test
def _hf_layer(li):
    """The HF Engram layer's hashing state (meta for the table: only buffers are materialised)."""
    from transformers import AutoConfig
    from transformers.models.hy_v4_flash import modeling_hy_v4_flash as M

    cfg = AutoConfig.from_pretrained(snapshot()).text_config
    with torch.device("meta"):
        layer = M.HYV4FlashEngramLayer(cfg, li + 1)
    lut = torch.from_numpy(np.asarray(tensor_memmap(f"model.layers.{li}.engram_layer.lookup_table")).copy())
    layer.lookup_table = lut
    layer.head_vocab_sizes = torch.tensor(layer.config_head_vocab_sizes)
    layer.hash_multipliers = layer.draw_hash_multipliers(int(lut.max()) + 1)
    offs = M.HYV4FlashEngramEmbedding.head_offsets(layer.multi_head_embedding)
    return M, cfg, layer, offs


def test():
    rng = np.random.default_rng(0)
    cfg = text_config()
    store = EngramStore()
    er = EngramRows()
    for li in LAYERS:
        M, hcfg, layer, offs = _hf_layer(li)
        h = store.hashers[li]
        assert np.array_equal(offs.numpy(), h.offsets), "head offsets"
        assert np.array_equal(layer.hash_multipliers.numpy(), h.multipliers), "multipliers"
        T = 4096
        ids = rng.integers(0, cfg["vocab_size"], T)
        ids[rng.integers(0, T, 40)] = cfg["engram_pad_id"]
        ids[:3] = [120000, 5, 7]
        ids_t = torch.from_numpy(ids)[None]
        # HF path: build_ngram_tokens (position_ids None) -> hash_ngrams -> embedding rows -> _fp8_round_trip
        tm = M.HYV4FlashTextModel.__new__(M.HYV4FlashTextModel)
        torch.nn.Module.__init__(tm)
        tm.config = hcfg
        ng_hf = M.HYV4FlashTextModel.build_ngram_tokens(tm, ids_t, None)[0].numpy()
        ng = h.ngram_tokens(ids)
        assert np.array_equal(ng, ng_hf), "ngram tokens"
        # incremental: positions 100.. given the history
        ng_inc = h.ngram_tokens(ids[100:], start=100, history=ids[:100])
        assert np.array_equal(ng_inc, ng_hf[100:]), "incremental ngram tokens"
        hid_hf = layer.hash_ngrams(torch.from_numpy(ng_hf)[None])[0].numpy()
        assert np.array_equal(hid_hf, h.hash_ids(ng)), "hash ids"
        assert hid_hf.min() >= 0
        flat = (hid_hf + offs.numpy()).reshape(-1)
        src = tensor_memmap(table_key(li))
        x = torch.from_numpy(np.array(src[flat])).view(torch.bfloat16).view(T, 16, 96)
        ref = M._fp8_round_trip(x)
        rows, mask = store.lookup(ids, layers=[li])
        same = torch.equal(rows[li].view(torch.int16), ref.view(torch.int16))
        diff = (rows[li].float() - ref.float()).abs().max().item()
        print(f"layer {li}: rows bit-identical {same} (max |diff| {diff:g}), masked {int((~mask).sum())}", flush=True)
        assert same
        # EngramRows: the HF per-head nn.RMSNorm(96) modules (bf16 weights, eps=None) after the round trip, flattened;
        # fed in contiguous pieces (prefill chunks, then single decode tokens)
        norms = [torch.nn.RMSNorm(96).to(torch.bfloat16) for _ in range(16)]
        for hh, nm in enumerate(norms):
            nm.weight.data = er.norm_w[li][hh].clone()
        want = torch.stack([nm(ref[:, hh]) for hh, nm in enumerate(norms)], 1).flatten(-2)
        cuts = [0, 1000, 1001, 1002, 2500, T]
        got = torch.cat([er.rows(li, torch.from_numpy(ids[a:b]), a) for a, b in zip(cuts[:-1], cuts[1:])])
        assert got.shape == (T, 1536) and got.dtype == torch.bfloat16
        same = torch.equal(got.view(torch.int16), want.view(torch.int16))
        print(f"layer {li}: EngramRows (chunked + incremental) bit-identical {same}", flush=True)
        assert same
    # timing
    for T in (1, 3, 8192):
        ids = rng.integers(0, cfg["vocab_size"], T)
        store.lookup(ids, start=10, history=rng.integers(0, 1000, 10))
        t0 = time.perf_counter()
        n = 20 if T < 100 else 3
        for _ in range(n):
            store.lookup(ids, start=10, history=rng.integers(0, 1000, 10))
        print(f"lookup T={T}: {(time.perf_counter() - t0) / n * 1e3:.2f} ms (4 layers)", flush=True)
        er.reset()
        t0 = time.perf_counter()
        for li in LAYERS:
            er.rows(li, torch.from_numpy(ids), 0)
        print(f"EngramRows T={T}: {(time.perf_counter() - t0) * 1e3:.2f} ms (4 layers, incl. norms)", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "test"
    if cmd == "convert":
        convert(tuple(int(x) for x in sys.argv[2:]) or LAYERS)
    else:
        test()

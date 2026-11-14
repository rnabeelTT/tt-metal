# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Host RAM check: does this host have the free memory the selected profile needs, before anything is loaded?

A hybrid layer keeps two host copies of its missable experts: the int4 library rows (HYB_LIB, resident for the CPU
kernel) and the device-format copies (HYB_DEV_EXPERTS, pinned for PCIe fetch / streaming). Their sizes are read from the
files themselves, so the estimate follows the library, not a constant. On top: BASE_GIB for everything else (vLLM API
server, engine process, host embedding table, drafter, staging slots; measured 10.3 GB on MiniMax-M2.7, 32K profile;
13 GiB keeps a margin now that only the missable rows of the library are counted).

needed(layers) = BASE_GIB + sum over hybrid layers of (int4 file + device-format files)

An elastic profile (HYB_ELASTIC_DEMOTE) starts with the short layout and demotes up to every listed layer for long
contexts, so its maximum is the layout with all of them hybrid; the profile's max_model_len is only reachable with that.

HYB_RAM_CHECK: error (default: refuse to start, naming what fits), warn (start anyway; elastic switches log when they
are about to exceed free RAM), off. Compared against MemAvailable (and the container's cgroup limit, if any): RAM that
is free right now, not swap. Running short makes the kernel swap the expert copies (the CPU path becomes several times
slower) or kill the server.
"""

import os
import sys

GIB = 1 << 30
BASE_GIB = float(os.environ.get("HYB_RAM_BASE_GIB", "13"))
_sizes = None


def available():
    """Bytes of RAM this process can still get: MemAvailable, capped by a cgroup v2 memory limit."""
    avail = None
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                avail = int(line.split()[1]) * 1024
    try:
        lim = open("/sys/fs/cgroup/memory.max").read().strip()
        if lim != "max":
            used = int(open("/sys/fs/cgroup/memory.current").read())
            avail = min(avail, int(lim) - used)
    except OSError:
        pass
    return avail


def _num_experts():
    """Routed experts per layer from HF_MODEL's config (num_experts / num_local_experts / n_routed_experts), or None."""
    try:
        import json

        from . import hfpath

        c = json.load(open(os.path.join(hfpath.snapshot(os.environ["HF_MODEL"]), "config.json")))
        c = c.get("text_config", c)
        for k in ("num_experts", "num_local_experts", "n_routed_experts"):
            if c.get(k):
                return int(c[k])
    except Exception:  # noqa: BLE001 - the estimate falls back to the whole library
        pass
    return None


def layer_bytes(lib_dir, dev_dir):
    """{layer: host bytes the server keeps for that layer's missable experts when it is hybrid} (cached):
    missable = E - (HYB_SLOTS - 1 - HYB_LRU) per layer (static slots; the fused path's sink and the LRU slots hold no
    static expert), capped at the library's experts; int4 copies counted when resident (HYB_LIB_RESIDENT, default 1:
    =0 serves them from the page cache), device-format copies (only with HYB_LRU > 0: the fetch) up to HYB_PIN_TOP per
    layer (0 = all)."""
    global _sizes
    if _sizes is None:
        lib, dev, n_dev = {}, {}, {}
        if lib_dir and os.path.isdir(lib_dir):
            for e in os.scandir(lib_dir):
                if e.name.startswith("layer_") and e.name.endswith(".pt"):
                    L = int(e.name[len("layer_") : -len(".pt")])
                    lib[L] = lib.get(L, 0) + e.stat().st_size
        if dev_dir and os.path.isdir(dev_dir):
            for e in os.scandir(dev_dir):  # <L>_<e>_gu.tensorbin / <L>_<e>_dn.tensorbin
                head = e.name.split("_", 1)[0]
                if head.isdigit() and e.name.endswith(".tensorbin"):
                    L = int(head)
                    dev[L] = dev.get(L, 0) + e.stat().st_size
                    n_dev[L] = n_dev.get(L, 0) + e.name.endswith("_gu.tensorbin")
        E, C = _num_experts(), int(os.environ.get("HYB_SLOTS", "0") or 0)
        lru = int(os.environ.get("HYB_LRU", "0") or 0)
        pin_top = int(os.environ.get("HYB_PIN_TOP", "0") or 0)
        resident = os.environ.get("HYB_LIB_RESIDENT", "1") == "1"
        sizes = {}
        for L in set(lib) | set(dev):
            n_lib = n_dev.get(L, 0)  # the library and the device copies hold the same experts (one plan, one S)
            cold = n_lib
            if E and C and n_lib:
                cold = min(n_lib, max(0, E - (C - 1 - lru)))
            b = 0
            if resident and L in lib:
                b += lib[L] * cold // n_lib if n_lib else lib[L]
            if lru > 0 and L in dev and n_lib:  # the device-format copies load only with the fetch on (HYB_LRU > 0)
                b += dev[L] * (min(cold, pin_top) if pin_top else cold) // n_lib
            sizes[L] = b
        _sizes = sizes
    return _sizes


def needed(hybrid_layers, sizes):
    return int(BASE_GIB * GIB) + sum(sizes.get(L, 0) for L in hybrid_layers)


def _gb(n):
    return f"{n / 1e9:.0f} GB"


def check_startup(lib_dir, dev_dir, full_layers, demote):
    """Called once, before the model loads. Raises (HYB_RAM_CHECK=error) or warns when the profile does not fit."""
    mode = os.environ.get("HYB_RAM_CHECK", "error")
    if mode == "off" or os.environ.get("_HYB_RAM_CHECKED") == "1":
        return
    os.environ["_HYB_RAM_CHECKED"] = "1"  # the EngineCore child inherits it: the check runs in the first process only
    sizes = layer_bytes(lib_dir, dev_dir)
    if not sizes:
        return
    lib_layers = set(sizes)
    start = sorted(lib_layers - set(full_layers))
    peak = sorted(lib_layers - (set(full_layers) - set(demote)))
    need_start, need_peak, avail = needed(start, sizes), needed(peak, sizes), available()
    msg = (
        f"[hybrid] host RAM: {_gb(avail)} free; this profile needs {_gb(need_start)} "
        f"({len(start)} hybrid layers x {sum(sizes[L] for L in start) / max(1, len(start)) / 1e9:.2f} GB of expert copies "
        f"+ {BASE_GIB:.0f} GiB base)"
    )
    if demote:
        msg += f", {_gb(need_peak)} at its full context ({len(peak)} hybrid layers)"
    sys.stderr.write(msg + "\n")
    if avail >= need_peak:
        return
    if avail >= need_start:  # elastic: starts, but cannot hold every demotion
        step = (need_peak - need_start) / max(1, len(peak) - len(start))
        fit = int((avail - need_start) // step) if step else 0
        short = int(os.environ.get("QWEN35_KV_TOKENS", "0") or 0)
        long_ = int(os.environ.get("QWEN35_MAX_TOKENS_ALL_USERS", "0") or 0)
        reach = f" (contexts up to ~{(short + fit * (long_ - short) // len(demote)) // 1024}K tokens)" if short and long_ else ""
        problem = (
            f"only {fit} of the {len(peak) - len(start)} elastic demotions fit{reach}: longer contexts would swap the "
            f"expert copies or get the server killed"
        )
    else:
        problem = "not enough free RAM for this profile" + (" even at short context" if demote else "")
    advice = (
        f"Free {_gb(need_peak - avail)} of host RAM (stop other jobs on this machine), or pick a profile with fewer "
        f"hybrid layers (`tt-model profiles <id>`: the 32K profile needs the least), or set HYB_RAM_CHECK=warn to start "
        f"anyway."
    )
    if mode == "warn":
        sys.stderr.write(f"[hybrid] WARNING: {problem}. {advice}\n")
        return
    raise RuntimeError(f"host RAM check failed: {_gb(avail)} free, {_gb(need_peak)} needed; {problem}. {advice}")


def warn_switch(lib_dir, dev_dir, layers):
    """Before an elastic demotion: log if the new layers' copies exceed the RAM free right now."""
    if os.environ.get("HYB_RAM_CHECK", "error") == "off":
        return
    sizes = layer_bytes(lib_dir, dev_dir)
    extra, avail = sum(sizes.get(L, 0) for L in layers), available()
    if extra > avail:
        sys.stderr.write(
            f"[hybrid] WARNING: demoting {len(layers)} layers needs {_gb(extra)} of host RAM, {_gb(avail)} is free: "
            f"expect swapping (slow CPU experts) or an out-of-memory kill\n"
        )

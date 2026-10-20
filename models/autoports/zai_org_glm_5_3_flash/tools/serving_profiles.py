# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Maintain serving_profiles.json (the single source of the serving profiles) and the files derived from it. Host only.

  show                         profiles, pool blocks per die, largest max_model_len, number source
  ingest CAPACITY_JSON [-w]    record a COMPLETED capacity probe (tools/memory_probe.py capacity --slots S) as
                               measured[S] of its DSA layout (dsa_shard -> layouts.sharded); a 1-slot probe also
                               updates that layout's budget free_after_load_bytes / dsa_bytes_per_block
  refresh [-w]                 every profile's max_model_len = the largest that fits (max_model_len_cap limits it),
                               status = measured when its slot count is measured
  render-yaml [-w]             write the profile numbers into tt-model.yaml (marked lines + the generated blocks)
  check                        data file consistent, tt-model.yaml in sync (exit 1 otherwise)
  get KEY [PROFILE]            one value for shell scripts: max_num_seqs, max_model_len, trace_region_bytes,
                               weights_revision, supported_context, dsa_layout, reserve_mib, env (K=V per line)

Without -w / --write the commands print what they would change. Rule and readers: tt/serving_profiles.py.
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
import sys
from pathlib import Path

from models.autoports.zai_org_glm_5_3_flash.tt import serving_profiles as sp

YAML = sp.MODEL_DIR / "tt-model.yaml"
BEGIN, END = "# BEGIN generated from serving_profiles.json", "# END generated from serving_profiles.json"
CARD_BEGIN, CARD_END = "<!-- BEGIN profiles: serving_profiles.json -->", "<!-- END profiles: serving_profiles.json -->"


def _rel(p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(sp.MODEL_DIR.parents[2]))
    except ValueError:
        return str(p)


# ------------------------------------------------------------------------------------------------ commands
def cmd_show(data, _a):
    print(
        f"{sp.path()}  default {data['default_profile']}  layouts {sp.layouts(data)}  precision {data['precision']['id']}"
    )
    print(f"budget: {data['budget']['status'][:110]}")
    print(f"{'profile':14} {'seqs':>4} {'max_model_len':>13} {'largest':>9} {'pool blk':>8} {'status':9} source")
    for p in sp.profiles(data):
        S, lay = int(p["max_num_seqs"]), sp.profile_layout(data, p)
        print(
            f"{p['name']:14} {S:4d} {int(p['max_model_len']):13,d} {sp.max_model_len_for(data, S, layout=lay):9,d} "
            f"{sp.pool_blocks(data, S, layout=lay):8d} {p.get('status', '?'):9} {lay[:4]} "
            f"{sp.pool_source(data, S, lay)[:56]}"
        )
    print(f"supported context (1 slot): {sp.supported_context(data):,d}")
    for e in sp.validate(data):
        print("PROBLEM:", e)


def cmd_ingest(data, a):
    c = json.loads(Path(a.capacity_json).read_text())
    if c.get("mode") != "capacity":
        sys.exit(f"{a.capacity_json}: not a capacity probe (mode {c.get('mode')})")
    missing = [k for k in ("num_blocks", "after_capture", "after_prefill", "copy_accuracy", "prompt_len") if k not in c]
    if missing:
        sys.exit(f"{a.capacity_json}: probe not complete (missing {missing}); only a completed probe is evidence")
    if float(c["copy_accuracy"]) < 1.0:
        print(f"WARNING: copy_accuracy {c['copy_accuracy']} < 1.0: check the probe before using it as evidence")
    if int(c["trace_region_bytes"]) != int(data["trace_region_bytes"]):
        sys.exit(
            f"probe trace region {c['trace_region_bytes']} != serving trace_region_bytes {data['trace_region_bytes']}"
        )
    S = int(c["S"])
    lay = "sharded" if c.get("dsa_shard") else sp.default_layout(data)
    if lay not in sp.layouts(data):
        sys.exit(f"{a.capacity_json}: DSA layout {lay!r} has no entry in serving_profiles.json layouts")
    ws = c.get("workspace_mib")
    reserve_mib = int(sp.layout_budget(data, lay)["reserve_bytes"]) / 2**20
    if ws is not None and abs(float(ws) - reserve_mib) > 0.5:
        print(f"WARNING: probe workspace {ws} MiB != {lay} layout reserve {reserve_mib:.0f} MiB")
    peaks = [int(v.get("peak_over_entry_bytes", 0)) for v in (c.get("prefill_peaks") or {}).values()]
    free_peak = int(c["after_capture"]["free"]) - (max(peaks) if peaks else 0)
    rec = {
        # bind_kv_cache(num_blocks) allocates num_blocks + 1 blocks (the dummy block): pool_blocks counts both
        "pool_blocks": int(c["num_blocks"]) + 1,
        "capped_by_max_seq_len": bool(c.get("capped_by_max_seq_len")),
        "max_model_len_tested": int(c["max_seq_len"]),
        "prompt_len": int(c["prompt_len"]),
        "copy_accuracy": float(c["copy_accuracy"]),
        "free_at_prefill_peak_bytes": free_peak,
        "trace_allocated_bytes": int(c["after_capture"].get("trace_allocated", 0)),
        "free_after_load_bytes": int(c["free_after_load_bytes"]),
        "workspace_mib": float(ws) if ws is not None else None,
        "evidence": _rel(Path(a.capacity_json)),
        "date": a.date or datetime.date.today().isoformat(),
    }
    print(f"{lay} measured[{S}] = {json.dumps(rec)}")
    if lay == sp.default_layout(data):
        data.setdefault("measured", {})[str(S)] = rec
        b = data["budget"]
    else:
        ent = data["layouts"][lay]
        ent.setdefault("measured", {})[str(S)] = rec
        b = ent.setdefault("budget", {})
        b.setdefault("evidence", {})
    if S == 1:
        b["free_after_load_bytes"] = int(c["free_after_load_bytes"])
        b["dsa_bytes_per_block"] = int(c["dsa_bytes_per_block_all_layers"])
        b["evidence"]["free_after_load_bytes"] = f"{rec['evidence']} after_load"
        b["evidence"]["dsa_bytes_per_block"] = f"{rec['evidence']} dsa_bytes_per_block_all_layers"
        print(
            f"budget: free_after_load {b['free_after_load_bytes']:,d}, dsa_bytes_per_block {b['dsa_bytes_per_block']:,d}"
        )
    if rec["capped_by_max_seq_len"]:
        print("NOTE: the pool was capped by --max-context: the measured value is a lower bound of the capacity")
    _save(data, a)


def cmd_refresh(data, a):
    for p in sp.profiles(data):
        S, lay = int(p["max_num_seqs"]), sp.profile_layout(data, p)
        L = sp.max_model_len_for(data, S, layout=lay)
        if p.get("max_model_len_cap"):
            L = min(L, int(p["max_model_len_cap"]))
        st = "measured" if sp._measured(data, S, lay) else "estimate"
        if (int(p["max_model_len"]), p.get("status")) != (L, st):
            print(f"{p['name']}: max_model_len {p['max_model_len']} -> {L}, status {p.get('status')} -> {st}")
        p["max_model_len"], p["status"] = L, st
    errs = sp.validate(data)
    if errs:
        sys.exit("\n".join(errs))
    _save(data, a)


def _profile_yaml(p: dict, default: dict, data: dict) -> list[str]:
    out = [f"  - name: {p['name']}"]
    status = "measured" if p.get("status") == "measured" else "ESTIMATE, pending the G9 capacity probe"
    out.append(f"    description: {p['description']} {p['max_num_seqs']} x {p['max_model_len']} tokens ({status}).")
    out.append(f"    hardware: {p['hardware']}")
    out.append(f"    mesh_device: {p['mesh_device']}")
    if p is not default:
        out.append(f"    max_num_seqs: {p['max_num_seqs']}")
        out.append(f"    max_model_len: {p['max_model_len']}")
    env = sp.layout_env(data, sp.profile_layout(data, p))
    if env:  # deep-merged over serve.env by tt-model (dicts merge)
        out.append("    env: {" + ", ".join(f'{k}: "{v}"' for k, v in env.items()) + "}")
    return out


def _card_lines(data: dict) -> list[str]:
    out = []
    for p in sp.profiles(data):
        S, L = int(p["max_num_seqs"]), int(p["max_model_len"])
        lay = sp.profile_layout(data, p)
        tag = " (default)" if p["name"] == data["default_profile"] else ""
        est = "" if p.get("status") == "measured" else " [estimate, pending measurement]"
        lay_txt = "" if lay == sp.default_layout(data) else f" ({lay} DSA cache)"
        out.append(
            f"- `{p['name']}`{tag}: {S} sequence{'s' if S > 1 else ''}, {L:,d}-token context, DSA pool "
            f"{(sp.pool_blocks(data, S, layout=lay) - 1) * sp.BLOCK:,d} tokens shared by all sequences{lay_txt}{est}."
        )
    return out


def render_yaml(data: dict, text: str) -> str:
    default = sp.profile(data)
    tr = int(data["trace_region_bytes"])
    marks = {
        "default.max_num_seqs": str(default["max_num_seqs"]),
        "default.max_model_len": str(default["max_model_len"]),
        "trace_region_bytes": str(tr),
        "trace_region_env": f'"{tr}"',
    }

    def sub(m):
        key = m.group(3)
        if key not in marks:
            raise KeyError(f"unknown tt-model.yaml mark @profiles:{key}")
        return f"{m.group(1)}{marks[key]}{m.group(2)}"

    # a marked line: "<prefix><value><spaces># @profiles:<key>"; the value is the token before the comment
    text = re.sub(r"^([^#\n]*?:\s*)[^\s#][^#\n]*?(\s+# @profiles:([\w.]+).*)$", sub, text, flags=re.M)
    rev = data["weights_revision"]
    text = re.sub(r'("--revision",\s*")[0-9a-f]+(")', lambda m: m.group(1) + rev + m.group(2), text)
    text = re.sub(r"(GLM53_WEIGHTS_REVISION:\s*)\S+", lambda m: m.group(1) + rev, text)
    text = re.sub(
        r"(^weights:\s*\{repo:\s*[^,]+,\s*revision:\s*)[0-9a-f]+", lambda m: m.group(1) + rev, text, flags=re.M
    )
    lines = ["serve_profiles:"]
    for p in sp.profiles(data):
        lines += _profile_yaml(p, default, data)
    lines.append(f"default_profile: {data['default_profile']}")
    block = "\n".join([BEGIN] + lines + [END])
    if BEGIN not in text or END not in text:
        raise ValueError(f"{YAML}: markers {BEGIN!r} / {END!r} missing")
    text = re.sub(re.escape(BEGIN) + r".*?" + re.escape(END), lambda m: block, text, flags=re.S)
    if CARD_BEGIN in text:
        ind = re.search(r"^(\s*)" + re.escape(CARD_BEGIN), text, flags=re.M).group(1)
        card = "\n".join([ind + CARD_BEGIN] + [ind + ln for ln in _card_lines(data)] + [ind + CARD_END])
        text = re.sub(
            r"^\s*" + re.escape(CARD_BEGIN) + r".*?" + re.escape(CARD_END), lambda m: card, text, flags=re.S | re.M
        )
    return text


def cmd_render(data, a):
    old = YAML.read_text()
    new = render_yaml(data, old)
    if new == old:
        print("tt-model.yaml is in sync")
        return
    import difflib

    sys.stdout.writelines(difflib.unified_diff(old.splitlines(True), new.splitlines(True), "tt-model.yaml", "rendered"))
    if a.write:
        YAML.write_text(new)
        print(f"wrote {YAML}")


def cmd_check(data, _a):
    errs = sp.validate(data)
    if YAML.exists() and render_yaml(data, YAML.read_text()) != YAML.read_text():
        errs.append("tt-model.yaml is out of sync with serving_profiles.json (run: render-yaml -w)")
    for e in errs:
        print("PROBLEM:", e)
    print("check:", "FAIL" if errs else "OK")
    sys.exit(1 if errs else 0)


def cmd_get(data, a):
    if a.key == "trace_region_bytes":
        print(data["trace_region_bytes"])
    elif a.key == "weights_revision":
        print(data["weights_revision"])
    elif a.key == "supported_context":
        print(sp.supported_context(data))
    elif a.key in ("dsa_layout", "reserve_mib", "env"):
        lay = sp.profile_layout(data, sp.profile(data, a.profile))
        if a.key == "dsa_layout":
            print(lay)
        elif a.key == "reserve_mib":
            print(int(sp.layout_budget(data, lay)["reserve_bytes"]) // 2**20)
        else:
            for k, v in sp.layout_env(data, lay).items():
                print(f"{k}={v}")
    else:
        print(sp.profile(data, a.profile)[a.key])


def _save(data, a):
    if not a.write:
        print("(dry run: add -w to write)")
        return
    sp.path().write_text(json.dumps(data, indent=1) + "\n")
    print(f"wrote {sp.path()}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("show")
    p = sub.add_parser("ingest")
    p.add_argument("capacity_json")
    p.add_argument("--date")
    p.add_argument("-w", "--write", action="store_true")
    p = sub.add_parser("refresh")
    p.add_argument("-w", "--write", action="store_true")
    p = sub.add_parser("render-yaml")
    p.add_argument("-w", "--write", action="store_true")
    sub.add_parser("check")
    p = sub.add_parser("get")
    p.add_argument(
        "key",
        choices=[
            "max_num_seqs",
            "max_model_len",
            "trace_region_bytes",
            "weights_revision",
            "supported_context",
            "mesh_device",
            "dsa_layout",
            "reserve_mib",
            "env",
        ],
    )
    p.add_argument("profile", nargs="?")
    a = ap.parse_args(argv)
    data = sp.load()
    {
        "show": cmd_show,
        "ingest": cmd_ingest,
        "refresh": cmd_refresh,
        "render-yaml": cmd_render,
        "check": cmd_check,
        "get": cmd_get,
    }[a.cmd](data, a)


if __name__ == "__main__":
    main()

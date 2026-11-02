#!/usr/bin/env -S /usr/bin/python3
"""Worker CLI for the Claude user-simulator relay. Instructions for Claude workers: ~/evals/usersim/WORKER.md

  worker.py claim   --worker NAME [--n 1] [--wait 120] [--conversation CID] [--full]
  worker.py answer  KEY --worker NAME  (--content TEXT | --tool-call NAME 'JSON' [...] | text on stdin)
  worker.py batch   --worker NAME   < JSONL  ({"key":..,"content":..} or {"key":..,"tool_calls":[{"name":..,"arguments":{..}}]})
  worker.py release KEY [...] --worker NAME      (give claimed jobs back)
  worker.py show    KEY [--full]                 (render a job again, any state)
  worker.py status                               (queue counts, answer rate, per-worker activity)
  worker.py reset [--force]                      (orchestrator, between runs: archive the answer cache + logs)

KEY may be any unique prefix (>= 6 hex chars) of the 64-hex job key.
Affinity: claim prefers (1) jobs of conversations this worker already handled, (2) conversations nobody owns or whose
owner has been idle > USERSIM_AFFINITY_TTL (180 s), (3) any job pending longer than USERSIM_STEAL_AFTER (60 s).
Rendering is incremental per worker: the role instructions and earlier turns of a conversation are printed only the
first time this worker sees it (or with --full).
"""
import argparse
import json
import os
import re
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import usersim_common as C  # noqa: E402

AFF_TTL = float(os.environ.get("USERSIM_AFFINITY_TTL", 180))
STEAL_AFTER = float(os.environ.get("USERSIM_STEAL_AFTER", 60))


def die(msg, code=2):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(code)


def resolve(prefix):
    prefix = prefix.strip().lower()
    if not re.fullmatch(r"[0-9a-f]{6,64}", prefix):
        die(f"bad key {prefix!r}")
    hits = set()
    for st in ("claimed", "pending", "abandoned", "answered", "done"):
        hits |= {p.stem for p in C.d(st).glob(f"{prefix}*.json")}
    if not hits:
        die(f"no job with key {prefix}")
    if len(hits) > 1:
        die(f"ambiguous key prefix {prefix}: {sorted(h[:16] for h in hits)}")
    return hits.pop()


def load_json(p, default):
    try:
        return C.read_json(p)
    except (OSError, json.JSONDecodeError):
        return default


def wstate_path(worker):
    return C.d("state/workers") / f"{re.sub(r'[^A-Za-z0-9_.-]', '_', worker)}.json"


# ---------------------------------------------------------------- rendering
def short_scenario(sys_txt):
    """One-line reminder of which customer this conversation is (reason for call + known info, or the persona)."""
    m = re.search(r"Reason for call:(.*?)(?:Task instructions:|</scenario>)", sys_txt, re.S) or \
        re.search(r"<scenario>(.*?)</scenario>", sys_txt, re.S)
    if m:
        s = m.group(1)
    else:   # VITA: the persona dict + the start of <instructions> (the prose mentions the tags first, so take the last match)
        per = re.findall(r"<persona>\s*(.*?)\s*</persona>", sys_txt, re.S)
        ins = re.findall(r"<instructions>\s*(.*?)\s*</instructions>", sys_txt, re.S)
        s = (per[-1] if per else sys_txt) + (" | " + ins[-1][:80] if ins else "")
    s = " ".join(s.split())
    return s[:260] + ("..." if len(s) > 260 else "")


def describe(v, defs):
    """Compact type of a JSON-schema property, resolving $ref / anyOf / enum (pydantic-style tau2 schemas)."""
    if "$ref" in v:
        v = {**defs.get(v["$ref"].split("/")[-1], {}), **{k: x for k, x in v.items() if k != "$ref"}}
    if "enum" in v:
        return f"one of {v['enum']}"
    if "anyOf" in v:
        alts = [describe(x, defs) for x in v["anyOf"]]
        return " | ".join(a for a in alts if a != "null") or "null"
    if v.get("type") == "object" and v.get("properties"):
        return "object{" + ", ".join(f"{k}: {describe(x, defs)}" for k, x in v["properties"].items()) + "}"
    if v.get("type") == "array" and v.get("items"):
        return f"array of {describe(v['items'], defs)}"
    return v.get("type") or "any"


def fmt_tools(tools):
    out = []
    for t in tools or []:
        f = t.get("function") or {}
        params = f.get("parameters") or {}
        req = set(params.get("required") or [])
        defs = params.get("$defs") or params.get("definitions") or {}
        ps = []
        for k, v in (params.get("properties") or {}).items():
            ps.append(f"{k}: {describe(v, defs)}{'' if k in req else ' (optional)'}")
        desc = " ".join((f.get("description") or "").split())
        out.append(f"  - {f.get('name')}({', '.join(ps)}): {desc[:200]}")
    return "\n".join(out)


def fmt_msg(i, m):
    role = m.get("role")
    txt = C.text_of(m.get("content"))
    if role == "user":
        return f"[{i}] AGENT: {txt}"
    if role == "assistant":
        lines = []
        if txt:
            lines.append(f"[{i}] YOU: {txt}")
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function") or {}
            lines.append(f"[{i}] YOU called tool {fn.get('name')}({fn.get('arguments')})")
        return "\n".join(lines) or f"[{i}] YOU: (empty)"
    if role == "tool":
        return f"[{i}] TOOL RESULT (your device): {txt}"
    return f"[{i}] {role}: {txt}"


def render(rec, worker, full, mark_seen=True):
    msgs = rec["messages"]
    sys_txt = C.text_of(msgs[0].get("content")) if msgs and msgs[0].get("role") == "system" else ""
    body = msgs[1:] if sys_txt else msgs
    cid = rec["conv"]
    ws = load_json(wstate_path(worker), {"convs": {}}) if worker else {"convs": {}}
    seen = ws["convs"].get(cid) if not full else None
    sys_h = C.sha(sys_txt)
    start = 0
    if seen and seen.get("sys") == sys_h and seen["n"] <= len(body) and \
            C.sha(C.normalize(body[:seen["n"]])) == seen["h"]:
        start = seen["n"]
    else:
        seen = None
    tools = rec.get("tools")
    out = [f"=== JOB {rec['key'][:16]}  conversation {cid}  {rec['bench']}  turn {len(body)} "
           f"({'continuing: only new messages shown' if seen else 'full'}) ==="]
    if seen:
        out.append(f"(persona reminder: {short_scenario(sys_txt)})")
    else:
        out += ["--- YOUR ROLE INSTRUCTIONS (official user-simulator system prompt, verbatim) ---", sys_txt,
                "--- END OF ROLE INSTRUCTIONS ---"]
        if tools:
            out += ["--- TOOLS ON YOUR SIDE (your own phone/device; one tool call per turn) ---", fmt_tools(tools)]
    out.append("--- CONVERSATION (AGENT = the customer-service agent; YOU = the user you play) ---"
               if not seen else "--- NEW MESSAGES ---")
    for i, m in enumerate(body[start:], start=start + 1):
        out.append(fmt_msg(i, m))
    lang = "Chinese" if rec["bench"].endswith("/zh") else None
    fmt = ("ONE user message (plain text"
           + (", one line" if rec["bench"].startswith("vita") else "")
           + (f", in {lang}" if lang else "") + ")")
    if tools:
        fmt += " OR one tool call on your device (not both)"
    if rec["bench"].startswith("vita"):
        end = ("No stop token: the official VITA user prompt has none; the AGENT ends the dialogue (answer its "
               "'anything else?' naturally, e.g. that you need nothing more)")
    else:
        end = "To end the conversation put ###STOP### / ###TRANSFER### / ###OUT-OF-SCOPE### in the message as your role instructions say"
    out.append(f"--- REPLY: {fmt}. {end}.\n    answer: worker.py answer {rec['key'][:16]} --worker "
               f"{worker or 'NAME'} <<'EOF' ... EOF" + ("   |   --tool-call NAME '{json}'" if tools else ""))
    if worker and mark_seen:
        ws["convs"][cid] = {"sys": sys_h, "n": len(body), "h": C.sha(C.normalize(body)), "t": time.time()}
        C.atomic_write(wstate_path(worker), ws)
    return "\n".join(out)


# ---------------------------------------------------------------- queue ops
def pick(worker, n, conversation):
    now = time.time()
    aff = load_json(C.d("state") / "affinity.json", {})
    cands = []
    for p in C.d("pending").glob("*.json"):
        rec = load_json(p, None)
        if not rec:
            continue
        cid = rec["conv"]
        if conversation and not cid.startswith(conversation):
            continue
        owner = aff.get(cid)
        age = now - rec.get("created", now)
        if conversation or (owner and owner["worker"] == worker):
            tier = 0
        elif not owner or now - owner["t"] > AFF_TTL or rec.get("lease_expired_by") == owner["worker"]:
            tier = 1
        elif age > STEAL_AFTER:
            tier = 2
        else:
            continue
        cands.append((tier, rec.get("created", now), p, rec))
    cands.sort(key=lambda c: (c[0], c[1]))
    got = []
    for _, _, p, rec in cands:
        if len(got) >= n:
            break
        if any(g["conv"] == rec["conv"] for g in got) and not conversation:
            continue   # one job per conversation per claim (VITA trials share a conversation id)
        dst = C.d("claimed") / p.name
        try:
            os.replace(p, dst)
        except FileNotFoundError:
            continue   # another worker took it
        C.atomic_write(C.d("claimed") / f"{rec['key']}.lease",
                       {"worker": worker, "t": now, "expires": now + C.LEASE_S})
        aff[rec["conv"]] = {"worker": worker, "t": now}
        got.append(rec)
    if got:
        C.atomic_write(C.d("state") / "affinity.json", aff)
    return got


def cmd_claim(a):
    C.ensure_dirs()
    deadline = time.time() + a.wait
    while True:
        with C.state_lock():
            C.requeue_expired_leases()
            got = pick(a.worker, a.n, a.conversation)
            if got:
                print("\n\n".join(render(r, a.worker, a.full) for r in got))
                print(f"\n[{len(got)} job(s) claimed by {a.worker}; lease {C.LEASE_S:.0f}s. Answer each with worker.py "
                      f"answer KEY --worker {a.worker}]")
                return
        if time.time() >= deadline:
            print(f"NO_JOBS (waited {a.wait:.0f}s). {status_line()}")
            return
        time.sleep(0.5)


def write_answer(key, worker, content, tool_calls, allow_stop=False):
    if (C.d("done") / f"{key}.json").exists():
        return f"already answered: {key[:16]} (nothing written)", False
    st, p, rec = C.find_request(key)
    if not rec:
        return f"no request record for {key[:16]}", False
    errs = C.validate_answer(rec, content, tool_calls)
    if content and rec["bench"].startswith("vita"):
        if any(t in content for t in C.STOP_TOKENS) and not allow_stop:
            errs.append("VITA's official user prompt defines no stop token (the agent ends the dialogue); reply in "
                        "words, or pass --allow-stop only if the orchestrator told you to")
        if "\n" in content:
            errs.append("VITA wants exactly one line (no line breaks)")
    if errs:
        return "REJECTED " + key[:16] + ": " + "; ".join(errs), False
    tcs = [{"id": "call_" + uuid.uuid4().hex[:24], "name": nm, "arguments": json.dumps(ar, ensure_ascii=False)}
           for nm, ar in tool_calls] or None
    done = {"key": key, "content": None if tcs else content, "tool_calls": tcs, "worker": worker, "t": time.time()}
    if not C.create_exclusive(C.d("done") / f"{key}.json", done):
        return f"already answered: {key[:16]} (another worker was first; nothing written)", False
    try:
        os.replace(p, C.d("answered") / f"{key}.json")
    except FileNotFoundError:
        pass
    (C.d("claimed") / f"{key}.lease").unlink(missing_ok=True)
    C.log_line("answers.jsonl", {"key": key, "conv": rec["conv"], "bench": rec["bench"], "n_msgs": rec["n_msgs"],
                                 "worker": worker, "t": done["t"], "created": rec.get("created"),
                                 "tool_call": bool(tcs), "len": len(content or "")})
    note = ""
    if st == "abandoned":
        note = " (note: the harness had stopped waiting for this job; it is cached in case it retries)"
    if content and any(t in content for t in C.STOP_TOKENS):
        note += " [conversation ends]"
    return f"OK {key[:16]} " + (f"tool_call {tcs[0]['name']}" if tcs else "message") + note, True


def parse_tool_calls(pairs):
    out = []
    for name, js in pairs or []:
        try:
            ar = json.loads(js) if js.strip() else {}
        except json.JSONDecodeError as e:
            die(f"tool-call arguments for {name} are not valid JSON: {e}")
        out.append((name, ar))
    return out


def cmd_answer(a):
    key = resolve(a.key)
    tcs = parse_tool_calls(a.tool_call)
    content = a.content
    if content is None and not tcs:
        if sys.stdin.isatty():
            die("give --content, --tool-call, or the message on stdin")
        content = sys.stdin.read()
    if content is not None:
        content = content.strip()
    msg, ok = write_answer(key, a.worker, content, tcs, a.allow_stop)
    print(msg)
    sys.exit(0 if ok else 1)


def cmd_batch(a):
    bad = 0
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            r = json.loads(line)
            key = resolve(r["key"])
        except (json.JSONDecodeError, KeyError) as e:
            print(f"REJECTED line: {e}: {line[:120]}")
            bad += 1
            continue
        tcs = [(t["name"], t.get("arguments") if isinstance(t.get("arguments"), dict) else json.loads(t.get("arguments") or "{}"))
               for t in r.get("tool_calls") or []]
        c = r.get("content")
        msg, ok = write_answer(key, a.worker, c.strip() if isinstance(c, str) else c, tcs, a.allow_stop)
        bad += not ok
        print(msg)
    sys.exit(1 if bad else 0)


def cmd_release(a):
    for k in a.keys:
        key = resolve(k)
        try:
            os.replace(C.d("claimed") / f"{key}.json", C.d("pending") / f"{key}.json")
            print(f"released {key[:16]}")
        except FileNotFoundError:
            print(f"{key[:16]} is not claimed")
        (C.d("claimed") / f"{key}.lease").unlink(missing_ok=True)


def cmd_show(a):
    key = resolve(a.key)
    st, p, rec = C.find_request(key)
    if not rec:
        die("request record gone")
    print(f"(state: {st})")
    print(render(rec, a.worker, a.full or not a.worker, mark_seen=False))
    dp = C.d("done") / f"{key}.json"
    if dp.exists():
        print("--- ANSWER ---\n" + json.dumps(C.read_json(dp), ensure_ascii=False))


def status_line():
    c = {st: sum(1 for _ in C.d(st).glob("*.json")) for st in ("pending", "claimed", "done", "abandoned")}
    return " ".join(f"{k}={v}" for k, v in c.items())


def cmd_status(a):
    C.ensure_dirs()
    now = time.time()
    print(status_line())
    pend = [load_json(p, {}) for p in C.d("pending").glob("*.json")]
    if pend:
        oldest = min(r.get("created", now) for r in pend)
        print(f"oldest pending: {now - oldest:.0f}s; pending by bench: "
              + ", ".join(f"{b}={sum(1 for r in pend if r.get('bench') == b)}" for b in sorted({r.get('bench') for r in pend})))
    for lp in sorted(C.d("claimed").glob("*.lease")):
        le = load_json(lp, {})
        print(f"  claimed {lp.stem[:16]} by {le.get('worker')} {now - le.get('t', now):.0f}s ago")
    ans = []
    ap_ = C.d("log") / "answers.jsonl"
    if ap_.exists():
        ans = [json.loads(x) for x in ap_.read_text().splitlines() if x.strip()]
    for win in (300, 3600):
        w = [x for x in ans if now - x["t"] < win]
        print(f"answers last {win // 60:>2d} min: {len(w)} ({len(w) / (win / 60):.2f}/min)"
              + (f", mean queue->answer {sum(x['t'] - x['created'] for x in w if x.get('created')) / len(w):.1f}s" if w else ""))
    byw = {}
    for x in ans:
        byw.setdefault(x["worker"], []).append(x["t"])
    for w, ts in sorted(byw.items()):
        print(f"  worker {w}: {len(ts)} answers, last {now - max(ts):.0f}s ago")
    print(f"total answers: {len(ans)}; conversations: {len({x['conv'] for x in ans})}")


def cmd_reset(a):
    """Archive done/answered/abandoned/log/state so a new run starts with an empty answer cache."""
    busy = [st for st in ("pending", "claimed") if any(C.d(st).glob("*.json"))]
    if busy and not a.force:
        die(f"jobs still in {busy}; finish or stop the harness first (or --force)")
    dst = C.SPOOL / "archive" / time.strftime("%Y%m%d-%H%M%S")
    dst.mkdir(parents=True)
    for st in ("done", "answered", "abandoned", "log", "state") + (("pending", "claimed") if a.force else ()):
        if C.d(st).exists():
            os.replace(C.d(st), dst / st)
    C.ensure_dirs()
    print(f"archived to {dst}; spool is empty")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)
    w_default = os.environ.get("USERSIM_WORKER")
    c = sp.add_parser("claim")
    c.add_argument("--worker", default=w_default, required=w_default is None)
    c.add_argument("--n", type=int, default=1)
    c.add_argument("--wait", type=float, default=0, help="long-poll up to this many seconds for a job")
    c.add_argument("--conversation", default=None)
    c.add_argument("--full", action="store_true")
    c.set_defaults(f=cmd_claim)
    an = sp.add_parser("answer")
    an.add_argument("key")
    an.add_argument("--worker", default=w_default, required=w_default is None)
    an.add_argument("--content", default=None)
    an.add_argument("--tool-call", nargs=2, action="append", metavar=("NAME", "JSON"))
    an.add_argument("--allow-stop", action="store_true", help="permit a stop token in a VITA reply")
    an.set_defaults(f=cmd_answer)
    b = sp.add_parser("batch")
    b.add_argument("--worker", default=w_default, required=w_default is None)
    b.add_argument("--allow-stop", action="store_true")
    b.set_defaults(f=cmd_batch)
    r = sp.add_parser("release")
    r.add_argument("keys", nargs="+")
    r.add_argument("--worker", default=w_default)
    r.set_defaults(f=cmd_release)
    s = sp.add_parser("show")
    s.add_argument("key")
    s.add_argument("--worker", default=None)
    s.add_argument("--full", action="store_true")
    s.set_defaults(f=cmd_show)
    rs = sp.add_parser("reset")
    rs.add_argument("--force", action="store_true")
    rs.set_defaults(f=cmd_reset)
    st = sp.add_parser("status")
    st.set_defaults(f=cmd_status)
    a = p.parse_args()
    C.ensure_dirs()
    a.f(a)


if __name__ == "__main__":
    main()

"""Shared spool layout, request keys and validation for the Claude user-simulator relay (shim.py / worker.py).

Spool (USERSIM_SPOOL, default ~/evals/usersim/spool):
  pending/<key>.json     request waiting for a worker   {key, conv, n_msgs, bench, created, model, seed, messages, tools}
  claimed/<key>.json     same record, leased to a worker; claimed/<key>.lease = {worker, t, expires}
  done/<key>.json        the answer {key, content, tool_calls:[{id,name,arguments}], worker, t}   (kept: retry cache)
  answered/<key>.json    request record after it was answered (audit)
  abandoned/<key>.json   request nobody waits for any more (harness gave up / shim restarted); revived on retry
  state/                 affinity.json (conv -> worker), workers/<name>.json (what each worker has already seen)
  log/served.jsonl       one line per HTTP reply (timings); log/answers.jsonl one line per worker answer
"""
import fcntl
import hashlib
import json
import os
import re
import time
from contextlib import contextmanager
from pathlib import Path

SPOOL = Path(os.environ.get("USERSIM_SPOOL", str(Path.home() / "evals/usersim/spool")))
DIRS = ["pending", "claimed", "done", "answered", "abandoned", "state", "state/workers", "log"]
STOP_TOKENS = ("###STOP###", "###TRANSFER###", "###OUT-OF-SCOPE###")
LEASE_S = float(os.environ.get("USERSIM_LEASE", 900))


def d(name):
    return SPOOL / name


def ensure_dirs():
    for n in DIRS:
        (SPOOL / n).mkdir(parents=True, exist_ok=True)


def text_of(content):
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return str(content)


def _norm_args(a):
    if isinstance(a, str):
        try:
            return json.loads(a)
        except Exception:  # noqa: BLE001
            return a
    return a


def normalize(messages):
    """Messages without volatile fields (tool-call ids), for hashing."""
    out = []
    for m in messages:
        r = {"role": m.get("role"), "content": text_of(m.get("content"))}
        if m.get("tool_calls"):
            r["tool_calls"] = [[(tc.get("function") or {}).get("name"), _norm_args((tc.get("function") or {}).get("arguments"))]
                               for tc in m["tool_calls"]]
        out.append(r)
    return out


def sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def request_key(body):
    """Hash of everything the simulator sees: identical input -> identical (cached) answer, so retries are idempotent."""
    msgs = body.get("messages") or []
    tools = sorted(((t.get("function") or {}).get("name") or "") for t in (body.get("tools") or []))
    return sha({"m": normalize(msgs), "seed": body.get("seed"), "tools": tools})


def conv_id(body):
    """Stable per conversation: system prompt + seed (tau2: unique per task x trial; VITA: per task, trials share it)."""
    msgs = body.get("messages") or []
    sys_txt = text_of(msgs[0].get("content")) if msgs and msgs[0].get("role") == "system" else ""
    return sha({"s": sys_txt, "seed": body.get("seed")})[:12]


def detect_bench(body):
    msgs = body.get("messages") or []
    s = text_of(msgs[0].get("content")) if msgs else ""
    if "<scenario>" in s:
        return "tau2/" + ("telecom(tools)" if body.get("tools") else "airline-or-retail")
    if "<persona>" in s:
        return "vita/" + ("zh" if re.search(r"[一-鿿]", s[:200]) else "en")
    return "unknown"


def atomic_write(path: Path, obj):
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.monotonic_ns()}.tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False))
    os.replace(tmp, path)


def create_exclusive(path: Path, obj):
    """Atomically create path with obj; False if it already exists (first writer wins)."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.monotonic_ns()}.tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False))
    try:
        os.link(tmp, path)
        return True
    except FileExistsError:
        return False
    finally:
        tmp.unlink(missing_ok=True)


def read_json(path: Path):
    return json.loads(path.read_text())


@contextmanager
def state_lock():
    ensure_dirs()
    with open(SPOOL / "state" / "lock", "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def find_request(key):
    """-> (state_dir_name, path, record) for a key (pending/claimed/abandoned/answered); robust to concurrent moves."""
    for _ in range(5):
        raced = False
        for st in ("claimed", "pending", "abandoned", "answered"):
            p = d(st) / f"{key}.json"
            if not p.exists():
                continue
            try:
                return st, p, read_json(p)
            except (OSError, json.JSONDecodeError):
                raced = True   # moved between exists() and read: look again
        if not raced:
            break
    return None, None, None


def requeue_expired_leases():
    """Claimed jobs whose lease ran out go back to pending, marked so that the lapsed worker loses its affinity."""
    now = time.time()
    for lp in d("claimed").glob("*.lease"):
        try:
            lease = read_json(lp)
        except (OSError, json.JSONDecodeError):
            continue
        if lease.get("expires", now + 1) >= now:
            continue
        cp = d("claimed") / f"{lp.stem}.json"
        try:
            rec = read_json(cp)
            rec["lease_expired_by"] = lease.get("worker")
            atomic_write(d("pending") / cp.name, rec)
            cp.unlink(missing_ok=True)
        except (OSError, json.JSONDecodeError):
            pass
        lp.unlink(missing_ok=True)


def tool_schemas(req):
    return {(t.get("function") or {}).get("name"): (t.get("function") or {}) for t in (req.get("tools") or [])}


def validate_answer(req, content, tool_calls):
    """-> list of error strings (empty = valid). tool_calls: [(name, args_dict)]."""
    errs = []
    if content is not None and tool_calls:
        errs.append("give EITHER a message OR tool call(s), not both")
    if not tool_calls:
        if content is None or not content.strip():
            errs.append("content must be a non-empty string")
        return errs
    schemas = tool_schemas(req)
    if not schemas:
        errs.append("this request offers no tools: reply with a text message")
        return errs
    for name, args in tool_calls:
        if name not in schemas:
            errs.append(f"unknown tool {name!r}; available: {', '.join(sorted(schemas))}")
            continue
        if not isinstance(args, dict):
            errs.append(f"{name}: arguments must be a JSON object")
            continue
        params = schemas[name].get("parameters") or {}
        props = params.get("properties") or {}
        for r in params.get("required") or []:
            if r not in args:
                errs.append(f"{name}: missing required argument {r!r}")
        for k, v in args.items():
            if props and k not in props:
                errs.append(f"{name}: unknown argument {k!r}; allowed: {', '.join(props)}")
            elif k in props and "enum" in props[k] and v not in props[k]["enum"]:
                errs.append(f"{name}: {k}={v!r} not in {props[k]['enum']}")
    return errs


def log_line(name, obj):
    ensure_dirs()
    with open(d("log") / name, "a") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

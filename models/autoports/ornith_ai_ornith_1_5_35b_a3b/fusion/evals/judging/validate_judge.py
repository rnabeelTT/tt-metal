"""validate_judge.py <chunk.jsonl> <verdicts.jsonl>: every chunk id needs exactly one verdict line that matches its
verdict_schema (required keys present, enums / types / ranges respected). Prints OK n or the problems; exit 1 on any."""
import json, sys
chunk = {json.loads(l)["id"]: json.loads(l)["verdict_schema"] for l in open(sys.argv[1])}
seen, bad = {}, []
try:
    lines = [l for l in open(sys.argv[2]) if l.strip()]
except FileNotFoundError:
    lines = []
for n, l in enumerate(lines, 1):
    try:
        v = json.loads(l)
    except Exception as e:
        bad.append(f"line {n}: not JSON ({e})"); continue
    i = v.get("id")
    if i not in chunk:
        bad.append(f"line {n}: unknown id {i!r}"); continue
    if i in seen:
        bad.append(f"line {n}: duplicate id {i}"); continue
    seen[i] = 1
    s = chunk[i]
    for k in s.get("required", []):
        if k not in v:
            bad.append(f"{i}: missing {k}")
    for k, p in s.get("properties", {}).items():
        if k not in v:
            continue
        x = v[k]
        if "enum" in p and x not in p["enum"]:
            bad.append(f"{i}: {k}={x!r} not in {p['enum']}")
        if p.get("type") == "integer" and not (isinstance(x, int) and not isinstance(x, bool)):
            bad.append(f"{i}: {k} must be an integer")
        if p.get("type") == "number" and not isinstance(x, (int, float)):
            bad.append(f"{i}: {k} must be a number")
        if p.get("type") == "string" and not isinstance(x, str):
            bad.append(f"{i}: {k} must be a string")
        if "minimum" in p and isinstance(x, (int, float)) and x < p["minimum"]:
            bad.append(f"{i}: {k} below {p['minimum']}")
        if "maximum" in p and isinstance(x, (int, float)) and x > p["maximum"]:
            bad.append(f"{i}: {k} above {p['maximum']}")
missing = [i for i in chunk if i not in seen]
if missing:
    bad.append(f"{len(missing)} ids without a verdict, e.g. {missing[:3]}")
print("OK", len(seen)) if not bad else print("PROBLEMS:\n" + "\n".join(bad[:40]))
sys.exit(1 if bad else 0)

#!/usr/bin/env bash
# TOOL CALLING, checked by making a tool call -- not by checking that the flag loaded.
#
# hermes shipped here for weeks and never produced one: it expects JSON inside <tool_call>,
# the model emits Qwen XML, so it raised JSONDecodeError, returned tools_called=False and
# leaked the markup into `content` as prose. The server starts cleanly either way, which is
# why a startup check cannot catch this and this script exists.
#
# Usage: verify_tools.sh <base-url> [model-id]
set -u
BASE="${1:?usage: verify_tools.sh <base-url> [model-id]}"
MID="${2:-}"
[ -z "$MID" ] && MID=$(curl -s -m 10 "$BASE/v1/models" | python3 -c 'import json,sys;print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null)
[ -z "$MID" ] && { echo "!!! no model at $BASE"; exit 1; }
echo "model: $MID"

req() {  # $1 = user content, $2 = tool_choice
python3 - "$MID" "$1" "$2" <<'PY'
import json,sys
print(json.dumps({
 "model": sys.argv[1],
 "messages":[{"role":"user","content":sys.argv[2]}],
 "tools":[{"type":"function","function":{"name":"get_weather","description":"Get current weather for a city",
   "parameters":{"type":"object","properties":{"city":{"type":"string"},"unit":{"type":"string","enum":["c","f"]}},"required":["city"]}}},
          {"type":"function","function":{"name":"add","description":"Add two numbers",
   "parameters":{"type":"object","properties":{"a":{"type":"number"},"b":{"type":"number"}},"required":["a","b"]}}}],
 "tool_choice": sys.argv[3], "max_tokens":1200, "temperature":0.0}))
PY
}

check() {  # $1 label, $2 prompt, $3 tool_choice, $4 expect_call(yes/no)
  R=$(curl -s -m 900 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d "$(req "$2" "$3")")
  echo "$R" | python3 -c '
import json,sys
exp=sys.argv[1]; label=sys.argv[2]
d=json.load(sys.stdin); ch=(d.get("choices") or [{}])[0]; m=ch.get("message") or {}
tc=m.get("tool_calls") or []; content=(m.get("content") or "")
leak = "<tool_call>" in content or "<function=" in content
ok = (len(tc)>0) if exp=="yes" else (len(tc)==0)
ok = ok and not leak
# Plain %-formatting, NOT an f-string: this source is carried inside a single-quoted
# shell argument, and the version that lived here escaped its inner quotes as \" --
# which is a SyntaxError, so every check exited 1 and the gate reported
# "TOOL CALLING FAILED" whatever the server did. It had never been run.
print("  [%s] %s: finish=%s calls=%d leak=%s"
      % ("PASS" if ok else "FAIL", label, ch.get("finish_reason"), len(tc), leak))
# tool_calls are parsed JSON dicts, not objects: c.function.name was an AttributeError.
for c in tc:
    fn = c.get("function") or {}
    print("         -> %s(%s)" % (fn.get("name"), fn.get("arguments")))
if not ok: print("         content=%r" % (content[:160],))
sys.exit(0 if ok else 1)
' "$4" "$1" || FAILED=1
}

FAILED=0
echo "### tool calling"
check "auto, weather"  "What is the weather in Paris right now? Use the tool."      auto     yes
check "auto, math"     "Use the add tool to add 17 and 25."                          auto     yes
check "no tool needed" "Say the single word: hello"                                  auto     no
echo
if [ "$FAILED" -ne 0 ]; then echo "@@@@ TOOL CALLING FAILED @@@@"; exit 1; fi
echo "@@@@ TOOL CALLING OK @@@@"

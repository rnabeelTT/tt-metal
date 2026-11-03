"""Bring-up smoke test against a served bundle (OpenAI API on URL, default http://localhost:20000).

    python smoke.py [--no-image]

Checks: a thinking answer (reasoning separated, right content), a thinking-off answer, a parsed tool call, and (unless
--no-image) an image question on a generated picture with known content. Prints one line per check and exits non-zero
if any fails.
"""

import base64
import io
import json
import os
import sys
import urllib.request

URL = os.environ.get("URL", "http://localhost:20000")
MODEL = os.environ.get("MODEL_ID", "osunlp/QUEST-35B-RL")


def chat(messages, **kw):
    body = {"model": MODEL, "messages": messages, "max_tokens": kw.pop("max_tokens", 4096), "temperature": 0, **kw}
    req = urllib.request.Request(
        URL + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.loads(r.read())["choices"][0]


def picture():
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (640, 400), "white")
    d = ImageDraw.Draw(img)
    d.ellipse((40, 100, 240, 300), fill=(220, 20, 20))
    d.rectangle((380, 100, 580, 300), fill=(20, 60, 220))
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 56)
    except OSError:
        font = ImageFont.load_default(size=56)
    d.text((230, 20), "QUEST 42", fill="black", font=font)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def main():
    ok = True

    def check(name, cond, detail):
        nonlocal ok
        ok &= bool(cond)
        print(f"{'PASS' if cond else 'FAIL'} {name}: {detail[:300]!r}", flush=True)

    c = chat([{"role": "user", "content": "What is 17 * 23? Answer with the number only."}])
    m = c["message"]
    reasoning = m.get("reasoning_content") or m.get("reasoning") or ""
    check("thinking", "391" in (m.get("content") or "") and len(reasoning) > 0,
          f"content={m.get('content')} reasoning_chars={len(reasoning)} finish={c['finish_reason']}")

    c = chat([{"role": "user", "content": "Name the capital of France in one word."}],
             chat_template_kwargs={"enable_thinking": False}, max_tokens=64)
    check("no-thinking", "paris" in (c["message"].get("content") or "").lower(), c["message"].get("content") or "")

    tools = [{"type": "function", "function": {
        "name": "web_search", "description": "Search the web and return the top results.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}}]
    c = chat([{"role": "user", "content": "Use the web_search tool to look up the population of Columbus, Ohio."}],
             tools=tools, tool_choice="auto")
    calls = c["message"].get("tool_calls") or []
    args = json.loads(calls[0]["function"]["arguments"]) if calls else {}
    check("tool-call", calls and calls[0]["function"]["name"] == "web_search" and "columbus" in args.get("query", "").lower(),
          json.dumps(calls))

    if "--no-image" not in sys.argv:
        c = chat([{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": picture()}},
            {"type": "text", "text": "What text is written in this image, and what colors are the circle and the square? "
                                     "Answer in one short sentence."}]}],
            chat_template_kwargs={"enable_thinking": False}, max_tokens=128)
        a = (c["message"].get("content") or "").lower()
        check("image", "42" in a and "red" in a and "blue" in a, a)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Served image / video test (OpenAI chat completions with a base64 image_url / video_url) against serve_hy4.sh.

    python models/autoports/tencent_hy4_flash/tools/vision_client.py [PORT] [--sizes 512,1024,640x480]
        [--text-users 0,3,7] [--max-tokens 128] [--videos 8f_640x480:ffv1,8f_640x480:mp4]

Per size: the tools/vision_ref.py test image ("TT 4821", red circle, blue square) asked "What text is in the image?
What shapes and colors do you see?" greedy, alone and then with N concurrent text users; prints the answers, TTFT,
decode tok/s and a PASS / FAIL (the reply must name 4821, red, blue, circle and square). Exit status 1 on any FAIL.
--videos TAG:CODEC: the tools/video_ref.py clip DIR/video_TAG.npy (a red ball moving left -> right) encoded at 2 fps
(ffv1: lossless .avi, mp4: mp4v) asked tools/video_ref.py's QUESTION; PASS: the reply names red, a ball / circle(ular) /
sphere, left and right. --sizes "" skips the images.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import threading
import time
import urllib.request

sys.path.insert(0, __file__.rsplit("/models/", 1)[0])
from models.autoports.tencent_hy4_flash.tools.vision_ref import QUESTION, make_test_image  # noqa: E402

TEXTS = ["Explain why the sky is blue.", "Write a haiku about rain.", "What is 17 * 23?", "Name three rivers in Europe.",
         "Describe a cat.", "List two prime numbers.", "Say hello in French."]


def stream(port, messages, max_tokens, out):
    body = {"model": "hy4-flash", "messages": messages, "max_tokens": max_tokens, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}, "chat_template_kwargs": {"reasoning_effort": "no_think"}}
    req = urllib.request.Request(f"http://localhost:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    ttft, text, usage = None, [], None
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            for line in r:
                line = line.decode().strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                ev = json.loads(line[5:])
                if ev.get("usage"):
                    usage = ev["usage"]
                for ch in ev.get("choices", []):
                    d = ch.get("delta", {}).get("content") or ""
                    if d:
                        ttft = ttft or time.time() - t0
                        text.append(d)
        out.update(text="".join(text), ttft=ttft, total=time.time() - t0, usage=usage)
    except Exception as e:  # noqa: BLE001
        out.update(error=repr(e), text="".join(text), ttft=ttft, total=time.time() - t0, usage=usage)


def image_msg(size):
    w, h = (int(v) for v in (size.split("x") if "x" in size else (size, size)))
    buf = io.BytesIO()
    make_test_image(w, h).save(buf, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    return [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}},
                                         {"type": "text", "text": QUESTION}]}]


def video_msg(spec, fps=2.0):
    import os
    import tempfile

    import cv2
    import numpy as np

    from models.autoports.tencent_hy4_flash.tools.video_ref import QUESTION as VQ

    tag, codec = (spec.split(":") + ["ffv1"])[:2]
    frames = np.load(os.path.expanduser(f"~/m27-work/hy4/vision/video_{tag}.npy"))
    fourcc, ext, mime = {"ffv1": ("FFV1", "avi", "video/x-msvideo"), "mp4": ("mp4v", "mp4", "video/mp4")}[codec]
    with tempfile.TemporaryDirectory() as d:
        path = f"{d}/clip.{ext}"
        w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc), fps, (frames.shape[2], frames.shape[1]))
        for x in frames:
            w.write(cv2.cvtColor(x, cv2.COLOR_RGB2BGR))
        w.release()
        data = open(path, "rb").read()
    url = f"data:{mime};base64," + base64.b64encode(data).decode()
    return [{"role": "user", "content": [{"type": "video_url", "video_url": {"url": url}},
                                         {"type": "text", "text": VQ}]}]


def check_video(text):
    t = (text or "").lower()
    return "red" in t and "left" in t and "right" in t and any(k in t for k in ("ball", "circle", "circular", "sphere", "dot"))


def check(text):
    t = (text or "").lower()
    return all(k in t for k in ("4821", "red", "blue", "circle", "square"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("port", nargs="?", default="8000")
    ap.add_argument("--sizes", default="512,1024,640x480")
    ap.add_argument("--text-users", default="0,7")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--videos", default="")
    args = ap.parse_args()
    fails = 0
    items = [("IMAGE", s) for s in args.sizes.split(",") if s] + [("VIDEO", v) for v in args.videos.split(",") if v]
    for kind, size in items:
        msgs = image_msg(size) if kind == "IMAGE" else video_msg(size)
        for n in [int(x) for x in args.text_users.split(",")]:
            outs = [{} for _ in range(n + 1)]
            ths = [threading.Thread(target=stream, args=(args.port, msgs, args.max_tokens, outs[0]))]
            ths += [threading.Thread(target=stream, args=(args.port, [{"role": "user", "content": TEXTS[i % 7]}],
                                                          args.max_tokens, outs[i + 1])) for i in range(n)]
            for t in ths:
                t.start()
            for t in ths:
                t.join()
            o = outs[0]
            ok = (check if kind == "IMAGE" else check_video)(o.get("text")) and "error" not in o
            fails += not ok
            u = o.get("usage") or {}
            nt = u.get("completion_tokens") or 0
            dec = (nt - 1) / max(1e-9, o["total"] - (o["ttft"] or 0)) if nt > 1 and o.get("ttft") else 0
            print(f"{kind} {size} with {n} text users: {'PASS' if ok else 'FAIL'} prompt {u.get('prompt_tokens')} tok, "
                  f"TTFT {o.get('ttft') or 0:.2f} s, {nt} tok at {dec:.1f} tok/s {o.get('error', '')}")
            print(f"  answer: {o.get('text')!r}")
            for i, x in enumerate(outs[1:]):
                bad = "error" in x or not x.get("text")
                fails += bad
                print(f"  text user {i}: {'ERR ' + x.get('error', '') if bad else ''}{(x.get('text') or '')[:80]!r} "
                      f"TTFT {x.get('ttft') or 0:.2f} s")
            sys.stdout.flush()
    print("SUMMARY", "PASS" if not fails else f"FAIL ({fails})")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()

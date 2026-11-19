# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Image / audio inputs through the running Inkling server (OpenAI chat API): each case's answer must contain one of
its expected words. Cases: a drawn image (shape, colour, text), a real logo, three short sound clips (bird, trumpet,
piano; informational), speech transcription (three spoken clips),
image + audio in one prompt, and the prefix cache with media (two large images that differ only in colour: the second
must not reuse the first's prefix; the first again may).
usage: python serve/inkling_mm.py [port=8000] [logo png] [sounds dir=/usr/share/sounds/sound-icons]"""
import base64
import io
import json
import os
import sys
import time
import urllib.request

import numpy as np
from PIL import Image, ImageDraw, ImageFont

port = sys.argv[1] if len(sys.argv) > 1 else "8000"
logo = sys.argv[2] if len(sys.argv) > 2 else "/home/raahem/tt-llama.cpp/media/llama1-logo.png"
snd = sys.argv[3] if len(sys.argv) > 3 else "/usr/share/sounds/sound-icons"
url = f"http://127.0.0.1:{port}/v1/chat/completions"
with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=60) as r:
    model = json.load(r)["data"][0]["id"]


def png_b64(img):
    b = io.BytesIO()
    img.save(b, format="PNG")
    return base64.b64encode(b.getvalue()).decode()


def drawn(size=(480, 320), bg=(30, 60, 200), fg=(220, 30, 30), text="TT 42"):
    img = Image.new("RGB", size, bg)
    d = ImageDraw.Draw(img)
    w, h = size
    d.ellipse([w * 0.08, h * 0.15, w * 0.08 + h * 0.7, h * 0.85], fill=fg)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", int(h * 0.25))
    except OSError:
        font = ImageFont.load_default()
    d.text((w * 0.55, h * 0.35), text, fill=(255, 230, 0), font=font)
    return img


def image_part(img):
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{png_b64(img)}"}}


def audio_part(path):
    return {"type": "input_audio", "input_audio": {"data": base64.b64encode(open(path, "rb").read()).decode(),
                                                   "format": "wav"}}


def ask(parts, max_tokens=80):
    body = {"model": model, "messages": [{"role": "user", "content": parts}], "max_tokens": max_tokens,
            "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"reasoning_effort": "none"}}
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    ttft, text, usage = None, "", None
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ev = json.loads(line[5:])
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices", []):
                piece = ch.get("delta", {}).get("content") or ""
                if piece and ttft is None:
                    ttft = time.time() - t0
                text += piece
    return text.strip(), ttft, (usage or {}).get("prompt_tokens", 0)


ok_all = True


ONLY = tuple(filter(None, os.environ.get("ONLY", "").split(",")))  # ONLY=audio,speech: those case-name prefixes


def case(name, parts, expect, gate=True):
    global ok_all
    if ONLY and not name.startswith(ONLY):
        return ""
    try:
        text, ttft, pt = ask(parts)
        ok = any(e in text.lower() for e in expect)
        if gate:
            ok_all &= ok
        tag = ("PASS" if ok else "FAIL") if gate else ("info match" if ok else "info miss")
        at = f"TTFT {ttft:.2f}s" if ttft is not None else "no answer within the token budget (reasoned instead)"
        print(f"{tag} {name}: {pt} prompt tokens, {at} | {text[:160]!r}", flush=True)
        return text
    except Exception as e:  # noqa: BLE001
        ok_all = False
        body = ""
        if hasattr(e, "read"):  # an HTTP error: the server's reason
            try:
                body = " | " + e.read().decode(errors="replace")
            except Exception:  # noqa: BLE001
                pass
        print(f"FAIL {name}: {e!r}{body}"[:600], flush=True)
        return ""


txt = lambda s: {"type": "text", "text": s}  # noqa: E731
case("drawn image: shape", [image_part(drawn()), txt("What shape is in this image? Answer in a few words.")],
     ["circle", "oval", "ellipse"])
case("drawn image: colour", [image_part(drawn()), txt("What colour is the round shape? One word.")], ["red"])
case("drawn image: text", [image_part(drawn()), txt("What text is written in the image? Reply with the text only.")],
     ["42"])
case("logo", [image_part(Image.open(logo).convert("RGB")), txt("What animal is shown? One word.")], ["llama", "alpaca"])
# sound clips of under 2 s (0.75 s for the canary): reported, not gating - the model names them unreliably
# (canary -> crickets, piano -> trumpet) while it transcribes speech exactly (the speech cases below gate)
case("audio: bird", [audio_part(f"{snd}/canary-long.wav"), txt("What makes this sound? A few words.")],
     ["bird", "canary", "chirp", "tweet", "whistl"], gate=False)
case("audio: trumpet", [audio_part(f"{snd}/trumpet-12.wav"), txt("Which musical instrument is this? One word.")],
     ["trumpet", "brass", "horn", "cornet", "bugle"], gate=False)
case("audio: piano", [audio_part(f"{snd}/piano-3.wav"), txt("Which musical instrument is this? One word.")],
     ["piano", "keyboard"], gate=False)
case("image + audio", [image_part(drawn()), audio_part(f"{snd}/canary-long.wav"),
                       txt("Answer both: 1) what colour is the circle in the image? 2) what animal makes the sound?")],
     ["red"])
# prefix cache with media: ~2,900-patch images (a snapshot needs >= 2048 prompt tokens), same text, different colour
big_r = drawn(size=(2400, 1600), fg=(220, 30, 30))
big_g = drawn(size=(2400, 1600), fg=(30, 200, 40))
q = txt("What colour is the round shape? One word.")
case("prefix cache 1: red", [image_part(big_r), q], ["red"])
case("prefix cache 2: green (must not reuse the red prefix)", [image_part(big_g), q], ["green"])
case("prefix cache 3: red again", [image_part(big_r), q], ["red"])
# speech: ALSA's channel-test clips (a voice saying the channel name)
for clip, words in (("Front_Left", ["front", "left"]), ("Rear_Center", ["rear", "center"]), ("Side_Right", ["side", "right"])):
    case(f"speech: {clip}", [audio_part(f"/usr/share/sounds/alsa/{clip}.wav"),
                             txt("Transcribe the speech in this audio exactly. Reply with the words only.")], words)
case("text only after media", [txt("What is 17 * 23? Reply with the number only.")], ["391"])
print("ALL PASS" if ok_all else "SOME FAILED")

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Gradio page for Qwen3-VL-Reranker-8B. Every score comes from the in-process ``RerankService`` (the path
``/v1/rerank`` uses), never from an HTTP call back into the server."""
from __future__ import annotations

import base64
import html
import io
import json
import time
from typing import Callable, Optional
from urllib.parse import quote

import gradio as gr
from PIL import Image

from models.autoports.qwen_qwen3_vl_reranker_8b.demo.library import (
    KINDS,
    Entry,
    Library,
    Ranked,
    chunk_size,
    make_query,
    rank,
)
from models.autoports.qwen_qwen3_vl_reranker_8b.server.media import RequestError
from models.autoports.qwen_qwen3_vl_reranker_8b.server.service import EngineError, sigmoid

FILES_PATH = "/demo-files"
MAX_CANDIDATES = 10
KIND_LABELS = {"image": "Images", "text": "Texts (multilingual)", "video": "Videos", "code": "Code"}
DEFAULT_KINDS = ["image", "text"]
DEFAULT_INSTRUCTION = "Given a search query, retrieve relevant candidates that answer the query."

CSS = """
.rr-prose{max-width:48rem}
.rr-prose p,.rr-prose li{font-size:1.02rem !important;line-height:1.55 !important}
.rr-meta{font-family:ui-monospace,monospace;font-size:.84rem;opacity:.85;margin:4px 0 10px}
.rr-note{border-left:4px solid #d97706;background:#d9770614;padding:6px 10px;margin:6px 0;font-size:.85rem;border-radius:4px}
.rr-progress{display:flex;align-items:center;gap:10px;font-size:.95rem;margin:6px 0 10px}
.rr-spin{width:14px;height:14px;border-radius:50%;border:2px solid var(--border-color-primary);border-top-color:var(--color-accent);animation:rr-spin .8s linear infinite}
@keyframes rr-spin{to{transform:rotate(360deg)}}
.rr-pbar{height:6px;border-radius:3px;background:var(--border-color-primary);overflow:hidden;max-width:28rem;margin-bottom:10px}
.rr-pbar>div{height:100%;background:var(--color-accent)}
.rr-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:12px}
.rr-card{border:1px solid var(--border-color-primary);border-radius:8px;padding:8px;background:var(--background-fill-secondary);display:flex;flex-direction:column;gap:5px;min-width:0}
.rr-card img,.rr-card video{width:100%;max-height:180px;object-fit:contain;border-radius:4px;background:#0002}
.rr-card pre{max-height:200px;overflow:auto;font-size:.72rem;white-space:pre;margin:0;padding:6px;border-radius:4px;background:var(--background-fill-primary)}
.rr-rank{display:flex;justify-content:space-between;align-items:baseline;font-family:ui-monospace,monospace;font-size:.8rem}
.rr-rel{font-weight:700;font-size:1rem}
.rr-bar{height:8px;border-radius:4px;background:var(--border-color-primary);overflow:hidden}
.rr-bar>div{height:100%;border-radius:4px}
.rr-title{font-weight:600;font-size:.88rem;overflow-wrap:anywhere}
.rr-cap{font-size:.8rem;opacity:.85;overflow-wrap:anywhere}
.rr-badge{display:inline-block;border-radius:4px;padding:0 6px;font-size:.72rem;background:var(--color-accent-soft);margin-right:4px}
.rr-lic{font-size:.7rem;opacity:.65;overflow-wrap:anywhere}
.rr-table{border-collapse:collapse;font-size:.85rem;width:100%;max-width:48rem}
.rr-table th,.rr-table td{border-bottom:1px solid var(--border-color-primary);padding:4px 8px;text-align:left;vertical-align:top}
.rr-code{font-family:ui-monospace,monospace;font-size:.78rem;white-space:pre-wrap;overflow-wrap:anywhere;padding:8px;border-radius:6px;background:var(--background-fill-secondary);max-width:48rem}
"""

INTRO = """
## Qwen3-VL-Reranker-8B on Tenstorrent

A **reranker** reads a query and one candidate *together* and says how well the candidate answers the query.
Qwen3-VL-Reranker-8B does this for text, images and video on either side: each (query, candidate) pair runs
through the 8B vision-language model, and the score is `logit("yes") - logit("no")` for "does this document meet
the requirement?". The relevance shown on each result is `sigmoid(score)`, from 0% to 100%.

Every score on this page is computed live, one pair at a time, by the same server that answers `/v1/rerank`, on the
Tenstorrent chip(s) listed under **About**. Nothing is cached and no keywords are matched.
"""


# ---- small html helpers -------------------------------------------------------------------------------------------
def esc(x) -> str:
    return html.escape(str(x if x is not None else ""))


def file_url(entry: Entry) -> str:
    return f"{FILES_PATH}/{quote(entry.file)}"


def thumb_data_url(img: Image.Image, side: int = 320) -> str:
    img = img.convert("RGB").copy()
    img.thumbnail((side, side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def bar_color(rel: float) -> str:
    hue = int(120 * max(0.0, min(1.0, rel)))  # red -> amber -> green
    return f"hsl({hue},65%,45%)"


def score_head(rank_no: int, score: float) -> str:
    rel = sigmoid(score)
    return (
        f'<div class="rr-rank"><span>#{rank_no}</span><span class="rr-rel">{rel * 100:.1f}%</span>'
        f"<span>score {score:+.3f}</span></div>"
        f'<div class="rr-bar"><div style="width:{rel * 100:.1f}%;background:{bar_color(rel)}"></div></div>'
    )


def entry_card(r: Ranked, rank_no: int) -> str:
    e = r.entry
    if e.kind == "image":
        body = f'<img loading="lazy" src="{file_url(e)}" alt="{esc(e.title)}">'
    elif e.kind == "video":
        poster = f' poster="{e.poster}"' if e.poster else ""
        secs = f" · {e.seconds:.0f} s" if e.seconds else ""
        body = (
            f'<video controls preload="none"{poster} src="{file_url(e)}"></video>'
            f'<div class="rr-cap"><a href="{file_url(e)}" target="_blank" rel="noopener">open mp4</a>{secs}</div>'
        )
    elif e.kind == "code":
        head = "\n".join((e.text or "").splitlines()[:40])
        body = f"<pre>{esc(head)}</pre>"
    else:
        snippet = (e.text or "")[:320]
        body = f'<div class="rr-cap">{esc(snippet)}{"…" if len(e.text or "") > 320 else ""}</div>'
    lang = f'<span class="rr-badge">{esc(e.lang)}</span>' if e.lang else ""
    caption = f'<div class="rr-cap">{esc(e.caption)}</div>' if e.caption and e.kind in ("image", "video") else ""
    lic = ""
    if e.license:
        lic = f'<div class="rr-lic">{esc(e.author)} · {esc(e.license)}'
        if e.source_url:
            lic += f' · <a href="{esc(e.source_url)}" target="_blank" rel="noopener">source</a>'
        lic += "</div>"
    return (
        f'<div class="rr-card" data-id="{esc(e.id)}" data-score="{r.score:.6f}">{score_head(rank_no, r.score)}'
        f'<div class="rr-title"><span class="rr-badge">{esc(e.kind)}</span>{lang}{esc(e.title)}</div>'
        f"{body}{caption}{lic}</div>"
    )


def candidate_card(idx: int, text: str, img: Optional[Image.Image], score: float, rank_no: int) -> str:
    body = ""
    if img is not None:
        body += f'<img src="{thumb_data_url(img)}" alt="candidate {idx}">'
    if text:
        body += f'<div class="rr-cap">{esc(text[:500])}{"…" if len(text) > 500 else ""}</div>'
    return (
        f'<div class="rr-card" data-index="{idx}" data-score="{score:.6f}">{score_head(rank_no, score)}'
        f'<div class="rr-title"><span class="rr-badge">candidate {idx}</span></div>{body}</div>'
    )


def progress_html(done: int, total: int, what: str, dies: int) -> str:
    pct = 100.0 * done / max(1, total)
    return (
        f'<div class="rr-progress"><div class="rr-spin"></div><span>Ranking {total} {what} on {dies} '
        f"die{'s' if dies != 1 else ''}… {done}/{total} scored</span></div>"
        f'<div class="rr-pbar"><div style="width:{pct:.0f}%"></div></div>'
    )


def stats_line(n: int, elapsed: float, dies: int, extra: str = "") -> str:
    rate = n / elapsed if elapsed > 0 else 0.0
    return (
        f'<div class="rr-meta">{n} pairs scored in {elapsed:.2f} s · {rate:.1f} pairs/s · {dies} '
        f"die{'s' if dies != 1 else ''}{extra}</div>"
    )


def fake_note(service) -> str:
    if not getattr(service, "fake", False):
        return ""
    return (
        '<div class="rr-note">This server runs with <code>QWEN3VL_RERANKER_FAKE=1</code>: the scores come from '
        "device-free fake engines (a hash of the tokens), not from the model.</div>"
    )


def table_html(header, rows) -> str:
    th = "".join(f"<th>{esc(h)}</th>" for h in header)
    trs = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<table class="rr-table"><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table>'


# ---- handlers -----------------------------------------------------------------------------------------------------
class Handlers:
    """The page's callbacks; plain async methods so tests can drive them without a browser."""

    def __init__(self, library: Library, get_service: Callable[[], object]):
        self.library = library
        self.get_service = get_service

    def service(self):
        service = self.get_service()
        if service is None or not getattr(service, "ready", False):
            raise gr.Error("The model is still loading; try again in a moment.")
        if not any(w.alive() for w in service.workers):
            raise gr.Error("No die is serving (see /health).")
        return service

    async def _scores(self, service, query, docs, instruction):
        """Yields (done, scores) from ``rank``; turns request / engine errors into a gr.Error."""
        try:
            async for done, scores in rank(service, query, docs, instruction, chunk_size(service)):
                yield done, scores
        except RequestError as exc:
            raise gr.Error(exc.message)
        except EngineError as exc:
            raise gr.Error(str(exc))

    async def library_search(self, text, image, kinds, instruction, top_k):
        """Async generator of result HTML: a progress view per step, then the final ranking."""
        service = self.service()
        query = make_query(text, image)
        if query is None:
            raise gr.Error("Type a query or add a query image.")
        kinds = [k for k in (kinds or []) if k in KINDS]
        if not kinds:
            raise gr.Error("Pick at least one kind of document to rank.")
        entries = self.library.select(kinds)
        docs = [e.document(self.library.root) for e in entries]
        top_k = max(1, int(top_k or 12))
        dies = len(service.workers)
        what = " + ".join(KIND_LABELS[k].split(" ")[0].lower() for k in kinds)
        qdesc = (
            "image + text"
            if isinstance(query, dict) and "text" in query
            else "image"
            if isinstance(query, dict)
            else "text"
        )
        yield fake_note(service) + progress_html(0, len(docs), what, dies)
        t0 = time.perf_counter()
        scores: list = []
        async for done, scores in self._scores(service, query, docs, instruction):
            if done < len(docs):
                partial = sorted((Ranked(e, s) for e, s in zip(entries, scores)), key=lambda r: -r.score)[:top_k]
                yield (
                    fake_note(service)
                    + progress_html(done, len(docs), what, dies)
                    + '<div class="rr-meta">best so far:</div>'
                    + self._grid(partial)
                )
        elapsed = time.perf_counter() - t0
        ranked = sorted((Ranked(e, s) for e, s in zip(entries, scores)), key=lambda r: (-r.score, r.entry.id))
        yield (
            fake_note(service)
            + stats_line(
                len(docs), elapsed, dies, f" · query: {qdesc} · ranked {what} · showing top {min(top_k, len(ranked))}"
            )
            + self._grid(ranked[:top_k])
        )

    @staticmethod
    def _grid(ranked: list) -> str:
        return '<div class="rr-grid">' + "".join(entry_card(r, i + 1) for i, r in enumerate(ranked)) + "</div>"

    async def own_candidates(self, q_text, q_image, instruction, n_rows, *cands):
        """``cands`` is MAX_CANDIDATES texts then MAX_CANDIDATES images; rows past ``n_rows`` are ignored."""
        service = self.service()
        query = make_query(q_text, q_image)
        if query is None:
            raise gr.Error("Type a query or add a query image.")
        n_rows = max(1, min(MAX_CANDIDATES, int(n_rows or MAX_CANDIDATES)))
        texts = list(cands[:MAX_CANDIDATES]) + [None] * MAX_CANDIDATES
        images = list(cands[MAX_CANDIDATES : 2 * MAX_CANDIDATES]) + [None] * MAX_CANDIDATES
        rows = []
        for i in range(n_rows):
            t = (texts[i] or "").strip()
            img = images[i]
            doc = make_query(t, img)
            if doc is not None:
                rows.append((i + 1, t, img, doc))
        if not rows:
            raise gr.Error("Fill in at least one candidate (text and/or image).")
        dies = len(service.workers)
        yield fake_note(service) + progress_html(0, len(rows), "candidates", dies)
        t0 = time.perf_counter()
        scores: list = []
        async for _, scores in self._scores(service, query, [r[3] for r in rows], instruction):
            pass
        elapsed = time.perf_counter() - t0
        order = sorted(range(len(rows)), key=lambda i: (-scores[i], i))
        cards = "".join(
            candidate_card(rows[i][0], rows[i][1], rows[i][2], scores[i], k + 1) for k, i in enumerate(order)
        )
        yield fake_note(service) + stats_line(len(rows), elapsed, dies) + f'<div class="rr-grid">{cards}</div>'

    def example_candidates(self):
        """Values for the "Your own candidates" inputs: q_text, q_image, instruction, n_rows, texts..., images..."""
        lib = self.library.by_id

        def img(i):
            e = lib.get(i)
            return e.image if e is not None else None

        rows = [
            ("A woman shares a joyful moment with her golden retriever on a beach at sunset.", None),
            ("", img("img-dog-labrador")),
            ("", img("img-beach")),
            ("The stock market fell sharply on Tuesday.", None),
            ("Ein Hund rennt am Strand den Wellen hinterher.", None),
        ]
        texts = [t for t, _ in rows] + [""] * (MAX_CANDIDATES - len(rows))
        images = [i for _, i in rows] + [None] * (MAX_CANDIDATES - len(rows))
        return ["A woman playing with her dog on a beach at sunset.", None, "", len(rows), *texts, *images]

    def about(self, request: Optional[gr.Request] = None) -> str:
        service = self.get_service()
        host = "localhost:8000"
        if request is not None:
            host = request.headers.get("host") or host
        if service is None:
            live = "<p>The model is still loading.</p>"
        else:
            s = service.summary()
            alive = sum(1 for e in s["engines"] if e["alive"])
            status = "ok" if alive == s["dies"] else "degraded" if alive else "down"
            st = s["stats"]
            live = table_html(
                ["field", "value"],
                [
                    ["status", esc(status) + (" (fake engines, no device)" if s["fake_engines"] else "")],
                    [
                        "mesh / dies",
                        f"{esc(s['mesh_shape'])} / {alive} of {s['dies']} serving (one model copy per die)",
                    ],
                    [
                        "max pair length",
                        f"{esc(s['max_tokens'])} tokens (text truncated to {esc(s['max_text_tokens'])})",
                    ],
                    ["startup (s)", esc(", ".join(f"{k} {v}" for k, v in (s.get("startup_s") or {}).items()))],
                    ["uptime (s)", esc(s["uptime_s"])],
                    ["requests / pairs / tokens", f"{st['requests']} / {st['pairs']} / {st['tokens']}"],
                    ["engine calls", esc(st["engine_calls"])],
                    [
                        "pairs per die",
                        esc(", ".join(f"die {e['die']}: {e['pairs']}" for e in s["engines"])),
                    ],
                    ["demo library", esc(", ".join(f"{k} {v}" for k, v in self.library.counts().items()))],
                ],
            )
        body = json.dumps(
            {
                "query": "A woman playing with her dog on a beach at sunset.",
                "documents": [
                    "A woman shares a joyful moment with her golden retriever on a beach at sunset.",
                    {"image": "<base64 jpeg>"},
                    {"video": {"data": "<base64 mp4>", "fps": 1}},
                    "The stock market fell sharply on Tuesday.",
                ],
                "instruction": DEFAULT_INSTRUCTION,
                "raw_scores": True,
            },
            indent=2,
        )
        curl = f"curl -s http://{host}/v1/rerank -H 'content-type: application/json' -d '{body}'"
        return (
            "<h3>What this is</h3>"
            "<p class='rr-prose'><a href='https://huggingface.co/Qwen/Qwen3-VL-Reranker-8B' target='_blank' "
            "rel='noopener'>Qwen/Qwen3-VL-Reranker-8B</a> ported to TTNN: the 36-layer Qwen3 text tower and the "
            "27-block vision tower run on Tenstorrent Blackhole, one full model copy per chip. The scores on this "
            "page come live from those chips through the same in-process service as the HTTP API.</p>"
            + fake_note(service)
            + "<h3>Live server (<code>/health</code>)</h3>"
            + live
            + "<h3>API</h3>"
            "<p class='rr-prose'><code>POST /v1/rerank</code> (Cohere / Jina style), <code>POST /v1/score</code> "
            "(vLLM style), <code>GET /health</code>, <code>GET /v1/models</code>; OpenAPI at "
            "<a href='/docs' target='_blank'>/docs</a>. Media is sent as base64 or data URLs.</p>"
            f'<div class="rr-code">{esc(curl)}</div>'
            "<p class='rr-meta'>relevance_score = sigmoid(score); with raw_scores the raw logit difference is returned "
            "too. Library media licences: <a href='" + FILES_PATH + "/LICENSES.md' target='_blank'>LICENSES.md</a>.</p>"
        )


# ---- page ---------------------------------------------------------------------------------------------------------
def build_ui(library: Library, get_service: Callable[[], object]) -> gr.Blocks:
    h = Handlers(library, get_service)
    counts = library.counts()
    with gr.Blocks(title="Qwen3-VL-Reranker-8B on Tenstorrent") as demo:
        gr.Markdown(INTRO, elem_classes="rr-prose")
        with gr.Tabs():
            with gr.Tab("Library search"):
                gr.Markdown(
                    elem_classes="rr-prose",
                    value=f"Rank the demo library ({counts['image']} images, {counts['text']} texts in 10 languages, "
                    f"{counts['video']} videos, {counts['code']} code snippets) against a text query, a query image, "
                    "or both. Each document is one (query, document) pair scored on the chip, so ranking a whole kind "
                    "takes seconds: about 60 ms per text pair, 120 ms per image and 200 ms per short video on one chip.",
                )
                with gr.Row():
                    with gr.Column(scale=3):
                        q_text = gr.Textbox(label="Text query", placeholder="a red double-decker bus")
                        kinds = gr.CheckboxGroup(
                            choices=[(KIND_LABELS[k], k) for k in KINDS], value=DEFAULT_KINDS, label="Rank"
                        )
                        with gr.Row():
                            top_k = gr.Slider(1, 48, value=12, step=1, label="Results shown")
                        with gr.Accordion("Instruction (optional)", open=False):
                            instr = gr.Textbox(label="Instruction", placeholder=DEFAULT_INSTRUCTION, lines=2)
                        go = gr.Button("Rank", variant="primary")
                    with gr.Column(scale=2):
                        q_image = gr.Image(
                            type="pil", label="or a query image", sources=["upload", "clipboard"], height=240
                        )
                ex_text = (
                    library.examples("media", n=12)
                    + library.examples("multilingual", n=5)
                    + library.examples("code", n=4)
                )
                gr.Examples(
                    [[q, k] for q, k in ex_text],
                    inputs=[q_text, kinds],
                    label="Example text queries (set the kinds to rank)",
                )
                ex_img = library.examples("media", kind="image")
                if ex_img:
                    gr.Examples([[p, k] for p, k in ex_img], inputs=[q_image, kinds], label="Example query images")
                out = gr.HTML()
                args = [q_text, q_image, kinds, instr, top_k]
                go.click(h.library_search, args, out, api_name="library_search")
                q_text.submit(h.library_search, args, out, api_visibility="private")

            with gr.Tab("Your own candidates"):
                gr.Markdown(
                    elem_classes="rr-prose",
                    value=f"Write a query (text and/or image) and up to {MAX_CANDIDATES} candidates (text and/or "
                    "image each). The candidates are ranked by relevance to the query.",
                )
                with gr.Row():
                    with gr.Column(scale=3):
                        c_q_text = gr.Textbox(label="Query", placeholder="A woman playing with her dog on a beach")
                        c_instr = gr.Textbox(label="Instruction (optional)", placeholder=DEFAULT_INSTRUCTION)
                        n_rows = gr.Slider(1, MAX_CANDIDATES, value=4, step=1, label="Candidates")
                    with gr.Column(scale=2):
                        c_q_image = gr.Image(
                            type="pil", label="Query image (optional)", sources=["upload", "clipboard"], height=200
                        )
                rows, c_texts, c_images = [], [], []
                for i in range(MAX_CANDIDATES):
                    with gr.Row(visible=i < 4) as row:
                        c_texts.append(gr.Textbox(label=f"Candidate {i + 1} text", lines=2, scale=3))
                        c_images.append(
                            gr.Image(
                                type="pil",
                                label=f"Candidate {i + 1} image",
                                sources=["upload", "clipboard"],
                                height=120,
                                scale=1,
                            )
                        )
                    rows.append(row)
                n_rows.change(lambda n: [gr.Row(visible=i < int(n)) for i in range(MAX_CANDIDATES)], n_rows, rows)
                with gr.Row():
                    c_example = gr.Button("Load an example")
                    c_go = gr.Button("Rank candidates", variant="primary")
                c_out = gr.HTML()
                c_inputs = [c_q_text, c_q_image, c_instr, n_rows, *c_texts, *c_images]
                c_example.click(h.example_candidates, [], c_inputs, api_visibility="private")
                c_go.click(h.own_candidates, c_inputs, c_out, api_name="own_candidates")

            with gr.Tab("About"):
                a_out = gr.HTML()
                a_go = gr.Button("Refresh")
                a_go.click(h.about, [], a_out, api_name="about")
                demo.load(h.about, [], a_out, api_visibility="private")
    demo.queue(default_concurrency_limit=4)
    return demo

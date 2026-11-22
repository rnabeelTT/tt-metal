# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Mounts the Gradio demo on the reranker's FastAPI app (only when QWEN3VL_RERANKER_DEMO=1; imports gradio)."""
from __future__ import annotations

import os

from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

DEMO_PATH = "/demo"


def demo_enabled() -> bool:
    return os.environ.get("QWEN3VL_RERANKER_DEMO", "0") == "1"


def mount_demo(app):
    """``/demo`` (the page), ``/demo-files`` (library media), ``/`` -> ``/demo/``. Scores come from
    ``app.state.service``, which the server's lifespan sets once the dies are warm."""
    import gradio as gr

    from models.autoports.qwen_qwen3_vl_reranker_8b.demo.library import Library
    from models.autoports.qwen_qwen3_vl_reranker_8b.demo.ui import CSS, FILES_PATH, build_ui

    library = Library()
    app.state.demo_library = library
    app.mount(FILES_PATH, StaticFiles(directory=str(library.root)), name="demo-files")

    @app.get("/", include_in_schema=False)
    async def root():
        return RedirectResponse(DEMO_PATH + "/")

    blocks = build_ui(library, get_service=lambda: getattr(app.state, "service", None))
    app.state.demo_blocks = blocks
    return gr.mount_gradio_app(
        app,
        blocks,
        path=DEMO_PATH,
        css=CSS,
        footer_links=[],
        ssr_mode=False,
        allowed_paths=[str(library.root / "queries")],
    )

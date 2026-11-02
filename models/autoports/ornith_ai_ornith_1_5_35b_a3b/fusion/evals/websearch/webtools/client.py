"""Thin clients for the webtools service (async for the agent loop, sync for BFCL)."""
from __future__ import annotations

import os
import time

WEBTOOLS_URL = os.environ.get("WEBTOOLS_URL", "http://127.0.0.1:28011")


class AsyncWebTools:
    def __init__(self, bench: str, url: str | None = None, blocklist: str | None = None):
        import httpx

        self.url = (url or WEBTOOLS_URL).rstrip("/")
        self.bench = bench
        self.blocklist = blocklist or os.environ.get("BLOCKLIST")
        self.http = httpx.AsyncClient(timeout=httpx.Timeout(600, connect=10))

    async def _post(self, path: str, body: dict) -> dict:
        if self.blocklist:
            body["blocklist"] = self.blocklist
        last = None
        for attempt in range(4):
            try:
                r = await self.http.post(self.url + path, json=body)
                r.raise_for_status()
                return r.json()
            except Exception as e:  # noqa: BLE001
                last = f"{type(e).__name__}: {e}"
                import asyncio

                await asyncio.sleep(3 * (attempt + 1))
        return {"error": f"webtools service unreachable ({last})"}

    async def search(self, query: str, tag: str, question: str | None = None, n: int = 10) -> dict:
        return await self._post("/search", {"query": query, "bench": self.bench, "tag": tag, "question": question, "n": n})

    async def fetch(self, url: str, tag: str, max_chars: int = 60000) -> dict:
        return await self._post("/fetch", {"url": url, "bench": self.bench, "tag": tag, "max_chars": max_chars})

    async def aclose(self):
        await self.http.aclose()


class SyncWebTools:
    def __init__(self, bench: str, url: str | None = None, blocklist: str | None = None):
        import requests

        self.s = requests.Session()
        self.url = (url or WEBTOOLS_URL).rstrip("/")
        self.bench = bench
        self.blocklist = blocklist or os.environ.get("BLOCKLIST")

    def _post(self, path: str, body: dict) -> dict:
        if self.blocklist:
            body["blocklist"] = self.blocklist
        last = None
        for attempt in range(4):
            try:
                r = self.s.post(self.url + path, json=body, timeout=600)
                r.raise_for_status()
                return r.json()
            except Exception as e:  # noqa: BLE001
                last = f"{type(e).__name__}: {e}"
                time.sleep(3 * (attempt + 1))
        return {"error": f"webtools service unreachable ({last})"}

    def search(self, query: str, tag: str = "", question: str | None = None, n: int = 10, lang: str | None = None) -> dict:
        return self._post("/search", {"query": query, "bench": self.bench, "tag": tag, "question": question, "n": n, "lang": lang})

    def fetch(self, url: str, tag: str = "", raw: bool = False, max_chars: int = 60000) -> dict:
        return self._post("/fetch", {"url": url, "bench": self.bench, "tag": tag, "raw": raw, "max_chars": max_chars})


def check_service(url: str | None = None, autostart: bool = True) -> str | None:
    """None if the service answers /health (starting it with run_webtools.sh if needed), else an error string."""
    import subprocess
    import urllib.request
    from pathlib import Path

    u = (url or WEBTOOLS_URL).rstrip("/") + "/health"
    err = None
    for attempt in range(2):
        try:
            with urllib.request.urlopen(u, timeout=5) as r:
                if r.status == 200:
                    return None
            err = f"HTTP {r.status}"
        except Exception as e:  # noqa: BLE001
            err = e
        if attempt == 0 and autostart and url in (None, "http://127.0.0.1:28011"):
            subprocess.run([str(Path(__file__).resolve().parent.parent / "run_webtools.sh"), "start"], check=False)
    return f"webtools service not reachable at {url or WEBTOOLS_URL} ({err}); start it with ~/evals/websearch/run_webtools.sh start"

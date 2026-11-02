"""Page fetching + extraction: httpx -> trafilatura markdown (HTML), pypdf text (PDF), plain text.

No summarizer: the page's main content is returned as markdown, truncated (default 60,000 chars, HLE tools spec)
with an explicit truncation indicator. Images / audio / video payloads are refused.
robots.txt is honoured for our user agent (explicit Disallow rules only; an unreachable robots.txt allows).
"""
from __future__ import annotations

import asyncio
import io
import os
import re
import time
import urllib.robotparser
from urllib.parse import urlsplit

import httpx

UA_TOKEN = "TTBenchEval"  # not a substring-match of common robots.txt bot groups ("Fetch", "bot", ...)
USER_AGENT = (f"Mozilla/5.0 (compatible; {UA_TOKEN}/1.0; non-commercial research benchmark evaluation; "
              "respects robots.txt)")
MAX_BYTES = 20 * 1024 * 1024
RESPECT_ROBOTS = os.environ.get("RESPECT_ROBOTS", "1") != "0"
TEXT_TYPES = ("text/html", "application/xhtml+xml", "text/plain", "application/json", "application/xml", "text/xml",
              "text/markdown", "text/csv", "application/ld+json", "application/rss+xml", "application/atom+xml")


class FetchError(Exception):
    def __init__(self, msg: str, cacheable: bool = False):
        super().__init__(msg)
        self.cacheable = cacheable


class Robots:
    """Per-host robots.txt cache (24 h)."""

    def __init__(self, http: httpx.AsyncClient):
        self.http = http
        self.cache: dict[str, tuple[float, urllib.robotparser.RobotFileParser | None]] = {}
        self.locks: dict[str, asyncio.Lock] = {}

    async def allowed(self, url: str) -> bool:
        p = urlsplit(url)
        base = f"{p.scheme}://{p.netloc}"
        lock = self.locks.setdefault(base, asyncio.Lock())
        async with lock:
            ent = self.cache.get(base)
            if ent is None or time.time() - ent[0] > 86400:
                rp = None
                try:
                    r = await self.http.get(base + "/robots.txt", timeout=10, follow_redirects=True)
                    if r.status_code == 200 and "html" not in r.headers.get("content-type", ""):
                        rp = urllib.robotparser.RobotFileParser()
                        rp.parse(r.text.splitlines())
                except Exception:  # noqa: BLE001  unreachable robots.txt -> allow
                    rp = None
                ent = (time.time(), rp)
                self.cache[base] = ent
        rp = ent[1]
        return True if rp is None else rp.can_fetch(UA_TOKEN, url)


def _ctype(headers) -> str:
    return (headers.get("content-type") or "").split(";")[0].strip().lower()


def _sniff(body: bytes, ctype: str, url: str) -> str:
    if body[:5] == b"%PDF-":
        return "application/pdf"
    if ctype in ("", "application/octet-stream", "binary/octet-stream"):
        if url.lower().split("?")[0].endswith(".pdf"):
            return "application/pdf"
        head = body[:2048].lower()
        if b"<html" in head or b"<!doctype html" in head:
            return "text/html"
        try:
            body[:4096].decode("utf-8")
            return "text/plain"
        except UnicodeDecodeError:
            return ctype or "application/octet-stream"
    return ctype


def _decode(body: bytes, headers) -> str:
    m = re.search(r"charset=([\w-]+)", headers.get("content-type", ""), re.I)
    for enc in ([m.group(1)] if m else []) + ["utf-8", "gb18030", "latin-1"]:
        try:
            return body.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return body.decode("utf-8", "replace")


def html_to_markdown(body: bytes, url: str) -> tuple[str, str]:
    import trafilatura
    from trafilatura.metadata import extract_metadata

    title = ""
    try:
        md = extract_metadata(body, default_url=url)
        title = (md.title if md else "") or ""
    except Exception:  # noqa: BLE001
        pass
    text = None
    try:
        text = trafilatura.extract(body, url=url, output_format="markdown", include_tables=True, include_links=True,
                                   include_images=False, include_comments=False, favor_recall=True, deduplicate=False)
    except Exception:  # noqa: BLE001
        text = None
    if not text or len(text) < 200:
        # fallback: visible text (scripts/styles removed), like BFCL's "truncate" mode
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(body, "html.parser")
        for t in soup(["script", "style", "noscript", "svg"]):
            t.extract()
        if not title and soup.title and soup.title.string:
            title = soup.title.string.strip()
        alt = soup.get_text(separator="\n", strip=True)
        if not text or len(alt) > 3 * len(text):
            text = alt
    return title.strip(), (text or "").strip()


def pdf_to_text(body: bytes, char_budget: int) -> tuple[str, str]:
    from pypdf import PdfReader

    r = PdfReader(io.BytesIO(body))
    title = ""
    try:
        title = (r.metadata.title or "") if r.metadata else ""
    except Exception:  # noqa: BLE001
        pass
    parts, n = [], 0
    for i, page in enumerate(r.pages):
        try:
            t = page.extract_text() or ""
        except Exception:  # noqa: BLE001
            t = ""
        parts.append(f"\n\n[page {i + 1}]\n{t}")
        n += len(t)
        if n > char_budget:
            parts.append(f"\n\n[... PDF has {len(r.pages)} pages; extraction stopped after page {i + 1} ...]")
            break
    return str(title).strip(), "".join(parts).strip()


async def fetch_document(http: httpx.AsyncClient, robots: Robots, url: str, check_url, timeout: float,
                         raw: bool = False, char_budget: int = 400_000) -> dict:
    """Returns {final_url, status, content_type, title, text}. Raises FetchError.
    check_url(u) -> rule|None is applied to every redirect hop."""
    if not re.match(r"^https?://", url, re.I):
        raise FetchError("Invalid URL: must start with http:// or https://", cacheable=True)
    if RESPECT_ROBOTS and not await robots.allowed(url):
        raise FetchError("Fetching this URL is disallowed by the site's robots.txt.", cacheable=False)
    cur = url
    t0 = time.time()
    for _hop in range(8):
        rule = check_url(cur)
        if rule:
            raise FetchError(f"BLOCKED:{rule}", cacheable=False)
        req = http.build_request("GET", cur, headers={"User-Agent": USER_AGENT})
        try:
            r = await http.send(req, stream=True, follow_redirects=False)
        except httpx.TimeoutException:
            raise FetchError(f"Timed out fetching {cur}")
        except httpx.HTTPError as e:
            raise FetchError(f"Could not fetch {cur}: {type(e).__name__}: {e}")
        try:
            if r.is_redirect and r.headers.get("location"):
                cur = str(r.url.join(r.headers["location"]))
                continue
            if r.status_code >= 400:
                raise FetchError(f"HTTP {r.status_code} fetching {cur}", cacheable=r.status_code in (404, 410))
            ct = _ctype(r.headers)
            if ct.startswith(("image/", "audio/", "video/")):
                raise FetchError(f"Unsupported content type {ct}: image/audio/video payloads are not returned.", cacheable=True)
            buf = bytearray()
            async for chunk in r.aiter_bytes():
                buf.extend(chunk)
                if len(buf) > MAX_BYTES:
                    break
                if time.time() - t0 > timeout:
                    raise FetchError(f"Timed out reading {cur}")
            body = bytes(buf)
            headers = r.headers
        finally:
            await r.aclose()
        break
    else:
        raise FetchError("Too many redirects")
    ct = _sniff(body, ct, cur)
    if raw:
        return {"final_url": cur, "status": 200, "content_type": ct, "title": "", "text": _decode(body, headers)}
    if ct == "application/pdf":
        title, text = await asyncio.wait_for(asyncio.to_thread(pdf_to_text, body, char_budget), timeout=90)
    elif ct in ("text/html", "application/xhtml+xml") or ct.endswith("+xml") and b"<html" in body[:4096].lower():
        title, text = await asyncio.wait_for(asyncio.to_thread(html_to_markdown, body, cur), timeout=60)
    elif ct.startswith("text/") or ct in TEXT_TYPES:
        title, text = "", _decode(body, headers)
    else:
        raise FetchError(f"Unsupported content type {ct or 'unknown'}.", cacheable=True)
    return {"final_url": cur, "status": 200, "content_type": ct, "title": title, "text": text}

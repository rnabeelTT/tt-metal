"""Blocklist enforcement following CAIS HLE docs/evaluation-with-tools.md (section 3), applied to all benchmarks.

Rules come from blocklist/hle_blocklist.json (verbatim CAIS copy) merged with blocklist/extra_blocklist.json:
  domains                  exact hostnames and all their subdomains
  url_prefixes             URL prefixes, case-insensitive, scheme and leading www. removed
  serp_url_patterns        raw search-engine result URLs (no fetching SERPs to bypass result filtering)
  url_patterns             case-insensitive regexes on queries, URLs, result titles/snippets, fetched titles/markdown
  patterns_case_sensitive  same checks, case-sensitive
  answer_title_patterns    case-insensitive, search-result titles only (also with a trailing site name removed)
  answer_title_allow_hosts hosts exempt from answer_title_patterns only
Queries are also refused when they contain a blocked `site:` domain or a long verbatim span of the question.
Every check returns None (allowed) or a short rule string (what matched), which callers log.
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from urllib.parse import unquote, urlsplit

HERE = Path(__file__).resolve().parent.parent / "blocklist"
DEFAULT_FILES = [HERE / "hle_blocklist.json", HERE / "extra_blocklist.json"]

_EMBEDDED_URL = re.compile(r"(?:https?:/{1,2}|(?<![a-z0-9])www\.)[^\s\"'<>]+", re.I)
_TITLE_SITE_SUFFIX = re.compile(r"\s+(?:[-|–—:·•]|::)\s+[^-|–—:·•]{1,80}$")
_CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯]")
_WORD = re.compile(r"\w+", re.U)


def _strip_scheme(u: str) -> str:
    return re.sub(r"^[a-z][a-z0-9+.-]*://", "", u.strip(), flags=re.I)


def _strip_www(u: str) -> str:
    return re.sub(r"^www\d?\.", "", u, flags=re.I)


def _host(u: str) -> str:
    s = u.strip()
    if not re.match(r"^[a-z][a-z0-9+.-]*://", s, re.I):
        s = "http://" + s
    try:
        h = urlsplit(s).hostname or ""
    except ValueError:
        h = ""
    return h.lower().rstrip(".")


class Blocklist:
    def __init__(self, files: list[Path] | None = None, verbatim_words: int = 12, verbatim_cjk_chars: int = 30):
        d: dict[str, list] = {}
        self.sources = []
        for f in files or DEFAULT_FILES:
            f = Path(f)
            if not f.exists():
                continue
            self.sources.append(str(f))
            for k, v in json.loads(f.read_text()).items():
                if isinstance(v, list):
                    d.setdefault(k, []).extend(v)
        self.domains = {x.lower().strip(".") for x in d.get("domains", [])}
        self.prefixes = [_strip_www(_strip_scheme(x.lower())) for x in d.get("url_prefixes", [])]
        self.serp = [re.compile(p, re.I) for p in d.get("serp_url_patterns", [])]
        self.pat_i = [re.compile(p, re.I) for p in d.get("url_patterns", [])]
        self.pat_cs = [re.compile(p) for p in d.get("patterns_case_sensitive", [])]
        self.title_pats = [re.compile(p, re.I) for p in d.get("answer_title_patterns", [])]
        self.title_allow = {x.lower() for x in d.get("answer_title_allow_hosts", [])}
        self.verbatim_words = verbatim_words
        self.verbatim_cjk_chars = verbatim_cjk_chars

    # ------------------------------------------------------------------ primitives
    def _domain_blocked(self, host: str) -> str | None:
        h = host
        while h:
            if h in self.domains:
                return f"domain:{h}"
            if "." not in h:
                break
            h = h.split(".", 1)[1]
        return None

    def check_text(self, text: str | None, where: str = "text") -> str | None:
        if not text:
            return None
        for p in self.pat_i:
            if p.search(text):
                return f"url_patterns[{where}]:{p.pattern[:80]}"
        for p in self.pat_cs:
            if p.search(text):
                return f"patterns_case_sensitive[{where}]:{p.pattern[:80]}"
        return None

    def check_url(self, url: str, _depth: int = 0) -> str | None:
        if not url:
            return None
        variants = [url]
        u = url
        for _ in range(3):  # percent-decoding (proxy / archive links)
            nu = unquote(u)
            if nu == u:
                break
            variants.append(nu)
            u = nu
        for v in variants:
            host = _host(v)
            r = self._domain_blocked(host)
            if r:
                return r
            bare = _strip_www(_strip_scheme(v)).lower()
            for p in self.prefixes:
                if bare.startswith(p):
                    return f"url_prefix:{p}"
            noscheme = _strip_scheme(v).lower()
            for p in self.serp:
                if p.search(noscheme):
                    return f"serp_url:{p.pattern[:60]}"
            r = self.check_text(v, "url")
            if r:
                return r
            if _depth < 2:  # URLs embedded in the path / query (archive.org/web/..., proxies, redirectors)
                body = _strip_scheme(v)
                slash = body.find("/")
                rest = body[slash:] if slash >= 0 else ""
                for m in _EMBEDDED_URL.finditer(rest):
                    r = self.check_url(m.group(0), _depth + 1)
                    if r:
                        return "embedded:" + r
        return None

    def check_result_title(self, title: str, url: str) -> str | None:
        host = _host(url)
        if any(host == a or host.endswith("." + a) for a in self.title_allow):
            return None
        cands = [title.strip()]
        t = title.strip()
        for _ in range(2):
            nt = _TITLE_SITE_SUFFIX.sub("", t)
            if nt == t:
                break
            cands.append(nt)
            t = nt
        for c in cands:
            for p in self.title_pats:
                if p.search(c):
                    return f"answer_title:{p.pattern[:60]}"
        return None

    # ------------------------------------------------------------------ verbatim question spans
    @staticmethod
    def _norm(s: str) -> str:
        return unicodedata.normalize("NFKC", s).lower()

    def verbatim_span(self, query: str, question: str | None) -> str | None:
        if not question:
            return None
        q, src = self._norm(query), self._norm(question)
        # word n-grams (Latin etc.)
        qw, sw = _WORD.findall(q), _WORD.findall(src)
        n = self.verbatim_words
        if len(qw) >= n and len(sw) >= n:
            grams = {tuple(sw[i : i + n]) for i in range(len(sw) - n + 1)}
            for i in range(len(qw) - n + 1):
                if tuple(qw[i : i + n]) in grams:
                    return f"verbatim_question_span:{n}_words"
        # CJK character spans (punctuation/whitespace removed)
        qc = "".join(_CJK.findall(q))
        sc = "".join(_CJK.findall(src))
        k = self.verbatim_cjk_chars
        if len(qc) >= k and len(sc) >= k:
            grams = {sc[i : i + k] for i in range(len(sc) - k + 1)}
            for i in range(len(qc) - k + 1):
                if qc[i : i + k] in grams:
                    return f"verbatim_question_span:{k}_cjk_chars"
        return None

    # ------------------------------------------------------------------ high-level
    def check_query(self, query: str, question: str | None = None) -> str | None:
        for m in re.finditer(r"\b(?:site|inurl|url):\s*([^\s)\"']+)", query, re.I):
            r = self.check_url(m.group(1))
            if r:
                return "site_operator:" + r
        for m in _EMBEDDED_URL.finditer(query):
            r = self.check_url(m.group(0))
            if r:
                return "query_url:" + r
        r = self.check_text(query, "query")
        if r:
            return r
        return self.verbatim_span(query, question)

    def check_result(self, url: str, title: str, snippet: str) -> str | None:
        return (self.check_url(url) or self.check_text(title, "title") or self.check_text(snippet, "snippet")
                or self.check_result_title(title or "", url))

    def check_page(self, title: str | None, text: str | None) -> str | None:
        return self.check_text(title, "page_title") or self.check_text(text, "page_text")

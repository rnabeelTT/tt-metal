"""Deferred stand-in for claw_eval.graders.llm_judge.LLMJudge (no LLM API is ever called).

Graders call  judge.evaluate / judge.evaluate_actions / judge.evaluate_visual  (and T001/T002 call
judge.client.chat.completions.create directly). Each call is turned into a judge *request*: exactly the
system prompt + user content the official LLMJudge would send to gemini-3-flash (multi_turn: claude-opus-4.6),
keyed by sha256 of (method, system, user content, image hashes).

  record mode (during the run):  request -> <judge_dir>/judge_requests/<key>.json (images -> judge_assets/),
                                 returns a placeholder verdict (score 0.0) so grading completes provisionally.
  replay mode (ingest):          returns the verdict from judgments.jsonl for that key; unknown keys are
                                 recorded as missing (the request is written too, so a follow-up round of
                                 judging can pick them up) and get the placeholder.

Verdict schemas:  evaluate*/visual: {"score": 0..1, "reasoning": str}   chat (raw prompt): {"response": str}
Parsing mirrors LLMJudge: score clamped to [0,1].
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path

from claw_eval.graders.llm_judge import (
    _ACTIONS_SYSTEM_PROMPT,
    _SYSTEM_PROMPT,
    _VISUAL_SYSTEM_PROMPT,
    JudgeResult,
)
from claw_eval.models.trace import _now

PLACEHOLDER = 0.0
SCORE_SCHEMA = {"type": "object", "required": ["score", "reasoning"],
                "properties": {"score": {"type": "number", "minimum": 0, "maximum": 1},
                               "reasoning": {"type": "string"}}}
CHAT_SCHEMA = {"type": "object", "required": ["response"],
               "properties": {"response": {"type": "string",
                                           "description": "the judge model's raw reply text (the prompt says what format)"}}}


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _ext(raw: bytes) -> str:
    if raw[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if raw[:3] == b"\xff\xd8\xff":
        return "jpg"
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "webp"
    if raw[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    return "bin"


class DeferredJudge:
    """Drop-in for LLMJudge(model_id, api_key, base_url)."""

    def __init__(self, model_id: str = "google/gemini-3-flash-preview", api_key: str | None = None,
                 base_url: str = "", *, mode: str | None = None, judge_dir: str | Path | None = None,
                 verdicts: dict | None = None) -> None:
        self.model_id = model_id
        self.api_key = api_key
        self.base_url = base_url
        self.mode = mode or os.environ.get("CLAWEVAL_JUDGE_MODE", "record")
        self.dir = Path(judge_dir or os.environ.get("CLAWEVAL_JUDGE_DIR", "."))
        self.verdicts = verdicts or {}
        self.missing: list[str] = []
        self.used: list[str] = []
        self._call_log: list[dict] = []
        self.client = _FakeClient(self)
        (self.dir / "judge_requests").mkdir(parents=True, exist_ok=True)
        (self.dir / "judge_assets").mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ plumbing
    def _asset(self, b64: str) -> str:
        raw = base64.b64decode(b64)
        h = hashlib.sha256(raw).hexdigest()
        p = self.dir / "judge_assets" / f"{h[:24]}.{_ext(raw)}"
        if not p.exists():
            tmp = p.with_name(p.name + f".{os.getpid()}.tmp")
            tmp.write_bytes(raw)
            os.replace(tmp, p)
        return h, str(p)

    def _request(self, method: str, system: str | None, user, extra: dict | None = None) -> tuple[str, dict]:
        """user: str, or list of OpenAI content parts (text / image_url data URIs)."""
        images, parts, user_hashable = [], [], user
        if isinstance(user, list):
            user_hashable = []
            for p in user:
                if p.get("type") == "image_url":
                    url = p["image_url"]["url"]
                    b64 = url.split(",", 1)[1] if url.startswith("data:") else url
                    h, path = self._asset(b64)
                    images.append(path)
                    user_hashable.append({"image_sha256": h})
                    parts.append(f"[IMAGE {len(images)}: {path}]")
                else:
                    user_hashable.append({"text": p.get("text", "")})
                    parts.append(p.get("text", ""))
            user_text = "\n".join(parts)
        else:
            user_text = user
        key = _sha({"m": method, "s": system, "u": user_hashable})
        req = {"key": key, "method": method, "system_prompt": system, "user_prompt": user_text,
               "images": images, "judge_model_official": self.model_id, **(extra or {})}
        p = self.dir / "judge_requests" / f"{key}.json"
        if not p.exists():
            tmp = p.with_name(p.name + f".{os.getpid()}.tmp")
            tmp.write_text(json.dumps(req, ensure_ascii=False))
            os.replace(tmp, p)
        return key, req

    def _verdict(self, key: str):
        self.used.append(key)
        if self.mode == "replay":
            v = self.verdicts.get(key)
            if v is not None:
                return v
            self.missing.append(key)
        return None

    def _score_result(self, method: str, key: str, rubric: str, extra_log: dict | None = None) -> JudgeResult:
        v = self._verdict(key)
        if v is None:
            res = JudgeResult(score=PLACEHOLDER, reasoning="DEFERRED (judge verdict pending)")
        else:
            try:
                score = max(0.0, min(1.0, float(v.get("score"))))
            except (TypeError, ValueError):
                score = PLACEHOLDER
            res = JudgeResult(score=score, reasoning=str(v.get("reasoning", "")))
        self._call_log.append({"method": method, "deferred_key": key, "deferred": v is None,
                               "rubric_preview": (rubric or "")[:300], "score": res.score,
                               "reasoning": res.reasoning, "timestamp": _now(), **(extra_log or {})})
        return res

    # ------------------------------------------------------------------ LLMJudge API
    def evaluate(self, task_prompt: str, conversation: str, actions_summary: str, rubric: str) -> JudgeResult:
        user_msg = (f"## Task Prompt\n{task_prompt}\n\n## Conversation\n{conversation}\n\n"
                    f"## Actions Taken\n{actions_summary}\n\n## Rubric\n{rubric}")
        key, _ = self._request("evaluate", _SYSTEM_PROMPT, user_msg)
        return self._score_result("evaluate", key, rubric)

    def evaluate_actions(self, task_prompt: str, artifacts: str, rubric: str) -> JudgeResult:
        user_msg = (f"## Task Prompt\n{task_prompt}\n\n## Agent Actions (from server audit log)\n{artifacts}\n\n"
                    f"## Rubric\n{rubric}")
        key, _ = self._request("evaluate_actions", _ACTIONS_SYSTEM_PROMPT, user_msg)
        return self._score_result("evaluate_actions", key, rubric)

    def evaluate_visual(self, rubric: str, reference_images_b64: list[str], candidate_images_b64: list[str],
                        context: str = "") -> JudgeResult:
        # identical content construction to LLMJudge.evaluate_visual
        content_parts: list[dict] = []
        header = "## Visual Evaluation\n"
        if context:
            header += f"{context}\n\n"
        header += f"## Rubric\n{rubric}\n\n"
        header += (
            "## Scoring Calibration\n"
            "- 0.0-0.2: Output is mostly wrong, unrecognizable, or missing most required content\n"
            "- 0.2-0.4: Some elements present but major content errors (wrong notes, wrong colors, wrong layout)\n"
            "- 0.4-0.6: General structure is right but significant content inaccuracies remain\n"
            "- 0.6-0.8: Most content is correct with some minor issues\n"
            "- 0.8-1.0: Content is substantially correct, matching reference closely\n\n"
            "IMPORTANT: Looking nice is NOT enough. The CONTENT must be accurate. "
            "Check each rubric criterion individually and sum up the weighted scores.\n\n"
        )
        header += "Below are reference images followed by candidate images.\n"
        header += 'Respond with JSON only: {"score": <float>, "reasoning": "<brief explanation>"}'
        content_parts.append({"type": "text", "text": header})
        if reference_images_b64:
            content_parts.append({"type": "text", "text": f"\n### Reference ({len(reference_images_b64)} images)"})
            for img in reference_images_b64:
                content_parts.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}})
        if candidate_images_b64:
            content_parts.append({"type": "text", "text": f"\n### Candidate ({len(candidate_images_b64)} images)"})
            for img in candidate_images_b64:
                content_parts.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}})
        key, _ = self._request("evaluate_visual", _VISUAL_SYSTEM_PROMPT, content_parts,
                               {"n_ref_images": len(reference_images_b64 or []),
                                "n_cand_images": len(candidate_images_b64 or [])})
        return self._score_result("evaluate_visual", key, rubric,
                                  {"n_ref_images": len(reference_images_b64 or []),
                                   "n_cand_images": len(candidate_images_b64 or []),
                                   "context_preview": (context or "")[:200]})

    def get_call_log(self) -> list[dict]:
        return list(self._call_log)

    def reset_call_log(self) -> None:
        self._call_log.clear()


# ---------------------------------------------------------------------- raw client (T001/T002 graders)
class _Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _Completions:
    def __init__(self, judge: DeferredJudge):
        self.j = judge

    def create(self, model=None, messages=None, **kw):
        msgs = messages or []
        system = "\n".join(m.get("content", "") for m in msgs if m.get("role") == "system") or None
        users = [m.get("content") for m in msgs if m.get("role") != "system"]
        user = users[0] if len(users) == 1 else json.dumps(users, ensure_ascii=False)
        key, _ = self.j._request("chat", system, user)
        v = self.j._verdict(key)
        text = "{}" if v is None else str(v.get("response", ""))
        self.j._call_log.append({"method": "chat", "deferred_key": key, "deferred": v is None,
                                 "rubric_preview": (user if isinstance(user, str) else "")[:300],
                                 "score": -1.0, "reasoning": text[:500], "timestamp": _now()})
        msg = _Obj(content=text, tool_calls=None)
        return _Obj(choices=[_Obj(message=msg, finish_reason="stop")], usage=None)


class _FakeClient:
    def __init__(self, judge: DeferredJudge):
        self.chat = _Obj(completions=_Completions(judge))

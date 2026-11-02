"""Python sandbox for HLE w/ tools: docker image evals-pysandbox:3.11.16 (Python 3.11.16 + CAIS HLE
docs/requirements-tools.txt pins), no network, CPU / memory / pids / time limits.

One container per agent episode (started lazily on the first call, removed at the end of the episode). Each call
runs `python -` in a fresh process inside it; files written to the working directory /work persist between calls of
the same episode, interpreter state does not. Host-wide parallelism of running code is capped (SANDBOX_PARALLEL) so
the host CPU stays available for the model server.
Env: SANDBOX_IMAGE, SANDBOX_CPUS (1), SANDBOX_MEM (2g), SANDBOX_TIMEOUT (120 s per call), SANDBOX_PARALLEL (4),
     SANDBOX_MAX_OUTPUT (20000 chars).
Orphans (e.g. after a crash): docker rm -f $(docker ps -aq --filter label=evals-pysandbox)
"""
from __future__ import annotations

import asyncio
import os
import uuid

IMAGE = os.environ.get("SANDBOX_IMAGE", "evals-pysandbox:3.11.16")
CPUS = os.environ.get("SANDBOX_CPUS", "1")
MEM = os.environ.get("SANDBOX_MEM", "2g")
TIMEOUT = float(os.environ.get("SANDBOX_TIMEOUT", "120"))
MAX_OUT = int(os.environ.get("SANDBOX_MAX_OUTPUT", "20000"))
_SEM: asyncio.Semaphore | None = None


def _sem() -> asyncio.Semaphore:
    global _SEM
    if _SEM is None:
        _SEM = asyncio.Semaphore(int(os.environ.get("SANDBOX_PARALLEL", "4")))
    return _SEM


async def _run(*args: str, stdin: bytes | None = None, timeout: float = 60) -> tuple[int, bytes, bytes]:
    p = await asyncio.create_subprocess_exec(*args, stdin=asyncio.subprocess.PIPE if stdin is not None else None,
                                             stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        out, err = await asyncio.wait_for(p.communicate(stdin), timeout=timeout)
    except asyncio.TimeoutError:
        p.kill()
        await p.wait()
        return -9, b"", b"host-side timeout"
    return p.returncode, out, err


def _clip(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n // 2] + f"\n... [{len(s) - n} characters omitted] ...\n" + s[-n // 2 :]


class PythonSandbox:
    def __init__(self, tag: str):
        self.name = "evals-pysbx-" + uuid.uuid4().hex[:12]
        self.tag = tag
        self.started = False
        self.calls = 0

    async def _start(self):
        rc, out, err = await _run(
            "docker", "run", "-d", "--rm", "--name", self.name, "--label", "evals-pysandbox", "--label", f"tag={self.tag[:60]}",
            "--network", "none", "--cpus", CPUS, "--memory", MEM, "--memory-swap", MEM, "--pids-limit", "256",
            "--read-only", "--tmpfs", "/work:rw,size=512m,uid=1000,mode=1777", "--tmpfs", "/tmp:rw,size=256m,mode=1777",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--ulimit", "nofile=1024:1024",
            IMAGE, "sleep", "infinity", timeout=120)
        if rc != 0:
            raise RuntimeError(f"sandbox start failed: {err.decode(errors='replace')[:500]}")
        self.started = True

    async def run(self, code: str) -> str:
        async with _sem():
            if not self.started:
                await self._start()
            self.calls += 1
            t = int(TIMEOUT)
            rc, out, err = await _run("docker", "exec", "-i", self.name, "timeout", "-s", "KILL", str(t), "python", "-",
                                      stdin=code.encode(), timeout=TIMEOUT + 30)
        o, e = out.decode(errors="replace"), err.decode(errors="replace")
        res = ""
        if o:
            res += _clip(o, MAX_OUT)
        if e:
            res += ("\n" if res else "") + "[stderr]\n" + _clip(e, MAX_OUT // 2)
        if rc in (137, -9):
            res += f"\n[execution killed: time limit {t}s or memory limit {MEM} exceeded]"
        elif rc != 0 and not e:
            res += f"\n[exit code {rc}]"
        return res.strip() or "[no output; use print() to see results]"

    async def close(self):
        if self.started:
            await _run("docker", "rm", "-f", self.name, timeout=60)
            self.started = False

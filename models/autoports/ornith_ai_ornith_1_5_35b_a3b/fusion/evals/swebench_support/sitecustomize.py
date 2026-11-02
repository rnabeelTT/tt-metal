"""Host-side patches for SWE-agent's SWE-ReX docker deployment (loaded via PYTHONPATH only
by the SWE-agent processes swe_driver.py starts; it does nothing elsewhere).

* No per-instance image build: SWE-ReX normally runs `docker build` to copy a standalone
  Python + swe-rex into every instance image (/root/python3.11). Builds through the shared
  default builder leave cache records that pin each instance's base layers on disk, and
  the fallback (pipx install at start) needs network. Instead the identical bundle, built
  once by the same recipe (python 3.11.8, --enable-shared, rpath /root/python3.11/lib), is
  mounted read-only at /root/python3.11 from swerex_py/<swe-rex version>/.
* SWE_AGENT_NONET=1: the container joins SWE_AGENT_NETWORK (an --internal docker network,
  no egress) and the runtime is reached on the container IP (no published ports there).
* SWE_AGENT_ENTRYPOINT_RESET=1: `--entrypoint ""` (SWE-bench Pro images start bash).
"""
import os

if os.environ.get("SWE_AGENT_PATCH", "1") == "1":
    try:
        import importlib.metadata as _md
        import subprocess as _sp

        import swerex.deployment.docker as _dd

        _VER = _md.version("swe-rex")
        _BUNDLE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "swerex_py", _VER, "python3.11")

        def _no_build(self):
            if not os.path.isdir(_BUNDLE):
                raise RuntimeError(f"missing swe-rex bundle {_BUNDLE}")
            args = list(self._config.docker_args)
            mount = f"{_BUNDLE}:{self._config.python_standalone_dir}/python3.11:ro"
            if mount not in args:
                args += ["-v", mount]
            if os.environ.get("SWE_AGENT_NONET") == "1" and "--network" not in args:
                args += ["--network", os.environ.get("SWE_AGENT_NETWORK", "swe-nonet")]
            if os.environ.get("SWE_AGENT_ENTRYPOINT_RESET") == "1" and "--entrypoint" not in args:
                args += ["--entrypoint", ""]
            self._config.docker_args = args
            return self._config.image

        _dd.DockerDeployment._build_image = _no_build

        if os.environ.get("SWE_AGENT_NONET") == "1":
            from swerex.runtime.remote import RemoteRuntime, RemoteRuntimeConfig

            _orig_from_config = RemoteRuntime.from_config.__func__

            _orig_wait = _dd.DockerDeployment._wait_until_alive

            async def _wait(self, timeout: float = 10.0):
                import asyncio
                ip = ""
                for _ in range(120):
                    ip = _sp.run(["docker", "inspect", "-f",
                                  "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}",
                                  self._container_name], capture_output=True, text=True).stdout.strip()
                    if ip:
                        break
                    await asyncio.sleep(0.5)
                if ip and self._runtime is not None:
                    cfg = self._runtime._config
                    self._runtime = RemoteRuntime.from_config(
                        RemoteRuntimeConfig(host=f"http://{ip}", port=8000, timeout=cfg.timeout,
                                            auth_token=cfg.auth_token))
                return await _orig_wait(self, timeout=timeout)

            _dd.DockerDeployment._wait_until_alive = _wait
    except Exception as _e:  # noqa: BLE001
        import sys

        print(f"[swebench_support/sitecustomize] patch not applied: {_e!r}", file=sys.stderr)

# SWE_AGENT_POST_STARTUP='["cmd", ...]': run after SWE-agent's repo reset, before the agent
# starts (run-batch has no CLI flag for env.post_startup_commands).
if os.environ.get("SWE_AGENT_POST_STARTUP"):
    try:
        import json as _json

        import sweagent.environment.swe_env as _se

        _cmds = _json.loads(os.environ["SWE_AGENT_POST_STARTUP"])
        _orig_start = _se.SWEEnv.start

        def _start(self):
            _orig_start(self)
            for c in _cmds:
                out = self.communicate(c, check="ignore", timeout=1800)
                self.logger.info(f"post-startup (swebench_support): {c[:60]}... -> {out[-200:]!r}")

        _se.SWEEnv.start = _start
    except Exception as _e:  # noqa: BLE001
        import sys

        print(f"[swebench_support/sitecustomize] post-startup patch not applied: {_e!r}", file=sys.stderr)

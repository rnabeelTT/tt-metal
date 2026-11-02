"""Run an OpenHands benchmarks entry point (swebench / swebenchmultilingual / swebenchpro
run_infer) with small host-side patches for this box. Nothing here changes the agent,
its prompts, tools or the patch extraction.

  SWE_DOCKER_NETWORK=swe-nonet  start the agent-server container on this docker network
                                (an --internal network = no egress; the card's "network
                                access is disabled") and talk to it via its container IP,
                                since ports are not published on internal networks.
  SWE_DOCKER_CPUS / SWE_DOCKER_MEM  per-container --cpus / --memory(-swap) limits.
  SWE_RUN_LABEL                 docker label swe-run=<label> on agent containers.
  Agent-server images are built with BUILDX_BUILDER (an isolated docker-container
  builder, so its cache can be pruned without touching other users' build cache); the
  local cache export is disabled (inline only) so no per-instance cache dirs pile up.
  The sphinx-doc "docutils<0.21 roman" wrapper layer is applied with docker run+commit
  instead of a docker build, for the same reason.

usage: oh_entry.py <swebench|swebenchmultilingual|swebenchpro> <run_infer args...>
"""
import json
import re
import os
import subprocess
import sys
import time

VARIANT = sys.argv.pop(1)

import openhands.agent_server.docker.build as sdk_build  # noqa: E402

sdk_build._active_buildx_driver = lambda: "docker"  # inline cache only, no local dir export
# Docker tags are limited to 128 chars after the colon; for SWE-bench Pro the SDK's
# full-SHA alias tag exceeds that and the whole build fails. Drop over-long aliases (the
# short-SHA tag the benchmark looks up is always kept).
_orig_all_tags = sdk_build.BuildOptions.all_tags.fget
sdk_build.BuildOptions.all_tags = property(
    lambda self: [t for t in _orig_all_tags(self) if len(t.rpartition(":")[2]) <= 128])

import openhands.workspace.docker.workspace as wsmod  # noqa: E402

_orig_exec = wsmod.execute_command
NET = os.environ.get("SWE_DOCKER_NETWORK", "")


def _exec(cmd, *a, **k):
    if isinstance(cmd, list) and cmd[:2] == ["docker", "run"]:
        extra = []
        if os.environ.get("SWE_DOCKER_CPUS"):
            extra += ["--cpus", os.environ["SWE_DOCKER_CPUS"]]
        if os.environ.get("SWE_DOCKER_MEM"):
            extra += ["--memory", os.environ["SWE_DOCKER_MEM"], "--memory-swap", os.environ["SWE_DOCKER_MEM"]]
        if os.environ.get("SWE_RUN_LABEL"):
            extra += ["--label", f"swe-run={os.environ['SWE_RUN_LABEL']}"]
        if NET and "--network" not in cmd:
            extra += ["--network", NET]
        cmd = cmd[:2] + extra + cmd[2:]
    return _orig_exec(cmd, *a, **k)


wsmod.execute_command = _exec
_orig_wait = wsmod.DockerWorkspace._wait_for_health


def _wait(self, *, timeout):
    if NET and self._container_id:
        for _ in range(60):
            ip = subprocess.run(
                ["docker", "inspect", "-f", "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}",
                 self._container_id], capture_output=True, text=True).stdout.strip()
            if ip:
                object.__setattr__(self, "host", f"http://{ip}:8000")
                break
            time.sleep(1)
    return _orig_wait(self, timeout=timeout)


wsmod.DockerWorkspace._wait_for_health = _wait


def _wrap_by_commit(agent_image, push=False):
    """Same effect as Dockerfile.swebench-deps, without a docker build."""
    from benchmarks.utils.build_utils import BuildOutput

    ep = subprocess.run(["docker", "image", "inspect", "-f", "{{json .Config.Entrypoint}}", agent_image],
                        capture_output=True, text=True).stdout.strip()
    name = f"swe-wrap-{os.getpid()}-{int(time.time())}"
    script = ("conda run -n testbed pip install --no-deps --force-reinstall 'docutils<0.21' 'roman' "
              "|| (source /opt/miniconda3/bin/activate testbed && pip install --no-deps --force-reinstall "
              "'docutils<0.21' 'roman'); pip install --no-deps --force-reinstall 'docutils<0.21' 'roman'")
    r = subprocess.run(["docker", "run", "--name", name, "-u", "root", "--entrypoint", "/bin/bash",
                        agent_image, "-c", script], capture_output=True, text=True)
    if r.returncode != 0:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
        return BuildOutput(base_image=agent_image, tags=[], error=f"wrap failed: {r.stderr[-2000:]}")
    changes = ["--change", "USER openhands"]
    if ep and ep != "null":
        changes += ["--change", f"ENTRYPOINT {ep}"]
    c = subprocess.run(["docker", "commit", *changes, name, agent_image], capture_output=True, text=True)
    subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    if c.returncode != 0:
        return BuildOutput(base_image=agent_image, tags=[], error=f"commit failed: {c.stderr[-2000:]}")
    return BuildOutput(base_image=agent_image, tags=[agent_image], error=None)


# --- anti-hacking git sanitization (SANITIZE_GIT=full|refs|off) -----------------------
# Applied right after the harness's own `cd <repo> ; git reset --hard` on the agent's copy
# of the repo, i.e. before the agent's first step.
#   refs: drop remotes, every branch/tag except HEAD, stash and reflog, then prune
#         unreachable objects (no commit after base_commit stays in the object store).
#   full: additionally squash history to one root commit with the current tree; the patch
#         is then diffed against that root ("git history removed").
SANITIZE = os.environ.get("SANITIZE_GIT", "off")
GIT_REFS = ("git remote | xargs -r -n1 git remote remove; "
            "git for-each-ref --format='%(refname)' refs/tags refs/remotes refs/stash refs/notes "
            "| xargs -r -n1 git update-ref -d; "
            "cur=$(git symbolic-ref -q HEAD || true); "
            "git for-each-ref --format='%(refname)' refs/heads | grep -vxF \"$cur\" | xargs -r -n1 git update-ref -d; "
            "git reflog expire --expire=now --all; git gc -q --prune=now")
GIT_FULL = ("git -c user.email=a@b -c user.name=base checkout -q --orphan __base && "
            "git -c user.email=a@b -c user.name=base commit -q --no-verify --allow-empty -m base && "
            "git for-each-ref --format='%(refname)' refs/heads | grep -vx refs/heads/__base "
            "| xargs -r -n1 git update-ref -d; git branch -q -m __base main; " + GIT_REFS)

from openhands.sdk.workspace.remote.base import RemoteWorkspace  # noqa: E402

_orig_ws_exec = RemoteWorkspace.execute_command


def _ws_exec(self, command, cwd=None, timeout=30.0):
    m = re.fullmatch(r"cd (\S+) ; git reset --hard", command.strip())
    if m and SANITIZE in ("refs", "full"):
        res = _orig_ws_exec(self, command, cwd=cwd, timeout=timeout)
        script = GIT_FULL if SANITIZE == "full" else GIT_REFS
        s = _orig_ws_exec(self, f"cd {m.group(1)} && ( {script} ) 2>&1 | tail -5; git log --oneline | wc -l",
                          timeout=1800)
        print(json.dumps({"sanitize_git": SANITIZE, "exit": s.exit_code, "out": (s.stdout or "")[-300:]}),
              flush=True)
        return res
    if SANITIZE == "full":
        # patch extraction diffs against base_commit, which no longer exists: use the root
        m2 = re.fullmatch(r"cd (\S+) ; git --no-pager diff --no-color (--cached )?([0-9a-f]{7,40})( HEAD)?", command.strip())
        if m2:
            command = (f"cd {m2.group(1)} ; git --no-pager diff --no-color {m2.group(2) or ''}"
                       f"$(git rev-list --max-parents=0 HEAD | tail -1){m2.group(4) or ''}")
    return _orig_ws_exec(self, command, cwd=cwd, timeout=timeout)


RemoteWorkspace.execute_command = _ws_exec

if VARIANT == "swebench":
    import benchmarks.swebench.run_infer as mod
elif VARIANT == "swebenchmultilingual":
    import benchmarks.swebenchmultilingual.run_infer as mod
elif VARIANT == "swebenchpro":
    import benchmarks.swebenchpro.run_infer as mod
else:
    sys.exit(f"unknown variant {VARIANT}")
if hasattr(mod, "wrap_image"):
    mod.wrap_image = _wrap_by_commit

print(json.dumps({"oh_entry": VARIANT, "network": NET, "builder": os.environ.get("BUILDX_BUILDER"),
                  "tag_prefix": os.environ.get("IMAGE_TAG_PREFIX")}), flush=True)
mod.main()

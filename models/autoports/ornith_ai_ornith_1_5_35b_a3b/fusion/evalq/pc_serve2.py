#!/usr/bin/env python3
"""pc_serve2.py <bundle> <container> <metal tree> [docker args...]: the bundle's published p300x2 launch command
(tt-model serve --print), renamed, with <metal tree>/models and the prefix-state-cache plugin branch mounted over the
image's, extra docker args before the image and $VARGS (e.g. --enable-prefix-caching) before --port."""
import os, shlex, subprocess, sys
bundle, name, tree, extra = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
cmd = subprocess.run(["tt-model", "serve", bundle, "--profile", os.environ.get("PROFILE", "p300x2"), "--print", "--no-update-check"],
                     capture_output=True, text=True, check=True).stdout.strip().splitlines()[-1]
a = shlex.split(cmd)
if "-d" not in a:
    a.insert(a.index("run") + 1, "-d")  # the printed command runs in the foreground
a[a.index("--name") + 1] = name
img = next(i for i, w in enumerate(a) if w.startswith("tt-model/") and ":" in w)
home = os.environ["HOME"]
mounts = ["--volume", f"{tree}/models:/opt/tt-metal/models:ro",
          "--volume", f"{home}/vllm-tt-plugin-pub/src/vllm_tt_plugin:/opt/tt-venv/lib/python3.12/site-packages/vllm_tt_plugin:ro"]
a = a[:img] + mounts + extra + a[img:]
p = a.index("--port")
a = a[:p] + shlex.split(os.environ.get("VARGS", "")) + a[p:]
subprocess.run(["docker", "rm", "-f", name], capture_output=True)
subprocess.run(a, check=True, stdout=subprocess.DEVNULL)
print(" ".join(shlex.quote(w) for w in a), file=sys.stderr)

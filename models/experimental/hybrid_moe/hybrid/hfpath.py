# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Paths into the local Hugging Face cache, so a package does not depend on where the host keeps it.

`snapshot(repo)` is the cached snapshot dir of `repo` at its pinned revision: tt-model exports the pins as
TT_MODEL_WEIGHTS_REVISION (the main weights, HF_MODEL) and TT_AUXILIARY_WEIGHTS (repo@rev,...); without a pin the
cache's refs/main. `resolve(v)` maps `hf://<org>/<repo>/<sub/path>` to `<snapshot>/<sub/path>` and leaves any other
value (a plain path) as it is. Used for HYB_LIB, HYB_DEV_EXPERTS, HYB_PLAN and the DFlash drafter.
"""

import glob
import os


def _pin(repo):
    for ref in os.environ.get("TT_AUXILIARY_WEIGHTS", "").split(","):
        name, _, rev = ref.strip().partition("@")
        if name == repo and rev:
            return rev
    if repo == os.environ.get("HF_MODEL") and os.environ.get("TT_MODEL_WEIGHTS_REVISION"):
        return os.environ["TT_MODEL_WEIGHTS_REVISION"]
    return None


def snapshot(repo):
    if os.path.isdir(repo):  # already a local snapshot dir (the generator may have resolved HF_MODEL to one)
        return repo
    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(repo, revision=_pin(repo), local_files_only=True)
    except Exception:  # an older cache layout without refs: any snapshot of the repo
        hub = os.environ.get("HF_HUB_CACHE") or os.path.join(os.environ.get("HF_HOME", "/hf"), "hub")
        found = sorted(glob.glob(os.path.join(hub, "models--" + repo.replace("/", "--"), "snapshots", "*")))
        if not found:
            raise
        return found[0]


def resolve(value):
    if not value or not value.startswith("hf://"):
        return value
    org, name, *rest = value[len("hf://") :].split("/")
    return os.path.join(snapshot(f"{org}/{name}"), *rest)

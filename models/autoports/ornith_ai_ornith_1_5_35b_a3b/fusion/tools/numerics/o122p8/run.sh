#!/bin/bash
# Serve Ornith-1.5-122B from the shipped image with the precision variant in policy.json (A/B only, not a package).
# Same env / command / mounts as `tt-model serve raahemnabeel/ornith-1.5-122b --profile p300x2`, separate tensor caches.
D=$(cd "$(dirname "$0")" && pwd)
C=${1:-tt-model-ornith-1.5-122b-p8-p300x2}
IMG=$(docker images tt-model/ornith-1.5-122b --format '{{.Repository}}:{{.Tag}}' | head -1)
mapfile -t CMD < <(python3 -c 'import json,sys;print("\n".join(json.load(open(sys.argv[1]))))' $D/cmd.json)
docker run -d --name $C --ipc host -u 1004:1004 --shm-size 64m \
  --device /dev/tenstorrent/0 --device /dev/tenstorrent/1 --device /dev/tenstorrent/2 --device /dev/tenstorrent/3 \
  -v /dev/hugepages-1G:/dev/hugepages-1G -p 20000:20000 --env-file $D/env.list \
  -v $HOME/.cache/huggingface:/hf -v $HOME/.cache/tt-model/ornith-1.5-122b/cache:/cache \
  -v $HOME/.cache/tt-model/ornith-1.5-122b-p8/weights:/weight-cache -v $HOME/.cache/tt-model/ornith-1.5-122b-p8/tensors:/tensor-cache \
  -v $D:/variant:ro --label org.tenstorrent.tt-model.variant=o122p8 $IMG "${CMD[@]}"
echo "variant $C image $IMG"

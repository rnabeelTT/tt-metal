#!/usr/bin/env bash
# Start (or restart) the pinned SearXNG container, bound to 127.0.0.1:${SEARXNG_PORT:-28010} only.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
IMG=searxng/searxng:2026.10.13-22c64b8d3
NAME=evals-searxng
PORT=${SEARXNG_PORT:-28010}
SECRET_FILE=$HERE/.secret; [ -f "$SECRET_FILE" ] || (umask 077; head -c 32 /dev/urandom | od -An -tx1 | tr -d ' \n' > "$SECRET_FILE")
docker rm -f $NAME >/dev/null 2>&1 || true
docker run -d --name $NAME --restart unless-stopped \
  -p 127.0.0.1:$PORT:8080 \
  -v "$HERE/settings.yml:/etc/searxng/settings.yml:ro" \
  -e SEARXNG_SECRET="$(cat "$SECRET_FILE")" \
  --cpus 2 --memory 1g \
  $IMG >/dev/null
for i in $(seq 60); do curl -sf "http://127.0.0.1:$PORT/healthz" >/dev/null 2>&1 && { echo "searxng up on 127.0.0.1:$PORT"; exit 0; }; sleep 1; done
echo "searxng did not come up"; docker logs --tail 50 $NAME; exit 1

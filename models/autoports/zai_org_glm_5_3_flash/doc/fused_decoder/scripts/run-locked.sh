#!/bin/bash
# Usage: run-locked.sh LOG cmd...   (one device job at a time on this host; tt-smi -r after unless NORESET=1)
# The lock file is shared with the earlier bring-ups on this host, so jobs from any workspace serialize.
LOG=$(realpath -m "$1"); shift
source ~/glm53-flash-bringup/env.sh >/dev/null
cd $TT_METAL_HOME
exec 9>~/qwen3-coder-next-bringup/.device.lock
flock 9
# reset first as well: a job killed mid-run (systemctl stop) leaves the board unreset for the next one
if [ "${NORESET:-0}" != 1 ]; then env -u PYTHONPATH -u VIRTUAL_ENV /home/raahem/.tenstorrent-venv/bin/tt-smi -r > /dev/null 2>&1; fi
# a process spinning in C++ can ignore timeout's SIGTERM and hold the lock for hours: escalate to SIGKILL after 60 s
if [ "$1" = timeout ] && [ "$2" != -k ]; then shift; set -- timeout -k 60 "$@"; fi
echo "=== start $(date -u) : $*" >> "$LOG"
"$@" >> "$LOG" 2>&1
rc=$?
echo "=== exit code: $rc $(date -u)" >> "$LOG"
if [ "${NORESET:-0}" != 1 ]; then env -u PYTHONPATH -u VIRTUAL_ENV /home/raahem/.tenstorrent-venv/bin/tt-smi -r >> "$LOG" 2>&1; fi
exit $rc

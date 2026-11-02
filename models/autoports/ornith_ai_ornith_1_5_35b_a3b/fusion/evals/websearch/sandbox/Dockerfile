# HLE-with-tools Python sandbox: Python 3.11.16 + the pinned requirements of CAIS hle docs/requirements-tools.txt.
# Run with --network none (no internet), CPU/memory/pids limits (see websearch/sandbox.py).
FROM python:3.11.16-slim-bookworm
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --only-binary=:all: -r /tmp/requirements.txt && rm /tmp/requirements.txt \
 && useradd -m -u 1000 sandbox && mkdir -p /work && chown sandbox /work
ENV MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
USER sandbox
WORKDIR /work
CMD ["sleep", "infinity"]

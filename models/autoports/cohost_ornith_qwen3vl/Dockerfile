# One co-host image: an Ornith-1.5 package image (its tt-metal build, vLLM plugin and model code, unchanged) plus
#   - the Qwen3-VL-Embedding-8B / Qwen3-VL-Reranker-8B model code (from the qwen3-vl-embedding-8b package image),
#     added as NEW packages under /opt/tt-metal/models (nothing the base ships is replaced),
#   - the co-host runtime with mixed groups (tt_kernel.cohost, branch raahem/cohost-mixed-groups),
#   - the adapter that runs both 8B models chip-sharded on the LLM's mesh (cohost_qwen3vl).
# Built by build.sh, which stages ./context and passes BASE.
ARG BASE
FROM ${BASE}
USER root
# the Qwen3-VL input pipeline's two extra wheels (pins from the qwen3-vl-embedding-8b package); --no-deps keeps the
# base image's torch / transformers / pillow untouched
# (the base image has no pip: the build host's uv installs them, and is not kept)
COPY context/uv /tmp/uv
RUN /tmp/uv pip install --python /opt/tt-venv/bin/python --no-cache --no-deps qwen-vl-utils==0.0.14 av==18.1.0 \
 && rm /tmp/uv
COPY context/models/ /opt/tt-metal/models/
COPY context/tt-cohost/ /opt/tt-cohost/
RUN chmod -R a+rX /opt/tt-cohost /opt/tt-metal/models
ENV PYTHONPATH=/opt/tt-cohost:/opt/tt-metal \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TT_MODEL_KIND=cohost
USER 1000:1000
WORKDIR /home/tt/work
# imports as the server's uid: the runtime, the adapter, both input pipelines (ttnn imports without a device)
RUN cd /opt/tt-metal && python -c "import tt_kernel.cohost.launch, cohost_qwen3vl.member, qwen_vl_utils, av; \
from models.autoports.qwen_qwen3_vl_embedding_8b.inputs import EmbedFormatter; \
from models.autoports.qwen_qwen3_vl_reranker_8b.tt.inputs import PairFormatter; \
from cohost_qwen3vl import split, engine; print('cohost image imports ok')"
CMD ["python", "-m", "tt_kernel.cohost.launch"]

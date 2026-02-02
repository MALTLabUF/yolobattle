FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

ARG APP_DIR=/opt/app
WORKDIR $APP_DIR

# Create user that can match the host IDs (supplied at build time)
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} appgroup \
 && useradd -m -u ${UID} -g ${GID} -s /bin/bash appuser

ENV DEBIAN_FRONTEND=noninteractive \
    PATH="/usr/local/cuda/bin:/opt/venv/bin:${PATH}" \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# Base deps
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
      build-essential git libopencv-dev file python3-venv python3-pip fio \
      ca-certificates wget gnupg lsb-release pkg-config; \
    rm -rf /var/lib/apt/lists/*

# Newer CMake
RUN set -eux; \
    wget -qO- https://apt.kitware.com/keys/kitware-archive-latest.asc \
      | gpg --dearmor -o /etc/apt/trusted.gpg.d/kitware.gpg; \
    echo "deb [signed-by=/etc/apt/trusted.gpg.d/kitware.gpg] https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main" \
      > /etc/apt/sources.list.d/kitware.list; \
    apt-get update; \
    apt-get install -y --no-install-recommends cmake; \
    rm -rf /var/lib/apt/lists/*

# Python venv + deps
RUN python3 -m venv /opt/venv \
 && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
 && /opt/venv/bin/pip install --no-cache-dir \
      pycocotools \
      imagesize \
      cloudmesh-common \
      PyYAML \
      Pillow \
      git+https://github.com/cloudmesh/cloudmesh-gpu.git

# Your Python package
COPY --chown=appuser:appgroup src/ ./src/
ENV PYTHONPATH=$APP_DIR/src

# Default in-image paths (binds will overlay these)
RUN mkdir -p /outputs /workspace/.cache/splits \
 && chown -R appuser:appgroup /outputs /workspace

# Optional: collaborative-friendly umask
RUN echo 'umask 0002' >> /etc/profile
ENV HOME=/home/appuser
USER appuser
WORKDIR /workspace

# Default command (can be overridden at runtime)
CMD ["bash", "-lc", "bash build_darknet.sh; python -u -m yolobattle.model_training.train --profile LegoGearsDarknet"]

FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

ARG UID=1000
ARG GID=1000
ARG TIANXIAOMO_REF=848e868675c83c7ee025676f9e5ebe122e9c659f
LABEL org.yolobattle.uid="${UID}" \
      org.yolobattle.gid="${GID}"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTORCH_YOLOV4_ROOT=/opt/pytorch-yolov4

RUN apt-get update \
 && apt-get install -y --no-install-recommends fio git libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /opt
RUN git init pytorch-yolov4 \
 && cd pytorch-yolov4 \
 && git remote add origin https://github.com/jpfleischer/pytorch-YOLOv4.git \
 && git fetch --depth 1 origin ${TIANXIAOMO_REF} \
 && git checkout --detach FETCH_HEAD

# The fork supplies modern PyTorch compatibility fixes.  Install only its
# runtime dependencies; do not reinstall the base image's CUDA-enabled torch.
RUN pip install --no-cache-dir \
      'numpy<2' \
      opencv-python-headless \
      tensorboardX \
      easydict \
      pycocotools \
      imagesize \
      cloudmesh-common \
      PyYAML \
      pillow \
      matplotlib \
      tqdm \
      git+https://github.com/cloudmesh/cloudmesh-gpu.git

COPY docker/pytorch_yolov4/smoke_test.py /opt/pytorch-yolov4/smoke_test.py
COPY src/ /opt/pytorch-yolov4/src/
ENV PYTHONPATH=/opt/pytorch-yolov4/src

WORKDIR /opt/pytorch-yolov4
RUN python smoke_test.py

# The training command writes to bind mounts.  Run it with the caller's UID/GID
# (provided as build arguments) so those artifacts remain writable on the host.
RUN groupadd -g ${GID} appgroup \
 && useradd -m -u ${UID} -g ${GID} -s /bin/bash appuser

ENV HOME=/home/appuser
USER appuser

CMD ["python", "-m", "yolobattle.model_training.train", "--profile", "LegoGearsPyTorchYOLOv4"]

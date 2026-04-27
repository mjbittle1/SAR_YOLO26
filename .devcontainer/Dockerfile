FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

# Install system libraries needed by OpenCV and other packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip (conda pip is fine, just make sure it's current)
RUN pip install --upgrade pip

# Install project dependencies.
# torch/torchvision/torchaudio are already in the base image — do NOT re-install.
# Install ultralytics without its torch dependency to avoid pip conflicts.
RUN pip install --no-deps ultralytics>=8.4.0

# Install the remaining ultralytics runtime deps (excluding torch)
RUN pip install \
    wandb>=0.17.0 \
    numpy \
    opencv-python-headless \
    pillow \
    pyyaml \
    tqdm \
    scipy \
    matplotlib \
    pandas \
    seaborn \
    psutil \
    py-cpuinfo \
    thop \
    requests \
    filelock \
    typing_extensions

WORKDIR /workspace

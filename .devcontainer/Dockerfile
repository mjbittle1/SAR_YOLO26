FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

# Install system libraries needed by OpenCV and other packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install ultralytics without torch dependency
RUN pip install --no-deps ultralytics>=8.4.0

# Install remaining dependencies
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

WORKDIR /workspaces/SAR_YOLO26

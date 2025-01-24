# Use the NVIDIA TensorRT image as the base
FROM nvcr.io/nvidia/tensorrt:24.04-py3

# Set environment variables
ENV LD_LIBRARY_PATH=/opt/tensorrt/lib:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/lib:$LD_LIBRARY_PATH

# Update system and install dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    git-all \
    ffmpeg \
    wget \
    gnupg && \
    # Add NVIDIA TensorRT repository
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    # Install the required libraries for TensorRT
    apt-get install -y \
    libnvinfer10 libnvinfer-plugin10 libnvonnxparsers10 \
    python3.11 python3.11-distutils python3.11-dev && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    apt-get clean

# Install pip for Python 3.11
RUN wget https://bootstrap.pypa.io/get-pip.py && python3.11 get-pip.py && rm get-pip.py

# Install Python dependencies
RUN python3.11 -m pip install --no-cache-dir \
    onnxruntime-gpu==1.20.0 \
    pandas \
    opencv-python-headless \
    tqdm \
    matplotlib \
    IPython \
    opencv-python \
    "numpy<2" \
    torchvision \
    requests

# Add NVIDIA repository for TensorRT
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update

# Copy local repository into the container
COPY . /workspace/NeuFlow_v2-Pytorch-Inference

# Set the working directory
WORKDIR /workspace/NeuFlow_v2-Pytorch-Inference

# Install the project as a package
RUN python3.11 -m pip install -U pip && python3.11 -m pip install -e .

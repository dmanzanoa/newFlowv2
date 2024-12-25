# Use the NVIDIA TensorRT image as the base
FROM nvcr.io/nvidia/tensorrt:24.10-py3

# Set environment variables
ENV LD_LIBRARY_PATH=/opt/tensorrt/lib:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/workspace/tensorrt/data

# Update system and install dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    git-all \
    ffmpeg \
    wget \
    gnupg && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && apt-get install -y \
    python3.11 python3.11-distutils python3.11-dev && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    apt-get clean

# Install pip for Python 3.11
RUN wget https://bootstrap.pypa.io/get-pip.py && python3.11 get-pip.py && rm get-pip.py

# Install PyTorch and other Python dependencies for Python 3.11
# Install other Python dependencies for Python 3.11
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

# Clone the repository
RUN git clone https://github.com/dmanzanoa/newFlowv2.git /workspace/NeuFlow_v2-Pytorch-Inference

# Set the working directory
WORKDIR /workspace/NeuFlow_v2-Pytorch-Inference


FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

EXPOSE 7860

ARG FACEFUSION_VERSION=3.1.1

ENV GRADIO_SERVER_NAME=0.0.0.0 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/python3.12/dist-packages/tensorrt_libs

WORKDIR /facefusion

# Install required system packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.12 \
        python-is-python3 \
        python3-pip \
        git \
        curl \
        ffmpeg \
        nano \
    && rm -rf /var/lib/apt/lists/*

# Install TensorRT from NVIDIA's PyPI
RUN python3 -m pip install --upgrade pip && \
    pip install tensorrt==10.6.0 --extra-index-url https://pypi.nvidia.com

# Clone and install FaceFusion
RUN git clone https://github.com/facefusion/facefusion.git --branch ${FACEFUSION_VERSION} --single-branch . && \
    python install.py --onnxruntime cuda --skip-conda

CMD ["python", "/facefusion/facefusion.py", "run"]

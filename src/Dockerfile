FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
RUN apt update && apt install -y \
    git \
    cmake \
    build-essential \
    ninja-build \
    pkg-config \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
RUN pip install --upgrade pip setuptools wheel
ENV CMAKE_ARGS="-DGGML_CUDA=ON"
ENV FORCE_CMAKE=1
ENV GGML_CUDA=1
ENV CUDA_ARCHITECTURES=89
RUN pip install --no-cache-dir llama-cpp-python==0.2.90
RUN pip install --no-cache-dir fastapi uvicorn[standard]
COPY app.py /workspace/app.py
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

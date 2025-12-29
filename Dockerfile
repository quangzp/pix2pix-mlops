FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

# 1️⃣ System + Python 3.10 (cố định version)
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-distutils \
    python3-pip \
    git \
    ca-certificates \
    install -y libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 2️⃣ Ép python = python3.10
RUN ln -sf /usr/bin/python3.10 /usr/bin/python

# 3️⃣ Upgrade pip (layer ổn định)
RUN pip install --upgrade pip setuptools wheel

# 4️⃣ Copy requirements trước để cache
COPY requirements.txt .

# 5️⃣ Install deps (layer nặng – cần cache)
RUN pip install --no-cache-dir -r requirements.txt

# 6️⃣ Copy source code (hay đổi)
COPY . .

ENTRYPOINT ["python", "mlops/modeling/train.py"]

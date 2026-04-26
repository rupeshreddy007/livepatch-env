FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip git && \
    rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python

WORKDIR /app

# Clone repo
RUN git clone https://github.com/rupeshreddy007/livepatch-env.git .

# Install Python deps
RUN pip install --no-cache-dir \
    unsloth trl transformers datasets matplotlib \
    flask numpy pyyaml sentencepiece accelerate bitsandbytes

# Run training
CMD ["python", "training/train.py"]

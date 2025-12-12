FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Install core packages only
RUN pip install --upgrade pip && \
    pip install --no-cache-dir requests tqdm pandas numpy

# Install torch (CPU) from PyTorch repo
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Try installing ML packages
RUN pip install --no-cache-dir transformers || true
RUN pip install --no-cache-dir rouge-score || true  
RUN pip install --no-cache-dir bert-score || true

COPY eval.py .
RUN mkdir -p /app/dataset /app/results

CMD ["python", "eval.py"]


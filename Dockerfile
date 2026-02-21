# ─────────────────────────────────────────────────────────────
#  ML Fin-Advisor — Production Serving Image
#  Used by Render.com (and Docker Compose via root context)
# ─────────────────────────────────────────────────────────────
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8

WORKDIR /app

# ── System dependencies ─────────────────────────────────────
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# ── Python dependencies ─────────────────────────────────────
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt && \
    pip uninstall -y nvidia-nccl-cu12 nvidia-cuda-runtime-cu12 \
       nvidia-cuda-cupti-cu12 nvidia-cuda-nvrtc-cu12 \
       nvidia-cublas-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 \
       nvidia-cusolver-cu12 nvidia-cusparse-cu12 nvidia-nvjitlink-cu12 \
       2>/dev/null || true

# ── Application code ────────────────────────────────────────
COPY src/ ./src/
COPY configs/ ./configs/

# ── Pre-trained model artefacts ─────────────────────────────
COPY models/serving/ ./models/serving/

EXPOSE 8000

# Render sets $PORT; fall back to 8000 for local use
CMD uvicorn src.serving.app:app \
    --host 0.0.0.0 \
    --port ${PORT:-8000} \
    --workers 1 \
    --log-level info \
    --timeout-keep-alive 65

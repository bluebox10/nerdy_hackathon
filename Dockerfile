FROM python:3.10-slim

WORKDIR /app

# Install dependencies for serving (CPU only, no torch, no CUDA)
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# Copy application and web static files
COPY serve/ ./serve/
COPY web/ ./web/

WORKDIR /app/serve

# Expose default port (7860 is default for Hugging Face Spaces; 8077 for local/custom)
EXPOSE 7860

ENV PORT=7860
ENV WW_RERANK_K=25
ENV WW_THREADS=1

CMD ["sh", "-c", "python -m uvicorn app:app --host 0.0.0.0 --port ${PORT}"]

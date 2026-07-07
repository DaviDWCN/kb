# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

# System dependencies for parsing libs (PyMuPDF, tesseract, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    tesseract-ocr \
    tesseract-ocr-chi-sim \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[parsing,reranking,providers,vector]" fastapi uvicorn

# Copy source code
COPY src/ src/
COPY scripts/ scripts/

# Default data & store dirs inside the container
ENV RAG_DATA_DIR=/app/data
ENV RAG_STORE_DIR=/app/.local_store

VOLUME ["/app/data", "/app/.local_store"]

ENTRYPOINT ["python", "scripts/run_rag.py"]
CMD ["server", "--host", "0.0.0.0", "--port", "8080"]
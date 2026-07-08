FROM python:3.12-slim

WORKDIR /app

# Install system dependencies for parsing libraries (e.g. OCR, docling)
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY scripts/ ./scripts/

# Install the package with all optional dependencies to ensure it works offline
RUN pip install --no-cache-dir ".[parsing,reranking,providers,server,vector]"

EXPOSE 8080

ENTRYPOINT ["python", "scripts/run_rag.py"]
CMD ["server", "--host", "0.0.0.0", "--port", "8080"]

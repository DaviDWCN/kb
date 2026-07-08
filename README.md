# Agentic-KB

**Agentic-KB** is a Python-based local RAG (Retrieval-Augmented Generation) system utilizing FAISS. It provides tools to parse various document formats, build a local vector database, retrieve relevant context, and answer user queries with large language models, either locally or through APIs.

## Features

- **Document Parsing**: Extract text from various formats (PDF, Markdown, etc.) including support for OCR and structured metadata.
- **Vector Database**: Build and query local FAISS-backed RAG stores.
- **Modular Pipeline**: Configurable chunking, contextualization, and embedding options.
- **Reranking & Expansion**: Optional query expansion, term mappings, and Cross-Encoder reranking for better search relevance.
- **API Server**: Run as a FastAPI local server for real-time query handling.

---

## Installation

### Local Installation

The project uses `pyproject.toml` and requires Python 3.11+.

To install the package with all optional dependencies (for parsing, reranking, API providers, and vector stores) so that it can run offline or fully-featured:

```bash
pip install '.[parsing,reranking,providers,vector]'
```

### Docker Setup

You can easily containerize the application using the provided `Dockerfile`. The Docker image comes with system-level dependencies pre-installed (e.g., `tesseract-ocr`, `libgl1`, `libglib2.0-0`), which are necessary for parsing libraries.

**Build the Docker image:**
```bash
docker build -t agentic-kb:latest .
```

**Run the Docker container:**
By default, running the container starts the RAG API server on port `8080`.
```bash
docker run -p 8080:8080 agentic-kb:latest
```

To run other commands using Docker, you can override the default command. Note that you may want to mount a local directory for data storage (`/app/data` and `/app/.local_store`):
```bash
# Example: Running the query command inside docker
docker run -v $(pwd)/data:/app/data -v $(pwd)/.local_store:/app/.local_store agentic-kb:latest python scripts/run_rag.py query "Your question here"
```

---

## Configuration / Continuous Integration

### Environment Variables (.env)
You can configure the behavior of the application by creating a `.env` file in the project root. Examples of configuration keys include:
- `RAG_CONTEXTUALIZATION`: `none`, `dummy`, `llm`
- `RAG_ANSWER_MODEL`: `echo`, `qwen`, `deepseek`
- `RAG_RERANKER`: Optional sentence-transformers cross-encoder model name.
- `RAG_EMBEDDING`: `hash` or `bge`

### CI/CD Pipeline (.gitlab-ci.yml)
The repository uses GitLab CI for continuous integration. The provided `.gitlab-ci.yml` defines the pipeline structure.

**How to write/use the `.gitlab-ci.yml` file:**
The file is structured to define jobs that run on GitLab Runners. Currently, it includes a `build-image` job in the `build` stage:

```yaml
stages:
  - build

variables:
  DOCKER_DRIVER: overlay2
  DOCKER_TLS_CERTDIR: ""

build-image:
  stage: build
  image: docker:24.0.5
  services:
    - docker:24.0.5-dind
  script:
    - docker login -u $CI_REGISTRY_USER -p $CI_REGISTRY_PASSWORD $CI_REGISTRY
    - docker build -t $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA -t $CI_REGISTRY_IMAGE:latest .
    - docker push $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA
    - docker push $CI_REGISTRY_IMAGE:latest
  rules:
    - if: $CI_COMMIT_BRANCH == "main"
```

- **stages**: Defines the order of execution.
- **variables**: Global settings for the jobs (like Docker configuration).
- **job (build-image)**: Defines the steps (`script`) to log in to the Docker registry, build the image, and push it. This job only runs on the `main` branch due to the `rules`.
You can add more stages like `test` or `deploy` to run unit tests automatically or deploy the service to your environment.

---

## Usage

The primary entry point is the `scripts/run_rag.py` script. It supports several subcommands to manage the RAG pipeline.

### 1. Convert
Parse all source files (in `data/`) and save them as intermediate Markdown files (in `data/.intermediate/`).
```bash
python scripts/run_rag.py convert --workers 4
```

### 2. Build
Parse intermediate Markdown files and build the vector database in `.local_store/`.
```bash
python scripts/run_rag.py build --embedding bge --workers 4
```

### 3. Query
Retrieve chunks from an existing vector database and generate an answer.
```bash
python scripts/run_rag.py query "How does chunking work?" --embedding bge --answer-model deepseek
```

### 4. Server
Start a local FastAPI RAG server.
```bash
python scripts/run_rag.py server --host 0.0.0.0 --port 8080 --embedding bge
```
Once the server is running, you can interact with it via HTTP POST at `/query` or view the simple UI at the root `/`.

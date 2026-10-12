# RAG Document Q&A Assistant — container image
# Build:  docker build -t rag-doc-qa .
# Run:    docker run -p 8501:8501 rag-doc-qa
#
# First container start will download the embedding model (~80MB) and
# the flan-t5-base generation model (~1GB) from huggingface.co, so the
# container needs outbound internet access at least once. Models are
# cached inside the container's HuggingFace cache after that.

FROM python:3.11-slim

WORKDIR /app

# System deps needed by faiss / torch wheels at runtime
# (curl is needed by the HEALTHCHECK below; slim images don't ship it)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Install the CPU-only torch wheel first (~200MB instead of ~2GB with CUDA
# libraries); the next line then sees torch as already satisfied.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY app/ ./app/
COPY data/raw/ ./data/raw/
COPY data/test_questions.json ./data/test_questions.json

# Pre-build the FAISS index at image build time so the container starts
# fast. (The app can also build it itself on first start if it's missing.)
# This step downloads the embedding model, so it requires network
# access during `docker build`.
RUN cd src && python chunking.py && python build_index.py

EXPOSE 8501

HEALTHCHECK --start-period=60s --interval=30s CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app/streamlit_app.py", \
            "--server.port=8501", "--server.address=0.0.0.0"]

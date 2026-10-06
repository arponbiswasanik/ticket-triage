FROM python:3.11-slim

WORKDIR /app

# Copy ONLY requirements first: cheap layer, everything below caches on it
COPY requirements.txt .

# Install CPU-only torch, pinned to our frozen version.
# (PyPI's default linux wheel drags in CUDA deps -> ~6GB image.
#  PEP 440: torch==2.x.y is satisfied by 2.x.y+cpu, so requirements.txt
#  won't re-download the CUDA build afterwards.)
RUN TORCH_VERSION=$(grep -i '^torch==' requirements.txt | cut -d= -f3) && \
    if [ -z "$TORCH_VERSION" ]; then \
        echo "ERROR: no torch== pin found in requirements.txt"; exit 1; \
    fi && \
    pip install --no-cache-dir "torch==${TORCH_VERSION}" \
      --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

# App code + trained artifacts (build context = local files, not git)
COPY src/ src/
COPY evaluation/ evaluation/
COPY models/ models/
COPY data/processed/retrieval_corpus.parquet data/processed/
COPY data/vectorstore/ data/vectorstore/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
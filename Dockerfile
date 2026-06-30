# =============================================================================
#  GraphRAG Research Assistant — Multi-Stage Dockerfile (Backend)
#
#  Stage 1 (builder): Install all Python dependencies into an isolated prefix.
#  Stage 2 (runtime): Lean image with non-root user; only copies what's needed.
#
#  Security notes:
#    - No secrets in any layer (all credentials injected at runtime via env_file)
#    - Non-root UID 1000 in the final image
#    - Only the virtual-env prefix and app source are copied to runtime stage
# =============================================================================

# ── Stage 1: builder ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

# Install OS-level build tools required by some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        libmagic1 \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy only the dependency manifest first (maximises Docker layer caching)
COPY backend/pyproject.toml ./pyproject.toml
# Create a minimal package stub so pip can resolve the project root
RUN mkdir -p backend && touch backend/__init__.py

# Create a virtual environment and install all dependencies into it
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir \
        fastapi>=0.115.0 \
        "uvicorn[standard]>=0.30.0" \
        pydantic>=2.7.0 \
        pydantic-settings>=2.3.0 \
        pymupdf>=1.24.0 \
        "mistralai>=1.0.0" \
        "neo4j>=5.20.0" \
        "pinecone>=4.0.0" \
        "sentence-transformers>=3.0.0" \
        "langchain>=0.3.0" \
        "langchain-community>=0.3.0" \
        "langchain-core>=0.3.0" \
        "python-multipart>=0.0.9" \
        "python-magic>=0.4.27" \
        "slowapi>=0.1.9" \
        "structlog>=24.0.0" \
        "python-dotenv>=1.0.0"

# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="GraphRAG Research Assistant"
LABEL org.opencontainers.image.description="FastAPI backend for the GraphRAG Research Assistant"
LABEL org.opencontainers.image.source="https://github.com/your-org/research-agent"

# Install only the minimal runtime OS libraries (libmagic for python-magic)
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user and group (UID/GID 1000)
RUN groupadd --gid 1000 appgroup \
    && useradd --uid 1000 --gid appgroup --shell /bin/bash --create-home appuser

WORKDIR /app

# Copy the virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy only the application source code (no secrets, no .env files)
COPY backend/ ./backend/

# Fix ownership so the non-root user can write logs / temp files
RUN chown -R appuser:appgroup /app

# Activate the virtual environment for all subsequent commands
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Prevent sentence-transformers from attempting to write to root-owned dirs
    HF_HOME=/app/.cache/huggingface \
    TORCH_HOME=/app/.cache/torch

# Pre-create cache directories with correct ownership
RUN mkdir -p /app/.cache/huggingface /app/.cache/torch \
    && chown -R appuser:appgroup /app/.cache

EXPOSE 8000

# Health check — hits the /health endpoint every 30 s, 3 retries, 10 s timeout
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Drop to non-root user
USER appuser

# Start the server with 4 Uvicorn workers (adjust via UVICORN_WORKERS env var)
CMD ["uvicorn", "backend.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "4", \
     "--log-config", "backend/logging_config.json"]

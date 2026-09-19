# NexusCRM AI — FastAPI Backend Dockerfile
#
# WHY MULTI-STAGE BUILD:
#   Stage 1 (builder): Install dependencies (large, includes build tools)
#   Stage 2 (runtime): Copy only the installed packages (smaller final image)
#   Result: ~600MB → ~350MB final image; no build tools in production

# ── Stage 1: Builder ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency specs
COPY pyproject.toml .

# Install dependencies into a prefix directory (not system Python)
RUN pip install --upgrade pip && \
    pip install --prefix=/install \
        fastapi \
        uvicorn[standard] \
        sqlalchemy[asyncio] \
        asyncpg \
        psycopg2-binary \
        alembic \
        motor \
        pydantic-settings \
        python-jose[cryptography] \
        passlib[bcrypt] \
        python-multipart \
        langchain \
        langchain-openai \
        langchain-community \
        langchain-huggingface \
        langchain-chroma \
        langchain-pinecone \
        langgraph \
        langsmith \
        openai \
        faiss-cpu \
        chromadb \
        tiktoken \
        sentence-transformers \
        httpx

# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Install only runtime system deps
RUN apt-get update && apt-get install -y \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY backend/ ./backend/
COPY migrations/ ./migrations/
COPY knowledge_base/ ./knowledge_base/
COPY scripts/ ./scripts/
COPY alembic.ini .

# Create non-root user (security best practice)
RUN useradd -m -u 1001 nexuscrm && \
    chown -R nexuscrm:nexuscrm /app
USER nexuscrm

# Create directories for persistent data
RUN mkdir -p /app/data/faiss /app/data/chroma /app/evaluation_results

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Environment
ENV PYTHONPATH=/app/backend
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

# WHY uvicorn --host 0.0.0.0:
#   Default 127.0.0.1 is only accessible within the container.
#   0.0.0.0 exposes to the container network (mapped by Docker to host port).
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--loop", "uvloop", \
     "--http", "h11"]

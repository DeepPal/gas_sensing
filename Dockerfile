# =============================================================================
# Au-MIP LSPR Gas Sensing Platform — Multi-Stage Dockerfile
# =============================================================================
#
# Build targets
# -------------
#   api          FastAPI inference server on port 8000  (default)
#   dashboard    Streamlit analytics dashboard on port 8501
#   test         pytest + mypy CI runner (exits 0/1)
#
# Examples
# --------
#   docker build --target api       -t gas-api:latest .
#   docker build --target dashboard -t gas-dashboard:latest .
#   docker build --target test      -t gas-test:latest .
#
#   docker compose up                       # api + dashboard
#   docker compose --profile test run test  # CI

# =============================================================================
# Stage 1 — base: OS packages shared by all targets
# =============================================================================
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Keeps matplotlib from trying to open a display
    MPLBACKEND=Agg

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# =============================================================================
# Stage 2 — deps: pip install (cached layer — only reruns when pyproject changes)
# =============================================================================
FROM base AS deps

# Copy only dependency manifests to maximise cache hits
COPY pyproject.toml ./

# Install core + all optional extras (no torch — too large for API/dashboard)
# torch is an optional extra; containers that need CNN inference should add:
#   RUN pip install "torch>=2.0.0" --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -e ".[all]"

# =============================================================================
# Stage 3 — source: add application code on top of frozen deps
# =============================================================================
FROM deps AS source

# Copy source packages (ordered from least-changed to most-changed)
COPY config/       ./config/
COPY src/          ./src/
COPY gas_analysis/ ./gas_analysis/
COPY dashboard/    ./dashboard/
COPY serve.py      ./serve.py
COPY run.py        ./run.py

# Pre-compile to .pyc for faster cold starts
RUN python -m compileall -q src/ config/ 2>/dev/null || true

# Runtime output directory (override with a named volume in production)
RUN mkdir -p /app/output /app/data

# =============================================================================
# Stage 4 — api: FastAPI inference server (DEFAULT target)
# =============================================================================
FROM source AS api

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# uvicorn with 2 workers; override via WORKERS env var
CMD uvicorn src.api.main:app \
        --host 0.0.0.0 \
        --port 8000 \
        --workers ${WORKERS:-2} \
        --log-level ${LOG_LEVEL:-info}

# =============================================================================
# Stage 5 — dashboard: Streamlit analytics UI
# =============================================================================
FROM source AS dashboard

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "dashboard/app.py", \
     "--server.address", "0.0.0.0", \
     "--server.port", "8501", \
     "--server.headless", "true", \
     "--browser.gatherUsageStats", "false"]

# =============================================================================
# Stage 6 — test: pytest + mypy CI runner
# =============================================================================
FROM source AS test

# Install dev/test extras on top (ruff, pytest-cov, mypy, etc.)
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e ".[dev]"

COPY tests/ ./tests/

# Run pytest then mypy; non-zero exit on any failure
CMD ["sh", "-c", \
     "pytest -q --tb=short tests/ && mypy src --no-site-packages --ignore-missing-imports"]

# =============================================================================
# SpectraAgent — Universal Agentic Spectroscopy Platform
# Multi-Stage Dockerfile
# =============================================================================
#
# Build targets
# -------------
#   spectraagent  SpectraAgent FastAPI + acquisition server on port 8765 (default)
#   dashboard     Streamlit scientific analysis dashboard on port 8501
#   test          pytest + mypy CI runner (exits 0/1)
#
# Examples
# --------
#   docker build --target spectraagent  -t spectraagent:latest .
#   docker build --target dashboard     -t spectraagent-dashboard:latest .
#   docker build --target test          -t spectraagent-test:latest .
#
#   docker compose up                           # spectraagent + dashboard
#   docker compose --profile test run test      # CI

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

# Create non-root user for runtime stages
RUN groupadd -r appuser && useradd --no-create-home -r -g appuser appuser

# =============================================================================
# Stage 2 — source: application tree required for editable installs
# =============================================================================
FROM base AS source

# Editable installs require project metadata and source tree to be present.
COPY README.md       ./README.md
COPY pyproject.toml  ./pyproject.toml
COPY .streamlit/     ./.streamlit/
COPY config/         ./config/
COPY src/            ./src/
COPY gas_analysis/   ./gas_analysis/
COPY spectraagent/   ./spectraagent/
COPY dashboard/      ./dashboard/
COPY spectraagent.toml ./spectraagent.toml
COPY serve.py        ./serve.py
COPY run.py          ./run.py

# Pre-compile to .pyc for faster cold starts
RUN python -m compileall -q src/ config/ spectraagent/ 2>/dev/null || true

# Runtime output directories (override with named volumes in production)
RUN mkdir -p /app/output/sessions /app/data && chown -R appuser:appuser /app/output /app/data

# =============================================================================
# Stage 3 — runtime-base: install runtime dependencies once for all targets
# =============================================================================
FROM source AS runtime-base

RUN pip install --no-cache-dir -e ".[all]"

# =============================================================================
# Stage 4 — spectraagent: primary FastAPI + acquisition server (DEFAULT target)
# =============================================================================
FROM runtime-base AS spectraagent

EXPOSE 8765

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8765/api/health || exit 1

USER appuser

# Simulation mode by default in Docker; mount real hardware via device passthrough
CMD ["python", "-m", "spectraagent", "start", "--simulate", "--host", "0.0.0.0", "--no-browser"]

# =============================================================================
# Stage 5 — api: standalone inference API on port 8000
# =============================================================================
FROM runtime-base AS api

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

USER appuser

CMD ["python", "serve.py", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

# =============================================================================
# Stage 6 — dashboard: Streamlit scientific analysis UI
# =============================================================================
FROM runtime-base AS dashboard

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

USER appuser

CMD ["streamlit", "run", "dashboard/app.py", \
     "--server.address", "0.0.0.0", \
     "--server.port", "8501", \
     "--server.headless", "true", \
     "--browser.gatherUsageStats", "false"]

# =============================================================================
# Stage 7 — test: pytest + mypy CI runner
# =============================================================================
FROM runtime-base AS test

# Install dev/test extras (ruff, pytest-cov, mypy, etc.)
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e ".[dev]"

COPY tests/ ./tests/

CMD ["sh", "-c", \
     "pytest -q --tb=short -m 'not reliability' tests/ && mypy src --ignore-missing-imports --follow-imports=skip"]

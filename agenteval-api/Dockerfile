# Build from the repository root so the local package is installable:
#   docker build -f agenteval-api/Dockerfile -t agenteval-api .
#
# Render free-tier settings:
#   Dockerfile Path  = agenteval-api/Dockerfile
#   Docker Context   = .   (repository root)

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

WORKDIR /app

# Install build tooling once
RUN pip install --upgrade pip

# Copy only what is needed to install nishanttyagi-agenteval from this monorepo.
# (Avoid copying node_modules / dashboard build artifacts.)
COPY pyproject.toml README.md LICENSE requirements.txt /tmp/agenteval-src/
COPY __init__.py __main__.py cli.py compat.py agents.yaml /tmp/agenteval-src/
COPY adapters /tmp/agenteval-src/adapters
COPY core /tmp/agenteval-src/core
COPY dashboard /tmp/agenteval-src/dashboard
COPY evaluators /tmp/agenteval-src/evaluators
COPY sql /tmp/agenteval-src/sql
COPY templates /tmp/agenteval-src/templates
COPY baselines /tmp/agenteval-src/baselines
COPY tests/golden /tmp/agenteval-src/tests/golden

RUN pip install /tmp/agenteval-src \
    && pip install \
        "fastapi==0.116.1" \
        "uvicorn[standard]==0.35.0" \
        "python-multipart==0.0.20" \
    && rm -rf /tmp/agenteval-src

COPY agenteval-api/main.py /app/main.py

RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Render injects $PORT; default 8000 for local runs.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]

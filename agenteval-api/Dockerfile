# Build with context = agenteval-api/ (Render Root Directory = agenteval-api):
#   cd agenteval-api && docker build -t agenteval-api .
#
# Render free-tier:
#   Root Directory   = agenteval-api
#   Dockerfile Path  = Dockerfile
#   Docker Context   = .   (relative to Root Directory)

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

WORKDIR /app

RUN pip install --upgrade pip

# Install API deps + published/git AgentEval package (see requirements.txt).
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY main.py .

RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Render injects $PORT; default 8000 for local runs.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]

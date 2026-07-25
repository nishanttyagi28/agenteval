# AgentEval SQL API

Minimal FastAPI service that exposes AgentEval’s SQL safety scanner over HTTP.

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness check |
| `POST` | `/scan` | Scan a JSONL corpus or JSON query list |
| `POST` | `/diff` | Behavioural diff of baseline vs candidate |

Stateless — no auth, no database. Portfolio / demo deployment only.

## Local run

```bash
# from repo root (editable package already installed)
cd agenteval-api
python -m pip install -r requirements.txt
python -m pip install httpx pytest   # for tests
uvicorn main:app --reload --port 8000
```

## Example

```bash
curl -s http://localhost:8000/health

curl -s -X POST http://localhost:8000/scan \
  -H "Content-Type: application/json" \
  -d '{"queries":[{"id":"q1","sql":"INSERT INTO t VALUES (1)"}],"dialect":"postgres"}'
```

## Deploy (Render free tier)

See the step-by-step instructions in the parent deployment notes, or use
`render.yaml` (Blueprint) / the `Dockerfile` with context = repository root.

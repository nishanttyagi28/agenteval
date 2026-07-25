# AgentEval SQL Dashboard

Minimal Vite + React + Tailwind dark UI for the SQL safety scanner demo.

## Dev

```bash
# terminal 1 — API (from repo root)
cd agenteval-api
uvicorn main:app --reload --port 8000

# terminal 2 — dashboard
cd agenteval-dashboard
cp .env.example .env   # optional; defaults to http://localhost:8000
npm install
npm run dev
```

`VITE_API_URL` controls the backend base URL (no trailing slash).

## Production build

```bash
npm run build   # outputs dist/
```

## Vercel

Import the GitHub repo, set **Root Directory** to `agenteval-dashboard`, add env
`VITE_API_URL` = your Render API URL, then deploy. See parent deployment steps.

# Operator — Breath & Wellness API

Backend service for the OPERATOR iOS app (FastAPI + MongoDB Atlas).

## Quick deploy to Render

1. Push this repo to GitHub.
2. In Render: **New** → **Blueprint** → select this repo.
3. Add these secret env vars in the Render dashboard:
   - `MONGO_URL`        (your Atlas SRV connection string)
   - `DB_NAME`          (e.g. `ember_breath`)
   - `JWT_SECRET`       (use the same value as the previous deployment to keep existing tokens valid)
   - `ADMIN_EMAIL`
   - `ADMIN_PASSWORD`
   - `EMERGENT_LLM_KEY`
4. Wait for the green deploy. Service URL appears at the top of the page.
5. Smoke-test: `curl https://YOUR-SERVICE.onrender.com/api/health` → should return `{"status":"ok","db":"ok"}`.

## Local development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then edit
uvicorn server:app --reload --port 8001
```

## Files

| File              | Purpose                                                  |
|-------------------|----------------------------------------------------------|
| `server.py`       | FastAPI app (all routes, MongoDB integration)            |
| `render.yaml`     | Render Blueprint — defines the web service + disk        |
| `Procfile`        | Fallback start command                                   |
| `runtime.txt`     | Python version pin                                       |
| `requirements.txt`| Production deps only                                     |
| `static/`         | Created at runtime; on Render mapped to persistent disk  |

## Health check

`GET /api/health` returns 200 when MongoDB is reachable, 503 otherwise.
Render uses this to auto-restart unhealthy instances.

## Migration script

See `scripts/migrate_to_atlas.py` for moving data from a local MongoDB
to MongoDB Atlas (run once, idempotent).

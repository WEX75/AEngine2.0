# AEngine2.0

A local-first pipeline for finding notable moments in video, generating
short-form clip scripts, and rendering vertical clips. It includes a local
FastAPI/React control panel for VOD ingestion and manual-window rendering.

## Requirements

- Python 3.12 or newer
- FFmpeg
- Node.js and npm (for the web control panel)
- Ollama with the configured language model for LLM-assisted features

## Setup

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Install the frontend dependencies:

```bash
cd webui/frontend
npm ci
```

Edit `settings.yaml` for your model, encoder, and media paths. Keep API
credentials in `.env`; do not commit that file. Source videos and generated
media belong in the ignored `input/` and `output/` directories and are not
included in this repository.

## Run the control panel

In one terminal, from the repository root:

```bash
source .venv/bin/activate
uvicorn webui.backend.main:app --reload --host 127.0.0.1 --port 8000
```

In another terminal:

```bash
cd webui/frontend
npm run dev
```

Open the Vite URL shown in the terminal (usually `http://localhost:5173`).
The backend is intended for local use only; keep it bound to `127.0.0.1`.

## Tests

Run the offline unit tests from the repository root:

```bash
source .venv/bin/activate
python -m pytest
```

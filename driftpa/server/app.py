"""
DriftPA — server/app.py
========================
FastAPI application entry point for OpenEnv 0.2.1.

create_fastapi_app takes a *factory callable* (not an instance), so we pass
DriftPAEnvironment (the class itself), which is callable and creates a new
environment on each WebSocket session.
"""

import sys
import os

# Ensure the driftpa package is importable when run from any working directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from openenv.core import create_fastapi_app
from server.environment import DriftPAEnvironment
from models import DriftPAAction, DriftPAObservation

# create_fastapi_app requires the class (factory), not an instance.
# OpenEnv manages session state internally.
app = create_fastapi_app(
    DriftPAEnvironment,
    DriftPAAction,
    DriftPAObservation,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <html><head><title>DriftPA</title>
    <style>body{font-family:monospace;background:#0f1117;color:#e2e8f0;padding:40px;max-width:600px;}
    h1{color:#7c3aed;}a{color:#93c5fd;}code{background:#1e2130;padding:2px 6px;border-radius:4px;}</style>
    </head><body>
    <h1>⚡ DriftPA Environment</h1>
    <p>Personal assistant RL environment — schema drift, time pressure, irreversible actions.</p>
    <h3>API Endpoints</h3>
    <p><code>GET /health</code> — health check</p>
    <p><code>POST /reset</code> — start episode <code>{"seed": 0}</code></p>
    <p><code>POST /step</code> — take action <code>{"action": {"tool_name": "list_tools", "payload": {}}}</code></p>
    <p><code>GET /state</code> — current state</p>
    <h3>Links</h3>
    <p><a href="/health">Health Check</a> &nbsp;|&nbsp;
    <a href="https://github.com/rajashekarcs2023/openenv-RL">GitHub</a></p>
    </body></html>
    """

# ---------------------------------------------------------------------------
# Run with: uvicorn driftpa.server.app:app --host 0.0.0.0 --port 8000
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

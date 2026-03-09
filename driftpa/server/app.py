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
    <html><head><title>DriftPA — Personal Assistant RL Environment</title>
    <style>
      *{box-sizing:border-box;margin:0;padding:0;}
      body{font-family:'Courier New',monospace;background:#0f1117;color:#e2e8f0;padding:40px 60px;max-width:860px;}
      h1{color:#7c3aed;font-size:1.6em;margin-bottom:6px;}
      h2{color:#94a3b8;font-size:0.75em;text-transform:uppercase;letter-spacing:2px;margin:28px 0 10px;}
      p{color:#cbd5e1;line-height:1.7;font-size:0.9em;}
      a{color:#93c5fd;text-decoration:none;}
      a:hover{text-decoration:underline;}
      code{background:#1e2130;padding:2px 8px;border-radius:4px;font-size:0.85em;color:#a78bfa;}
      .tagline{color:#64748b;font-size:0.85em;margin-bottom:24px;}
      .stats{display:flex;gap:24px;margin:16px 0;}
      .stat{background:#1e2130;border:1px solid #2d3748;border-radius:8px;padding:14px 20px;text-align:center;}
      .stat-val{font-size:1.5em;font-weight:bold;display:block;}
      .stat-val.neg{color:#f87171;}
      .stat-val.pos{color:#34d399;}
      .stat-val.neu{color:#7c3aed;}
      .stat-label{color:#64748b;font-size:0.7em;margin-top:4px;display:block;}
      .mechanic{background:#1e2130;border-left:3px solid #7c3aed;border-radius:4px;padding:10px 14px;margin-bottom:8px;}
      .mechanic strong{color:#a78bfa;}
      .endpoint{display:flex;align-items:baseline;gap:12px;padding:8px 0;border-bottom:1px solid #1e2130;}
      .endpoint:last-child{border:none;}
      .method{background:#1e3a5f;color:#93c5fd;padding:2px 8px;border-radius:4px;font-size:0.75em;min-width:45px;text-align:center;}
      .method.get{background:#1a3320;color:#6ee7b7;}
      .links{display:flex;gap:16px;margin-top:8px;}
      .btn{background:#1e2130;border:1px solid #374151;color:#e2e8f0;padding:8px 16px;border-radius:6px;font-family:monospace;font-size:0.8em;}
      .btn:hover{border-color:#7c3aed;color:#a78bfa;}
    </style>
    </head><body>

    <h1>⚡ DriftPA</h1>
    <p class="tagline">OpenEnv 0.2.1 · Track 3.2 Personalized Tasks · Patronus AI Partner Track</p>

    <p>Trains LLM agents to act as personal executive assistants in a world that <strong>changes mid-task</strong>. The agent manages cascading real-life conflicts — calendar clashes, urgent emails, dinner bookings, ride scheduling — while four failure modes fire without warning.</p>

    <div class="stats">
      <div class="stat"><span class="stat-val neg">−9.55</span><span class="stat-label">Untrained mean reward</span></div>
      <div class="stat"><span class="stat-val pos">+22.0</span><span class="stat-label">Optimal episode reward</span></div>
      <div class="stat"><span class="stat-val neu">31 pts</span><span class="stat-label">Training gap</span></div>
      <div class="stat"><span class="stat-val neu">24,000</span><span class="stat-label">GRPO rollouts (H100)</span></div>
    </div>

    <h2>Four Novel Mechanics</h2>
    <div class="mechanic"><strong>Schema Drift</strong> — API field names change mid-episode (<code>party_size → guests</code>). Agent must call <code>list_tools()</code> to discover new schema or get penalised.</div>
    <div class="mechanic"><strong>Time Pressure</strong> — Tasks expire if not resolved within N steps. Boss email expires at step 4. Missing it triggers a cascade.</div>
    <div class="mechanic"><strong>Irreversible Actions</strong> — <code>reply_message</code>, <code>book_restaurant</code>, <code>book_ride</code> cannot be undone. Wrong commits create cascade failures.</div>
    <div class="mechanic"><strong>Policy Drift</strong> — Cancellation window tightens from 2hr → 4hr post-drift. Late cancellation = policy violation.</div>

    <h2>API Endpoints</h2>
    <div class="endpoint"><span class="method get">GET</span><code>/health</code><span style="color:#64748b;font-size:0.8em;">— liveness check</span></div>
    <div class="endpoint"><span class="method">POST</span><code>/reset</code><span style="color:#64748b;font-size:0.8em;">— start episode &nbsp;<code>{"seed": 0}</code></span></div>
    <div class="endpoint"><span class="method">POST</span><code>/step</code><span style="color:#64748b;font-size:0.8em;">— take action &nbsp;<code>{"action": {"tool_name": "list_tools", "payload": {}}}</code></span></div>
    <div class="endpoint"><span class="method get">GET</span><code>/state</code><span style="color:#64748b;font-size:0.8em;">— episode metadata</span></div>

    <h2>Quick Start</h2>
    <p style="background:#1e2130;padding:12px 16px;border-radius:6px;font-size:0.8em;line-height:2;">
      curl <a href="/health">/health</a><br>
      curl -X POST /reset -H "Content-Type: application/json" -d '{"seed": 0}'<br>
      curl -X POST /step -H "Content-Type: application/json" -d '{"action": {"tool_name": "list_tools", "payload": {}}}'
    </p>

    <h2>Links</h2>
    <div class="links">
      <a class="btn" href="/health" target="_blank">✓ Health Check</a>
      <a class="btn" href="https://github.com/rajashekarcs2023/openenv-RL" target="_blank">⌥ GitHub</a>
      <a class="btn" href="https://colab.research.google.com/github/rajashekarcs2023/openenv-RL/blob/main/driftpa/colab_training.ipynb" target="_blank">▶ Training Notebook</a>
    </div>

    </body></html>
    """

# ---------------------------------------------------------------------------
# Run with: uvicorn driftpa.server.app:app --host 0.0.0.0 --port 8000
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

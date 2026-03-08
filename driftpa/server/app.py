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

from openenv.core import create_fastapi_app
from server.environment import DriftPAEnvironment
from models import DriftPAAction, DriftPAObservation

# create_fastapi_app signature:
#   create_fastapi_app(env_factory, action_cls, observation_cls) -> FastAPI
app = create_fastapi_app(
    DriftPAEnvironment,   # factory callable — called per session
    DriftPAAction,
    DriftPAObservation,
)

# ---------------------------------------------------------------------------
# Run with: uvicorn driftpa.server.app:app --host 0.0.0.0 --port 8000
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

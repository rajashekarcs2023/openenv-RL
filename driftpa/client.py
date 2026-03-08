"""
DriftPA — client.py
====================
Typed client for connecting to a deployed DriftPA environment server.

Usage:
    from driftpa.client import DriftPAClient
    from driftpa.models import DriftPAAction

    with DriftPAClient(base_url="ws://localhost:8000") as env:
        result = env.reset(seed=0)
        while not result.done:
            action = DriftPAAction(tool_name="list_tools", payload={})
            result = env.step(action)
            print(result.observation)

    # Or connect to a deployed HuggingFace Space:
    env = DriftPAClient.from_env("rajashekarcs2023/driftpa")
"""

from typing import Any

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from driftpa.models import DriftPAAction, DriftPAObservation, DriftPAState


class DriftPAClient(EnvClient[DriftPAAction, DriftPAObservation, DriftPAState]):
    """
    Typed WebSocket client for DriftPA.

    Connects to a running DriftPA server (local or HuggingFace Spaces)
    and exposes reset() / step() with full type annotations.
    """

    def _step_payload(self, action: DriftPAAction) -> dict[str, Any]:
        """Serialize a DriftPAAction for the server."""
        return action.model_dump()

    def _parse_result(self, payload: dict[str, Any]) -> StepResult[DriftPAObservation]:
        """Deserialize server response into a typed StepResult."""
        obs = DriftPAObservation(**payload.get("observation", {}))
        return StepResult(
            observation=obs,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict[str, Any]) -> DriftPAState:
        """Deserialize server state response."""
        return DriftPAState(**payload)

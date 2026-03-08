"""
DriftPA — models.py
====================
Pydantic models for the DriftPA OpenEnv environment.

Inherits from openenv.core base classes (Pydantic BaseModel, NOT dataclasses).
  - Action   : base already has `metadata: dict`
  - Observation: base already has `done: bool`, `reward`, `metadata: dict`
  - State    : base already has `episode_id: str | None`, `step_count: int`
"""

from typing import Any
from pydantic import Field
from openenv.core import Action, Observation, State


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class DriftPAAction(Action):
    """A single tool call the agent wants to make.

    tool_name : one of the 11 allowed actions (e.g. 'read_message', 'finish')
    payload   : key-value arguments for that tool (may be empty dict)
    """
    tool_name: str = Field(..., description="Name of the tool to call")
    payload: dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments for the tool call"
    )


# ---------------------------------------------------------------------------
# Observation  (base already provides: done, reward, metadata)
# ---------------------------------------------------------------------------

class DriftPAObservation(Observation):
    """Everything the agent can see at each step.

    inbox            : list of message dicts (id, sender, subject, preview,
                       priority, expires_at_step)
    calendar         : list of event dicts (id, title/event_name, begins_at/
                       start_time, end_time, confirmed) — field names change
                       on schema drift!
    available_tools  : current tool schemas; agent MUST re-read after drift
    policy           : current policy rules (reimbursement, cancellation, etc.)
    time_step        : current step number (max 15)
    urgent_expiring  : task ids expiring within the next 2 steps
    last_action_result: human-readable result / error from the previous action
    """
    inbox: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Unread and open messages"
    )
    calendar: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Current calendar events"
    )
    available_tools: dict[str, Any] = Field(
        default_factory=dict,
        description="Current tool schemas — may change after schema drift"
    )
    policy: dict[str, Any] = Field(
        default_factory=dict,
        description="Current policy rules — may change after policy drift"
    )
    time_step: int = Field(default=0, description="Current step (max 15)")
    urgent_expiring: list[str] = Field(
        default_factory=list,
        description="Task IDs expiring within 2 steps"
    )
    last_action_result: str = Field(
        default="",
        description="Result / error message from the last action"
    )


# ---------------------------------------------------------------------------
# State  (base already provides: episode_id, step_count)
# ---------------------------------------------------------------------------

class DriftPAState(State):
    """Internal episode metadata (not shown to agent directly).

    schema_version            : increments on each schema drift (starts at 0)
    drift_log                 : list of {step, api, field_old, field_new}
    policy_log                : list of {step, rule, old_value, new_value}
    irreversible_actions_taken: list of {step, tool_name, payload}
    total_reward              : cumulative reward so far
    tasks_resolved            : count of tasks successfully resolved
    tasks_expired             : count of tasks that expired without resolution
    cascade_failures          : count of conflicts caused by irreversible actions
    """
    schema_version: int = Field(default=0, description="Increments on each drift")
    drift_log: list[dict[str, Any]] = Field(default_factory=list)
    policy_log: list[dict[str, Any]] = Field(default_factory=list)
    irreversible_actions_taken: list[dict[str, Any]] = Field(default_factory=list)
    total_reward: float = Field(default=0.0)
    tasks_resolved: int = Field(default=0)
    tasks_expired: int = Field(default=0)
    cascade_failures: int = Field(default=0)

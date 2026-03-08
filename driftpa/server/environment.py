"""
DriftPA — server/environment.py
================================
Core RL environment implementing OpenEnv 0.2.1 interface.

Implements three novel mechanics:
  1. Schema Drift     — API field names change mid-episode (steps 3-6)
  2. Time Pressure    — tasks expire if not resolved within N steps
  3. Irreversible Actions — reply/book/cancel cannot be undone

OpenEnv interface contract:
  - reset(seed, episode_id, **kwargs) -> DriftPAObservation
  - step(action, timeout_s, **kwargs) -> DriftPAObservation
  - state (property)                 -> DriftPAState
"""

import uuid
import random
from typing import Optional

from openenv.core import Environment

from driftpa.models import DriftPAAction, DriftPAObservation, DriftPAState
from driftpa.scenarios.generator import generate_scenario

# ---------------------------------------------------------------------------
# Reward constants — NEVER change without flagging
# ---------------------------------------------------------------------------
R_URGENT_RESOLVED   = +3.0
R_NORMAL_RESOLVED   = +2.0
R_CORRECT_DRIFT     = +2.0
R_FEASIBLE_SCHEDULE = +2.0
R_POLICY_QUERIED    = +1.0
R_CORRECT_FORMAT    = +1.0
R_EPISODE_BONUS     = +5.0

R_IRREVERSIBLE_CONFLICT = -4.0
R_DOUBLE_BOOKING        = -3.0
R_POLICY_VIOLATION      = -3.0
R_TASK_EXPIRED          = -3.0
R_STALE_SCHEMA          = -2.0
R_MISSED_URGENT         = -2.0
R_REDUNDANT_CALL        = -1.0
R_CONTRADICTORY_REPLY   = -1.0
R_CATASTROPHIC          = -5.0

MAX_STEPS = 15

# All 11 allowed tool names
VALID_TOOLS = {
    "read_message", "read_calendar", "reply_message",
    "move_event", "book_restaurant", "cancel_booking",
    "confirm_booking", "book_ride", "query_policy",
    "list_tools", "finish",
}

# Tools that are IRREVERSIBLE
IRREVERSIBLE_TOOLS = {
    "reply_message", "book_restaurant",
    "cancel_booking", "confirm_booking", "book_ride",
}

# Maps drift-spec API name → tool name in _active_schema
API_TO_TOOL = {
    "restaurant": "book_restaurant",
    "ride":       "book_ride",
    "calendar":   "move_event",
    "email":      "reply_message",
    "booking":    "confirm_booking",
}


class DriftPAEnvironment(Environment[DriftPAAction, DriftPAObservation, DriftPAState]):
    """Personal assistant RL environment with schema drift and time pressure."""

    SUPPORTS_CONCURRENT_SESSIONS = False  # single-session environment

    def __init__(self):
        super().__init__()
        # Episode state — initialised in reset()
        self._episode_id: str = ""
        self._step_count: int = 0
        self._scenario: dict = {}          # loaded from generator
        self._schema_version: int = 0
        self._drift_log: list = []
        self._policy_log: list = []
        self._irreversible_taken: list = []
        self._total_reward: float = 0.0
        self._tasks_resolved: int = 0
        self._tasks_expired: int = 0
        self._cascade_failures: int = 0

        # Mutable episode data
        self._inbox: list = []             # list of message dicts
        self._calendar: list = []          # list of event dicts
        self._bookings: dict = {}          # booking_id -> booking dict
        self._resolved_tasks: set = set()  # message ids resolved
        self._policy_queried_this_step: bool = False
        self._list_tools_called: bool = False
        self._last_action_result: str = ""
        self._done: bool = False

        # Current schema (mutates on drift)
        self._active_schema: dict = {}
        self._active_policy: dict = {}

        # Drift schedule from scenario
        self._schema_drift_steps: dict = {}  # step -> list of drift specs
        self._policy_drift_step: int = -1
        self._policy_drift_spec: dict = {}

        # Tracking
        self._read_messages: set = set()
        self._calendar_read: bool = False
        self._reply_targets: dict = {}     # message_id -> reply text
        self._ride_booked: bool = False
        self._restaurant_booked: bool = False
        self._conflicts: list = []
        # Fix 2: query_policy +1 only once per drift event (prevents farming)
        self._policy_reward_claimed: bool = False

    # -----------------------------------------------------------------------
    # OpenEnv interface — reset
    # -----------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs,
    ) -> DriftPAObservation:
        """Start a new episode. Loads a random (or seeded) scenario."""

        if seed is not None:
            random.seed(seed)

        self._episode_id = episode_id or str(uuid.uuid4())
        self._step_count = 0
        self._schema_version = 0
        self._drift_log = []
        self._policy_log = []
        self._irreversible_taken = []
        self._total_reward = 0.0
        self._tasks_resolved = 0
        self._tasks_expired = 0
        self._cascade_failures = 0
        self._done = False
        self._last_action_result = "Episode started. Check inbox and calendar."
        self._read_messages = set()
        self._calendar_read = False
        self._reply_targets = {}
        self._ride_booked = False
        self._restaurant_booked = False
        self._conflicts = []
        self._policy_queried_this_step = False
        self._list_tools_called = False
        self._bookings = {}
        self._resolved_tasks = set()
        self._policy_reward_claimed = False

        # Load scenario
        self._scenario = generate_scenario(seed)
        self._inbox = [dict(m) for m in self._scenario["inbox"]]
        self._calendar = [dict(e) for e in self._scenario["calendar"]]
        self._active_policy = dict(self._scenario["initial_policy"])
        self._schema_drift_steps = self._scenario["schema_drift_steps"]
        self._policy_drift_step = self._scenario["policy_drift_step"]
        self._policy_drift_spec = self._scenario["policy_drift_spec"]

        # Build initial tool schemas (pre-drift)
        self._active_schema = self._build_tool_schemas()

        return self._make_observation()

    # -----------------------------------------------------------------------
    # OpenEnv interface — step
    # -----------------------------------------------------------------------

    def step(
        self,
        action: DriftPAAction,
        timeout_s: Optional[float] = None,
        **kwargs,
    ) -> DriftPAObservation:
        """Execute one action and advance the environment by one step."""

        if self._done:
            return self._make_observation(
                result="Episode is already done. Call reset() to start a new one."
            )

        self._step_count += 1
        self._policy_queried_this_step = False
        step_reward = 0.0

        # ---- Validate tool name ----
        tool = action.tool_name
        if tool not in VALID_TOOLS:
            step_reward += R_REDUNDANT_CALL
            result = f"ERROR: Unknown tool '{tool}'. Call list_tools() to see valid tools."
            self._total_reward += step_reward
            return self._make_observation(result=result, reward=step_reward)

        # ---- Apply schema drift (if scheduled for this step) ----
        if self._step_count in self._schema_drift_steps:
            self._apply_schema_drift(self._step_count)

        # ---- Apply policy drift (if scheduled for this step) ----
        if self._step_count == self._policy_drift_step:
            self._apply_policy_drift()

        # ---- Check task expiry before executing action ----
        expired = self._check_expiry()
        for msg_id in expired:
            step_reward += R_TASK_EXPIRED
            self._tasks_expired += 1
            self._drift_log  # just accessing to keep linter happy
        if expired:
            result_prefix = f"[EXPIRED: {expired}] "
        else:
            result_prefix = ""

        # ---- Dispatch to action handler ----
        action_reward, action_result = self._dispatch(tool, action.payload)
        step_reward += action_reward
        result = result_prefix + action_result

        # ---- Check for episode termination ----
        if tool == "finish" or self._step_count >= MAX_STEPS:
            episode_bonus, bonus_msg = self._compute_episode_bonus()
            step_reward += episode_bonus
            result += bonus_msg
            self._done = True

        self._total_reward += step_reward
        return self._make_observation(result=result, reward=step_reward)

    # -----------------------------------------------------------------------
    # OpenEnv interface — state (property)
    # -----------------------------------------------------------------------

    @property
    def state(self) -> DriftPAState:
        return DriftPAState(
            episode_id=self._episode_id,
            step_count=self._step_count,
            schema_version=self._schema_version,
            drift_log=list(self._drift_log),
            policy_log=list(self._policy_log),
            irreversible_actions_taken=list(self._irreversible_taken),
            total_reward=self._total_reward,
            tasks_resolved=self._tasks_resolved,
            tasks_expired=self._tasks_expired,
            cascade_failures=self._cascade_failures,
        )

    # -----------------------------------------------------------------------
    # Action handlers
    # -----------------------------------------------------------------------

    def _dispatch(self, tool: str, payload: dict) -> tuple[float, str]:
        """Route to the correct handler. Returns (reward_delta, result_string)."""
        handlers = {
            "read_message":    self._h_read_message,
            "read_calendar":   self._h_read_calendar,
            "reply_message":   self._h_reply_message,
            "move_event":      self._h_move_event,
            "book_restaurant": self._h_book_restaurant,
            "cancel_booking":  self._h_cancel_booking,
            "confirm_booking": self._h_confirm_booking,
            "book_ride":       self._h_book_ride,
            "query_policy":    self._h_query_policy,
            "list_tools":      self._h_list_tools,
            "finish":          self._h_finish,
        }
        return handlers[tool](payload)

    def _h_read_message(self, payload: dict) -> tuple[float, str]:
        msg_id = payload.get("message_id", "")
        msg = next((m for m in self._inbox if m["id"] == msg_id), None)
        if msg is None:
            return R_REDUNDANT_CALL, f"ERROR: Message '{msg_id}' not found."
        if msg_id in self._read_messages:
            return R_REDUNDANT_CALL, f"Already read message '{msg_id}'."
        self._read_messages.add(msg_id)
        return 0.0, (
            f"FROM: {msg['sender']}\n"
            f"SUBJECT: {msg['subject']}\n"
            f"PRIORITY: {msg['priority']}\n"
            f"EXPIRES AT STEP: {msg.get('expires_at_step', 'none')}\n"
            f"BODY: {msg['body']}"
        )

    def _h_read_calendar(self, payload: dict) -> tuple[float, str]:
        if self._calendar_read and not self._calendar:
            return R_REDUNDANT_CALL, "Calendar already read and empty."
        self._calendar_read = True
        if not self._calendar:
            return 0.0, "Calendar is empty."
        lines = ["CALENDAR:"]
        for e in self._calendar:
            # Support both pre- and post-drift field names
            title     = e.get("title") or e.get("event_name", "Unknown")
            begins_at = e.get("begins_at") or e.get("start_time", "?")
            lines.append(
                f"  [{e['id']}] {title} @ {begins_at} — {e.get('end_time','?')} "
                f"({'confirmed' if e.get('confirmed') else 'tentative'})"
            )
        return 0.0, "\n".join(lines)

    def _h_reply_message(self, payload: dict) -> tuple[float, str]:
        """IRREVERSIBLE — sends a reply to a message."""
        # Check schema drift: after email drift, field is 'respond_to' not 'reply_to'
        msg_id = payload.get("message_id", "")
        text   = payload.get("text", "")

        reward, schema_note = self._check_schema_payload("email", payload)

        msg = next((m for m in self._inbox if m["id"] == msg_id), None)
        if msg is None:
            return R_CATASTROPHIC, f"CATASTROPHIC: Replied to unknown message '{msg_id}'."

        if msg_id in self._reply_targets:
            return R_CONTRADICTORY_REPLY, (
                f"WARNING: Sent contradictory reply to '{msg_id}'. "
                f"Previous reply already sent. {schema_note}"
            )

        self._reply_targets[msg_id] = text
        self._irreversible_taken.append({
            "step": self._step_count,
            "tool_name": "reply_message",
            "payload": payload,
        })

        # Mark as resolved if this message had a task
        task_reward = self._resolve_task(msg_id, msg.get("priority", "normal"))
        return reward + task_reward + R_CORRECT_FORMAT, (
            f"SENT reply to {msg['sender']}: '{text}'. "
            f"IRREVERSIBLE. {schema_note}"
        )

    def _h_move_event(self, payload: dict) -> tuple[float, str]:
        event_id = payload.get("event_id", "")
        # Support both pre- and post-drift field names for new_time
        new_time = payload.get("new_time") or payload.get("begins_at", "")

        event = next((e for e in self._calendar if e["id"] == event_id), None)
        if event is None:
            return R_REDUNDANT_CALL, f"ERROR: Event '{event_id}' not found."

        # Check for double-booking
        conflict = self._has_time_conflict(event_id, new_time)
        if conflict:
            self._cascade_failures += 1
            return R_DOUBLE_BOOKING, (
                f"ERROR: Moving '{event_id}' to {new_time} conflicts with "
                f"'{conflict}'. Double-booking penalty applied."
            )

        old_time = event.get("begins_at") or event.get("start_time", "?")
        # Update using current schema field name
        if "begins_at" in event:
            event["begins_at"] = new_time
        else:
            event["start_time"] = new_time

        return 0.0, (
            f"Moved '{event_id}' from {old_time} to {new_time}. "
            f"Schedule updated."
        )

    def _h_book_restaurant(self, payload: dict) -> tuple[float, str]:
        """IRREVERSIBLE — books a restaurant table."""
        reward, schema_note = self._check_schema_payload("restaurant", payload)

        if self._restaurant_booked:
            self._cascade_failures += 1
            return R_DOUBLE_BOOKING, (
                f"ERROR: Restaurant already booked. Double-booking penalty. {schema_note}"
            )

        # Policy check 1: guest limit
        guests = payload.get("guests") or payload.get("party_size", 0)
        try:
            guests = int(guests)
        except (TypeError, ValueError):
            guests = 0

        if guests > self._active_policy.get("guest_limit", 6):
            return R_POLICY_VIOLATION, (
                f"POLICY VIOLATION: guest_limit is "
                f"{self._active_policy['guest_limit']}, requested {guests}. {schema_note}"
            )

        # Fix 1: Policy check 2 — max_reimbursement enforcement.
        # The restaurant cost per person is known from the scenario (visible in
        # the friend's message body). Agent must query_policy() to know the limit.
        cost_per_person = self._scenario.get("restaurant_cost_per_person", 0)
        total_cost = cost_per_person * max(guests, 1)
        max_reimb = self._active_policy.get("max_reimbursement", 999)
        if cost_per_person > 0 and total_cost > max_reimb * max(guests, 1):
            # Cost per person exceeds per-person reimbursement cap
            return R_POLICY_VIOLATION, (
                f"POLICY VIOLATION: dinner costs ${cost_per_person}/person but "
                f"max_reimbursement is ${max_reimb}/person. "
                f"Call query_policy() before booking. {schema_note}"
            )

        booking_id = f"rest_{self._step_count}"
        self._bookings[booking_id] = {
            "type": "restaurant", "payload": payload, "confirmed": False
        }
        self._restaurant_booked = True
        self._irreversible_taken.append({
            "step": self._step_count,
            "tool_name": "book_restaurant",
            "payload": payload,
        })
        return reward + R_NORMAL_RESOLVED, (
            f"Restaurant booked (id={booking_id}). "
            f"Awaiting confirmation. IRREVERSIBLE. {schema_note}"
        )

    def _h_cancel_booking(self, payload: dict) -> tuple[float, str]:
        """IRREVERSIBLE — cancels an existing booking."""
        booking_id = payload.get("booking_id", "")
        booking = self._bookings.get(booking_id)
        if booking is None:
            return R_CATASTROPHIC, (
                f"CATASTROPHIC: Tried to cancel unknown booking '{booking_id}'."
            )

        # Policy check: cancellation_window enforcement.
        # The window string ("2hr", "4hr") maps to a step-based deadline.
        # Each step ≈ 30 min of the ~7.5hr evening; must cancel at least
        # (hours × 2) steps before the end of the episode.
        # Example: "2hr" → cutoff step 11; "4hr" (post-drift) → cutoff step 7.
        window_str = self._active_policy.get("cancellation_window", "2hr")
        try:
            hours = float(window_str.replace("hr", "").strip())
        except (ValueError, AttributeError):
            hours = 2.0
        steps_needed = int(hours * 2)
        cutoff_step = MAX_STEPS - steps_needed
        if self._step_count > cutoff_step:
            return R_POLICY_VIOLATION, (
                f"POLICY VIOLATION: Too late to cancel. cancellation_window is "
                f"{window_str} — must cancel by step {cutoff_step}, "
                f"but already at step {self._step_count}."
            )

        booking["cancelled"] = True
        self._irreversible_taken.append({
            "step": self._step_count,
            "tool_name": "cancel_booking",
            "payload": payload,
        })
        return 0.0, f"Booking '{booking_id}' cancelled. IRREVERSIBLE."

    def _h_confirm_booking(self, payload: dict) -> tuple[float, str]:
        """IRREVERSIBLE — confirms a pending booking."""
        booking_id = payload.get("booking_id", "")
        booking = self._bookings.get(booking_id)
        if booking is None:
            return R_CATASTROPHIC, (
                f"CATASTROPHIC: Tried to confirm unknown booking '{booking_id}'."
            )
        if booking.get("confirmed"):
            return R_REDUNDANT_CALL, f"Booking '{booking_id}' already confirmed."

        booking["confirmed"] = True
        self._irreversible_taken.append({
            "step": self._step_count,
            "tool_name": "confirm_booking",
            "payload": payload,
        })
        return 0.0, f"Booking '{booking_id}' confirmed. IRREVERSIBLE."

    def _h_book_ride(self, payload: dict) -> tuple[float, str]:
        """IRREVERSIBLE — books a ride."""
        reward, schema_note = self._check_schema_payload("ride", payload)

        if self._ride_booked:
            self._cascade_failures += 1
            return R_DOUBLE_BOOKING, (
                f"ERROR: Ride already booked. Double-booking penalty. {schema_note}"
            )

        booking_id = f"ride_{self._step_count}"
        self._bookings[booking_id] = {
            "type": "ride", "payload": payload, "confirmed": True
        }
        self._ride_booked = True
        self._irreversible_taken.append({
            "step": self._step_count,
            "tool_name": "book_ride",
            "payload": payload,
        })
        return reward + R_NORMAL_RESOLVED, (
            f"Ride booked (id={booking_id}). IRREVERSIBLE. {schema_note}"
        )

    def _h_query_policy(self, payload: dict) -> tuple[float, str]:
        self._policy_queried_this_step = True
        lines = ["CURRENT POLICIES:"]
        for k, v in self._active_policy.items():
            lines.append(f"  {k}: {v}")

        # Fix 2: +1 only the first time after a policy drift (prevents farming)
        if not self._policy_reward_claimed and self._policy_log:
            self._policy_reward_claimed = True
            reward = R_POLICY_QUERIED
            lines.append("  [+1 reward: queried policy after drift]")
        elif not self._policy_log:
            # Before any drift: still reward once to teach the behaviour
            if not self._policy_reward_claimed:
                self._policy_reward_claimed = True
                reward = R_POLICY_QUERIED
            else:
                reward = 0.0
        else:
            reward = 0.0  # already claimed this drift window

        return reward, "\n".join(lines)

    def _h_list_tools(self, payload: dict) -> tuple[float, str]:
        self._list_tools_called = True
        lines = ["AVAILABLE TOOLS (current schema):"]
        for tool_name, schema in self._active_schema.items():
            params = ", ".join(
                f"{p}:{t}" for p, t in schema.get("params", {}).items()
            )
            irrev = " [IRREVERSIBLE]" if tool_name in IRREVERSIBLE_TOOLS else ""
            lines.append(f"  {tool_name}({params}){irrev}")
        return 0.0, "\n".join(lines)

    def _h_finish(self, payload: dict) -> tuple[float, str]:
        # Episode bonus computed in step() after calling this handler
        return 0.0, "Agent called finish(). Computing final score..."

    # -----------------------------------------------------------------------
    # Drift helpers
    # -----------------------------------------------------------------------

    def _apply_schema_drift(self, step: int):
        """Apply all schema drifts scheduled for this step."""
        for spec in self._schema_drift_steps.get(step, []):
            api      = spec["api"]
            old_key  = spec["old"]
            new_key  = spec["new"]

            # Translate API name → tool name for schema lookup
            tool_key = API_TO_TOOL.get(api, api)
            if tool_key in self._active_schema:
                params = self._active_schema[tool_key].get("params", {})
                if old_key in params:
                    params[new_key] = params.pop(old_key)

            # Rename field in live calendar events (for calendar drifts)
            if api == "calendar":
                for event in self._calendar:
                    if old_key in event:
                        event[new_key] = event.pop(old_key)

            self._schema_version += 1
            self._drift_log.append({
                "step": step, "api": api,
                "field_old": old_key, "field_new": new_key,
            })

    def _apply_policy_drift(self):
        """Apply the scheduled policy drift."""
        for rule, new_val in self._policy_drift_spec.items():
            old_val = self._active_policy.get(rule)
            self._active_policy[rule] = new_val
            self._policy_log.append({
                "step": self._step_count, "rule": rule,
                "old_value": old_val, "new_value": new_val,
            })
        # Fix 2: reset so agent can earn +1 again by querying the new policy
        self._policy_reward_claimed = False

    def _check_schema_payload(self, api: str, payload: dict) -> tuple[float, str]:
        """Returns (reward, note) — +2 for correct post-drift fields, -2 for stale."""
        if self._schema_version == 0:
            return 0.0, ""  # no drift yet, any field is fine

        tool_key = API_TO_TOOL.get(api, api)
        current_params = self._active_schema.get(tool_key, {}).get("params", {})
        if not current_params:
            return 0.0, ""

        # Check if any payload key is a stale (old) field name
        drifted_apis = {d["api"] for d in self._drift_log}
        if api not in drifted_apis:
            return 0.0, ""

        # Get old→new mapping for this api
        old_to_new = {
            d["field_old"]: d["field_new"]
            for d in self._drift_log if d["api"] == api
        }
        stale_keys = [k for k in payload if k in old_to_new]
        correct_keys = [k for k in payload if k in current_params]

        if stale_keys:
            return R_STALE_SCHEMA, (
                f"DRIFT ERROR: Used stale field(s) {stale_keys} for '{api}'. "
                f"Correct fields: {list(current_params.keys())}. Stale schema penalty."
            )
        if correct_keys:
            return R_CORRECT_DRIFT, f"[Correct post-drift schema used for '{api}']"
        return 0.0, ""

    # -----------------------------------------------------------------------
    # Task expiry
    # -----------------------------------------------------------------------

    def _check_expiry(self) -> list[str]:
        """Mark tasks expired if their deadline has passed. Returns list of expired ids."""
        expired = []
        for msg in self._inbox:
            exp = msg.get("expires_at_step")
            if exp is None:
                continue
            if self._step_count > exp and msg["id"] not in self._resolved_tasks:
                expired.append(msg["id"])
                self._resolved_tasks.add(msg["id"])  # prevent double-counting
        return expired

    def _resolve_task(self, msg_id: str, priority: str) -> float:
        """Award task resolution reward. Returns reward delta."""
        if msg_id in self._resolved_tasks:
            return R_REDUNDANT_CALL  # already resolved

        msg = next((m for m in self._inbox if m["id"] == msg_id), None)
        exp = msg.get("expires_at_step") if msg else None

        self._resolved_tasks.add(msg_id)
        self._tasks_resolved += 1

        if priority == "critical" or priority == "urgent":
            if exp is not None and self._step_count <= exp:
                return R_URGENT_RESOLVED  # resolved before expiry
            return R_MISSED_URGENT  # too late
        return R_NORMAL_RESOLVED

    # -----------------------------------------------------------------------
    # Calendar conflict detection
    # -----------------------------------------------------------------------

    def _has_time_conflict(self, moving_id: str, new_time: str) -> Optional[str]:
        """Check if placing event at new_time conflicts with another event.
        Returns conflicting event id or None."""
        for event in self._calendar:
            if event["id"] == moving_id:
                continue
            existing_time = event.get("begins_at") or event.get("start_time", "")
            if existing_time == new_time:
                return event["id"]
        return None

    # -----------------------------------------------------------------------
    # Episode bonus
    # -----------------------------------------------------------------------

    def _compute_episode_bonus(self) -> tuple[float, str]:
        """+5 if ALL tasks resolved, +2 for clean schedule, -4 per calendar conflict if irreversible actions taken."""
        total_tasks = len([m for m in self._inbox if m.get("expires_at_step")])
        bonus = 0.0
        notes = []

        # Detect remaining calendar conflicts (same time slot occupied twice)
        times = [
            e.get("begins_at") or e.get("start_time")
            for e in self._calendar
        ]
        non_null_times = [t for t in times if t]
        conflict_count = len(non_null_times) - len(set(non_null_times))

        # Fix 3: Cascade penalty — irreversible commitments made but calendar
        # still in conflict = agent caused a real-world cascade failure.
        # This is the mechanical core of the "irreversible actions" mechanic.
        if conflict_count > 0 and self._irreversible_taken:
            cascade_penalty = R_IRREVERSIBLE_CONFLICT * conflict_count
            bonus += cascade_penalty
            self._cascade_failures += conflict_count
            notes.append(
                f"{cascade_penalty:.0f} CASCADE: {conflict_count} calendar conflict(s) "
                f"after irreversible actions"
            )

        # +2 for feasible schedule (zero conflicts at end)
        if conflict_count == 0:
            bonus += R_FEASIBLE_SCHEDULE
            notes.append(f"+{R_FEASIBLE_SCHEDULE} feasible schedule")

        # +5 episode bonus if all tasks resolved with no expiry
        if self._tasks_resolved >= total_tasks and self._tasks_expired == 0:
            bonus += R_EPISODE_BONUS
            notes.append(f"+{R_EPISODE_BONUS} EPISODE BONUS: all tasks resolved!")

        note_str = (" EPISODE END: " + ", ".join(notes)) if notes else " EPISODE END."
        return bonus, note_str

    # -----------------------------------------------------------------------
    # Tool schema builder
    # -----------------------------------------------------------------------

    def _build_tool_schemas(self) -> dict:
        """Build the tool schema dict shown to the agent via list_tools()."""
        return {
            "read_message":    {"params": {"message_id": "str"}},
            "read_calendar":   {"params": {}},
            "reply_message":   {"params": {"message_id": "str", "text": "str"}},
            "move_event":      {"params": {"event_id": "str", "new_time": "str"}},
            "book_restaurant": {"params": {
                "party_size": "int",   # drifts to 'guests'
                "date": "str",         # drifts to 'reservation_date'
                "time": "str",
                "restaurant": "str",
            }},
            "cancel_booking":  {"params": {"booking_id": "str"}},
            "confirm_booking": {"params": {"booking_id": "str"}},
            "book_ride":       {"params": {
                "pickup_time": "str",  # drifts to 'eta_minutes'
                "destination": "str",  # drifts to 'drop_off'
            }},
            "query_policy":    {"params": {}},
            "list_tools":      {"params": {}},
            "finish":          {"params": {}},
        }

    # -----------------------------------------------------------------------
    # Observation builder
    # -----------------------------------------------------------------------

    def _make_observation(
        self,
        result: str = "",
        reward: float = 0.0,
    ) -> DriftPAObservation:
        if result:
            self._last_action_result = result

        # Compute urgent_expiring: tasks expiring in next 2 steps
        urgent = [
            m["id"] for m in self._inbox
            if m.get("expires_at_step") is not None
            and m["id"] not in self._resolved_tasks
            and m["expires_at_step"] <= self._step_count + 2
        ]

        return DriftPAObservation(
            inbox=list(self._inbox),
            calendar=list(self._calendar),
            available_tools=dict(self._active_schema),
            policy=dict(self._active_policy),
            time_step=self._step_count,
            urgent_expiring=urgent,
            last_action_result=self._last_action_result,
            done=self._done,
            reward=reward,
        )

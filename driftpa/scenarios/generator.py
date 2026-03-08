"""
DriftPA — scenarios/generator.py
==================================
Generates randomised (or seeded) episode scenarios.

Seed 0 always returns the canonical "hero scenario" from context.md,
used for demos and reproducible evaluation.

Every scenario dict contains:
  inbox              : list of message dicts
  calendar           : list of event dicts
  initial_policy     : dict of policy rules (pre-drift)
  schema_drift_steps : {step_number: [drift_spec, ...]}
  policy_drift_step  : int
  policy_drift_spec  : {rule: new_value}
"""

import random
from typing import Optional

# ---------------------------------------------------------------------------
# Schema drift pool — from CLAUDE.md (never simplify)
# ---------------------------------------------------------------------------
SCHEMA_DRIFT_POOL = [
    {"api": "restaurant", "old": "party_size",  "new": "guests"},
    {"api": "restaurant", "old": "date",         "new": "reservation_date"},
    {"api": "ride",       "old": "pickup_time",  "new": "eta_minutes"},
    {"api": "ride",       "old": "destination",  "new": "drop_off"},
    {"api": "calendar",   "old": "start_time",   "new": "begins_at"},
    {"api": "calendar",   "old": "event_name",   "new": "title"},
    {"api": "email",      "old": "reply_to",     "new": "respond_to"},
    {"api": "booking",    "old": "confirm_code", "new": "reservation_id"},
]

# ---------------------------------------------------------------------------
# Policy drift pool — from context.md
# ---------------------------------------------------------------------------
POLICY_DRIFT_POOL = [
    {"max_reimbursement": 30},  # $50 → $30
    {"cancellation_window": "4hr"},  # 2hr → 4hr
    {"booking_lead_time": "1hr"},
    {"guest_limit": 4},  # 6 → 4
    {"ride_advance_booking": "30min"},
]

# ---------------------------------------------------------------------------
# Message templates
# ---------------------------------------------------------------------------
_MSG_POOL = [
    {
        "id": "msg_boss",
        "sender": "Sarah (Boss)",
        "subject": "URGENT: Move investor call to 7pm tonight",
        "preview": "The LP wants to move the call…",
        "body": (
            "Hey, the LP just messaged. They need to move the investor call "
            "to 7pm tonight. Can you handle the reschedule ASAP? "
            "This is time-sensitive."
        ),
        "priority": "critical",
        "expires_at_step": 4,
        "requires": ["move_event", "reply_message"],
    },
    {
        "id": "msg_friend",
        "sender": "Alex (Friend)",
        "subject": "Still on for dinner at 7? Booking for 2?",
        "preview": "Hey! Are we still on for dinner…",
        "body": (
            "Hey! Are we still on for dinner tonight at 7? "
            "Should I book for 2 at Nobu or are you doing it? "
            "It's around $45/person by the way."
        ),
        "priority": "high",
        "expires_at_step": 5,
        "requires": ["book_restaurant", "reply_message"],
    },
    {
        "id": "msg_finance",
        "sender": "Finance Team",
        "subject": "New expense policy effective immediately",
        "preview": "Please note the updated expense policy…",
        "body": (
            "Please note that effective immediately, the max dinner "
            "reimbursement is $30 (previously $50). The cancellation window "
            "is now 4 hours. Please query the policy system for full details."
        ),
        "priority": "high",
        "expires_at_step": None,
        "requires": ["query_policy"],
    },
    {
        "id": "msg_partner",
        "sender": "Jamie (Partner)",
        "subject": "Can you book a ride home after dinner?",
        "preview": "Could you grab an Uber home for us…",
        "body": (
            "Could you book a ride home for us after dinner? "
            "We'll be done around 9pm. Heading back to the apartment."
        ),
        "priority": "normal",
        "expires_at_step": 12,
        "requires": ["book_ride"],
    },
    {
        "id": "msg_client",
        "sender": "Chen Wei (Client)",
        "subject": "Confirmation on tomorrow's meeting?",
        "preview": "Just wanted to confirm we're still on…",
        "body": (
            "Hi, just wanted to confirm we're still on for tomorrow's "
            "10am strategy session. Please let me know."
        ),
        "priority": "low",
        "expires_at_step": None,
        "requires": ["reply_message"],
    },
    # Extra messages for non-hero scenarios
    {
        "id": "msg_vendor",
        "sender": "TechVendor",
        "subject": "Invoice #4821 due today",
        "preview": "Reminder: invoice due end of business…",
        "body": "Reminder: Invoice #4821 for $2,400 is due today. Please confirm receipt.",
        "priority": "normal",
        "expires_at_step": 10,
        "requires": ["reply_message"],
    },
    {
        "id": "msg_hr",
        "sender": "HR (Jordan)",
        "subject": "Team offsite — confirm attendance",
        "preview": "Please RSVP for the Q2 offsite by today…",
        "body": "Please RSVP for the Q2 offsite by end of day. Venue: Napa Valley.",
        "priority": "low",
        "expires_at_step": None,
        "requires": ["reply_message"],
    },
    {
        "id": "msg_investor",
        "sender": "Marcus (Investor)",
        "subject": "Quick question on the deck",
        "preview": "One clarification before the call tonight…",
        "body": "One quick clarification before tonight's call — slide 12, what's the ARR basis?",
        "priority": "high",
        "expires_at_step": 6,
        "requires": ["reply_message"],
    },
]

# ---------------------------------------------------------------------------
# Calendar event templates
# ---------------------------------------------------------------------------
_CAL_POOL = [
    {
        "id": "cal_teamsync",
        "event_name": "Team Sync",
        "start_time": "18:00",
        "end_time": "19:00",
        "confirmed": True,
    },
    {
        "id": "cal_investor",
        "event_name": "Investor Call (LP)",
        "start_time": "19:00",
        "end_time": "20:00",
        "confirmed": False,
    },
    {
        "id": "cal_dinner",
        "event_name": "Dinner — Nobu",
        "start_time": "19:00",
        "end_time": "21:00",
        "confirmed": False,
    },
    {
        "id": "cal_gym",
        "event_name": "Gym",
        "start_time": "17:00",
        "end_time": "18:00",
        "confirmed": True,
    },
]


# ---------------------------------------------------------------------------
# Default initial policy
# ---------------------------------------------------------------------------
_DEFAULT_POLICY = {
    "max_reimbursement": 50,   # $ max dinner expense
    "cancellation_window": "2hr",
    "booking_lead_time": "30min",
    "guest_limit": 6,
    "ride_advance_booking": "15min",
    "reply_tone": "professional",
}


# ---------------------------------------------------------------------------
# Hero scenario (seed 0) — canonical demo scenario from context.md
# ---------------------------------------------------------------------------
def _hero_scenario() -> dict:
    inbox = [
        dict(_MSG_POOL[0]),  # msg_boss     — critical, expires step 4
        dict(_MSG_POOL[1]),  # msg_friend   — high, expires step 5
        dict(_MSG_POOL[2]),  # msg_finance  — high, no expiry
        dict(_MSG_POOL[3]),  # msg_partner  — normal, expires step 12
        dict(_MSG_POOL[4]),  # msg_client   — low, no expiry
    ]
    calendar = [
        dict(_CAL_POOL[0]),  # Team Sync @ 18:00
        dict(_CAL_POOL[1]),  # Investor Call @ 19:00 (conflict zone)
        dict(_CAL_POOL[2]),  # Dinner — Nobu @ 19:00 (conflict zone)
    ]
    # Drift triggers at step 3 (schema) and step 4 (policy) — from context.md
    schema_drift_steps = {
        3: [
            {"api": "restaurant", "old": "party_size", "new": "guests"},
            {"api": "ride",       "old": "pickup_time", "new": "eta_minutes"},
            {"api": "calendar",   "old": "start_time",  "new": "begins_at"},
        ]
    }
    policy_drift_step = 4
    policy_drift_spec = {
        "max_reimbursement": 30,
        "cancellation_window": "4hr",
    }
    # Hero restaurant cost: $45/person × 2 = $90 total.
    # Pre-drift policy allows $50/person → fine. Post-drift $30 → violation.
    return {
        "inbox": inbox,
        "calendar": calendar,
        "initial_policy": dict(_DEFAULT_POLICY),
        "schema_drift_steps": schema_drift_steps,
        "policy_drift_step": policy_drift_step,
        "policy_drift_spec": policy_drift_spec,
        "restaurant_cost_per_person": 45,  # agent reads this from msg_friend body
    }


# ---------------------------------------------------------------------------
# Random scenario generator
# ---------------------------------------------------------------------------
def generate_scenario(seed: Optional[int] = None) -> dict:
    """Return a scenario dict. seed=0 always returns the hero scenario."""
    if seed == 0:
        return _hero_scenario()

    if seed is not None:
        random.seed(seed)

    # ---- Inbox: always include boss + friend (core conflict), add 3-5 more ----
    core_msgs = [dict(_MSG_POOL[0]), dict(_MSG_POOL[1]), dict(_MSG_POOL[2])]
    extra_pool = list(_MSG_POOL[3:])
    random.shuffle(extra_pool)
    n_extra = random.randint(2, min(4, len(extra_pool)))
    inbox = core_msgs + extra_pool[:n_extra]

    # Randomise expiry windows slightly (core messages keep fixed expiry for fairness)
    for msg in inbox[3:]:
        if msg.get("expires_at_step") is not None:
            msg["expires_at_step"] = random.randint(8, 13)

    # ---- Calendar: always team sync + investor (conflict), optionally gym ----
    calendar = [dict(_CAL_POOL[0]), dict(_CAL_POOL[1])]
    if random.random() > 0.5:
        calendar.append(dict(_CAL_POOL[3]))  # gym — extra conflict potential

    # ---- Schema drifts: pick 1-2 from pool, trigger between step 3-6 ----
    pool = list(SCHEMA_DRIFT_POOL)
    random.shuffle(pool)
    chosen_drifts = pool[:random.randint(1, 2)]
    drift_step_1 = random.randint(3, 5)
    drift_step_2 = random.randint(drift_step_1 + 1, 6) if len(chosen_drifts) > 1 else None

    schema_drift_steps: dict = {}
    schema_drift_steps[drift_step_1] = [chosen_drifts[0]]
    if drift_step_2 and len(chosen_drifts) > 1:
        schema_drift_steps[drift_step_2] = [chosen_drifts[1]]

    # ---- Policy drift: pick 1, trigger between step 3-6 ----
    policy_drift_step = random.randint(3, 6)
    policy_drift_spec = dict(random.choice(POLICY_DRIFT_POOL))

    # Random restaurant cost — sometimes above, sometimes below max_reimbursement
    # to create genuine policy-drift dilemmas across episodes
    restaurant_cost_per_person = random.choice([25, 30, 35, 40, 45, 55])

    return {
        "inbox": inbox,
        "calendar": calendar,
        "initial_policy": dict(_DEFAULT_POLICY),
        "schema_drift_steps": schema_drift_steps,
        "policy_drift_step": policy_drift_step,
        "policy_drift_spec": policy_drift_spec,
        "restaurant_cost_per_person": restaurant_cost_per_person,
    }

---
title: DriftPA Environment Server
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - rl
  - agent
---

# DriftPA

**Personal assistant RL environment with schema drift, time pressure, and irreversible actions.**

OpenEnv Hackathon | Track 3.2 (Personalized Tasks) + Patronus AI (Consumer Workflows)

---

## What is DriftPA?

DriftPA trains LLM agents to act as personal executive assistants in a world that changes mid-task. The agent must resolve cascading real-life conflicts — calendar clashes, urgent emails, dinner bookings — while four simultaneous failure modes fire without warning:

1. **Schema Drift** — API field names change mid-episode (`party_size → guests`, `pickup_time → eta_minutes`). Agent must call `list_tools()` to discover new schema or get penalised for stale fields.
2. **Time Pressure** — Tasks expire if not resolved within N steps. Boss email expires at step 4. Missing it is a cascade.
3. **Irreversible Actions** — `reply_message`, `book_restaurant`, `book_ride`, `cancel_booking` cannot be undone. Wrong commitments create cascade failures at episode end.
4. **Cancellation Window** — Policy controls how late a booking can be cancelled. Tightens post-drift (`2hr → 4hr`). Late cancellation = policy violation.

## Reward Range

| Agent | Mean Episode Reward |
|---|---|
| Untrained baseline | −3.15 (min −29) |
| Optimal sequence | +22.0 |

## Quick Start

```python
from driftpa.client import DriftPAClient
from driftpa.models import DriftPAAction

# Connect to this HF Space
env = DriftPAClient.from_env("rajv24/driftpa")
result = env.reset(seed=0)  # seed=0 = canonical hero scenario

action = DriftPAAction(tool_name="list_tools", payload={})
result = env.step(action)
print(result.observation)
env.close()
```

## Hero Scenario (seed=0)

The canonical demo: 5 inbox messages, 3 calendar events (with a 19:00 conflict), schema drift at step 3, policy drift at step 4. Optimal 10-step sequence scores +22.

## Training

See `colab_training.ipynb` — trains Qwen2.5-14B with TRL GRPO on an H100 using 1,000 diverse mid-episode states and exact state replay in the reward function.

## Environment API

```
POST /reset   {"seed": 0}
POST /step    {"tool_name": "list_tools", "payload": {}}
GET  /state
GET  /health
```

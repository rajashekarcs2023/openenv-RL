# DriftPA — Personal Assistant RL Environment

**Trains LLM agents to handle real-world chaos: schema drift, policy changes, time pressure, and irreversible decisions.**

OpenEnv Hackathon | Track 3.2 (Personalized Tasks) + Patronus AI (Consumer Workflows)

---

## The Problem

Real AI assistants break in production because:
1. **APIs change schema mid-session** — field names rename without warning
2. **Tasks expire** — the boss's urgent email can't wait forever
3. **Wrong actions can't be undone** — a bad booking is a bad booking
4. **Policies tighten** — cancellation windows shrink after drift

No existing RL environment trains agents against all four simultaneously.

## DriftPA

A personal executive assistant environment where the agent must resolve cascading real-life conflicts — calendar clashes, urgent emails, dinner bookings, ride scheduling — while four failure modes fire without warning.

| Agent | Mean Episode Reward |
|---|---|
| Untrained baseline | −9.55 (min −22) |
| Optimal sequence | +22.0 |

## Four Novel Mechanics

1. **Schema Drift** — `party_size → guests`, `pickup_time → eta_minutes`. Agent must call `list_tools()` to discover new schema or get penalised for stale fields.
2. **Time Pressure** — Boss email expires at step 4. Missing it triggers a cascade.
3. **Irreversible Actions** — `reply_message`, `book_restaurant`, `book_ride`, `cancel_booking` cannot be undone.
4. **Policy Drift** — Cancellation window tightens from 2hr → 4hr post-drift.

## Figure: GRPO Training on H100 — Qwen2.5-14B · 
  **24,000 rollouts · Baseline −9.55 → Post-GRPO −1.10 (Δ = +8.45)** 
  
<img width="2286" height="1379" alt="graph_v2" src="https://github.com/user-attachments/assets/dbe4f998-8f01-4304-bb57-5ba430b3eab8" />

   

## Live Environment

```
https://rajv24-driftpa.hf.space
```

```bash
# Health check
curl https://rajv24-driftpa.hf.space/health

# Start episode (hero scenario)
curl -X POST https://rajv24-driftpa.hf.space/reset \
  -H "Content-Type: application/json" -d '{"seed": 0}'
```

## Training

```python
from driftpa.client import DriftPAClient
from driftpa.models import DriftPAAction

env = DriftPAClient.from_env("rajv24/driftpa")
result = env.reset(seed=0)
action = DriftPAAction(tool_name="list_tools", payload={})
result = env.step(action)
```

Training notebook (Unsloth GRPO on H100): [`driftpa/colab_training.ipynb`](driftpa/colab_training.ipynb)

## Structure

```
driftpa/
├── models.py              # Action, Observation, State dataclasses
├── server/
│   ├── environment.py     # Core RL environment (reset/step/state)
│   └── app.py             # FastAPI server
├── scenarios/
│   └── generator.py       # 50 diverse episode scenarios
├── eval_baseline.py       # Untrained agent evaluation
└── colab_training.ipynb   # GRPO training on H100
```

## Reward Function

| Action | Reward |
|---|---|
| Urgent task resolved before expiry | +3 |
| Normal task resolved | +2 |
| Correct post-drift schema | +2 |
| Called query_policy() before policy action | +1 |
| Episode bonus: everything resolved | +5 |
| Irreversible action causing conflict | −4 |
| Policy violation | −3 |
| Task expired | −3 |
| Stale schema after drift | −2 |

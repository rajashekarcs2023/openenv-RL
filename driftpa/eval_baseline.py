"""
DriftPA — eval_baseline.py
============================
Evaluates a random/heuristic baseline agent over N episodes.

Usage:
    python eval_baseline.py [--episodes 20] [--seed 42] [--out baseline_rewards.json]

Outputs:
    baseline_rewards.json  — per-episode rewards for plotting
    Prints per-episode summary and aggregate stats to stdout.
"""

import sys
import os

# Make the parent directory importable regardless of working directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import random
import argparse

from driftpa.server.environment import DriftPAEnvironment
from driftpa.models import DriftPAAction

# All valid tool names
VALID_TOOLS = [
    "read_message", "read_calendar", "reply_message",
    "move_event", "book_restaurant", "cancel_booking",
    "confirm_booking", "book_ride", "query_policy",
    "list_tools", "finish",
]

# Irreversible tools (random agent avoids these with 80% prob)
IRREVERSIBLE = {"reply_message", "book_restaurant", "cancel_booking",
                "confirm_booking", "book_ride"}


def random_payload(tool: str, obs) -> dict:
    """Generate a (possibly wrong) payload for a tool call."""
    if tool == "read_message":
        msgs = obs.inbox
        if msgs:
            # Random — may pick low-priority first (bad agent behaviour)
            return {"message_id": random.choice(msgs)["id"]}
        return {"message_id": "unknown"}

    if tool == "reply_message":
        msgs = obs.inbox
        if msgs:
            return {
                "message_id": random.choice(msgs)["id"],
                "text": "Got it, will handle.",
            }
        return {"message_id": "unknown", "text": "OK"}

    if tool == "move_event":
        events = obs.calendar
        times = ["17:00", "17:30", "18:00", "18:30", "19:00", "20:00"]
        if events:
            return {
                "event_id": random.choice(events)["id"],
                "new_time": random.choice(times),
            }
        return {"event_id": "unknown", "new_time": "18:00"}

    if tool == "book_restaurant":
        # Intentionally uses STALE field names sometimes to show baseline failing
        use_stale = random.random() > 0.5
        if use_stale:
            return {"party_size": 2, "date": "2026-03-08",
                    "time": "19:00", "restaurant": "Nobu"}
        return {"guests": 2, "reservation_date": "2026-03-08",
                "time": "19:00", "restaurant": "Nobu"}

    if tool == "cancel_booking":
        return {"booking_id": "rest_0"}

    if tool == "confirm_booking":
        return {"booking_id": "rest_0"}

    if tool == "book_ride":
        # Intentionally uses stale fields sometimes
        use_stale = random.random() > 0.5
        if use_stale:
            return {"pickup_time": "21:00", "destination": "Home"}
        return {"eta_minutes": "15", "drop_off": "Home"}

    return {}


def run_episode(env: DriftPAEnvironment, max_steps: int = 15) -> dict:
    """Run one episode with the random baseline agent."""
    obs = env.reset()
    total_reward = 0.0
    steps_taken = 0
    called_list_tools = False
    called_query_policy = False

    for _ in range(max_steps):
        if obs.done:
            break

        # ---- Heuristic baseline behaviour (intentionally bad) ----
        # 1. Rarely checks list_tools after drift
        if not called_list_tools and random.random() < 0.2:
            tool = "list_tools"
            called_list_tools = True
        # 2. Rarely checks policy
        elif not called_query_policy and random.random() < 0.15:
            tool = "query_policy"
            called_query_policy = True
        # 3. Otherwise pick a random tool (biased away from irreversible)
        else:
            safe = [t for t in VALID_TOOLS if t not in IRREVERSIBLE]
            pool = safe + list(IRREVERSIBLE)[:2]  # small chance of irreversible
            tool = random.choice(pool)

        payload = random_payload(tool, obs)
        action = DriftPAAction(tool_name=tool, payload=payload)
        obs = env.step(action)
        total_reward += obs.reward or 0.0
        steps_taken += 1

        if tool == "finish":
            break

    # Force finish if max_steps reached without calling finish
    if not obs.done:
        obs = env.step(DriftPAAction(tool_name="finish", payload={}))
        total_reward += obs.reward or 0.0

    state = env.state
    return {
        "total_reward":      total_reward,
        "steps_taken":       steps_taken,
        "tasks_resolved":    state.tasks_resolved,
        "tasks_expired":     state.tasks_expired,
        "cascade_failures":  state.cascade_failures,
        "schema_version":    state.schema_version,
        "irreversible_taken": len(state.irreversible_actions_taken),
    }


def main():
    parser = argparse.ArgumentParser(description="DriftPA baseline evaluation")
    parser.add_argument("--episodes", type=int, default=20,
                        help="Number of evaluation episodes")
    parser.add_argument("--seed", type=int, default=42,
                        help="Master random seed")
    parser.add_argument("--out", type=str, default="baseline_rewards.json",
                        help="Output JSON file")
    args = parser.parse_args()

    random.seed(args.seed)
    env = DriftPAEnvironment()

    print(f"DriftPA Baseline Evaluation — {args.episodes} episodes\n")
    print(f"{'Ep':>3}  {'Reward':>8}  {'Resolved':>8}  {'Expired':>7}  {'Cascades':>8}")
    print("-" * 44)

    results = []
    for ep in range(1, args.episodes + 1):
        episode_seed = args.seed + ep
        random.seed(episode_seed)
        # Pass different seeds to generate variety
        env.reset(seed=None)
        r = run_episode(env)
        results.append(r)
        print(
            f"{ep:>3}  {r['total_reward']:>8.2f}  "
            f"{r['tasks_resolved']:>8}  {r['tasks_expired']:>7}  "
            f"{r['cascade_failures']:>8}"
        )

    rewards = [r["total_reward"] for r in results]
    print("-" * 44)
    print(f"Mean reward:  {sum(rewards)/len(rewards):.2f}")
    print(f"Min reward:   {min(rewards):.2f}")
    print(f"Max reward:   {max(rewards):.2f}")
    print(f"Mean expired: {sum(r['tasks_expired'] for r in results)/len(results):.1f}")

    # Save results
    output = {
        "episodes": results,
        "rewards": rewards,
        "mean_reward": sum(rewards) / len(rewards),
        "min_reward": min(rewards),
        "max_reward": max(rewards),
        "config": {"episodes": args.episodes, "seed": args.seed},
    }
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.out}")


if __name__ == "__main__":
    main()

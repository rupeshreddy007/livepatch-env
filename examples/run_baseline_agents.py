"""Example: run baseline agents against LivePatch environment."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tasks import run_task
from baseline import random_agent, heuristic_agent


def main():
    print("=" * 60)
    print("LivePatch — Baseline Agent Comparison")
    print("=" * 60)

    for task_name in ["easy", "medium", "hard"]:
        print(f"\n{'─' * 60}")
        print(f"Task: {task_name}")
        print(f"{'─' * 60}")

        # Random agent
        result = run_task(task_name, random_agent, n_episodes=3)
        print(f"\n  Random Agent:")
        print(f"    Score:       {result['avg_score']:.4f}")
        print(f"    Fix Quality: {result['avg_fix_quality']:.4f}")
        print(f"    Uptime:      {result['avg_uptime']:.4f}")
        print(f"    Efficiency:  {result['avg_efficiency']:.4f}")
        print(f"    Safety:      {result['avg_safety']:.4f}")
        print(f"    Reward:      {result['avg_reward']:.4f}")

        # Heuristic agent
        result = run_task(task_name, heuristic_agent, n_episodes=3)
        print(f"\n  Heuristic Agent:")
        print(f"    Score:       {result['avg_score']:.4f}")
        print(f"    Fix Quality: {result['avg_fix_quality']:.4f}")
        print(f"    Uptime:      {result['avg_uptime']:.4f}")
        print(f"    Efficiency:  {result['avg_efficiency']:.4f}")
        print(f"    Safety:      {result['avg_safety']:.4f}")
        print(f"    Reward:      {result['avg_reward']:.4f}")


if __name__ == "__main__":
    main()

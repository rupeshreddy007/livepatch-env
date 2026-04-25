"""Task definitions and grading for LivePatch environment."""
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from .config import (
    EnvironmentConfig, DIFFICULTY_EASY, DIFFICULTY_MEDIUM,
    DIFFICULTY_HARD, DIFFICULTY_EXPERT,
)
from .models import TaskResult
from .environment import LivePatchEnv
from .fault_injector import check_fault_resolved


# ── Task presets ─────────────────────────────────────────────────────

TASKS = {
    "easy": EnvironmentConfig(
        difficulty=DIFFICULTY_EASY,
        max_steps=20,
        traffic_rps=50,
        seed=2001,
    ),
    "medium": EnvironmentConfig(
        difficulty=DIFFICULTY_MEDIUM,
        max_steps=30,
        traffic_rps=100,
        seed=2002,
    ),
    "hard": EnvironmentConfig(
        difficulty=DIFFICULTY_HARD,
        max_steps=35,
        traffic_rps=200,
        seed=2003,
    ),
    "expert": EnvironmentConfig(
        difficulty=DIFFICULTY_EXPERT,
        max_steps=40,
        traffic_rps=500,
        seed=2004,
    ),
}


def grade_episode(env: LivePatchEnv) -> TaskResult:
    """Grade a completed episode. Returns TaskResult with scores in [0, 1]."""
    if not env.done or not env.db:
        return TaskResult(score=0.0, fix_quality=0.0, uptime=0.0,
                          efficiency=0.0, safety=0.0)

    snapshot = env.db.snapshot()

    # Fix quality: cost improvement ratio
    fix_quality = 0.0
    if env.initial_snapshot and env.initial_snapshot.total_explain_cost > 0:
        improvement = 1.0 - (snapshot.total_explain_cost / env.initial_snapshot.total_explain_cost)
        fix_quality = max(0.0, min(1.0, improvement))

    # Uptime: fraction of traffic that succeeded
    total_ok = sum(t.requests_ok for t in env.traffic_history)
    total_req = sum(t.requests_total for t in env.traffic_history)
    uptime = total_ok / max(1, total_req)

    # Efficiency: steps saved
    efficiency = max(0, (env.config.max_steps - env.step_num) / env.config.max_steps)

    # Safety: ratio of safe to total DDL operations
    total_ops = env.safe_ops + env.unsafe_ops
    safety = 1.0 if total_ops == 0 else env.safe_ops / total_ops

    # Faults resolved
    resolved = [f for f in env.faults if check_fault_resolved(env.db, f)]

    # Weighted score: 35% fix quality, 30% uptime, 20% efficiency, 15% safety
    score = (fix_quality * 0.35 + uptime * 0.30 + efficiency * 0.20 + safety * 0.15)

    # Bonus for resolving all faults
    if len(resolved) == len(env.faults) and len(env.faults) > 0:
        score = min(1.0, score + 0.1)

    return TaskResult(
        score=round(score, 4),
        fix_quality=round(fix_quality, 4),
        uptime=round(uptime, 4),
        efficiency=round(efficiency, 4),
        safety=round(safety, 4),
        faults_injected=[f.fault_type for f in env.faults],
        faults_resolved=[f.fault_type for f in resolved],
        steps_used=env.step_num,
        total_reward=round(env.total_reward, 4),
    )


def run_task(task_name: str, agent_fn, n_episodes: int = 3) -> Dict[str, Any]:
    """Run an agent on a task for multiple episodes and return averaged results."""
    config = TASKS.get(task_name)
    if not config:
        raise ValueError(f"Unknown task: {task_name}. Available: {list(TASKS.keys())}")

    results = []
    for ep in range(n_episodes):
        ep_config = EnvironmentConfig(
            difficulty=config.difficulty,
            max_steps=config.max_steps,
            traffic_rps=config.traffic_rps,
            seed=(config.seed or 0) + ep,
            sla_latency_ms=config.sla_latency_ms,
            step_duration_seconds=config.step_duration_seconds,
        )
        env = LivePatchEnv(ep_config)
        obs = env.reset()

        while not obs["done"]:
            action = agent_fn(obs, env.state())
            obs = env.step(action)

        result = grade_episode(env)
        results.append(result)

    # Average results
    avg_score = sum(r.score for r in results) / len(results)
    avg_fix = sum(r.fix_quality for r in results) / len(results)
    avg_uptime = sum(r.uptime for r in results) / len(results)
    avg_eff = sum(r.efficiency for r in results) / len(results)
    avg_safety = sum(r.safety for r in results) / len(results)
    avg_reward = sum(r.total_reward for r in results) / len(results)

    return {
        "task": task_name,
        "difficulty": config.difficulty,
        "episodes": n_episodes,
        "avg_score": round(avg_score, 4),
        "avg_fix_quality": round(avg_fix, 4),
        "avg_uptime": round(avg_uptime, 4),
        "avg_efficiency": round(avg_eff, 4),
        "avg_safety": round(avg_safety, 4),
        "avg_reward": round(avg_reward, 4),
        "per_episode": [
            {
                "score": r.score,
                "fix_quality": r.fix_quality,
                "uptime": r.uptime,
                "efficiency": r.efficiency,
                "safety": r.safety,
                "total_reward": r.total_reward,
                "faults_injected": r.faults_injected,
                "faults_resolved": r.faults_resolved,
                "steps_used": r.steps_used,
            }
            for r in results
        ],
    }

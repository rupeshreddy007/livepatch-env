"""Baseline agents for LivePatch environment."""
import random
import os
from typing import Dict, Any, Optional


def random_agent(obs: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, str]:
    """Agent that sends random psql commands."""
    commands = [
        "\\dt", "\\di", "\\d orders", "\\d users", "\\d payments",
        "EXPLAIN ANALYZE SELECT * FROM orders WHERE user_id = 1",
        "SELECT * FROM pg_stat_activity",
        "SELECT * FROM pg_locks",
        "SELECT * FROM pg_stat_user_tables",
        "SHOW work_mem",
        "SHOW max_connections",
        "VACUUM orders",
        "ANALYZE orders",
        "submit",
    ]
    return {"command": random.choice(commands)}


def heuristic_agent(obs: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, str]:
    """Rule-based agent that follows a diagnostic workflow."""
    step = state.get("step", 0)

    # Phase 1: Discovery (steps 0-3)
    discovery_cmds = [
        "\\dt",
        "\\di",
        "SELECT * FROM pg_stat_activity",
        "SELECT * FROM pg_stat_user_tables",
    ]
    if step < len(discovery_cmds):
        return {"command": discovery_cmds[step]}

    # Phase 2: Diagnose (steps 4-7)
    diagnose_cmds = [
        "SELECT * FROM pg_locks",
        "SELECT * FROM pg_stat_statements",
        "SHOW work_mem",
        "SHOW max_connections",
    ]
    diag_idx = step - len(discovery_cmds)
    if diag_idx < len(diagnose_cmds):
        return {"command": diagnose_cmds[diag_idx]}

    # Phase 3: Explain hot queries (steps 8-11)
    explain_cmds = [
        "EXPLAIN ANALYZE SELECT * FROM orders WHERE user_id = 1 AND status = 'active'",
        "EXPLAIN ANALYZE SELECT * FROM payments WHERE order_id = 1 ORDER BY created_at DESC",
        "EXPLAIN ANALYZE SELECT * FROM sessions WHERE token = 'abc' AND expires_at > NOW()",
        "EXPLAIN ANALYZE SELECT * FROM order_items WHERE order_id = 1",
    ]
    exp_idx = step - len(discovery_cmds) - len(diagnose_cmds)
    if exp_idx < len(explain_cmds):
        return {"command": explain_cmds[exp_idx]}

    # Phase 4: Fix — apply common fixes (steps 12+)
    fix_cmds = [
        "CREATE INDEX CONCURRENTLY idx_fix_orders_user ON orders(user_id)",
        "CREATE INDEX CONCURRENTLY idx_fix_orders_status ON orders(status)",
        "CREATE INDEX CONCURRENTLY idx_fix_orders_created ON orders(created_at)",
        "CREATE INDEX CONCURRENTLY idx_fix_payments_order ON payments(order_id)",
        "CREATE INDEX CONCURRENTLY idx_fix_sessions_token ON sessions(token)",
        "CREATE INDEX CONCURRENTLY idx_fix_oi_order ON order_items(order_id)",
        "VACUUM ANALYZE orders",
        "VACUUM ANALYZE payments",
        "ANALYZE sessions",
        "submit",
    ]
    fix_idx = step - len(discovery_cmds) - len(diagnose_cmds) - len(explain_cmds)
    if fix_idx < len(fix_cmds):
        return {"command": fix_cmds[fix_idx]}

    return {"command": "submit"}


def make_openai_agent(model: str = "gpt-4o-mini"):
    """Create an agent that uses OpenAI API for decision making."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package required: pip install openai")

    client = OpenAI(
        api_key=os.environ.get("API_KEY", os.environ.get("OPENAI_API_KEY", "")),
        base_url=os.environ.get("API_BASE_URL", "https://api.openai.com/v1"),
    )
    model_name = os.environ.get("MODEL_NAME", model)
    history = []

    from .environment import SYSTEM_PROMPT

    def agent(obs: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, str]:
        nonlocal history

        if not history:
            history = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Build user message with observation + traffic
        traffic = obs.get("traffic", {})
        msg = obs["observation"]
        msg += f"\n\n[Traffic] OK: {traffic.get('requests_ok', 0)} | "
        msg += f"Failed: {traffic.get('requests_failed', 0)} | "
        msg += f"p99: {traffic.get('p99_latency_ms', 0):.1f}ms"
        msg += f"\n[Step {obs['step']}/{obs['max_steps']}] "
        msg += f"Reward so far: {obs.get('total_reward', 0):.2f}"

        history.append({"role": "user", "content": msg})

        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=history,
                max_tokens=200,
                temperature=0.3,
            )
            reply = resp.choices[0].message.content.strip()
            history.append({"role": "assistant", "content": reply})

            # Extract command — take first line that looks like a command
            for line in reply.split("\n"):
                line = line.strip().strip("`")
                if line and not line.startswith(("#", "//", "--", "I ", "The ", "Let")):
                    return {"command": line}
            return {"command": reply.split("\n")[0].strip()}
        except Exception as e:
            # Fallback to heuristic
            return heuristic_agent(obs, state)

    return agent

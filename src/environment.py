"""LivePatch OpenEnv environment.

The agent receives a PagerDuty alert about a sick database under live traffic.
It must diagnose and fix the issue using psql commands — without causing downtime.
"""
import random
from typing import Dict, List, Any, Optional
from .config import EnvironmentConfig, HOT_QUERIES, DIFFICULTY_EASY
from .models import (
    FaultSpec, TrafficMetrics, DatabaseSnapshot,
    StepResult, TaskResult,
)
from .database import SimulatedDatabase
from .fault_injector import (
    select_faults, inject_fault, check_fault_resolved,
)


SYSTEM_PROMPT = """You are an expert PostgreSQL DBA responding to a production incident.

You have access to a psql prompt connected to a live production database.
The database is serving real traffic — your actions have immediate consequences.

Available commands:
  \\dt                              — list all tables
  \\di                              — list all indexes
  \\d <table>                       — describe a table (columns, indexes, stats)
  EXPLAIN ANALYZE SELECT ...        — show query plan and cost
  SELECT * FROM pg_stat_activity    — show running queries
  SELECT * FROM pg_locks            — show current locks
  SELECT * FROM pg_stat_user_tables — show table statistics
  SELECT * FROM pg_stat_statements  — show query performance stats
  SELECT count(*) FROM <table>      — count rows
  SHOW <param>                      — show config parameter
  CREATE INDEX [CONCURRENTLY] <name> ON <table>(<cols>) [WHERE ...]
  DROP INDEX <name>
  ALTER SYSTEM SET <param> = '<value>'
  SELECT pg_reload_conf()           — apply config changes
  SELECT pg_terminate_backend(<pid>) — kill a backend process
  SELECT pg_cancel_backend(<pid>)    — cancel a query (graceful)
  VACUUM [FULL] [ANALYZE] <table>   — reclaim dead tuples
  ANALYZE <table>                   — update planner statistics
  submit                            — submit your fix (ends episode)

⚠️  CRITICAL SAFETY RULES:
  - CREATE INDEX (without CONCURRENTLY) LOCKS the table — all writes fail
  - VACUUM FULL LOCKS the table — all reads AND writes fail
  - Use CONCURRENTLY and non-blocking operations when possible
  - The database is under LIVE TRAFFIC — downtime = penalty
"""


def _generate_alert(faults: List[FaultSpec], config: EnvironmentConfig) -> str:
    """Generate a realistic PagerDuty alert from injected faults."""
    # Don't reveal the fault directly — just symptoms
    lines = [
        "🚨 PAGERDUTY ALERT — PRODUCTION DATABASE",
        f"Severity: P0 — SLA violation detected",
        f"Service: checkout-api, payments-service",
        f"SLA threshold: {config.sla_latency_ms:.0f}ms",
        "",
        "Symptoms:",
    ]

    symptom_map = {
        "missing_index": "  - Query latency spiked to >10s on multiple endpoints",
        "lock_storm": "  - Write operations timing out, transactions piling up",
        "table_bloat": "  - Sequential scan times increasing, disk I/O elevated",
        "bad_config": "  - Connection errors and memory pressure reported",
        "connection_flood": "  - 'too many connections' errors from app servers",
        "runaway_query": "  - CPU at 100%, one core pinned, other queries starving",
        "stale_stats": "  - Planner choosing wrong query plans, intermittent slowness",
        "sort_spill": "  - Disk I/O spikes during ORDER BY and JOIN operations",
    }

    seen_symptoms = set()
    for f in faults:
        symptom = symptom_map.get(f.fault_type, "  - Unknown degradation detected")
        if symptom not in seen_symptoms:
            lines.append(symptom)
            seen_symptoms.add(symptom)

    lines.extend([
        "",
        f"Traffic: ~{config.traffic_rps} requests/second hitting the database",
        "Impact: Revenue loss estimated at $52K/hour",
        "",
        "You have psql access. Diagnose and fix the issue.",
        "⚠️  The database is under LIVE TRAFFIC. Unsafe operations cause downtime.",
    ])

    return "\n".join(lines)


class LivePatchEnv:
    """OpenEnv-compatible environment for live database incident response."""

    def __init__(self, config: Optional[EnvironmentConfig] = None):
        self.config = config or EnvironmentConfig()
        self.db: Optional[SimulatedDatabase] = None
        self.faults: List[FaultSpec] = []
        self.step_num = 0
        self.done = False
        self.commands_used: List[str] = []
        self.total_reward = 0.0
        self.unsafe_ops = 0
        self.safe_ops = 0
        self.initial_snapshot: Optional[DatabaseSnapshot] = None
        self.traffic_history: List[TrafficMetrics] = []
        self._alert_text = ""

    def reset(self, config: Optional[EnvironmentConfig] = None) -> Dict[str, Any]:
        """Reset environment with new faults. Returns initial observation."""
        if config:
            self.config = config

        seed = self.config.seed or random.randint(0, 999999)
        self.db = SimulatedDatabase(seed=seed)
        rng = random.Random(seed)

        # Inject faults
        self.faults = select_faults(self.config.difficulty, rng)
        for fault in self.faults:
            inject_fault(self.db, fault)

        self.step_num = 0
        self.done = False
        self.commands_used = []
        self.total_reward = 0.0
        self.unsafe_ops = 0
        self.safe_ops = 0
        self.traffic_history = []

        # Take initial snapshot (the sick state)
        self.initial_snapshot = self.db.snapshot()

        # Simulate initial traffic to show the problem
        initial_traffic = self.db.simulate_traffic(
            self.config.traffic_rps,
            self.config.step_duration_seconds,
            self.config.sla_latency_ms,
        )
        self.traffic_history.append(initial_traffic)

        self._alert_text = _generate_alert(self.faults, self.config)

        return {
            "observation": self._alert_text,
            "traffic": {
                "requests_total": initial_traffic.requests_total,
                "requests_ok": initial_traffic.requests_ok,
                "requests_failed": initial_traffic.requests_failed,
                "avg_latency_ms": initial_traffic.avg_latency_ms,
                "p99_latency_ms": initial_traffic.p99_latency_ms,
            },
            "step": 0,
            "max_steps": self.config.max_steps,
            "done": False,
            "reward": 0.0,
        }

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute agent's psql command. Returns observation with traffic impact."""
        if self.done:
            return self._make_obs("Episode already finished.", 0.0, True)

        command = action.get("command", "").strip()
        if not command:
            return self._make_obs("ERROR: no command provided", -0.1, False)

        self.step_num += 1
        self.db.advance_step()

        # Check for submit
        if command.lower() == "submit":
            return self._handle_submit()

        # Execute the command
        output, is_unsafe = self.db.execute(command)

        # Track safety
        if is_unsafe:
            self.unsafe_ops += 1
        elif command.upper().startswith(("CREATE INDEX CONCURRENTLY", "VACUUM ANALYZE", "VACUUM ")):
            if not command.upper().startswith("VACUUM FULL"):
                self.safe_ops += 1

        # Simulate traffic during this step
        traffic = self.db.simulate_traffic(
            self.config.traffic_rps,
            self.config.step_duration_seconds,
            self.config.sla_latency_ms,
        )
        self.traffic_history.append(traffic)

        # Compute step reward
        reward = self._compute_step_reward(command, is_unsafe, traffic)
        self.total_reward += reward
        self.commands_used.append(command)

        # Check if max steps reached
        if self.step_num >= self.config.max_steps:
            self.done = True
            output += "\n\n⏰ Max steps reached. Episode ending."

        return self._make_obs(output, reward, self.done, traffic)

    def state(self) -> Dict[str, Any]:
        """Return full environment state."""
        snapshot = self.db.snapshot() if self.db else DatabaseSnapshot()
        return {
            "step": self.step_num,
            "max_steps": self.config.max_steps,
            "done": self.done,
            "total_reward": round(self.total_reward, 3),
            "difficulty": self.config.difficulty,
            "faults_injected": [f.fault_type for f in self.faults],
            "faults_resolved": [
                f.fault_type for f in self.faults
                if check_fault_resolved(self.db, f)
            ] if self.db else [],
            "db_snapshot": {
                "total_explain_cost": snapshot.total_explain_cost,
                "worst_query_cost": snapshot.worst_query_cost,
                "worst_query_name": snapshot.worst_query_name,
                "active_lock_count": snapshot.active_lock_count,
                "idx_scan_ratio": snapshot.idx_scan_ratio,
            },
            "traffic_latest": {
                "requests_ok": self.traffic_history[-1].requests_ok if self.traffic_history else 0,
                "requests_failed": self.traffic_history[-1].requests_failed if self.traffic_history else 0,
                "avg_latency_ms": self.traffic_history[-1].avg_latency_ms if self.traffic_history else 0,
            },
            "commands_used": len(self.commands_used),
            "unsafe_ops": self.unsafe_ops,
            "safe_ops": self.safe_ops,
        }

    # ── Internal ─────────────────────────────────────────────────────

    def _handle_submit(self) -> Dict[str, Any]:
        """Handle episode submission."""
        self.done = True
        snapshot = self.db.snapshot()

        # Check which faults are resolved
        resolved = [f for f in self.faults if check_fault_resolved(self.db, f)]
        unresolved = [f for f in self.faults if not check_fault_resolved(self.db, f)]

        # Final traffic check
        final_traffic = self.db.simulate_traffic(
            self.config.traffic_rps,
            self.config.step_duration_seconds,
            self.config.sla_latency_ms,
        )
        self.traffic_history.append(final_traffic)

        # Compute submission reward
        reward = 0.0

        # Cost improvement: compare initial vs final
        if self.initial_snapshot and self.initial_snapshot.total_explain_cost > 0:
            improvement = 1.0 - (snapshot.total_explain_cost / self.initial_snapshot.total_explain_cost)
            reward += max(0, improvement) * self.config.reward_cost_improvement

        # SLA met?
        if final_traffic.p99_latency_ms < self.config.sla_latency_ms:
            reward += self.config.reward_sla_met

        # Uptime bonus: what fraction of total traffic succeeded?
        total_ok = sum(t.requests_ok for t in self.traffic_history)
        total_req = sum(t.requests_total for t in self.traffic_history)
        uptime = total_ok / max(1, total_req)
        reward += uptime * self.config.reward_uptime_bonus

        # Efficiency bonus
        steps_used = self.step_num
        efficiency = max(0, (self.config.max_steps - steps_used) / self.config.max_steps)
        reward += efficiency * self.config.bonus_efficiency

        # Safety bonus: used safe operations
        if self.unsafe_ops == 0 and self.safe_ops > 0:
            reward += self.config.bonus_safe_operation

        # Penalty for unresolved faults
        reward -= len(unresolved) * 1.0

        self.total_reward += reward

        # Build summary
        lines = [
            "=" * 60,
            "EPISODE COMPLETE — INCIDENT REPORT",
            "=" * 60,
            f"Faults injected:  {len(self.faults)}",
            f"Faults resolved:  {len(resolved)}/{len(self.faults)}",
        ]
        for f in resolved:
            lines.append(f"  ✅ {f.fault_type} on '{f.target_table}'")
        for f in unresolved:
            lines.append(f"  ❌ {f.fault_type} on '{f.target_table}' — UNRESOLVED")
        lines.extend([
            f"\nCost improvement: {snapshot.total_explain_cost:.0f} (was {self.initial_snapshot.total_explain_cost:.0f})",
            f"Final p99 latency: {final_traffic.p99_latency_ms:.1f}ms (SLA: {self.config.sla_latency_ms:.0f}ms)",
            f"Overall uptime:   {uptime:.1%}",
            f"Steps used:       {steps_used}/{self.config.max_steps}",
            f"Unsafe operations: {self.unsafe_ops}",
            f"Safe operations:  {self.safe_ops}",
            f"\nSubmission reward: {reward:+.2f}",
            f"Total reward:      {self.total_reward:+.2f}",
        ])

        return self._make_obs("\n".join(lines), reward, True, final_traffic)

    def _compute_step_reward(self, command: str, is_unsafe: bool,
                              traffic: TrafficMetrics) -> float:
        """Compute reward for a single step."""
        reward = 0.0

        # Downtime penalty: failed requests during this step (capped per step)
        if traffic.requests_failed > 0:
            traffic_penalty = traffic.requests_failed * self.config.penalty_downtime_per_error
            reward += max(traffic_penalty, -0.5)

        # Unsafe operation penalty
        if is_unsafe:
            reward += self.config.penalty_unsafe_op

        # Repeated command penalty
        if command in self.commands_used:
            reward += self.config.penalty_repeated_cmd

        # Safe operation bonus (small per-step signal)
        if command.upper().startswith("CREATE INDEX CONCURRENTLY"):
            reward += 0.2  # small reward for choosing safe variant

        return round(reward, 4)

    def _make_obs(self, output: str, reward: float, done: bool,
                   traffic: Optional[TrafficMetrics] = None) -> Dict[str, Any]:
        """Build observation dict."""
        t = traffic or (self.traffic_history[-1] if self.traffic_history else TrafficMetrics())
        return {
            "observation": output,
            "traffic": {
                "requests_total": t.requests_total,
                "requests_ok": t.requests_ok,
                "requests_failed": t.requests_failed,
                "avg_latency_ms": t.avg_latency_ms,
                "p99_latency_ms": t.p99_latency_ms,
            },
            "step": self.step_num,
            "max_steps": self.config.max_steps,
            "done": done,
            "reward": round(reward, 4),
            "total_reward": round(self.total_reward, 4),
        }

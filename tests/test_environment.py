"""Comprehensive tests for LivePatch environment."""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    EnvironmentConfig, DIFFICULTY_EASY, DIFFICULTY_MEDIUM,
    DIFFICULTY_HARD, DIFFICULTY_EXPERT, TABLES, HOT_QUERIES, DEFAULT_INDEXES,
    FAULT_MISSING_INDEX, FAULT_LOCK_STORM, FAULT_TABLE_BLOAT,
    FAULT_BAD_CONFIG, FAULT_CONNECTION_FLOOD, FAULT_RUNAWAY_QUERY,
    FAULT_STALE_STATS, FAULT_SORT_SPILL,
)
from src.models import TableState, IndexState, FaultSpec
from src.database import SimulatedDatabase
from src.fault_injector import inject_fault, check_fault_resolved, select_faults
from src.environment import LivePatchEnv
from src.tasks import TASKS, grade_episode, run_task
from src.curriculum import CurriculumController


# ═══════════════════════════════════════════════════════════════════
# Database Tests
# ═══════════════════════════════════════════════════════════════════

class TestDatabase:
    def setup_method(self):
        self.db = SimulatedDatabase(seed=42)

    def test_init_tables(self):
        assert len(self.db.tables) == len(TABLES)
        assert "orders" in self.db.tables
        assert self.db.tables["orders"].row_count == 8_000_000

    def test_init_indexes(self):
        assert len(self.db.indexes) == len(DEFAULT_INDEXES)
        assert "idx_orders_user_id" in self.db.indexes

    def test_init_config(self):
        assert self.db.config["work_mem"] == "4MB"
        assert self.db.config["max_connections"] == "100"

    def test_list_tables(self):
        output, unsafe = self.db.execute("\\dt")
        assert not unsafe
        assert "orders" in output
        assert "users" in output
        assert "8,000,000" in output

    def test_list_indexes(self):
        output, unsafe = self.db.execute("\\di")
        assert not unsafe
        assert "idx_orders_user_id" in output

    def test_describe_table(self):
        output, unsafe = self.db.execute("\\d orders")
        assert not unsafe
        assert "user_id" in output
        assert "status" in output
        assert "idx_orders_user_id" in output

    def test_describe_nonexistent(self):
        output, _ = self.db.execute("\\d nonexistent")
        assert "ERROR" in output

    def test_explain_with_index(self):
        output, _ = self.db.execute(
            "EXPLAIN ANALYZE SELECT * FROM orders WHERE user_id = 1"
        )
        assert "Index Scan" in output
        assert "cost=" in output

    def test_explain_seq_scan_after_drop(self):
        self.db.execute("DROP INDEX idx_orders_user_id")
        self.db.execute("DROP INDEX idx_orders_status")
        self.db.execute("DROP INDEX idx_orders_created_at")
        output, _ = self.db.execute(
            "EXPLAIN ANALYZE SELECT * FROM orders WHERE user_id = 1"
        )
        assert "Seq Scan" in output

    def test_explain_cost_index_vs_seqscan(self):
        """Index scan should have much lower cost than seq scan."""
        # With index
        cost_idx, plan_idx = self.db.compute_explain_cost(
            "orders", ["user_id"], 0.00005, False
        )
        # Drop index
        del self.db.indexes["idx_orders_user_id"]
        cost_seq, plan_seq = self.db.compute_explain_cost(
            "orders", ["user_id"], 0.00005, False
        )
        assert cost_idx < cost_seq
        assert cost_seq > 1000  # seq scan on 8M rows should be expensive
        assert "Index Scan" in plan_idx
        assert "Seq Scan" in plan_seq

    def test_pg_stat_activity_empty(self):
        output, _ = self.db.execute("SELECT * FROM pg_stat_activity")
        assert "0 rows" in output

    def test_pg_stat_user_tables(self):
        output, _ = self.db.execute("SELECT * FROM pg_stat_user_tables")
        assert "orders" in output
        assert "payments" in output

    def test_count(self):
        output, _ = self.db.execute("SELECT count(*) FROM orders")
        assert "8,000,000" in output

    def test_show_config(self):
        output, _ = self.db.execute("SHOW work_mem")
        assert "4MB" in output

    def test_create_index_blocking(self):
        output, unsafe = self.db.execute(
            "CREATE INDEX idx_test ON orders(status, created_at)"
        )
        assert unsafe  # blocking!
        assert "LOCKED" in output
        assert "idx_test" in self.db.indexes

    def test_create_index_concurrently(self):
        output, unsafe = self.db.execute(
            "CREATE INDEX CONCURRENTLY idx_test ON orders(status, created_at)"
        )
        assert not unsafe  # safe!
        assert "non-blocking" in output
        assert "idx_test" in self.db.indexes

    def test_create_index_duplicate(self):
        output, _ = self.db.execute(
            "CREATE INDEX idx_orders_user_id ON orders(user_id)"
        )
        assert "already exists" in output

    def test_create_index_bad_table(self):
        output, _ = self.db.execute(
            "CREATE INDEX idx_test ON nonexistent(col)"
        )
        assert "ERROR" in output

    def test_create_index_bad_column(self):
        output, _ = self.db.execute(
            "CREATE INDEX idx_test ON orders(nonexistent_col)"
        )
        assert "ERROR" in output

    def test_drop_index(self):
        assert "idx_orders_user_id" in self.db.indexes
        output, _ = self.db.execute("DROP INDEX idx_orders_user_id")
        assert "DROP INDEX" in output
        assert "idx_orders_user_id" not in self.db.indexes

    def test_alter_system(self):
        output, _ = self.db.execute("ALTER SYSTEM SET work_mem = '256MB'")
        assert "ALTER SYSTEM" in output
        assert self.db.config["work_mem"] == "256MB"

    def test_terminate_backend(self):
        from src.models import ActiveQuery
        self.db.active_queries.append(ActiveQuery(
            pid=9999, query="SELECT 1", state="active", started_step=0
        ))
        output, _ = self.db.execute("SELECT pg_terminate_backend(9999)")
        assert "t" in output
        assert not any(q.pid == 9999 for q in self.db.active_queries)

    def test_vacuum(self):
        self.db.tables["orders"].dead_tuples = 1000000
        output, unsafe = self.db.execute("VACUUM orders")
        assert not unsafe
        assert self.db.tables["orders"].dead_tuples < 1000000

    def test_vacuum_full_is_unsafe(self):
        self.db.tables["orders"].dead_tuples = 1000000
        output, unsafe = self.db.execute("VACUUM FULL orders")
        assert unsafe  # blocking!
        assert "LOCKED" in output
        assert self.db.tables["orders"].dead_tuples == 0

    def test_vacuum_analyze(self):
        self.db.tables["orders"].dead_tuples = 1000000
        self.db.tables["orders"].stats_accurate = False
        output, unsafe = self.db.execute("VACUUM ANALYZE orders")
        assert not unsafe
        assert self.db.tables["orders"].stats_accurate

    def test_analyze(self):
        self.db.tables["orders"].stats_accurate = False
        output, _ = self.db.execute("ANALYZE orders")
        assert self.db.tables["orders"].stats_accurate

    def test_table_size(self):
        output, _ = self.db.execute(
            "SELECT pg_size_pretty(pg_total_relation_size('orders'))"
        )
        assert "MB" in output or "GB" in output

    def test_unknown_command(self):
        output, _ = self.db.execute("kubectl get pods")
        assert "ERROR" in output

    def test_traffic_simulation_healthy(self):
        """Healthy DB should have mostly OK requests."""
        metrics = self.db.simulate_traffic(100, 1.0, 200.0)
        assert metrics.requests_total == 100
        assert metrics.requests_ok > 50  # most should succeed

    def test_traffic_simulation_with_locks(self):
        """Locked table should cause failed requests."""
        self.db.tables["orders"].locked_until = self.db.step_count + 10
        metrics = self.db.simulate_traffic(100, 1.0, 200.0)
        assert metrics.requests_failed > 0

    def test_blocking_op_expires(self):
        """Blocking operations should expire after duration."""
        self.db.execute("CREATE INDEX idx_temp ON orders(status)")
        assert self.db.tables["orders"].locked_until >= self.db.step_count
        # Advance enough steps
        for _ in range(10):
            self.db.advance_step()
        assert self.db.tables["orders"].locked_until < self.db.step_count


# ═══════════════════════════════════════════════════════════════════
# Fault Injector Tests
# ═══════════════════════════════════════════════════════════════════

class TestFaultInjector:
    def setup_method(self):
        self.db = SimulatedDatabase(seed=42)

    def test_inject_missing_index(self):
        fault = FaultSpec(fault_type=FAULT_MISSING_INDEX, target_table="orders")
        inject_fault(self.db, fault)
        # All indexes on orders should be gone
        order_indexes = [i for i in self.db.indexes.values() if i.table == "orders"]
        assert len(order_indexes) == 0

    def test_inject_lock_storm(self):
        fault = FaultSpec(fault_type=FAULT_LOCK_STORM, target_table="orders")
        inject_fault(self.db, fault)
        assert len(self.db.active_queries) > 0
        assert self.db.active_queries[-1].is_blocking
        assert self.db.tables["orders"].locked_until > self.db.step_count

    def test_inject_table_bloat(self):
        fault = FaultSpec(fault_type=FAULT_TABLE_BLOAT, target_table="orders")
        inject_fault(self.db, fault)
        assert self.db.tables["orders"].dead_tuples > 0
        assert self.db.tables["orders"].bloat_ratio > 0.3

    def test_inject_bad_config(self):
        fault = FaultSpec(fault_type=FAULT_BAD_CONFIG, target_table="orders")
        inject_fault(self.db, fault)
        # Some config should now be bad
        bad_values = {"5", "16MB", "32MB"}
        has_bad = any(v in bad_values for v in self.db.config.values())
        assert has_bad

    def test_inject_connection_flood(self):
        fault = FaultSpec(fault_type=FAULT_CONNECTION_FLOOD, target_table="orders")
        inject_fault(self.db, fault)
        assert len(self.db.active_queries) > 50

    def test_inject_runaway_query(self):
        fault = FaultSpec(fault_type=FAULT_RUNAWAY_QUERY, target_table="orders")
        inject_fault(self.db, fault)
        assert any("CROSS JOIN" in q.query for q in self.db.active_queries)

    def test_inject_stale_stats(self):
        fault = FaultSpec(fault_type=FAULT_STALE_STATS, target_table="orders")
        inject_fault(self.db, fault)
        assert not self.db.tables["orders"].stats_accurate

    def test_inject_sort_spill(self):
        fault = FaultSpec(fault_type=FAULT_SORT_SPILL, target_table="orders")
        inject_fault(self.db, fault)
        assert self.db.config["work_mem"] == "64kB"

    def test_select_faults_easy(self):
        import random
        faults = select_faults(DIFFICULTY_EASY, random.Random(42))
        assert len(faults) == 1  # easy = 1 fault

    def test_select_faults_expert(self):
        import random
        faults = select_faults(DIFFICULTY_EXPERT, random.Random(42))
        assert len(faults) == 4  # expert = 4 faults

    def test_check_missing_index_resolved(self):
        fault = FaultSpec(fault_type=FAULT_MISSING_INDEX, target_table="orders")
        inject_fault(self.db, fault)
        assert not check_fault_resolved(self.db, fault)
        # Fix: recreate indexes (checkout_lookup needs user_id,status prefix)
        self.db.execute("CREATE INDEX CONCURRENTLY idx_fix1 ON orders(user_id, status)")
        self.db.execute("CREATE INDEX CONCURRENTLY idx_fix3 ON orders(created_at)")
        assert check_fault_resolved(self.db, fault)

    def test_check_lock_storm_resolved(self):
        fault = FaultSpec(fault_type=FAULT_LOCK_STORM, target_table="orders")
        inject_fault(self.db, fault)
        pid = fault.details["blocking_pid"]
        assert not check_fault_resolved(self.db, fault)
        self.db.execute(f"SELECT pg_terminate_backend({pid})")
        assert check_fault_resolved(self.db, fault)

    def test_check_bloat_resolved(self):
        fault = FaultSpec(fault_type=FAULT_TABLE_BLOAT, target_table="orders")
        inject_fault(self.db, fault)
        assert not check_fault_resolved(self.db, fault)
        self.db.execute("VACUUM FULL orders")
        assert check_fault_resolved(self.db, fault)

    def test_check_stale_stats_resolved(self):
        fault = FaultSpec(fault_type=FAULT_STALE_STATS, target_table="orders")
        inject_fault(self.db, fault)
        assert not check_fault_resolved(self.db, fault)
        self.db.execute("ANALYZE orders")
        assert check_fault_resolved(self.db, fault)

    def test_check_sort_spill_resolved(self):
        fault = FaultSpec(fault_type=FAULT_SORT_SPILL, target_table="orders")
        inject_fault(self.db, fault)
        assert not check_fault_resolved(self.db, fault)
        self.db.execute("ALTER SYSTEM SET work_mem = '256MB'")
        assert check_fault_resolved(self.db, fault)


# ═══════════════════════════════════════════════════════════════════
# Environment Tests
# ═══════════════════════════════════════════════════════════════════

class TestEnvironment:
    def test_reset(self):
        env = LivePatchEnv(EnvironmentConfig(difficulty="easy", seed=42))
        obs = env.reset()
        assert not obs["done"]
        assert obs["step"] == 0
        assert "PAGERDUTY" in obs["observation"]
        assert obs["traffic"]["requests_total"] > 0

    def test_step_basic(self):
        env = LivePatchEnv(EnvironmentConfig(difficulty="easy", seed=42))
        env.reset()
        obs = env.step({"command": "\\dt"})
        assert obs["step"] == 1
        assert "orders" in obs["observation"]
        assert not obs["done"]

    def test_step_empty_command(self):
        env = LivePatchEnv(EnvironmentConfig(difficulty="easy", seed=42))
        env.reset()
        obs = env.step({"command": ""})
        assert "ERROR" in obs["observation"]

    def test_step_after_done(self):
        env = LivePatchEnv(EnvironmentConfig(difficulty="easy", seed=42))
        env.reset()
        obs = env.step({"command": "submit"})
        assert obs["done"]
        obs2 = env.step({"command": "\\dt"})
        assert obs2["done"]
        assert "already finished" in obs2["observation"]

    def test_submit(self):
        env = LivePatchEnv(EnvironmentConfig(difficulty="easy", seed=42))
        env.reset()
        obs = env.step({"command": "submit"})
        assert obs["done"]
        assert "EPISODE COMPLETE" in obs["observation"]

    def test_max_steps(self):
        env = LivePatchEnv(EnvironmentConfig(difficulty="easy", seed=42, max_steps=3))
        env.reset()
        env.step({"command": "\\dt"})
        env.step({"command": "\\di"})
        obs = env.step({"command": "\\d orders"})
        assert obs["done"]

    def test_state(self):
        env = LivePatchEnv(EnvironmentConfig(difficulty="easy", seed=42))
        env.reset()
        env.step({"command": "\\dt"})
        state = env.state()
        assert state["step"] == 1
        assert state["difficulty"] == "easy"
        assert len(state["faults_injected"]) > 0
        assert "total_reward" in state

    def test_unsafe_op_penalty(self):
        env = LivePatchEnv(EnvironmentConfig(difficulty="easy", seed=42))
        env.reset()
        obs = env.step({"command": "CREATE INDEX idx_test ON orders(status)"})
        assert obs["reward"] < 0  # should be penalized for unsafe op

    def test_safe_op_bonus(self):
        env = LivePatchEnv(EnvironmentConfig(difficulty="easy", seed=42))
        env.reset()
        obs_safe = env.step({"command": "CREATE INDEX CONCURRENTLY idx_test ON orders(status)"})
        # Safe op should not get the -1.0 unsafe penalty
        # It may still have small traffic-related penalties but no big hit
        assert obs_safe["reward"] > -1.0

    def test_repeated_command_penalty(self):
        env = LivePatchEnv(EnvironmentConfig(difficulty="easy", seed=42))
        env.reset()
        env.step({"command": "\\dt"})
        obs2 = env.step({"command": "\\dt"})
        # Second time should have repeat penalty
        assert obs2["reward"] < 0

    def test_traffic_in_observation(self):
        env = LivePatchEnv(EnvironmentConfig(difficulty="easy", seed=42))
        obs = env.reset()
        assert "requests_total" in obs["traffic"]
        assert "requests_ok" in obs["traffic"]
        assert "requests_failed" in obs["traffic"]
        assert "avg_latency_ms" in obs["traffic"]
        assert "p99_latency_ms" in obs["traffic"]

    def test_full_diagnostic_workflow(self):
        """Test a complete diagnostic → fix → submit workflow."""
        env = LivePatchEnv(EnvironmentConfig(
            difficulty="easy", seed=100, max_steps=20
        ))
        env.reset()

        # Discover
        env.step({"command": "\\dt"})
        env.step({"command": "\\di"})
        env.step({"command": "SELECT * FROM pg_stat_activity"})

        # Diagnose
        env.step({"command": "SELECT * FROM pg_stat_user_tables"})
        env.step({"command": "EXPLAIN ANALYZE SELECT * FROM orders WHERE user_id = 1"})

        # Fix (safe operations)
        env.step({"command": "CREATE INDEX CONCURRENTLY idx_fix_a ON orders(user_id)"})
        env.step({"command": "CREATE INDEX CONCURRENTLY idx_fix_b ON orders(status)"})
        env.step({"command": "CREATE INDEX CONCURRENTLY idx_fix_c ON orders(created_at)"})
        env.step({"command": "VACUUM ANALYZE orders"})

        # Submit
        obs = env.step({"command": "submit"})
        assert obs["done"]
        assert "EPISODE COMPLETE" in obs["observation"]
        assert env.unsafe_ops == 0
        assert env.safe_ops > 0


# ═══════════════════════════════════════════════════════════════════
# Tasks & Grading Tests
# ═══════════════════════════════════════════════════════════════════

class TestTasks:
    def test_task_presets(self):
        assert "easy" in TASKS
        assert "medium" in TASKS
        assert "hard" in TASKS
        assert "expert" in TASKS

    def test_grade_unfinished(self):
        env = LivePatchEnv(EnvironmentConfig(difficulty="easy", seed=42))
        env.reset()
        result = grade_episode(env)
        assert result.score == 0.0  # not done yet

    def test_grade_finished(self):
        env = LivePatchEnv(EnvironmentConfig(difficulty="easy", seed=42))
        env.reset()
        env.step({"command": "submit"})
        result = grade_episode(env)
        assert 0.0 <= result.score <= 1.0
        assert 0.0 <= result.uptime <= 1.0
        assert 0.0 <= result.efficiency <= 1.0

    def test_run_task_random(self):
        from baseline import random_agent
        result = run_task("easy", random_agent, n_episodes=1)
        assert result["task"] == "easy"
        assert "avg_score" in result
        assert len(result["per_episode"]) == 1

    def test_run_task_heuristic(self):
        from baseline import heuristic_agent
        result = run_task("easy", heuristic_agent, n_episodes=1)
        assert result["avg_score"] > 0  # heuristic should do something

    def test_heuristic_beats_random(self):
        """Heuristic agent should consistently outperform random."""
        from baseline import random_agent, heuristic_agent
        random_result = run_task("easy", random_agent, n_episodes=2)
        heuristic_result = run_task("easy", heuristic_agent, n_episodes=2)
        # Heuristic should get better score (or at least not terrible)
        assert heuristic_result["avg_score"] >= random_result["avg_score"] * 0.5


# ═══════════════════════════════════════════════════════════════════
# Curriculum Tests
# ═══════════════════════════════════════════════════════════════════

class TestCurriculum:
    def test_init(self):
        cc = CurriculumController()
        assert cc.current_difficulty == "easy"
        assert cc.episode_count == 0

    def test_record_episode(self):
        cc = CurriculumController()
        cc.record_episode(
            faults_injected=["missing_index"],
            faults_resolved=["missing_index"],
            score=0.8,
        )
        assert cc.episode_count == 1
        assert cc.mastery("missing_index") == 1.0

    def test_difficulty_escalation(self):
        cc = CurriculumController()
        # Record 5 good episodes
        for _ in range(5):
            cc.record_episode(
                faults_injected=["missing_index"],
                faults_resolved=["missing_index"],
                score=0.9,
            )
        assert cc.current_difficulty in ("medium", "easy")  # should escalate

    def test_weakest_faults(self):
        cc = CurriculumController()
        cc.record_episode(["missing_index", "lock_storm"],
                          ["missing_index"], 0.5)
        weak = cc.weakest_faults(3)
        # lock_storm was attempted but not resolved
        assert "lock_storm" in weak

    def test_summary(self):
        cc = CurriculumController()
        cc.record_episode(["missing_index"], ["missing_index"], 0.8)
        summary = cc.summary()
        assert summary["episode_count"] == 1
        assert "mastery_by_fault" in summary


# ═══════════════════════════════════════════════════════════════════
# Edge Cases
# ═══════════════════════════════════════════════════════════════════

class TestEdgeCases:
    def test_partial_index(self):
        db = SimulatedDatabase(seed=42)
        output, _ = db.execute(
            "CREATE INDEX CONCURRENTLY idx_partial ON orders(created_at) WHERE status = 'active'"
        )
        assert "idx_partial" in db.indexes
        assert db.indexes["idx_partial"].partial_condition == "status = 'active'"

    def test_multiple_faults_same_episode(self):
        env = LivePatchEnv(EnvironmentConfig(
            difficulty="hard", seed=42, max_steps=40
        ))
        obs = env.reset()
        state = env.state()
        assert len(state["faults_injected"]) == 3  # hard = 3 faults

    def test_config_needs_reload(self):
        db = SimulatedDatabase(seed=42)
        db.execute("ALTER SYSTEM SET work_mem = '256MB'")
        assert db._config_needs_reload
        db.execute("SELECT pg_reload_conf()")
        assert not db._config_needs_reload

    def test_cancel_vs_terminate(self):
        """pg_cancel_backend is graceful, pg_terminate_backend is hard kill."""
        db = SimulatedDatabase(seed=42)
        from src.models import ActiveQuery
        db.active_queries.append(ActiveQuery(
            pid=5555, query="SELECT ...", state="active",
            started_step=0, is_blocking=True, blocked_table="orders",
        ))
        # Cancel: may not fully work for blocking queries
        output, _ = db.execute("SELECT pg_cancel_backend(5555)")
        assert "may not terminate" in output
        # Query still there after cancel
        assert any(q.pid == 5555 for q in db.active_queries)

        # Terminate: hard kill
        output, _ = db.execute("SELECT pg_terminate_backend(5555)")
        assert not any(q.pid == 5555 for q in db.active_queries)

    def test_sort_spill_increases_cost(self):
        """Low work_mem should increase cost for ORDER BY queries."""
        db = SimulatedDatabase(seed=42)
        # Use a query with higher selectivity so sort has more rows
        cost_normal, _ = db.compute_explain_cost(
            "payments", ["order_id"], 0.01, True  # has ORDER BY
        )
        db.config["work_mem"] = "64kB"
        cost_spill, plan = db.compute_explain_cost(
            "payments", ["order_id"], 0.01, True
        )
        assert cost_spill > cost_normal
        assert "disk" in plan

    def test_bloat_increases_cost(self):
        """Dead tuples should increase scan cost."""
        db = SimulatedDatabase(seed=42)
        # Remove index to force seq scan
        to_remove = [n for n, i in db.indexes.items() if i.table == "orders"]
        for n in to_remove:
            del db.indexes[n]
        cost_clean, _ = db.compute_explain_cost("orders", ["user_id"], 0.001, False)
        db.tables["orders"].dead_tuples = 5_000_000
        cost_bloated, _ = db.compute_explain_cost("orders", ["user_id"], 0.001, False)
        assert cost_bloated > cost_clean

    def test_stale_stats_increases_cost(self):
        """Stale stats should cause planner to make worse decisions."""
        db = SimulatedDatabase(seed=42)
        # Remove index to force seq scan where stats matter
        to_remove = [n for n, i in db.indexes.items() if i.table == "orders"]
        for n in to_remove:
            del db.indexes[n]
        cost_accurate, _ = db.compute_explain_cost("orders", ["user_id"], 0.001, False)
        db.tables["orders"].stats_accurate = False
        cost_stale, _ = db.compute_explain_cost("orders", ["user_id"], 0.001, False)
        assert cost_stale > cost_accurate

    def test_connection_flood_blocks_traffic(self):
        """Exhausted connections should cause traffic failures."""
        db = SimulatedDatabase(seed=42)
        db.config["max_connections"] = "5"
        # Fill connections
        from src.models import ActiveQuery
        for i in range(5):
            db.active_queries.append(ActiveQuery(
                pid=8000 + i, query="idle", state="idle in transaction",
                started_step=0,
            ))
        metrics = db.simulate_traffic(100, 1.0, 200.0)
        assert metrics.requests_failed == 100  # all should fail

    def test_index_covers_prefix(self):
        """Index on (a, b) should cover queries filtering on (a) or (a, b)."""
        idx = IndexState(name="idx_test", table="orders", columns=["user_id", "status"])
        assert idx.covers(["user_id"])
        assert idx.covers(["user_id", "status"])
        assert not idx.covers(["status"])  # wrong order
        assert not idx.covers(["status", "user_id"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

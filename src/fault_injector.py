"""Fault injector — injects realistic PostgreSQL pathologies."""
import random
from typing import List, Optional
from .config import (
    ALL_FAULT_TYPES, FAULT_MISSING_INDEX, FAULT_LOCK_STORM,
    FAULT_TABLE_BLOAT, FAULT_BAD_CONFIG, FAULT_CONNECTION_FLOOD,
    FAULT_RUNAWAY_QUERY, FAULT_STALE_STATS, FAULT_SORT_SPILL,
    DIFFICULTY_EASY, DIFFICULTY_MEDIUM, DIFFICULTY_HARD, DIFFICULTY_EXPERT,
    HOT_QUERIES,
)
from .models import FaultSpec, ActiveQuery
from .database import SimulatedDatabase


# Which tables are most impactful to break
HIGH_IMPACT_TABLES = ["orders", "order_items", "payments", "sessions"]
MEDIUM_IMPACT_TABLES = ["users", "inventory", "products"]
LOW_IMPACT_TABLES = ["audit_log"]


def get_fault_count(difficulty: str) -> int:
    """How many faults to inject based on difficulty."""
    return {
        DIFFICULTY_EASY: 1,
        DIFFICULTY_MEDIUM: 2,
        DIFFICULTY_HARD: 3,
        DIFFICULTY_EXPERT: 4,
    }.get(difficulty, 2)


def get_fault_pool(difficulty: str) -> List[str]:
    """Which fault types are available at each difficulty."""
    if difficulty == DIFFICULTY_EASY:
        return [FAULT_MISSING_INDEX, FAULT_TABLE_BLOAT, FAULT_STALE_STATS]
    if difficulty == DIFFICULTY_MEDIUM:
        return [FAULT_MISSING_INDEX, FAULT_TABLE_BLOAT, FAULT_STALE_STATS,
                FAULT_BAD_CONFIG, FAULT_SORT_SPILL]
    if difficulty == DIFFICULTY_HARD:
        return [FAULT_MISSING_INDEX, FAULT_LOCK_STORM, FAULT_TABLE_BLOAT,
                FAULT_BAD_CONFIG, FAULT_RUNAWAY_QUERY, FAULT_SORT_SPILL]
    return ALL_FAULT_TYPES  # expert: everything


def select_faults(difficulty: str, rng: random.Random,
                  exclude: Optional[List[str]] = None) -> List[FaultSpec]:
    """Select faults to inject based on difficulty."""
    pool = get_fault_pool(difficulty)
    count = get_fault_count(difficulty)
    exclude = exclude or []

    # Don't repeat fault types unless expert
    available = [f for f in pool if f not in exclude]
    if not available:
        available = pool

    selected_types = []
    for _ in range(count):
        if not available:
            break
        ft = rng.choice(available)
        selected_types.append(ft)
        if difficulty != DIFFICULTY_EXPERT:
            available = [f for f in available if f != ft]

    # Assign target tables
    faults = []
    used_tables = set()
    for ft in selected_types:
        # Pick a high-impact table for harder faults
        if ft in (FAULT_LOCK_STORM, FAULT_CONNECTION_FLOOD, FAULT_RUNAWAY_QUERY):
            candidates = HIGH_IMPACT_TABLES
        else:
            candidates = HIGH_IMPACT_TABLES + MEDIUM_IMPACT_TABLES
        # Avoid injecting two faults on same table (unless expert)
        if difficulty != DIFFICULTY_EXPERT:
            candidates = [t for t in candidates if t not in used_tables] or candidates
        target = rng.choice(candidates)
        used_tables.add(target)
        faults.append(FaultSpec(fault_type=ft, target_table=target))

    return faults


def inject_fault(db: SimulatedDatabase, fault: FaultSpec) -> str:
    """Inject a single fault into the database. Returns description."""
    handler = {
        FAULT_MISSING_INDEX: _inject_missing_index,
        FAULT_LOCK_STORM: _inject_lock_storm,
        FAULT_TABLE_BLOAT: _inject_table_bloat,
        FAULT_BAD_CONFIG: _inject_bad_config,
        FAULT_CONNECTION_FLOOD: _inject_connection_flood,
        FAULT_RUNAWAY_QUERY: _inject_runaway_query,
        FAULT_STALE_STATS: _inject_stale_stats,
        FAULT_SORT_SPILL: _inject_sort_spill,
    }.get(fault.fault_type)

    if handler is None:
        return f"Unknown fault type: {fault.fault_type}"
    return handler(db, fault)


def _inject_missing_index(db: SimulatedDatabase, fault: FaultSpec) -> str:
    """Drop indexes on the target table so queries do seq scans."""
    table = fault.target_table
    dropped = []
    to_remove = [name for name, idx in db.indexes.items() if idx.table == table]
    for name in to_remove:
        del db.indexes[name]
        dropped.append(name)
    fault.details["dropped_indexes"] = dropped
    fault.description = f"Dropped {len(dropped)} indexes on '{table}' — queries will do sequential scans"
    return fault.description


def _inject_lock_storm(db: SimulatedDatabase, fault: FaultSpec) -> str:
    """Add long-running blocking queries holding locks."""
    table = fault.target_table
    pid = db._next_pid
    db._next_pid += 1
    blocking_query = ActiveQuery(
        pid=pid,
        query=f"SELECT * FROM {table} WHERE id IN (...) FOR UPDATE",
        state="active",
        started_step=db.step_count,
        is_blocking=True,
        blocked_table=table,
        duration_ms=180000,  # 3 minute query
    )
    db.active_queries.append(blocking_query)
    # Table is partially locked
    db.tables[table].locked_until = db.step_count + 100  # effectively forever until killed
    fault.details["blocking_pid"] = pid
    fault.description = f"Long-running query (PID {pid}) holding locks on '{table}'"
    return fault.description


def _inject_table_bloat(db: SimulatedDatabase, fault: FaultSpec) -> str:
    """Add massive dead tuples to a table."""
    table = db.tables[fault.target_table]
    dead = int(table.row_count * db.rng.uniform(0.7, 0.95))
    table.dead_tuples = dead
    table.last_vacuum = None
    fault.details["dead_tuples"] = dead
    fault.description = f"Table '{fault.target_table}' has {dead:,} dead tuples ({table.bloat_ratio:.0%} bloat)"
    return fault.description


def _inject_bad_config(db: SimulatedDatabase, fault: FaultSpec) -> str:
    """Set configuration parameters to bad values."""
    # Pick a problematic config
    bad_configs = db.rng.choice([
        {"max_connections": "5"},
        {"shared_buffers": "16MB"},
        {"effective_cache_size": "32MB"},
    ])
    for k, v in bad_configs.items():
        db.config[k] = v
    fault.details["bad_config"] = bad_configs
    fault.description = f"Bad configuration: {bad_configs}"
    return fault.description


def _inject_connection_flood(db: SimulatedDatabase, fault: FaultSpec) -> str:
    """Exhaust connection pool with idle connections."""
    max_conn = int(db.config.get("max_connections", "100"))
    # Fill up connections with idle queries
    n_idle = max_conn - 2  # leave barely any room
    for i in range(n_idle):
        pid = db._next_pid
        db._next_pid += 1
        db.active_queries.append(ActiveQuery(
            pid=pid,
            query="<idle in transaction>",
            state="idle in transaction",
            started_step=db.step_count,
            is_blocking=False,
            duration_ms=db.rng.uniform(60000, 300000),
        ))
    fault.details["idle_connections"] = n_idle
    fault.description = f"Connection pool exhausted: {n_idle}/{max_conn} connections idle"
    return fault.description


def _inject_runaway_query(db: SimulatedDatabase, fault: FaultSpec) -> str:
    """Add a CPU-burning runaway query."""
    table = fault.target_table
    pid = db._next_pid
    db._next_pid += 1
    db.active_queries.append(ActiveQuery(
        pid=pid,
        query=f"SELECT t1.*, t2.* FROM {table} t1 CROSS JOIN {table} t2 WHERE t1.id < 1000",
        state="active",
        started_step=db.step_count,
        is_blocking=False,
        duration_ms=600000,  # 10 minute query
    ))
    fault.details["runaway_pid"] = pid
    fault.description = f"Runaway cartesian join on '{table}' (PID {pid}) burning CPU"
    return fault.description


def _inject_stale_stats(db: SimulatedDatabase, fault: FaultSpec) -> str:
    """Corrupt table statistics so planner makes bad decisions."""
    table = db.tables[fault.target_table]
    table.stats_accurate = False
    table.last_analyze = None
    fault.details["table"] = fault.target_table
    fault.description = f"Stale statistics on '{fault.target_table}' — planner making bad estimates"
    return fault.description


def _inject_sort_spill(db: SimulatedDatabase, fault: FaultSpec) -> str:
    """Set work_mem too low causing sorts to spill to disk."""
    db.config["work_mem"] = "64kB"
    fault.description = "work_mem set to 64kB — sorts spilling to disk"
    return fault.description


def check_fault_resolved(db: SimulatedDatabase, fault: FaultSpec) -> bool:
    """Check if a fault has been resolved."""
    ft = fault.fault_type
    table = fault.target_table

    if ft == FAULT_MISSING_INDEX:
        # Check if indexes have been created for the affected hot queries
        for hq in HOT_QUERIES:
            if hq["table"] == table:
                has_idx = any(
                    idx.table == table and idx.covers(hq["filter_columns"])
                    for idx in db.indexes.values()
                )
                if not has_idx:
                    return False
        return True

    if ft == FAULT_LOCK_STORM:
        pid = fault.details.get("blocking_pid")
        return not any(q.pid == pid for q in db.active_queries)

    if ft == FAULT_TABLE_BLOAT:
        tbl = db.tables.get(table)
        return tbl is not None and tbl.bloat_ratio < 0.3

    if ft == FAULT_BAD_CONFIG:
        bad = fault.details.get("bad_config", {})
        for k, v in bad.items():
            if db.config.get(k) == v:
                return False
        return True

    if ft == FAULT_CONNECTION_FLOOD:
        max_conn = int(db.config.get("max_connections", "100"))
        active = len(db.active_queries)
        return active < max_conn * 0.5

    if ft == FAULT_RUNAWAY_QUERY:
        pid = fault.details.get("runaway_pid")
        return not any(q.pid == pid for q in db.active_queries)

    if ft == FAULT_STALE_STATS:
        tbl = db.tables.get(table)
        return tbl is not None and tbl.stats_accurate

    if ft == FAULT_SORT_SPILL:
        wm = db._parse_mem(db.config.get("work_mem", "64kB"))
        return wm >= 4 * 1024 * 1024  # at least 4MB

    return False

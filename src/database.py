"""Simulated PostgreSQL database engine.

Models the key behaviors of a real PostgreSQL instance:
- EXPLAIN cost computation based on indexes and table stats
- Blocking operations (CREATE INDEX vs CREATE INDEX CONCURRENTLY)
- Lock tracking and pg_stat_activity
- Configuration effects (work_mem, shared_buffers, max_connections)
- Table bloat and vacuum behavior
"""
import math
import re
import random
from typing import Dict, List, Optional, Tuple, Any
from .config import TABLES, DEFAULT_INDEXES, HOT_QUERIES
from .models import (
    TableState, IndexState, ActiveQuery, BlockingOp,
    DatabaseSnapshot, TrafficMetrics,
)


class SimulatedDatabase:
    """A simulated PostgreSQL instance with realistic cost modeling."""

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        self.step_count = 0
        self.tables: Dict[str, TableState] = {}
        self.indexes: Dict[str, IndexState] = {}
        self.config: Dict[str, str] = {}
        self.active_queries: List[ActiveQuery] = []
        self.blocking_ops: List[BlockingOp] = []
        self.seq_scan_counts: Dict[str, int] = {}
        self.idx_scan_counts: Dict[str, int] = {}
        self._next_pid = 1000
        self._config_needs_reload = False
        self._init_healthy_state()

    def _init_healthy_state(self):
        """Initialize database to a healthy baseline."""
        for name, spec in TABLES.items():
            self.tables[name] = TableState(
                name=name,
                columns=list(spec["columns"]),
                row_count=spec["row_count"],
                avg_row_width=spec["avg_row_width"],
                pk=spec.get("pk", "id"),
            )
            self.seq_scan_counts[name] = 0
            self.idx_scan_counts[name] = 0

        for idx_spec in DEFAULT_INDEXES:
            idx = IndexState(
                name=idx_spec["name"],
                table=idx_spec["table"],
                columns=list(idx_spec["columns"]),
            )
            row_count = TABLES[idx.table]["row_count"]
            idx.size_pages = max(1, int(math.log2(row_count) * 10))
            self.indexes[idx.name] = idx

        self.config = {
            "work_mem": "4MB",
            "shared_buffers": "256MB",
            "max_connections": "100",
            "effective_cache_size": "1GB",
            "maintenance_work_mem": "64MB",
            "random_page_cost": "4.0",
            "seq_page_cost": "1.0",
            "enable_seqscan": "on",
        }

    def advance_step(self):
        """Advance simulated time by one step."""
        self.step_count += 1
        # Expire completed blocking operations
        expired = [op for op in self.blocking_ops if op.end_step <= self.step_count]
        for op in expired:
            if op.table in self.tables:
                self.tables[op.table].locked_until = -1
        self.blocking_ops = [op for op in self.blocking_ops if op.end_step > self.step_count]

    # ── Cost computation ─────────────────────────────────────────────

    def compute_explain_cost(self, table_name: str, filter_columns: List[str],
                              selectivity: float, has_order_by: bool) -> Tuple[float, str]:
        """Compute EXPLAIN ANALYZE cost and plan type for a query."""
        table = self.tables.get(table_name)
        if not table:
            return 0.0, "Error"

        row_count = table.row_count + table.dead_tuples
        pages = table.pages

        # Find best matching index
        best_index = None
        for idx in self.indexes.values():
            if idx.table == table_name and idx.covers(filter_columns):
                best_index = idx
                break

        # Stats accuracy multiplier: if stats are stale, planner makes bad choices
        stats_penalty = 1.0
        if not table.stats_accurate:
            stats_penalty = 5.0 + self.rng.uniform(0, 10)

        # Bloat penalty: dead tuples increase scan cost
        bloat_mult = 1.0 + table.bloat_ratio * 3.0

        work_mem_bytes = self._parse_mem(self.config.get("work_mem", "4MB"))

        if best_index and table.stats_accurate:
            # Index scan
            matched_rows = max(1, int(row_count * selectivity))
            cost = matched_rows * 0.01 + math.log2(max(2, row_count)) * float(self.config.get("random_page_cost", "4.0"))
            plan_type = f"Index Scan using {best_index.name}"
            self.idx_scan_counts[table_name] = self.idx_scan_counts.get(table_name, 0) + 1
        else:
            # Sequential scan
            seq_page_cost = float(self.config.get("seq_page_cost", "1.0"))
            cost = pages * seq_page_cost * bloat_mult + row_count * 0.01
            cost *= stats_penalty
            plan_type = "Seq Scan"
            self.seq_scan_counts[table_name] = self.seq_scan_counts.get(table_name, 0) + 1

        # Sort cost
        if has_order_by:
            matched_rows = max(1, int(row_count * selectivity))
            sort_bytes = matched_rows * table.avg_row_width
            if sort_bytes > work_mem_bytes:
                spill_factor = max(1.0, sort_bytes / work_mem_bytes)
                cost += (sort_bytes / 1024.0) * math.log2(spill_factor)  # more passes when work_mem is smaller
                plan_type += " (Sort: disk)"
            else:
                cost += matched_rows * 0.05
                plan_type += " (Sort: memory)"

        return round(cost, 2), plan_type

    def compute_hot_query_costs(self) -> Dict[str, Tuple[float, str]]:
        """Compute EXPLAIN cost for all hot queries."""
        results = {}
        for q in HOT_QUERIES:
            cost, plan = self.compute_explain_cost(
                q["table"], q["filter_columns"],
                q["selectivity"], q["has_order_by"],
            )
            results[q["name"]] = (cost, plan)
        return results

    def cost_to_latency_ms(self, cost: float) -> float:
        """Convert EXPLAIN cost to approximate latency in ms."""
        return max(0.1, cost * 0.02)

    # ── Traffic simulation ───────────────────────────────────────────

    def simulate_traffic(self, rps: int, duration_seconds: float,
                          sla_ms: float) -> TrafficMetrics:
        """Simulate app traffic for a time period and return metrics."""
        total_requests = int(rps * duration_seconds)
        metrics = TrafficMetrics(requests_total=total_requests)

        query_costs = self.compute_hot_query_costs()
        total_weight = sum(q["weight"] for q in HOT_QUERIES)
        latencies = []

        # Check connection availability
        max_conn = int(self.config.get("max_connections", "100"))
        active_conn = len(self.active_queries)
        conn_available = max_conn - active_conn

        for _ in range(total_requests):
            # Pick a random hot query weighted by frequency
            r = self.rng.uniform(0, total_weight)
            cumulative = 0
            chosen = HOT_QUERIES[0]
            for q in HOT_QUERIES:
                cumulative += q["weight"]
                if r <= cumulative:
                    chosen = q
                    break

            table_name = chosen["table"]
            table = self.tables.get(table_name)

            # Check if table is locked by a blocking operation
            if table and table.locked_until >= self.step_count:
                metrics.requests_failed += 1
                metrics.errors_by_query[chosen["name"]] = \
                    metrics.errors_by_query.get(chosen["name"], 0) + 1
                continue

            # Check connection limit
            if conn_available <= 0:
                metrics.requests_failed += 1
                metrics.errors_by_query[chosen["name"]] = \
                    metrics.errors_by_query.get(chosen["name"], 0) + 1
                continue

            cost, _ = query_costs.get(chosen["name"], (100.0, "Unknown"))
            latency = self.cost_to_latency_ms(cost)
            # Add jitter
            latency *= self.rng.uniform(0.8, 1.3)
            latencies.append(latency)

            if latency > sla_ms:
                metrics.requests_failed += 1
                metrics.errors_by_query[chosen["name"]] = \
                    metrics.errors_by_query.get(chosen["name"], 0) + 1
            else:
                metrics.requests_ok += 1

        if latencies:
            metrics.avg_latency_ms = round(sum(latencies) / len(latencies), 2)
            sorted_lat = sorted(latencies)
            p99_idx = min(len(sorted_lat) - 1, int(len(sorted_lat) * 0.99))
            metrics.p99_latency_ms = round(sorted_lat[p99_idx], 2)

        return metrics

    # ── Snapshot ─────────────────────────────────────────────────────

    def snapshot(self) -> DatabaseSnapshot:
        """Take a snapshot of current database health."""
        costs = self.compute_hot_query_costs()
        total_cost = sum(c for c, _ in costs.values())
        worst_name = max(costs, key=lambda k: costs[k][0]) if costs else ""
        worst_cost = costs[worst_name][0] if worst_name else 0.0

        total_seq = sum(self.seq_scan_counts.values())
        total_idx = sum(self.idx_scan_counts.values())
        ratio = total_idx / max(1, total_seq + total_idx)

        total_dead = sum(t.dead_tuples for t in self.tables.values())
        lock_count = sum(1 for q in self.active_queries if q.is_blocking)
        blocked_count = len(self.blocking_ops)

        return DatabaseSnapshot(
            total_explain_cost=round(total_cost, 2),
            worst_query_cost=round(worst_cost, 2),
            worst_query_name=worst_name,
            active_lock_count=lock_count,
            blocked_query_count=blocked_count,
            idx_scan_ratio=round(ratio, 4),
            total_dead_tuples=total_dead,
        )

    # ── Command execution ────────────────────────────────────────────

    def execute(self, command: str) -> Tuple[str, bool]:
        """Execute a psql command. Returns (output, is_unsafe).

        is_unsafe = True if the operation causes blocking/downtime.
        """
        cmd = command.strip().rstrip(";")
        if not cmd:
            return "ERROR: empty command", False

        # Dispatch to handler
        handlers = [
            (r"^\\dt\s*$", self._handle_list_tables),
            (r"^\\di\s*$", self._handle_list_indexes),
            (r"^\\d\+?\s+(\w+)\s*$", self._handle_describe_table),
            (r"^EXPLAIN\s+(ANALYZE\s+)?(.+)$", self._handle_explain),
            (r"^SELECT\s+.*FROM\s+pg_stat_activity", self._handle_pg_stat_activity),
            (r"^SELECT\s+.*FROM\s+pg_locks", self._handle_pg_locks),
            (r"^SELECT\s+.*FROM\s+pg_stat_user_tables", self._handle_pg_stat_user_tables),
            (r"^SELECT\s+.*FROM\s+pg_stat_statements", self._handle_pg_stat_statements),
            (r"^SELECT\s+pg_terminate_backend\((\d+)\)", self._handle_terminate_backend),
            (r"^SELECT\s+pg_cancel_backend\((\d+)\)", self._handle_cancel_backend),
            (r"^SELECT\s+pg_reload_conf\(\)", self._handle_reload_conf),
            (r"^SELECT\s+pg_size_pretty\(pg_total_relation_size\('(\w+)'\)\)", self._handle_table_size),
            (r"^SELECT\s+count\(\*\)\s+FROM\s+(\w+)", self._handle_count),
            (r"^CREATE\s+INDEX\s+CONCURRENTLY\s+(.+)$", self._handle_create_index_concurrently),
            (r"^CREATE\s+INDEX\s+(.+)$", self._handle_create_index),
            (r"^DROP\s+INDEX\s+(?:IF\s+EXISTS\s+)?(\w+)", self._handle_drop_index),
            (r"^ALTER\s+SYSTEM\s+SET\s+(\w+)\s*=\s*'?([^']+)'?", self._handle_alter_system),
            (r"^VACUUM\s+FULL\s+(\w+)", self._handle_vacuum_full),
            (r"^VACUUM\s+ANALYZE\s+(\w+)", self._handle_vacuum_analyze),
            (r"^VACUUM\s+(\w+)", self._handle_vacuum),
            (r"^ANALYZE\s+(\w+)", self._handle_analyze),
            (r"^SHOW\s+(\w+)", self._handle_show),
            (r"^SELECT\s+current_setting\('(\w+)'\)", self._handle_show_setting),
        ]

        for pattern, handler in handlers:
            m = re.match(pattern, cmd, re.IGNORECASE)
            if m:
                return handler(m)

        return f"ERROR: unrecognized command: {cmd}", False

    # ── Command handlers ─────────────────────────────────────────────

    def _handle_list_tables(self, m) -> Tuple[str, bool]:
        header = f"{'Schema':<10} {'Name':<20} {'Type':<10} {'Rows':>12}"
        sep = "-" * len(header)
        lines = [header, sep]
        for t in sorted(self.tables.values(), key=lambda x: x.name):
            lines.append(f"{'public':<10} {t.name:<20} {'table':<10} {t.row_count:>12,}")
        lines.append(f"({len(self.tables)} rows)")
        return "\n".join(lines), False

    def _handle_list_indexes(self, m) -> Tuple[str, bool]:
        header = f"{'Name':<35} {'Table':<20} {'Columns':<30}"
        sep = "-" * len(header)
        lines = [header, sep]
        for idx in sorted(self.indexes.values(), key=lambda x: x.name):
            cols = ", ".join(idx.columns)
            partial = f" WHERE {idx.partial_condition}" if idx.partial_condition else ""
            lines.append(f"{idx.name:<35} {idx.table:<20} {cols + partial:<30}")
        lines.append(f"({len(self.indexes)} rows)")
        return "\n".join(lines), False

    def _handle_describe_table(self, m) -> Tuple[str, bool]:
        table_name = m.group(1).lower()
        table = self.tables.get(table_name)
        if not table:
            return f'ERROR: relation "{table_name}" does not exist', False

        lines = [f"Table: public.{table_name}"]
        header = f"  {'Column':<20} {'Type':<15}"
        lines.append(header)
        lines.append("  " + "-" * 35)
        type_map = {
            "id": "integer", "email": "text", "name": "text", "token": "text",
            "created_at": "timestamptz", "updated_at": "timestamptz",
            "expires_at": "timestamptz", "status": "text", "total": "numeric",
            "user_id": "integer", "order_id": "integer", "product_id": "integer",
            "warehouse_id": "integer", "quantity": "integer", "unit_price": "numeric",
            "price": "numeric", "stock": "integer", "method": "text",
            "amount": "numeric", "category": "text", "table_name": "text",
            "row_id": "integer", "action": "text", "payload": "jsonb",
        }
        for col in table.columns:
            ctype = type_map.get(col, "text")
            lines.append(f"  {col:<20} {ctype:<15}")

        # Show indexes on this table
        table_indexes = [i for i in self.indexes.values() if i.table == table_name]
        if table_indexes:
            lines.append(f"\nIndexes:")
            for idx in table_indexes:
                cols = ", ".join(idx.columns)
                lines.append(f'  "{idx.name}" btree ({cols})')

        lines.append(f"\nRows: {table.row_count:,}  Dead tuples: {table.dead_tuples:,}  Bloat: {table.bloat_ratio:.1%}")
        if table.locked_until >= self.step_count:
            lines.append("⚠️  TABLE IS LOCKED (blocking operation in progress)")
        return "\n".join(lines), False

    def _handle_explain(self, m) -> Tuple[str, bool]:
        query = m.group(2).strip()
        # Parse the query to find table and filter columns
        table_match = re.search(r"FROM\s+(\w+)", query, re.IGNORECASE)
        if not table_match:
            return "ERROR: could not parse query", False
        table_name = table_match.group(1).lower()
        if table_name not in self.tables:
            return f'ERROR: relation "{table_name}" does not exist', False

        # Extract WHERE columns
        filter_cols = []
        where_match = re.search(r"WHERE\s+(.+?)(?:ORDER|GROUP|LIMIT|$)", query, re.IGNORECASE)
        if where_match:
            where_clause = where_match.group(1)
            col_matches = re.findall(r"(\w+)\s*[=><]", where_clause)
            filter_cols = [c.lower() for c in col_matches
                          if c.lower() in self.tables[table_name].columns]

        has_order = bool(re.search(r"ORDER\s+BY", query, re.IGNORECASE))
        selectivity = 0.001  # default guess
        # Try to match a known hot query for better selectivity
        for hq in HOT_QUERIES:
            if hq["table"] == table_name and set(hq["filter_columns"]) == set(filter_cols):
                selectivity = hq["selectivity"]
                break

        cost, plan_type = self.compute_explain_cost(
            table_name, filter_cols, selectivity, has_order,
        )
        table = self.tables[table_name]
        matched_rows = max(1, int(table.row_count * selectivity))
        latency = self.cost_to_latency_ms(cost)

        lines = [
            f"QUERY PLAN",
            f"-" * 60,
            f" {plan_type} on {table_name}  (cost=0.00..{cost:.2f} rows={matched_rows} width={table.avg_row_width})",
        ]
        if filter_cols:
            lines.append(f"   Filter: ({' AND '.join(c + ' = $1' for c in filter_cols)})")
        if has_order:
            lines.append(f"   Sort Key: specified column(s)")
        lines.append(f" Planning Time: 0.12 ms")
        lines.append(f" Execution Time: {latency:.2f} ms")
        return "\n".join(lines), False

    def _handle_pg_stat_activity(self, m) -> Tuple[str, bool]:
        if not self.active_queries:
            return "pid | state | query | duration\n-----+-------+-------+---------\n(0 rows)", False

        lines = [f"{'pid':>6} | {'state':<20} | {'duration_ms':>12} | query"]
        lines.append("-" * 80)
        for q in self.active_queries:
            dur = (self.step_count - q.started_step) * 5000 + q.duration_ms
            query_preview = q.query[:50]
            lines.append(f"{q.pid:>6} | {q.state:<20} | {dur:>12.0f} | {query_preview}")
        lines.append(f"({len(self.active_queries)} rows)")
        return "\n".join(lines), False

    def _handle_pg_locks(self, m) -> Tuple[str, bool]:
        lock_entries = [q for q in self.active_queries if q.is_blocking]
        if not lock_entries and not self.blocking_ops:
            return "pid | locktype | relation | mode | granted\n(0 rows)", False

        lines = [f"{'pid':>6} | {'locktype':<15} | {'relation':<20} | {'mode':<20} | granted"]
        lines.append("-" * 85)
        for q in lock_entries:
            tbl = q.blocked_table or "unknown"
            lines.append(f"{q.pid:>6} | {'relation':<15} | {tbl:<20} | {'RowExclusiveLock':<20} | t")
        for op in self.blocking_ops:
            lines.append(f"{'DDL':>6} | {'relation':<15} | {op.table:<20} | {'AccessExclusiveLock':<20} | t")
        lines.append(f"({len(lock_entries) + len(self.blocking_ops)} rows)")
        return "\n".join(lines), False

    def _handle_pg_stat_user_tables(self, m) -> Tuple[str, bool]:
        lines = [f"{'table':<20} | {'seq_scan':>10} | {'idx_scan':>10} | {'n_live':>12} | {'n_dead':>12} | {'bloat':>6} | last_vacuum"]
        lines.append("-" * 100)
        for t in sorted(self.tables.values(), key=lambda x: x.name):
            seq = self.seq_scan_counts.get(t.name, 0)
            idx = self.idx_scan_counts.get(t.name, 0)
            vac = f"step {t.last_vacuum}" if t.last_vacuum is not None else "never"
            lines.append(
                f"{t.name:<20} | {seq:>10,} | {idx:>10,} | {t.row_count:>12,} | "
                f"{t.dead_tuples:>12,} | {t.bloat_ratio:>5.1%} | {vac}"
            )
        lines.append(f"({len(self.tables)} rows)")
        return "\n".join(lines), False

    def _handle_pg_stat_statements(self, m) -> Tuple[str, bool]:
        lines = [f"{'query':<55} | {'calls':>8} | {'mean_ms':>10} | {'total_ms':>12}"]
        lines.append("-" * 95)
        for q in HOT_QUERIES:
            cost, _ = self.compute_explain_cost(
                q["table"], q["filter_columns"], q["selectivity"], q["has_order_by"],
            )
            latency = self.cost_to_latency_ms(cost)
            calls = q["weight"] * 1000
            lines.append(f"{q['sql'][:55]:<55} | {calls:>8,} | {latency:>10.2f} | {latency * calls:>12,.0f}")
        lines.append(f"({len(HOT_QUERIES)} rows)")
        return "\n".join(lines), False

    def _handle_terminate_backend(self, m) -> Tuple[str, bool]:
        pid = int(m.group(1))
        removed = [q for q in self.active_queries if q.pid == pid]
        if not removed:
            return f"WARNING: PID {pid} is not a backend process", False
        self.active_queries = [q for q in self.active_queries if q.pid != pid]
        # If this was a blocking query, release locks
        for q in removed:
            if q.blocked_table and q.blocked_table in self.tables:
                self.tables[q.blocked_table].locked_until = -1
        return f" pg_terminate_backend\n----------------------\n t\n(1 row)", False

    def _handle_cancel_backend(self, m) -> Tuple[str, bool]:
        pid = int(m.group(1))
        found = [q for q in self.active_queries if q.pid == pid]
        if not found:
            return f"WARNING: PID {pid} is not a backend process", False
        # pg_cancel_backend is graceful — may not work on all queries
        if found[0].is_blocking:
            return " pg_cancel_backend\n-------------------\n t\n(1 row)\nNOTICE: cancel request sent, but query may not terminate immediately", False
        self.active_queries = [q for q in self.active_queries if q.pid != pid]
        return " pg_cancel_backend\n-------------------\n t\n(1 row)", False

    def _handle_reload_conf(self, m) -> Tuple[str, bool]:
        self._config_needs_reload = False
        return " pg_reload_conf\n----------------\n t\n(1 row)", False

    def _handle_table_size(self, m) -> Tuple[str, bool]:
        table_name = m.group(1).lower()
        table = self.tables.get(table_name)
        if not table:
            return f'ERROR: relation "{table_name}" does not exist', False
        size_bytes = table.pages * 8192
        if size_bytes > 1_073_741_824:
            pretty = f"{size_bytes / 1_073_741_824:.1f} GB"
        elif size_bytes > 1_048_576:
            pretty = f"{size_bytes / 1_048_576:.0f} MB"
        else:
            pretty = f"{size_bytes / 1024:.0f} kB"
        return f" pg_size_pretty\n----------------\n {pretty}\n(1 row)", False

    def _handle_count(self, m) -> Tuple[str, bool]:
        table_name = m.group(1).lower()
        table = self.tables.get(table_name)
        if not table:
            return f'ERROR: relation "{table_name}" does not exist', False
        return f" count\n--------\n {table.row_count:,}\n(1 row)", False

    def _handle_create_index_concurrently(self, m) -> Tuple[str, bool]:
        return self._create_index_impl(m.group(1), concurrent=True)

    def _handle_create_index(self, m) -> Tuple[str, bool]:
        return self._create_index_impl(m.group(1), concurrent=False)

    def _create_index_impl(self, definition: str, concurrent: bool) -> Tuple[str, bool]:
        # Parse: [IF NOT EXISTS] idx_name ON table(col1, col2) [WHERE ...]
        pattern = r"(?:IF\s+NOT\s+EXISTS\s+)?(\w+)\s+ON\s+(\w+)\s*\(([^)]+)\)(?:\s+WHERE\s+(.+))?"
        m2 = re.match(pattern, definition.strip(), re.IGNORECASE)
        if not m2:
            return "ERROR: syntax error in CREATE INDEX", False

        idx_name = m2.group(1).lower()
        table_name = m2.group(2).lower()
        columns = [c.strip().lower() for c in m2.group(3).split(",")]
        partial = m2.group(4).strip() if m2.group(4) else None

        if table_name not in self.tables:
            return f'ERROR: relation "{table_name}" does not exist', False

        table = self.tables[table_name]
        for col in columns:
            if col not in table.columns:
                return f'ERROR: column "{col}" does not exist', False

        if idx_name in self.indexes:
            return f'ERROR: relation "{idx_name}" already exists', False

        # Check if table is currently locked
        if table.locked_until >= self.step_count:
            return "ERROR: could not obtain lock on relation — another operation in progress", False

        idx = IndexState(
            name=idx_name, table=table_name, columns=columns,
            partial_condition=partial,
            size_pages=max(1, int(math.log2(max(2, table.row_count)) * 10)),
        )
        self.indexes[idx_name] = idx

        if concurrent:
            # Non-blocking: takes longer but doesn't lock
            return f"CREATE INDEX CONCURRENTLY\n-- Index {idx_name} created (non-blocking, safe for production)", False
        else:
            # Blocking: locks the table for writes
            lock_duration = max(1, int(math.log10(max(10, table.row_count))))  # 1-3 steps
            table.locked_until = self.step_count + lock_duration
            self.blocking_ops.append(BlockingOp(
                table=table_name, operation="CREATE INDEX",
                started_step=self.step_count, duration_steps=lock_duration,
            ))
            return f"CREATE INDEX\n-- ⚠️  Table '{table_name}' LOCKED for ~{lock_duration * 5}s (use CONCURRENTLY to avoid this)", True

    def _handle_drop_index(self, m) -> Tuple[str, bool]:
        idx_name = m.group(1).lower()
        if idx_name not in self.indexes:
            return f'ERROR: index "{idx_name}" does not exist', False
        del self.indexes[idx_name]
        return "DROP INDEX", False

    def _handle_alter_system(self, m) -> Tuple[str, bool]:
        param = m.group(1).lower()
        value = m.group(2).strip().strip("'")
        if param not in self.config:
            return f'ERROR: unrecognized configuration parameter "{param}"', False
        self.config[param] = value
        self._config_needs_reload = True
        return f"ALTER SYSTEM SET\n-- Configuration will take effect after pg_reload_conf()", False

    def _handle_vacuum_full(self, m) -> Tuple[str, bool]:
        table_name = m.group(1).lower()
        table = self.tables.get(table_name)
        if not table:
            return f'ERROR: relation "{table_name}" does not exist', False

        # VACUUM FULL: exclusive lock, rewrites table
        lock_duration = max(1, int(math.log10(max(10, table.row_count + table.dead_tuples))))
        table.locked_until = self.step_count + lock_duration
        self.blocking_ops.append(BlockingOp(
            table=table_name, operation="VACUUM FULL",
            started_step=self.step_count, duration_steps=lock_duration,
        ))
        old_dead = table.dead_tuples
        table.dead_tuples = 0
        table.last_vacuum = self.step_count
        table.last_analyze = self.step_count
        table.stats_accurate = True
        return (
            f"VACUUM FULL\n-- Removed {old_dead:,} dead tuples from {table_name}\n"
            f"-- ⚠️  Table LOCKED for ~{lock_duration * 5}s (use VACUUM without FULL to avoid this)"
        ), True

    def _handle_vacuum_analyze(self, m) -> Tuple[str, bool]:
        table_name = m.group(1).lower()
        table = self.tables.get(table_name)
        if not table:
            return f'ERROR: relation "{table_name}" does not exist', False

        old_dead = table.dead_tuples
        table.dead_tuples = max(0, table.dead_tuples - int(table.dead_tuples * 0.8))
        table.last_vacuum = self.step_count
        table.last_analyze = self.step_count
        table.stats_accurate = True
        return f"VACUUM ANALYZE\n-- Reclaimed ~{old_dead - table.dead_tuples:,} dead tuples, updated statistics", False

    def _handle_vacuum(self, m) -> Tuple[str, bool]:
        table_name = m.group(1).lower()
        table = self.tables.get(table_name)
        if not table:
            return f'ERROR: relation "{table_name}" does not exist', False

        old_dead = table.dead_tuples
        table.dead_tuples = max(0, table.dead_tuples - int(table.dead_tuples * 0.8))
        table.last_vacuum = self.step_count
        return f"VACUUM\n-- Reclaimed ~{old_dead - table.dead_tuples:,} dead tuples from {table_name}", False

    def _handle_analyze(self, m) -> Tuple[str, bool]:
        table_name = m.group(1).lower()
        table = self.tables.get(table_name)
        if not table:
            return f'ERROR: relation "{table_name}" does not exist', False
        table.last_analyze = self.step_count
        table.stats_accurate = True
        return f"ANALYZE\n-- Updated statistics for {table_name}", False

    def _handle_show(self, m) -> Tuple[str, bool]:
        param = m.group(1).lower()
        value = self.config.get(param)
        if value is None:
            return f'ERROR: unrecognized configuration parameter "{param}"', False
        return f" {param}\n{'-' * max(len(param), len(str(value)))}\n {value}\n(1 row)", False

    def _handle_show_setting(self, m) -> Tuple[str, bool]:
        param = m.group(1).lower()
        value = self.config.get(param)
        if value is None:
            return f'ERROR: unrecognized configuration parameter "{param}"', False
        return f" current_setting\n-----------------\n {value}\n(1 row)", False

    # ── Helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _parse_mem(value: str) -> int:
        """Parse PostgreSQL memory string to bytes."""
        value = value.strip().upper()
        if value.endswith("GB"):
            return int(float(value[:-2]) * 1024 * 1024 * 1024)
        if value.endswith("MB"):
            return int(float(value[:-2]) * 1024 * 1024)
        if value.endswith("KB"):
            return int(float(value[:-2]) * 1024)
        try:
            return int(value)
        except ValueError:
            return 4 * 1024 * 1024  # default 4MB

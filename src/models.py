"""Data models for LivePatch environment."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class TableState:
    """State of a single database table."""
    name: str
    columns: List[str]
    row_count: int
    avg_row_width: int
    dead_tuples: int = 0
    last_vacuum: Optional[int] = None   # step number
    last_analyze: Optional[int] = None  # step number
    stats_accurate: bool = True
    locked_until: int = -1  # step until which table is exclusively locked
    pk: str = "id"

    @property
    def pages(self) -> int:
        total_rows = self.row_count + self.dead_tuples
        return max(1, (total_rows * self.avg_row_width) // 8192)

    @property
    def bloat_ratio(self) -> float:
        if self.row_count == 0:
            return 0.0
        return self.dead_tuples / (self.row_count + self.dead_tuples)


@dataclass
class IndexState:
    """State of a database index."""
    name: str
    table: str
    columns: List[str]
    partial_condition: Optional[str] = None
    size_pages: int = 0

    def covers(self, filter_columns: List[str]) -> bool:
        """Check if this index covers the given filter columns (prefix match)."""
        if not filter_columns:
            return False
        for i, col in enumerate(filter_columns):
            if i >= len(self.columns):
                return False
            if self.columns[i] != col:
                return False
        return True


@dataclass
class ActiveQuery:
    """A running query in pg_stat_activity."""
    pid: int
    query: str
    state: str  # active, idle, idle in transaction
    started_step: int
    is_blocking: bool = False
    blocked_table: Optional[str] = None
    duration_ms: float = 0.0


@dataclass
class BlockingOp:
    """A blocking DDL operation in progress."""
    table: str
    operation: str  # CREATE INDEX, VACUUM FULL
    started_step: int
    duration_steps: int  # how many steps it takes

    @property
    def end_step(self) -> int:
        return self.started_step + self.duration_steps


@dataclass
class TrafficMetrics:
    """Traffic metrics for one step."""
    requests_total: int = 0
    requests_ok: int = 0
    requests_failed: int = 0
    avg_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    errors_by_query: Dict[str, int] = field(default_factory=dict)


@dataclass
class FaultSpec:
    """Specification for an injected fault."""
    fault_type: str
    target_table: str
    details: Dict[str, Any] = field(default_factory=dict)
    description: str = ""


@dataclass
class DatabaseSnapshot:
    """Full snapshot of database state for reward computation."""
    total_explain_cost: float = 0.0
    worst_query_cost: float = 0.0
    worst_query_name: str = ""
    active_lock_count: int = 0
    blocked_query_count: int = 0
    idx_scan_ratio: float = 0.0
    total_dead_tuples: int = 0
    sla_violations: int = 0  # queries above SLA threshold


@dataclass
class StepResult:
    """Result of a single environment step."""
    command_output: str
    reward: float
    done: bool
    traffic: TrafficMetrics
    db_snapshot: DatabaseSnapshot


@dataclass
class TaskResult:
    """Final result of a completed episode."""
    score: float
    fix_quality: float      # 0-1: how much EXPLAIN cost improved
    uptime: float           # 0-1: fraction of traffic that succeeded
    efficiency: float       # 0-1: how few steps used
    safety: float           # 0-1: how few unsafe operations
    faults_injected: List[str] = field(default_factory=list)
    faults_resolved: List[str] = field(default_factory=list)
    steps_used: int = 0
    total_reward: float = 0.0

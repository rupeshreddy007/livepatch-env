"""Configuration for LivePatch environment."""
from dataclasses import dataclass, field
from typing import Optional, List

# ── Fault types ──────────────────────────────────────────────────────
FAULT_MISSING_INDEX = "missing_index"
FAULT_LOCK_STORM = "lock_storm"
FAULT_TABLE_BLOAT = "table_bloat"
FAULT_BAD_CONFIG = "bad_config"
FAULT_CONNECTION_FLOOD = "connection_flood"
FAULT_RUNAWAY_QUERY = "runaway_query"
FAULT_STALE_STATS = "stale_stats"
FAULT_SORT_SPILL = "sort_spill"

ALL_FAULT_TYPES = [
    FAULT_MISSING_INDEX, FAULT_LOCK_STORM, FAULT_TABLE_BLOAT,
    FAULT_BAD_CONFIG, FAULT_CONNECTION_FLOOD, FAULT_RUNAWAY_QUERY,
    FAULT_STALE_STATS, FAULT_SORT_SPILL,
]

# ── Difficulty tiers ─────────────────────────────────────────────────
DIFFICULTY_EASY = "easy"
DIFFICULTY_MEDIUM = "medium"
DIFFICULTY_HARD = "hard"
DIFFICULTY_EXPERT = "expert"

ALL_DIFFICULTIES = [DIFFICULTY_EASY, DIFFICULTY_MEDIUM, DIFFICULTY_HARD, DIFFICULTY_EXPERT]

# ── Schema: e-commerce database ──────────────────────────────────────
TABLES = {
    "users": {
        "columns": ["id", "email", "name", "created_at", "status"],
        "row_count": 500_000,
        "avg_row_width": 120,
        "pk": "id",
    },
    "orders": {
        "columns": ["id", "user_id", "status", "total", "created_at", "updated_at"],
        "row_count": 8_000_000,
        "avg_row_width": 96,
        "pk": "id",
    },
    "order_items": {
        "columns": ["id", "order_id", "product_id", "quantity", "unit_price"],
        "row_count": 25_000_000,
        "avg_row_width": 48,
        "pk": "id",
    },
    "products": {
        "columns": ["id", "name", "category", "price", "stock", "created_at"],
        "row_count": 50_000,
        "avg_row_width": 140,
        "pk": "id",
    },
    "inventory": {
        "columns": ["id", "product_id", "warehouse_id", "quantity", "updated_at"],
        "row_count": 200_000,
        "avg_row_width": 52,
        "pk": "id",
    },
    "payments": {
        "columns": ["id", "order_id", "method", "amount", "status", "created_at"],
        "row_count": 8_000_000,
        "avg_row_width": 80,
        "pk": "id",
    },
    "sessions": {
        "columns": ["id", "user_id", "token", "created_at", "expires_at"],
        "row_count": 2_000_000,
        "avg_row_width": 108,
        "pk": "id",
    },
    "audit_log": {
        "columns": ["id", "table_name", "row_id", "action", "payload", "created_at"],
        "row_count": 5_000_000,
        "avg_row_width": 256,
        "pk": "id",
    },
}

# Hot queries the "app" continuously runs
HOT_QUERIES = [
    {
        "name": "checkout_lookup",
        "sql": "SELECT * FROM orders WHERE user_id = $1 AND status = 'active'",
        "table": "orders",
        "filter_columns": ["user_id", "status"],
        "selectivity": 0.00005,
        "has_order_by": False,
        "weight": 3,  # relative traffic frequency
    },
    {
        "name": "order_detail",
        "sql": "SELECT oi.*, p.name FROM order_items oi JOIN products p ON oi.product_id = p.id WHERE oi.order_id = $1",
        "table": "order_items",
        "filter_columns": ["order_id"],
        "selectivity": 0.0001,
        "has_order_by": False,
        "weight": 2,
    },
    {
        "name": "stock_check",
        "sql": "SELECT * FROM inventory WHERE product_id = $1 AND quantity > 0",
        "table": "inventory",
        "filter_columns": ["product_id"],
        "selectivity": 0.005,
        "has_order_by": False,
        "weight": 3,
    },
    {
        "name": "payment_history",
        "sql": "SELECT * FROM payments WHERE order_id = $1 ORDER BY created_at DESC",
        "table": "payments",
        "filter_columns": ["order_id"],
        "selectivity": 0.0001,
        "has_order_by": True,
        "weight": 1,
    },
    {
        "name": "auth_check",
        "sql": "SELECT * FROM sessions WHERE token = $1 AND expires_at > NOW()",
        "table": "sessions",
        "filter_columns": ["token"],
        "selectivity": 0.000001,
        "has_order_by": False,
        "weight": 5,
    },
    {
        "name": "daily_orders",
        "sql": "SELECT count(*) FROM orders WHERE created_at > NOW() - INTERVAL '24 hours'",
        "table": "orders",
        "filter_columns": ["created_at"],
        "selectivity": 0.003,
        "has_order_by": False,
        "weight": 1,
    },
    {
        "name": "user_profile",
        "sql": "SELECT u.*, count(o.id) FROM users u JOIN orders o ON u.id = o.user_id WHERE u.email = $1 GROUP BY u.id",
        "table": "users",
        "filter_columns": ["email"],
        "selectivity": 0.000002,
        "has_order_by": False,
        "weight": 2,
    },
    {
        "name": "audit_recent",
        "sql": "SELECT * FROM audit_log WHERE table_name = $1 AND created_at > $2 ORDER BY created_at DESC LIMIT 100",
        "table": "audit_log",
        "filter_columns": ["table_name", "created_at"],
        "selectivity": 0.001,
        "has_order_by": True,
        "weight": 1,
    },
]

# Default healthy indexes (present in a healthy database)
DEFAULT_INDEXES = [
    {"name": "idx_orders_user_id", "table": "orders", "columns": ["user_id"]},
    {"name": "idx_orders_status", "table": "orders", "columns": ["status"]},
    {"name": "idx_orders_created_at", "table": "orders", "columns": ["created_at"]},
    {"name": "idx_order_items_order_id", "table": "order_items", "columns": ["order_id"]},
    {"name": "idx_order_items_product_id", "table": "order_items", "columns": ["product_id"]},
    {"name": "idx_inventory_product_id", "table": "inventory", "columns": ["product_id"]},
    {"name": "idx_payments_order_id", "table": "payments", "columns": ["order_id"]},
    {"name": "idx_payments_created_at", "table": "payments", "columns": ["created_at"]},
    {"name": "idx_sessions_token", "table": "sessions", "columns": ["token"]},
    {"name": "idx_sessions_expires_at", "table": "sessions", "columns": ["expires_at"]},
    {"name": "idx_users_email", "table": "users", "columns": ["email"]},
    {"name": "idx_audit_log_table_name", "table": "audit_log", "columns": ["table_name"]},
    {"name": "idx_audit_log_created_at", "table": "audit_log", "columns": ["created_at"]},
]


@dataclass
class EnvironmentConfig:
    """Configuration for a LivePatch episode."""
    difficulty: str = DIFFICULTY_MEDIUM
    max_steps: int = 25
    seed: Optional[int] = None

    # Traffic simulation
    traffic_rps: int = 100
    sla_latency_ms: float = 200.0
    step_duration_seconds: float = 5.0  # simulated time per agent step

    # Rewards
    reward_cost_improvement: float = 5.0
    reward_sla_met: float = 2.0
    reward_uptime_bonus: float = 1.5
    penalty_downtime_per_error: float = -0.01
    penalty_repeated_cmd: float = -0.15
    penalty_unsafe_op: float = -1.0
    bonus_efficiency: float = 1.5
    bonus_safe_operation: float = 0.5
    bonus_correct_phase: float = 0.3

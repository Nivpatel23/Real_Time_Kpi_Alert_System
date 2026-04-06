"""
Database schema definitions and initialization.

WHY: Idempotent schema creation (IF NOT EXISTS) means the pipeline can
bootstrap itself on first run without manual DB setup. Indexes on
frequently queried columns (kpi_name, timestamp) prevent slow scans.
"""

from src.database.connection import get_connection
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# ─── DDL Statements ───────────────────────────────────────────────

CREATE_KPI_READINGS = """
CREATE TABLE IF NOT EXISTS kpi_readings (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    kpi_name        TEXT NOT NULL,
    value           REAL NOT NULL,
    timestamp       DATETIME NOT NULL,
    source          TEXT NOT NULL DEFAULT 'unknown',
    symbol          TEXT,                          -- e.g., 'AAPL' for stock data
    metadata_json   TEXT,                          -- flexible field for extra context
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP
);
"""

CREATE_ALERTS = """
CREATE TABLE IF NOT EXISTS alerts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    kpi_name        TEXT NOT NULL,
    alert_type      TEXT NOT NULL,                 -- 'z_score', 'threshold', 'rolling_avg', 'pct_change'
    severity        TEXT NOT NULL DEFAULT 'MEDIUM', -- 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
    message         TEXT NOT NULL,
    kpi_value       REAL NOT NULL,
    threshold_value REAL,
    z_score         REAL,
    symbol          TEXT,
    triggered_at    DATETIME NOT NULL,
    acknowledged    BOOLEAN DEFAULT 0,
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP
);
"""

CREATE_KPI_THRESHOLDS = """
CREATE TABLE IF NOT EXISTS kpi_thresholds (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    kpi_name            TEXT NOT NULL UNIQUE,
    lower_bound         REAL,
    upper_bound         REAL,
    z_score_threshold   REAL DEFAULT 2.5,
    rolling_window      INTEGER DEFAULT 20,
    pct_change_threshold REAL DEFAULT 0.10,
    is_active           BOOLEAN DEFAULT 1,
    updated_at          DATETIME DEFAULT CURRENT_TIMESTAMP
);
"""

# ─── Indexes for query performance ────────────────────────────────

CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_readings_kpi_ts ON kpi_readings(kpi_name, timestamp);",
    "CREATE INDEX IF NOT EXISTS idx_readings_symbol ON kpi_readings(symbol, timestamp);",
    "CREATE INDEX IF NOT EXISTS idx_alerts_kpi ON alerts(kpi_name, triggered_at);",
    "CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(severity, acknowledged);",
]


def initialize_database() -> None:
    """
    Create all tables and indexes. Safe to call multiple times.
    This is the 'migration' step — in production you'd use Alembic or Flyway.
    """
    with get_connection() as conn:
        conn.execute(CREATE_KPI_READINGS)
        conn.execute(CREATE_ALERTS)
        conn.execute(CREATE_KPI_THRESHOLDS)

        for idx_sql in CREATE_INDEXES:
            conn.execute(idx_sql)

    logger.info("Database schema initialized successfully.")

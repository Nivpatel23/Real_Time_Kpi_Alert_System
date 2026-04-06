"""
Database CRUD operations for KPI readings and alerts.

WHY: Encapsulating SQL in a dedicated module prevents SQL from leaking
into business logic. Parameterized queries prevent SQL injection.
Batch inserts improve performance for bulk data loads.
"""

import sqlite3
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd

from src.database.connection import get_connection
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class DatabaseOperations:
    """Handles all database read/write operations for the KPI system."""

    # ─── KPI Readings ──────────────────────────────────────────

    @staticmethod
    def insert_kpi_readings(readings: pd.DataFrame) -> int:
        """
        Bulk insert KPI readings from a DataFrame.

        Expected columns: kpi_name, value, timestamp, source, symbol (optional)
        Returns: Number of rows inserted
        """
        if readings.empty:
            logger.warning("No readings to insert — empty DataFrame.")
            return 0

        required_cols = {"kpi_name", "value", "timestamp", "source"}
        if not required_cols.issubset(readings.columns):
            missing = required_cols - set(readings.columns)
            raise ValueError(f"Missing required columns: {missing}")

        records = []
        for _, row in readings.iterrows():
            records.append((
                row["kpi_name"],
                float(row["value"]),
                str(row["timestamp"]),
                row["source"],
                row.get("symbol", None),
                row.get("metadata_json", None),
            ))

        with get_connection() as conn:
            conn.executemany(
                """INSERT INTO kpi_readings 
                   (kpi_name, value, timestamp, source, symbol, metadata_json)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                records
            )

        logger.info(f"Inserted {len(records)} KPI readings into database.")
        return len(records)

    @staticmethod
    def get_kpi_history(
        kpi_name: str,
        symbol: Optional[str] = None,
        limit: int = 100
    ) -> pd.DataFrame:
        """
        Retrieve historical KPI readings for anomaly detection.

        WHY we need history: Z-score and rolling average calculations
        require a baseline of past values to determine what's 'normal'.
        """
        with get_connection() as conn:
            if symbol:
                query = """
                    SELECT kpi_name, value, timestamp, source, symbol
                    FROM kpi_readings
                    WHERE kpi_name = ? AND symbol = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """
                df = pd.read_sql_query(query, conn, params=(kpi_name, symbol, limit))
            else:
                query = """
                    SELECT kpi_name, value, timestamp, source, symbol
                    FROM kpi_readings
                    WHERE kpi_name = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """
                df = pd.read_sql_query(query, conn, params=(kpi_name, limit))

        # Reverse to chronological order (oldest first)
        return df.iloc[::-1].reset_index(drop=True)

    @staticmethod
    def get_latest_reading(kpi_name: str, symbol: Optional[str] = None) -> Optional[Dict]:
        """Get the most recent reading for a KPI."""
        with get_connection() as conn:
            if symbol:
                row = conn.execute(
                    """SELECT * FROM kpi_readings 
                       WHERE kpi_name = ? AND symbol = ?
                       ORDER BY timestamp DESC LIMIT 1""",
                    (kpi_name, symbol)
                ).fetchone()
            else:
                row = conn.execute(
                    """SELECT * FROM kpi_readings 
                       WHERE kpi_name = ? 
                       ORDER BY timestamp DESC LIMIT 1""",
                    (kpi_name,)
                ).fetchone()

        return dict(row) if row else None

    # ─── Alerts ────────────────────────────────────────────────

    @staticmethod
    def insert_alert(alert: Dict[str, Any]) -> int:
        """
        Log an alert to the database for audit trail.

        WHY: Email can fail, be deleted, or missed. Database alerts create
        a permanent, queryable record for dashboards and compliance.
        """
        with get_connection() as conn:
            cursor = conn.execute(
                """INSERT INTO alerts 
                   (kpi_name, alert_type, severity, message, kpi_value, 
                    threshold_value, z_score, symbol, triggered_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    alert["kpi_name"],
                    alert["alert_type"],
                    alert.get("severity", "MEDIUM"),
                    alert["message"],
                    alert["kpi_value"],
                    alert.get("threshold_value"),
                    alert.get("z_score"),
                    alert.get("symbol"),
                    alert.get("triggered_at", datetime.now().isoformat()),
                )
            )
            alert_id = cursor.lastrowid

        logger.info(f"Alert #{alert_id} logged: [{alert.get('severity')}] {alert['kpi_name']}")
        return alert_id

    @staticmethod
    def get_recent_alerts(hours: int = 24, severity: Optional[str] = None) -> pd.DataFrame:
        """Retrieve recent alerts for dashboard or reporting."""
        with get_connection() as conn:
            if severity:
                query = """
                    SELECT * FROM alerts 
                    WHERE triggered_at >= datetime('now', ?) AND severity = ?
                    ORDER BY triggered_at DESC
                """
                df = pd.read_sql_query(
                    query, conn,
                    params=(f"-{hours} hours", severity)
                )
            else:
                query = """
                    SELECT * FROM alerts 
                    WHERE triggered_at >= datetime('now', ?)
                    ORDER BY triggered_at DESC
                """
                df = pd.read_sql_query(query, conn, params=(f"-{hours} hours",))

        return df

    @staticmethod
    def get_alert_summary() -> pd.DataFrame:
        """Aggregate alert counts by KPI and severity — useful for dashboards."""
        with get_connection() as conn:
            query = """
                SELECT 
                    kpi_name,
                    severity,
                    COUNT(*) as alert_count,
                    MIN(triggered_at) as first_alert,
                    MAX(triggered_at) as last_alert
                FROM alerts
                GROUP BY kpi_name, severity
                ORDER BY alert_count DESC
            """
            return pd.read_sql_query(query, conn)

    # ─── Deduplication Check ───────────────────────────────────

    @staticmethod
    def reading_exists(kpi_name: str, timestamp: str, symbol: Optional[str] = None) -> bool:
        """
        Check if a reading already exists to prevent duplicates.

        WHY: If the pipeline runs multiple times (retries, overlapping schedules),
        we don't want duplicate data skewing anomaly detection.
        """
        with get_connection() as conn:
            if symbol:
                row = conn.execute(
                    """SELECT COUNT(*) as cnt FROM kpi_readings 
                       WHERE kpi_name = ? AND timestamp = ? AND symbol = ?""",
                    (kpi_name, timestamp, symbol)
                ).fetchone()
            else:
                row = conn.execute(
                    """SELECT COUNT(*) as cnt FROM kpi_readings 
                       WHERE kpi_name = ? AND timestamp = ?""",
                    (kpi_name, timestamp)
                ).fetchone()

        return row["cnt"] > 0

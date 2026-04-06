"""
Database connection manager with context manager support.

WHY: Raw connection handling leads to leaked connections and locks.
Context managers guarantee cleanup. Factory pattern lets us swap
SQLite for PostgreSQL by changing one function.
"""

import sqlite3
from contextlib import contextmanager
from config.settings import settings
from src.utils.logger import setup_logger
import os

logger = setup_logger(__name__)


@contextmanager
def get_connection(db_path: str = None):
    """
    Context manager for database connections.
    Ensures connections are properly closed even if exceptions occur.

    Usage:
        with get_connection() as conn:
            conn.execute("SELECT ...")

    Yields:
        sqlite3.Connection with row_factory set for dict-like access
    """
    db_path = db_path or settings.database.db_path

    # Ensure the data directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    conn = None
    try:
        conn = sqlite3.connect(
            db_path,
            detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES
        )
        # Enable dict-like row access: row["column_name"]
        conn.row_factory = sqlite3.Row
        # Enable foreign keys (off by default in SQLite)
        conn.execute("PRAGMA foreign_keys = ON")
        # WAL mode for better concurrent read performance
        conn.execute("PRAGMA journal_mode = WAL")

        yield conn
        conn.commit()

    except sqlite3.Error as e:
        logger.error(f"Database error: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()

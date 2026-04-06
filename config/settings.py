"""
Centralized configuration for the KPI Alert System.

WHY: Decouples environment-specific values (DB paths, SMTP credentials,
thresholds) from business logic. Enables easy testing, deployment to
different environments, and prevents secrets from leaking into code.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from dotenv import load_dotenv

# Load .env file if present (development); in production, use real env vars
load_dotenv()


@dataclass
class DatabaseConfig:
    """Database connection settings."""
    db_path: str = os.getenv("DB_PATH", "data/kpi_alerts.db")
    # For PostgreSQL migration, you'd add:
    # pg_host: str = os.getenv("PG_HOST", "localhost")
    # pg_port: int = int(os.getenv("PG_PORT", "5432"))
    # pg_database: str = os.getenv("PG_DATABASE", "kpi_alerts")


@dataclass
class EmailConfig:
    """SMTP email configuration."""
    smtp_server: str = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    smtp_port: int = int(os.getenv("SMTP_PORT", "587"))
    username: str = os.getenv("SMTP_USERNAME", "")
    password: str = os.getenv("SMTP_PASSWORD", "")
    recipients: List[str] = field(default_factory=lambda: [
        r.strip() for r in os.getenv("ALERT_RECIPIENTS", "").split(",") if r.strip()
    ])
    enabled: bool = os.getenv("EMAIL_ALERTS_ENABLED", "false").lower() == "true"


@dataclass
class KPIThreshold:
    """Threshold configuration for a single KPI."""
    kpi_name: str
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    z_score_threshold: float = 2.5         # Flag if |z| > this
    rolling_window: int = 20               # Lookback period for rolling stats
    pct_change_threshold: float = 0.10     # 10% day-over-day change
    is_active: bool = True


@dataclass
class DetectionConfig:
    """Anomaly detection settings."""
    # Default thresholds per KPI — override in DB for dynamic control
    default_thresholds: Dict[str, KPIThreshold] = field(default_factory=lambda: {
        "daily_revenue": KPIThreshold(
            kpi_name="daily_revenue",
            lower_bound=5000.0,
            upper_bound=50000.0,
            z_score_threshold=2.5,
            pct_change_threshold=0.15,
        ),
        "conversion_rate": KPIThreshold(
            kpi_name="conversion_rate",
            lower_bound=0.01,
            upper_bound=0.15,
            z_score_threshold=2.0,
            pct_change_threshold=0.20,
        ),
        "stock_close": KPIThreshold(
            kpi_name="stock_close",
            z_score_threshold=2.5,
            rolling_window=20,
            pct_change_threshold=0.05,
        ),
        "stock_volume": KPIThreshold(
            kpi_name="stock_volume",
            z_score_threshold=3.0,
            rolling_window=20,
            pct_change_threshold=0.50,
        ),
    })
    min_history_points: int = 10  # Minimum data points needed for z-score


@dataclass
class PipelineConfig:
    """Pipeline orchestration settings."""
    run_interval_minutes: int = int(os.getenv("RUN_INTERVAL_MINUTES", "30"))
    data_source: str = os.getenv("DATA_SOURCE", "simulated")  # "simulated" or "yahoo_finance"
    yahoo_symbols: List[str] = field(default_factory=lambda: ["AAPL", "MSFT", "GOOGL"])
    yahoo_period: str = "3mo"      # How far back to fetch
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    max_retries: int = 3
    retry_delay_seconds: int = 5


@dataclass
class Settings:
    """Master configuration — single source of truth."""
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    email: EmailConfig = field(default_factory=EmailConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)


# Singleton instance — import this everywhere
settings = Settings()

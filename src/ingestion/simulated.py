"""
Simulated data generator for testing and demos.

WHY: You need deterministic, controllable data to:
  1. Test anomaly detection without waiting for real anomalies
  2. Demo the system without API dependencies
  3. Run CI/CD pipelines without external calls
  
The generator intentionally injects anomalies so you can verify
the detection engine catches them.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional

from src.ingestion.base import BaseDataSource
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class SimulatedDataSource(BaseDataSource):
    """Generates realistic KPI data with controllable anomalies."""

    def __init__(
        self,
        num_days: int = 90,
        anomaly_rate: float = 0.05,
        seed: Optional[int] = 42
    ):
        """
        Args:
            num_days: Number of days of history to generate
            anomaly_rate: Fraction of data points that are anomalous (0.05 = 5%)
            seed: Random seed for reproducibility (set None for true randomness)
        """
        self.num_days = num_days
        self.anomaly_rate = anomaly_rate
        self.rng = np.random.RandomState(seed)

    def fetch_data(self) -> pd.DataFrame:
        """Generate multi-KPI simulated data with injected anomalies."""
        logger.info(f"Generating {self.num_days} days of simulated KPI data...")

        all_data = []

        # ─── KPI 1: Daily Revenue ──────────────────────────────
        all_data.append(self._generate_kpi(
            kpi_name="daily_revenue",
            base_value=15000.0,
            noise_std=2000.0,
            trend_per_day=10.0,          # Slight upward trend
            seasonality_amplitude=3000.0, # Weekly seasonality
            anomaly_magnitude=3.0,        # Anomalies are 3x normal noise
        ))

        # ─── KPI 2: Conversion Rate ───────────────────────────
        all_data.append(self._generate_kpi(
            kpi_name="conversion_rate",
            base_value=0.045,
            noise_std=0.008,
            trend_per_day=0.0,
            seasonality_amplitude=0.005,
            anomaly_magnitude=4.0,
        ))

        # ─── KPI 3: API Response Time (ms) ────────────────────
        all_data.append(self._generate_kpi(
            kpi_name="api_response_time",
            base_value=120.0,
            noise_std=20.0,
            trend_per_day=0.1,            # Slowly degrading
            seasonality_amplitude=15.0,
            anomaly_magnitude=5.0,        # Spikes are dramatic
        ))

        # ─── KPI 4: Order Count ───────────────────────────────
        all_data.append(self._generate_kpi(
            kpi_name="order_count",
            base_value=250.0,
            noise_std=40.0,
            trend_per_day=0.5,
            seasonality_amplitude=50.0,
            anomaly_magnitude=3.0,
        ))

        result = pd.concat(all_data, ignore_index=True)
        result = self.validate(result)

        logger.info(f"Generated {len(result)} data points across {result['kpi_name'].nunique()} KPIs.")
        return result

    def _generate_kpi(
        self,
        kpi_name: str,
        base_value: float,
        noise_std: float,
        trend_per_day: float,
        seasonality_amplitude: float,
        anomaly_magnitude: float,
    ) -> pd.DataFrame:
        """
        Generate a single KPI time series with trend, seasonality, noise, and anomalies.

        The formula: value = base + trend + seasonality + noise + anomaly_spike
        This mimics real-world KPIs which have all these components.
        """
        dates = [datetime.now() - timedelta(days=self.num_days - i) for i in range(self.num_days)]
        n = len(dates)

        # Components
        trend = np.array([trend_per_day * i for i in range(n)])
        seasonality = seasonality_amplitude * np.sin(2 * np.pi * np.arange(n) / 7)  # Weekly cycle
        noise = self.rng.normal(0, noise_std, n)

        values = base_value + trend + seasonality + noise

        # Inject anomalies at random positions
        num_anomalies = max(1, int(n * self.anomaly_rate))
        anomaly_indices = self.rng.choice(n, size=num_anomalies, replace=False)

        for idx in anomaly_indices:
            # Randomly spike up or drop down
            direction = self.rng.choice([-1, 1])
            values[idx] += direction * anomaly_magnitude * noise_std

        # Ensure non-negative values for metrics that can't be negative
        if kpi_name in ("daily_revenue", "order_count", "api_response_time"):
            values = np.maximum(values, 0)
        if kpi_name == "conversion_rate":
            values = np.clip(values, 0, 1)

        return pd.DataFrame({
            "kpi_name": kpi_name,
            "value": values,
            "timestamp": dates,
            "source": "simulated",
            "symbol": None,
        })

    def validate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate simulated data — check for NaN, correct types."""
        initial_len = len(data)

        # Drop any NaN values
        data = data.dropna(subset=["kpi_name", "value", "timestamp"])

        # Ensure correct types
        data["value"] = pd.to_numeric(data["value"], errors="coerce")
        data = data.dropna(subset=["value"])

        # Ensure timestamps are strings (for consistent DB storage)
        data["timestamp"] = pd.to_datetime(data["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")

        dropped = initial_len - len(data)
        if dropped > 0:
            logger.warning(f"Validation dropped {dropped} invalid rows.")

        return data.reset_index(drop=True)

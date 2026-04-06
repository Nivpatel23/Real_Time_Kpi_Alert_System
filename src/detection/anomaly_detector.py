"""
Multi-strategy anomaly detection engine.

WHY multiple strategies: No single method is universally best.
  - Static thresholds catch absolute violations (e.g., revenue < 0)
  - Z-score catches statistical outliers relative to history
  - Rolling average catches drift from recent trend
  - % change catches sudden shocks

Running all four and deduplicating gives robust, production-grade detection.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from config.settings import settings, KPIThreshold
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class AnomalyResult:
    """Structured representation of a detected anomaly."""
    kpi_name: str
    alert_type: str          # 'threshold_breach', 'z_score', 'rolling_avg', 'pct_change'
    severity: str            # 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
    message: str
    kpi_value: float
    threshold_value: Optional[float] = None
    z_score: Optional[float] = None
    symbol: Optional[str] = None
    triggered_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kpi_name": self.kpi_name,
            "alert_type": self.alert_type,
            "severity": self.severity,
            "message": self.message,
            "kpi_value": self.kpi_value,
            "threshold_value": self.threshold_value,
            "z_score": self.z_score,
            "symbol": self.symbol,
            "triggered_at": self.triggered_at,
        }


class AnomalyDetector:
    """
    Runs multiple anomaly detection strategies against KPI data.

    Usage:
        detector = AnomalyDetector()
        anomalies = detector.analyze(kpi_name="daily_revenue", history=df, latest_value=12500.0)
    """

    def __init__(self):
        self.detection_config = settings.detection

    def analyze(
        self,
        kpi_name: str,
        history: pd.DataFrame,
        latest_value: Optional[float] = None,
        symbol: Optional[str] = None,
    ) -> List[AnomalyResult]:
        """
        Run all detection strategies on a KPI's data.

        Args:
            kpi_name: Name of the KPI being analyzed
            history: DataFrame with at least a 'value' column, ordered chronologically
            latest_value: The most recent value to check (if None, uses last row of history)
            symbol: Optional ticker/entity identifier

        Returns:
            List of AnomalyResult objects (empty if no anomalies detected)
        """
        if history.empty:
            logger.warning(f"No history for {kpi_name}. Skipping detection.")
            return []

        values = history["value"].astype(float).values

        if latest_value is None:
            latest_value = float(values[-1])

        # Get thresholds (from config; could also query DB for dynamic thresholds)
        threshold = self._get_threshold(kpi_name)
        if not threshold.is_active:
            logger.debug(f"Detection disabled for {kpi_name}.")
            return []

        anomalies: List[AnomalyResult] = []

        # ─── Strategy 1: Static Threshold Breach ──────────────
        anomalies.extend(
            self._check_static_threshold(kpi_name, latest_value, threshold, symbol)
        )

        # ─── Strategy 2: Z-Score Outlier Detection ────────────
        if len(values) >= self.detection_config.min_history_points:
            anomalies.extend(
                self._check_z_score(kpi_name, values, latest_value, threshold, symbol)
            )
        else:
            logger.debug(
                f"Insufficient history for z-score ({len(values)} < "
                f"{self.detection_config.min_history_points}). Skipping."
            )

        # ─── Strategy 3: Rolling Average Deviation ────────────
        if len(values) >= threshold.rolling_window:
            anomalies.extend(
                self._check_rolling_average(kpi_name, values, latest_value, threshold, symbol)
            )

        # ─── Strategy 4: Percentage Change Spike ──────────────
        if len(values) >= 2:
            anomalies.extend(
                self._check_pct_change(kpi_name, values, latest_value, threshold, symbol)
            )

        if anomalies:
            logger.info(f"Detected {len(anomalies)} anomalies for {kpi_name}" +
                        (f" [{symbol}]" if symbol else ""))

        return anomalies

    # ─── Detection Strategies ──────────────────────────────────────

    def _check_static_threshold(
        self,
        kpi_name: str,
        value: float,
        threshold: KPIThreshold,
        symbol: Optional[str],
    ) -> List[AnomalyResult]:
        """Check if value falls outside fixed upper/lower bounds."""
        results = []

        if threshold.lower_bound is not None and value < threshold.lower_bound:
            severity = "CRITICAL" if value < threshold.lower_bound * 0.5 else "HIGH"
            results.append(AnomalyResult(
                kpi_name=kpi_name,
                alert_type="threshold_breach",
                severity=severity,
                message=(
                    f"{kpi_name} = {value:.4f} is BELOW lower bound "
                    f"({threshold.lower_bound:.4f})"
                ),
                kpi_value=value,
                threshold_value=threshold.lower_bound,
                symbol=symbol,
            ))

        if threshold.upper_bound is not None and value > threshold.upper_bound:
            severity = "CRITICAL" if value > threshold.upper_bound * 2 else "HIGH"
            results.append(AnomalyResult(
                kpi_name=kpi_name,
                alert_type="threshold_breach",
                severity=severity,
                message=(
                    f"{kpi_name} = {value:.4f} is ABOVE upper bound "
                    f"({threshold.upper_bound:.4f})"
                ),
                kpi_value=value,
                threshold_value=threshold.upper_bound,
                symbol=symbol,
            ))

        return results

    def _check_z_score(
        self,
        kpi_name: str,
        values: np.ndarray,
        latest_value: float,
        threshold: KPIThreshold,
        symbol: Optional[str],
    ) -> List[AnomalyResult]:
        """
        Z-score measures how many standard deviations a value is from the mean.
        
        WHY z-score: It automatically adapts to the scale and variance of each KPI.
        A \$500 deviation means nothing for revenue (\$15K avg) but is catastrophic
        for a conversion rate (0.04 avg). Z-score normalizes this.
        """
        mean = np.mean(values)
        std = np.std(values)

        # Guard against zero std (all identical values)
        if std == 0:
            logger.debug(f"Zero std for {kpi_name}. Z-score check skipped.")
            return []

        z = (latest_value - mean) / std

        if abs(z) > threshold.z_score_threshold:
            direction = "above" if z > 0 else "below"
            severity = self._z_score_to_severity(abs(z))

            return [AnomalyResult(
                kpi_name=kpi_name,
                alert_type="z_score",
                severity=severity,
                message=(
                    f"{kpi_name} z-score = {z:.2f} ({direction} mean). "
                    f"Value: {latest_value:.4f}, Mean: {mean:.4f}, Std: {std:.4f}"
                ),
                kpi_value=latest_value,
                threshold_value=threshold.z_score_threshold,
                z_score=round(z, 4),
                symbol=symbol,
            )]

        return []

    def _check_rolling_average(
        self,
        kpi_name: str,
        values: np.ndarray,
        latest_value: float,
        threshold: KPIThreshold,
        symbol: Optional[str],
    ) -> List[AnomalyResult]:
        """
        Compare latest value against rolling mean ± k * rolling_std.

        WHY: Z-score uses ALL history, so it's slow to adapt to trends.
        Rolling average focuses on RECENT behavior — catching anomalies
        relative to what the KPI has been doing lately, not historically.
        """
        window = threshold.rolling_window
        recent_values = values[-window:]

        rolling_mean = np.mean(recent_values)
        rolling_std = np.std(recent_values)

        if rolling_std == 0:
            return []

        deviation = abs(latest_value - rolling_mean) / rolling_std

        if deviation > threshold.z_score_threshold:
            direction = "above" if latest_value > rolling_mean else "below"
            return [AnomalyResult(
                kpi_name=kpi_name,
                alert_type="rolling_avg_deviation",
                severity=self._z_score_to_severity(deviation),
                message=(
                    f"{kpi_name} deviates {deviation:.2f}σ {direction} "
                    f"{window}-period rolling average. "
                    f"Value: {latest_value:.4f}, Rolling Mean: {rolling_mean:.4f}"
                ),
                kpi_value=latest_value,
                threshold_value=rolling_mean,
                z_score=round(deviation, 4),
                symbol=symbol,
            )]

        return []

    def _check_pct_change(
        self,
        kpi_name: str,
        values: np.ndarray,
        latest_value: float,
        threshold: KPIThreshold,
        symbol: Optional[str],
    ) -> List[AnomalyResult]:
        """
        Check if the latest value changed by more than X% from the previous value.

        WHY: Catches sudden shocks that might not breach absolute thresholds
        or z-score limits (e.g., a stock drops 

"""
Tests for the anomaly detection engine.

WHY these tests: They verify each detection strategy independently,
test edge cases (zero std, empty data, boundary values), and confirm
the severity classification logic — the most critical business logic.
"""

import unittest
import pandas as pd
import numpy as np

from src.detection.anomaly_detector import AnomalyDetector, AnomalyResult


class TestAnomalyDetector(unittest.TestCase):

    def setUp(self):
        """Create a detector and sample data for each test."""
        self.detector = AnomalyDetector()

        # Normal data: mean ≈ 100, std ≈ 10
        np.random.seed(42)
        normal_values = np.random.normal(100, 10, 50)
        self.normal_history = pd.DataFrame({
            "kpi_name": "test_kpi",
            "value": normal_values,
            "timestamp": pd.date_range("2024-01-01", periods=50, freq="D").astype(str),
            "source": "test",
        })

    def test_no_anomaly_on_normal_value(self):
        """A value near the mean should NOT trigger any alerts."""
        anomalies = self.detector.analyze(
            kpi_name="stock_close",
            history=self.normal_history,
            latest_value=100.0,
        )
        # Should be empty or very few (no threshold breach expected)
        z_score_alerts = [a for a in anomalies if a.alert_type == "z_score"]
        self.assertEqual(len(z_score_alerts), 0)

    def test_z_score_detects_extreme_value(self):
        """A value 5 std devs away should trigger a z-score alert."""
        extreme_value = 100 + 5 * 10  # 5 sigma above mean
        anomalies = self.detector.analyze(
            kpi_name="stock_close",
            history=self.normal_history,
            latest_value=extreme_value,
        )
        z_alerts = [a for a in anomalies if a.alert_type == "z_score"]
        self.assertGreater(len(z_alerts), 0)
        self.assertIn(z_alerts[0].severity, ["HIGH", "CRITICAL"])

    def test_pct_change_detects_spike(self):
        """A 50% jump from the previous value should trigger pct_change alert."""
        # Modify the last value to create a known previous value
        history = self.normal_history.copy()
        history.iloc[-1, history.columns.get_loc("value")] = 100.0

        anomalies = self.detector.analyze(
            kpi_name="stock_close",
            history=history,
            latest_value=155.0,  # 55% increase from 100
        )
        pct_alerts = [a for a in anomalies if a.alert_type == "pct_change_spike"]
        self.assertGreater(len(pct_alerts), 0)

    def test_empty_history_returns_no_anomalies(self):
        """Empty DataFrame should return empty results, not crash."""
        anomalies = self.detector.analyze(
            kpi_name="test_kpi",
            history=pd.DataFrame(),
            latest_value=100.0,
        )
        self.assertEqual(len(anomalies), 0)

    def test_insufficient_history_skips_z_score(self):
        """With < min_history_points, z-score should be skipped gracefully."""
        tiny_history = self.normal_history.head(3)
        anomalies = self.detector.analyze(
            kpi_name="stock_close",
            history=tiny_history,
            latest_value=200.0,
        )
        z_alerts = [a for a in anomalies if a.alert_type == "z_score"]
        self.assertEqual(len(z_alerts), 0)  # Skipped, not errored

    def test_zero_std_handles_gracefully(self):
        """All identical values (std=0) should not cause division by zero."""
        constant_history = pd.DataFrame({
            "kpi_name": "test_kpi",
            "value": [50.0] * 30,
            "timestamp": pd.date_range("2024-01-01", periods=30, freq="D").astype(str),
            "source": "test",
        })
        # Should not raise ZeroDivisionError
        anomalies = self.detector.analyze(
            kpi_name="stock_close",
            history=constant_history,
            latest_value=50.0,
        )
        self.assertIsInstance(anomalies, list)

    def test_batch_analysis(self):
        """Batch analysis should process multiple KPIs and return combined results."""
        data = pd.concat([
            self.normal_history,
            self.normal_history.assign(kpi_name="another_kpi"),
        ], ignore_index=True)

        anomalies = self.detector.analyze_batch(data)
        self.assertIsInstance(anomalies, list)

    def test_severity_classification(self):
        """Verify severity levels are correctly mapped."""
        self.assertEqual(self.detector._z_score_to_severity(5.0), "CRITICAL")
        self.assertEqual(self.detector._z_score_to_severity(3.5), "HIGH")
        self.assertEqual(self.detector._z_score_to_severity(2.7), "MEDIUM")
        self.assertEqual(self.detector._z_score_to_severity(2.0), "LOW")

    def test_anomaly_result_to_dict(self):
        """AnomalyResult should serialize to dict for database storage."""
        result = AnomalyResult(
            kpi_name="test",
            alert_type="z_score",
            severity="HIGH",
            message="Test alert",
            kpi_value=150.0,
            z_score=3.5,
        )
        d = result.to_dict()
        self.assertEqual(d["kpi_name"], "test")
        self.assertEqual(d["z_score"], 3.5)
        self.assertIn("triggered_at", d)


if __name__ == "__main__":
    unittest.main()

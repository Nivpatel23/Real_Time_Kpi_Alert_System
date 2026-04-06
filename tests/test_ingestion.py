"""Tests for data ingestion layer."""

import unittest
import pandas as pd
from src.ingestion.simulated import SimulatedDataSource


class TestSimulatedDataSource(unittest.TestCase):

    def setUp(self):
        self.source = SimulatedDataSource(num_days=30, seed=42)

    def test_fetch_returns_dataframe(self):
        """Fetch should return a non-empty DataFrame."""
        data = self.source.fetch_data()
        self.assertIsInstance(data, pd.DataFrame)
        self.assertGreater(len(data), 0)

    def test_required_columns_present(self):
        """Output must have the standardized columns."""
        data = self.source.fetch_data()
        required = {"kpi_name", "value", "timestamp", "source"}
        self.assertTrue(required.issubset(data.columns))

    def test_multiple_kpis_generated(self):
        """Should generate data for multiple KPI types."""
        data = self.source.fetch_data()
        self.assertGreater(data["kpi_name"].nunique(), 1)

    def test_no_nan_values(self):
        """Validated data should have no NaN in critical columns."""
        data = self.source.fetch_data()
        self.assertEqual(data["value"].isna().sum(), 0)
        self.assertEqual(data["kpi_name"].isna().sum(), 0)

    def test_reproducibility_with_seed(self):
        """Same seed should produce identical data."""
        source1 = SimulatedDataSource(num_days=10, seed=123)
        source2 = SimulatedDataSource(num_days=10, seed=123)
        data1 = source1.fetch_data()
        data2 = source2.fetch_data()
        pd.testing.assert_frame_equal(data1, data2)

    def test_anomalies_are_injected(self):
        """With anomaly_rate > 0, some values should be statistical outliers."""
        source = SimulatedDataSource(num_days=100, anomaly_rate=0.10, seed=42)
        data = source.fetch_data()

        # Check one KPI for high variance (anomalies increase spread)
        revenue = data[data["kpi_name"] == "daily_revenue"]["value"]
        z_scores = (revenue - revenue.mean()) / revenue.std()
        extreme_count = (z_scores.abs() > 2.0).sum()

        self.assertGreater(extreme_count, 0, "Expected some anomalous values.")


if __name__ == "__main__":
    unittest.main()

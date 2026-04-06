"""Tests for database operations using an in-memory SQLite database."""

import unittest
import os
import pandas as pd
from datetime import datetime

from src.database.connection import get_connection
from src.database.schema import initialize_database
from src.database.operations import DatabaseOperations


class TestDatabaseOperations(unittest.TestCase):

    TEST_DB = "data/test_kpi_alerts.db"

    def setUp(self):
        """Create a fresh test database for each test."""
        os.makedirs("data", exist_ok=True)
        os.environ["DB_PATH"] = self.TEST_DB

        # Re-import settings to pick up test DB path
        from config.settings import Settings
        self.settings = Settings()

        initialize_database()
        self.db_ops = DatabaseOperations()

    def tearDown(self):
        """Remove the test database after each test."""
        if os.path.exists(self.TEST_DB):
            os.remove(self.TEST_DB)

    def test_insert_and_retrieve_readings(self):
        """Should insert readings and retrieve them correctly."""
        data = pd.DataFrame({
            "kpi_name": ["test_kpi"] * 3,
            "value": [100.0, 110.0, 105.0],
            "timestamp": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "source": ["test"] * 3,
        })

        inserted = self.db_ops.insert_kpi_readings(data)
        self.assertEqual(inserted, 3)

        history = self.db_ops.get_kpi_history("test_kpi", limit=10)
        self.assertEqual(len(history), 3)

    def test_insert_alert(self):
        """Should insert an alert and assign an ID."""
        alert = {
            "kpi_name": "test_kpi",
            "alert_type": "z_score",
            "severity": "HIGH",
            "message": "Test alert message",
            "kpi_value": 150.0,
            "z_score": 3.5,
            "triggered_at": datetime.now().isoformat(),
        }

        alert_id = self.db_ops.insert_alert(alert)
        self.assertIsInstance(alert_id, int)
        self.assertGreater(alert_id, 0)

    def test_deduplication_check(self):
        """reading_exists should return True for duplicate entries."""
        data = pd.DataFrame({
            "kpi_name": ["test_kpi"],
            "value": [100.0],
            "timestamp": ["2024-01-01"],
            "source": ["test"],
        })
        self.db_ops.insert_kpi_readings(data)

        self.assertTrue(self.db_ops.reading_exists("test_kpi", "2024-01-01"))
        self.assertFalse(self.db_ops.reading_exists("test_kpi", "2024-01-02"))

    def test_empty_dataframe_insert(self):
        """Inserting an empty DataFrame should return 0, not crash."""
        result = self.db_ops.insert_kpi_readings(pd.DataFrame())
        self.assertEqual(result, 0)

    def test_alert_summary_aggregation(self):
        """Alert summary should group by KPI and severity."""
        for severity in ["HIGH", "HIGH", "MEDIUM"]:
            self.db_ops.insert_alert({
                "kpi_name": "test_kpi",
                "alert_type": "z_score",
                "severity": severity,
                "message": "Test",
                "kpi_value": 100.0,
                "triggered_at": datetime.now().isoformat(),
            })

        summary = self.db_ops.get_alert_summary()
        self.assertGreater(len(summary), 0)
        self.assertIn("alert_count", summary.columns)


if __name__ == "__main__":
    unittest.main()

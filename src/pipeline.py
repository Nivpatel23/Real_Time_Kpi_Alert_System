"""
Pipeline orchestrator — coordinates the full ingest → detect → alert cycle.

WHY: Without a central orchestrator, you'd have a fragile script with
interleaved concerns. The pipeline pattern gives you:
  - Clear execution order
  - Error isolation (one step fails, others can still run)
  - Retry logic for transient failures
  - Run metadata for observability
  - Easy scheduling (just call pipeline.run())
"""

import time
from datetime import datetime
from typing import Dict, Any, Optional

import pandas as pd

from config.settings import settings
from src.database import initialize_database, DatabaseOperations
from src.ingestion.simulated import SimulatedDataSource
from src.ingestion.yahoo_finance import YahooFinanceDataSource
from src.detection.anomaly_detector import AnomalyDetector
from src.alerts.alert_manager import AlertManager
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class KPIPipeline:
    """
    End-to-end KPI monitoring pipeline.

    Lifecycle:
        1. Initialize (DB schema, components)
        2. Ingest (fetch + validate data)
        3. Store (deduplicate + persist to SQL)
        4. Detect (run anomaly analysis)
        5. Alert (dispatch through all channels)
        6. Report (log run metadata)
    """

    def __init__(self, data_source: str = None):
        """
        Args:
            data_source: Override for data source ('simulated' or 'yahoo_finance').
                         Defaults to settings.pipeline.data_source.
        """
        self.data_source_type = data_source or settings.pipeline.data_source

        # Initialize components
        self.db_ops = DatabaseOperations()
        self.detector = AnomalyDetector()
        self.alert_manager = AlertManager()

        # Resolve data source
        self.data_source = self._create_data_source()

        logger.info(
            f"Pipeline initialized | Source: {self.data_source_type} | "
            f"DB: {settings.database.db_path}"
        )

    def _create_data_source(self):
        """Factory method — instantiate the correct data source."""
        if self.data_source_type == "yahoo_finance":
            return YahooFinanceDataSource()
        elif self.data_source_type == "simulated":
            return SimulatedDataSource()
        else:
            logger.warning(
                f"Unknown data source '{self.data_source_type}'. Falling back to simulated."
            )
            return SimulatedDataSource()

    def run(self) -> Dict[str, Any]:
        """
        Execute the full pipeline cycle with retry logic.

        Returns:
            Run metadata dict with stats and status.
        """
        run_start = datetime.now()
        run_id = run_start.strftime("%Y%m%d_%H%M%S")

        logger.info(f"{'='*60}")
        logger.info(f"  PIPELINE RUN: {run_id}")
        logger.info(f"  Started: {run_start.isoformat()}")
        logger.info(f"{'='*60}")

        run_metadata = {
            "run_id": run_id,
            "started_at": run_start.isoformat(),
            "data_source": self.data_source_type,
            "status": "UNKNOWN",
            "rows_ingested": 0,
            "rows_stored": 0,
            "anomalies_detected": 0,
            "alerts_dispatched": {},
            "errors": [],
        }

        try:
            # ─── Step 1: Initialize Database ──────────────────
            logger.info("[1/5] Initializing database schema...")
            initialize_database()

            # ─── Step 2: Ingest Data ──────────────────────────
            logger.info("[2/5] Ingesting data...")
            data = self._ingest_with_retry()

            if data.empty:
                run_metadata["status"] = "NO_DATA"
                logger.warning("Pipeline completed with no data.")
                return run_metadata

            run_metadata["rows_ingested"] = len(data)

            # ─── Step 3: Store to Database ────────────────────
            logger.info("[3/5] Storing data to database...")
            rows_stored = self._store_data(data)
            run_metadata["rows_stored"] = rows_stored

            # ─── Step 4: Detect Anomalies ─────────────────────
            logger.info("[4/5] Running anomaly detection...")
            anomalies = self.detector.analyze_batch(data)
            anomaly_dicts = [a.to_dict() for a in anomalies]
            run_metadata["anomalies_detected"] = len(anomaly_dicts)

            # ─── Step 5: Dispatch Alerts ──────────────────────
            logger.info("[5/5] Dispatching alerts...")
            dispatch_summary = self.alert_manager.dispatch(anomaly_dicts)
            run_metadata["alerts_dispatched"] = dispatch_summary

            run_metadata["status"] = "SUCCESS"

        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            run_metadata["status"] = "FAILED"
            run_metadata["errors"].append(str(e))

        finally:
            run_end = datetime.now()
            run_metadata["ended_at"] = run_end.isoformat()
            run_metadata["duration_seconds"] = (run_end - run_start).total_seconds()

            logger.info(f"\n{'─'*60}")
            logger.info(f"  RUN COMPLETE: {run_metadata['status']}")
            logger.info(f"  Duration: {run_metadata['duration_seconds']:.2f}s")
            logger.info(f"  Ingested: {run_metadata['rows_ingested']} rows")
            logger.info(f"  Stored: {run_metadata['rows_stored']} rows")
            logger.info(f"  Anomalies: {run_metadata['anomalies_detected']}")
            logger.info(f"{'─'*60}\n")

        return run_metadata

    def _ingest_with_retry(self) -> pd.DataFrame:
        """
        Fetch data with retry logic for transient failures.

        WHY: Network calls fail. APIs throttle. Servers have hiccups.
        Retrying with exponential backoff handles these gracefully
        without manual intervention.
        """
        max_retries = settings.pipeline.max_retries
        delay = settings.pipeline.retry_delay_seconds

        for attempt in range(1, max_retries + 1):
            try:
                data = self.data_source.fetch_data()
                if not data.empty:
                    return data
                logger.warning(f"Attempt {attempt}: Empty data returned.")
            except Exception as e:
                logger.warning(f"Attempt {attempt}/{max_retries} failed: {e}")

            if attempt < max_retries:
                wait_time = delay * (2 ** (attempt - 1))  # Exponential backoff
                logger.info(f"Retrying in {wait_time}s...")
                time.sleep(wait_time)

        logger.error(f"All {max_retries} ingestion attempts failed.")
        return pd.DataFrame()

    def _store_data(self, data: pd.DataFrame) -> int:
        """
        Store data with deduplication.

        WHY: Without dedup, re-running the pipeline (retries, scheduler
        overlap) would insert duplicate rows, inflating counts and
        skewing statistical calculations for anomaly detection.
        """
        # Filter out rows that already exist in the database
        new_rows = []

        for _, row in data.iterrows():
            if not self.db_ops.reading_exists(
                kpi_name=row["kpi_name"],
                timestamp=str(row["timestamp"]),
                symbol=row.get("symbol"),
            ):
                new_rows.append(row)

        if not new_rows:
            logger.info("All data already exists in database. No new rows to store.")
            return 0

        new_data = pd.DataFrame(new_rows)
        return self.db_ops.insert_kpi_readings(new_data)


class ScheduledPipeline:
    """
    Wraps KPIPipeline with scheduling for continuous monitoring.

    WHY: In production, the pipeline runs on a schedule (every 30 min,
    hourly, etc.). This class handles the scheduling loop, graceful
    shutdown, and run history — without requiring external tools like cron.
    """

    def __init__(self, data_source: str = None):
        self.pipeline = KPIPipeline(data_source=data_source)
        self.run_history = []
        self._running = False

    def start(self, interval_minutes: Optional[int] = None):
        """
        Start the scheduled pipeline loop.

        Args:
            interval_minutes: Override for run interval (default from settings)
        """
        import schedule

        interval = interval_minutes or settings.pipeline.run_interval_minutes

        logger.info(f"Starting scheduled pipeline (every {interval} minutes)...")
        logger.info("Press Ctrl+C to stop.\n")

        # Run immediately on start, then on schedule
        self._execute_run()

        schedule.every(interval).minutes.do(self._execute_run)

        self._running = True
        try:
            while self._running:
                schedule.run_pending()
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Scheduled pipeline stopped by user.")
            self._running = False

    def _execute_run(self):
        """Execute a single pipeline run and track history."""
        metadata = self.pipeline.run()
        self.run_history.append(metadata)

        # Keep only last 100 runs in memory
        if len(self.run_history) > 100:
            self.run_history = self.run_history[-100:]

    def stop(self):
        """Gracefully stop the scheduler."""
        self._running = False
        logger.info("Pipeline stop requested.")

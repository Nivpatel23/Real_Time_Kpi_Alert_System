"""
Entry point for the Real-Time KPI Alert System.

Usage:
    # Single run with simulated data
    python main.py --source simulated --mode single

    # Single run with Yahoo Finance data
    python main.py --source yahoo_finance --mode single

    # Scheduled continuous monitoring
    python main.py --source yahoo_finance --mode scheduled --interval 60

    # Quick demo (simulated data, verbose)
    python main.py --mode demo
"""

import argparse
import json
import sys
import os

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import KPIPipeline, ScheduledPipeline
from src.database import initialize_database, DatabaseOperations
from src.utils.logger import setup_logger

logger = setup_logger("main")


def run_single(data_source: str) -> None:
    """Execute a single pipeline run."""
    logger.info(f"Running single pipeline execution (source={data_source})...")

    pipeline = KPIPipeline(data_source=data_source)
    metadata = pipeline.run()

    logger.info("\n📊 Run Metadata:")
    logger.info(json.dumps(metadata, indent=2, default=str))

    # Print alert summary from database
    print_alert_summary()


def run_scheduled(data_source: str, interval: int) -> None:
    """Run the pipeline on a repeating schedule."""
    scheduler = ScheduledPipeline(data_source=data_source)
    scheduler.start(interval_minutes=interval)


def run_demo() -> None:
    """
    Demo mode: runs simulated data, shows results, prints summary.
    Perfect for portfolio presentations and interviews.
    """
    logger.info("🎯 DEMO MODE — Running with simulated data + anomaly injection...\n")

    pipeline = KPIPipeline(data_source="simulated")
    metadata = pipeline.run()

    print("\n" + "=" * 60)
    print("  📊 DEMO RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Status:           {metadata['status']}")
    print(f"  Rows Ingested:    {metadata['rows_ingested']}")
    print(f"  Rows Stored:      {metadata['rows_stored']}")
    print(f"  Anomalies Found:  {metadata['anomalies_detected']}")
    print(f"  Duration:         {metadata.get('duration_seconds', 0):.2f}s")
    print("=" * 60)

    print_alert_summary()
    print_recent_alerts()


def print_alert_summary() -> None:
    """Print aggregated alert statistics from the database."""
    try:
        db_ops = DatabaseOperations()
        summary = db_ops.get_alert_summary()

        if summary.empty:
            print("\n  ℹ️  No alerts in database yet.")
            return

        print("\n  📋 ALERT SUMMARY (All Time):")
        print("  " + "-" * 55)
        print(f"  {'KPI':<25} {'Severity':<12} {'Count':<8}")
        print("  " + "-" * 55)

        for _, row in summary.iterrows():
            print(f"  {row['kpi_name']:<25} {row['severity']:<12} {row['alert_count']:<8}")

        print("  " + "-" * 55)
        print(f"  Total: {summary['alert_count'].sum()} alerts\n")

    except Exception as e:
        logger.error(f"Could not print alert summary: {e}")


def print_recent_alerts(hours: int = 24) -> None:
    """Print the most recent alerts."""
    try:
        db_ops = DatabaseOperations()
        recent = db_ops.get_recent_alerts(hours=hours)

        if recent.empty:
            return

        print(f"\n  🕐 RECENT ALERTS (Last {hours}h):")
        print("  " + "-" * 70)

        for _, row in recent.head(10).iterrows():
            print(
                f"  [{row['severity']:<8}] {row['kpi_name']:<20} "
                f"| {row['alert_type']:<22} | Value: {row['kpi_value']:.4f}"
            )

        if len(recent) > 10:
            print(f"  ... and {len(recent) - 10} more alerts")

        print("  " + "-" * 70 + "\n")

    except Exception as e:
        logger.error(f"Could not print recent alerts: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Real-Time KPI Alert System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode demo
  python main.py --source yahoo_finance --mode single
  python main.py --source simulated --mode scheduled --interval 15
        """
    )

    parser.add_argument(
        "--source",
        choices=["simulated", "yahoo_finance"],
        default="simulated",
        help="Data source to use (default: simulated)"
    )
    parser.add_argument(
        "--mode",
        choices=["single", "scheduled", "demo"],
        default="demo",
        help="Execution mode (default: demo)"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=30,
        help="Minutes between scheduled runs (default: 30)"
    )

    args = parser.parse_args()

    # Ensure DB is ready
    initialize_database()

    if args.mode == "demo":
        run_demo()
    elif args.mode == "single":
        run_single(args.source)
    elif args.mode == "scheduled":
        run_scheduled(args.source, args.interval)


if __name__ == "__main__":
    main()

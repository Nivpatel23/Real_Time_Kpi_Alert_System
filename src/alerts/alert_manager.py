"""
Multi-channel alert dispatcher.

WHY: Production systems need redundant alerting. If email fails (SMTP
down, rate-limited, wrong credentials), you still need a record of
what happened. SQL logging provides the permanent audit trail.
Console/log output helps during development and debugging.

Architecture: Each channel is independent — failure in one doesn't
block the others (fail-open pattern for alerting).
"""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import List, Dict, Any

from config.settings import settings
from src.database.operations import DatabaseOperations
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class AlertManager:
    """
    Dispatches anomaly alerts through multiple channels.

    Channels:
        1. Console/Log — always active
        2. SQL Database — always active (audit trail)
        3. Email (SMTP) — active when configured and enabled
    """

    def __init__(self):
        self.email_config = settings.email
        self.db_ops = DatabaseOperations()

    def dispatch(self, anomalies: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Send alerts through all active channels.

        Args:
            anomalies: List of anomaly dicts (from AnomalyResult.to_dict())

        Returns:
            Summary dict: {"console": N, "database": N, "email": N}
        """
        if not anomalies:
            logger.info("No anomalies to dispatch.")
            return {"console": 0, "database": 0, "email": 0}

        summary = {"console": 0, "database": 0, "email": 0}

        # ─── Channel 1: Console / Log (always runs) ───────────
        summary["console"] = self._dispatch_console(anomalies)

        # ─── Channel 2: SQL Database (always runs) ────────────
        summary["database"] = self._dispatch_database(anomalies)

        # ─── Channel 3: Email (only if configured) ────────────
        if self.email_config.enabled and self.email_config.recipients:
            summary["email"] = self._dispatch_email(anomalies)
        else:
            logger.debug("Email alerts disabled or no recipients configured.")

        logger.info(
            f"Alert dispatch summary: "
            f"Console={summary['console']}, "
            f"Database={summary['database']}, "
            f"Email={summary['email']}"
        )
        return summary

    # ─── Channel Implementations ───────────────────────────────────

    def _dispatch_console(self, anomalies: List[Dict[str, Any]]) -> int:
        """
        Log alerts to console with severity-appropriate formatting.

        WHY: Even in production with fancy dashboards, console logging
        is your lifeline when debugging. Structured log format means
        these are greppable and parseable by log aggregation tools.
        """
        count = 0

        logger.info("=" * 70)
        logger.info(f"  ALERT REPORT — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"  Total Anomalies Detected: {len(anomalies)}")
        logger.info("=" * 70)

        for anomaly in anomalies:
            severity = anomaly.get("severity", "UNKNOWN")
            icon = self._severity_icon(severity)

            logger.warning(
                f"{icon} [{severity}] {anomaly['kpi_name']}"
                + (f" ({anomaly['symbol']})" if anomaly.get('symbol') else "")
                + f" | Type: {anomaly['alert_type']}"
                + f" | Value: {anomaly['kpi_value']:.4f}"
                + (f" | Z-Score: {anomaly['z_score']:.2f}" if anomaly.get('z_score') else "")
                + f" | {anomaly['message']}"
            )
            count += 1

        logger.info("=" * 70)
        return count

    def _dispatch_database(self, anomalies: List[Dict[str, Any]]) -> int:
        """
        Persist alerts to SQL for audit trail and dashboard queries.

        WHY: This is the most important channel. Emails get lost,
        consoles close, but database records persist. Dashboards
        (Power BI, Streamlit) query this table for alert history.
        """
        count = 0

        for anomaly in anomalies:
            try:
                self.db_ops.insert_alert(anomaly)
                count += 1
            except Exception as e:
                logger.error(f"Failed to log alert to DB: {e} | Alert: {anomaly['kpi_name']}")

        return count

    def _dispatch_email(self, anomalies: List[Dict[str, Any]]) -> int:
        """
        Send email digest of all anomalies (batched, not one email per alert).

        WHY: One email per anomaly = inbox spam = alert fatigue = ignored alerts.
        Batching into a single digest email with a summary table is professional
        and respects the recipient's attention.
        """
        try:
            # Build HTML email body
            html_body = self._build_email_html(anomalies)

            # Count by severity for subject line
            severity_counts = {}
            for a in anomalies:
                sev = a.get("severity", "UNKNOWN")
                severity_counts[sev] = severity_counts.get(sev, 0) + 1

            severity_summary = ", ".join(f"{v} {k}" for k, v in severity_counts.items())
            subject = f"🚨 KPI Alert: {len(anomalies)} anomalies detected ({severity_summary})"

            # Compose message
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = self.email_config.username
            msg["To"] = ", ".join(self.email_config.recipients)

            # Plain text fallback
            plain_text = self._build_email_plaintext(anomalies)
            msg.attach(MIMEText(plain_text, "plain"))
            msg.attach(MIMEText(html_body, "html"))

            # Send via SMTP
            with smtplib.SMTP(self.email_config.smtp_server, self.email_config.smtp_port) as server:
                server.ehlo()
                server.starttls()
                server.ehlo()
                server.login(self.email_config.username, self.email_config.password)
                server.send_message(msg)

            logger.info(f"Alert email sent to {len(self.email_config.recipients)} recipients.")
            return len(anomalies)

        except smtplib.SMTPAuthenticationError:
            logger.error(
                "Email authentication failed. Check SMTP_USERNAME and SMTP_PASSWORD. "
                "For Gmail, use App Passwords (not your regular password)."
            )
            return 0
        except smtplib.SMTPException as e:
            logger.error(f"SMTP error sending alert email: {e}")
            return 0
        except Exception as e:
            logger.error(f"Unexpected error sending email: {e}")
            return 0

    # ─── Email Body Builders ───────────────────────────────────────

    def _build_email_html(self, anomalies: List[Dict[str, Any]]) -> str:
        """
        Build a professional HTML email with a summary table.

        WHY HTML: Plain text emails look amateur. A clean HTML table
        makes alerts scannable at a glance — critical when an executive
        is reading this on their phone at 2 AM.
        """
        rows = ""
        for a in anomalies:
            severity = a.get("severity", "UNKNOWN")
            color = {
                "CRITICAL": "#dc3545",
                "HIGH": "#fd7e14",
                "MEDIUM": "#ffc107",
                "LOW": "#17a2b8",
            }.get(severity, "#6c757d")

            rows += f"""
            <tr>
                <td style="padding:8px;border:1px solid #dee2e6;">
                    <span style="color:{color};font-weight:bold;">{severity}</span>
                </td>
                <td style="padding:8px;border:1px solid #dee2e6;">{a['kpi_name']}</td>
                <td style="padding:8px;border:1px solid #dee2e6;">{a.get('symbol', '—')}</td>
                <td style="padding:8px;border:1px solid #dee2e6;">{a['alert_type']}</td>
                <td style="padding:8px;border:1px solid #dee2e6;">{a['kpi_value']:.4f}</td>
                <td style="padding:8px;border:1px solid #dee2e6;font-size:0.9em;">
                    {a['message']}
                </td>
            </tr>
            """

        html = f"""
        <html>
        <body style="font-family:Arial,sans-serif;color:#333;">
            <h2 style="color:#dc3545;"> KPI Alert Report</h2>
            <p><strong>Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>Total Anomalies:</strong> {len(anomalies)}</p>
            <table style="border-collapse:collapse;width:100%;margin-top:10px;">
                <thead>
                    <tr style="background:#f8f9fa;">
                        <th style="padding:8px;border:1px solid #dee2e6;text-align:left;">Severity</th>
                        <th style="padding:8px;border:1px solid #dee2e6;text-align:left;">KPI</th>
                        <th style="padding:8px;border:1px solid #dee2e6;text-align:left;">Symbol</th>
                        <th style="padding:8px;border:1px solid #dee2e6;text-align:left;">Type</th>
                        <th style="padding:8px;border:1px solid #dee2e6;text-align:left;">Value</th>
                        <th style="padding:8px;border:1px solid #dee2e6;text-align:left;">Details</th>
                    </tr>
                </thead>
                <tbody>
                    {rows}
                </tbody>
            </table>
            <hr style="margin-top:20px;">
            <p style="font-size:0.85em;color:#888;">
                Generated by Real-Time KPI Alert System | 
                <a href="https://github.com/YOUR_USERNAME/real-time-kpi-alert-system">GitHub</a>
            </p>
        </body>
        </html>
        """
        return html

    def _build_email_plaintext(self, anomalies: List[Dict[str, Any]]) -> str:
        """Plain text fallback for email clients that don't render HTML."""
        lines = [
            "=" * 50,
            " KPI ALERT REPORT",
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total Anomalies: {len(anomalies)}",
            "=" * 50,
            "",
        ]

        for i, a in enumerate(anomalies, 1):
            lines.append(
                f"{i}. [{a.get('severity', '?')}] {a['kpi_name']}"
                + (f" ({a['symbol']})" if a.get('symbol') else "")
            )
            lines.append(f"   Type: {a['alert_type']} | Value: {a['kpi_value']:.4f}")
            lines.append(f"   {a['message']}")
            lines.append("")

        return "\n".join(lines)

    # ─── Helpers ───────────────────────────────────────────────────

    @staticmethod
    def _severity_icon(severity: str) -> str:
        """Map severity to emoji for console output."""
        return {
            "CRITICAL": "🔴",
            "HIGH": "🟠",
            "MEDIUM": "🟡",
            "LOW": "🔵",
        }.get(severity, "⚪")

# Real_Time_Kpi_Alert_System
Automated KPI monitoring system with anomaly detection and alerting (Python + SQL)
# 🚨 Real-Time KPI Alert System

A production-grade anomaly detection and alerting system that monitors KPI data, 
detects statistical anomalies using multiple strategies, stores historical data in SQL, 
and sends automated alerts via email and database logging.

**Built as a portfolio project demonstrating Data Engineering, Python, SQL, 
and system design skills.**

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![SQLite](https://img.shields.io/badge/Database-SQLite-003B57?logo=sqlite)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📐 Architecture

```text
Data Sources (Yahoo Finance / Simulated)
        │
        ▼
  Ingestion Layer (fetch, validate, normalize)
        │
        ▼
  SQL Database (kpi_readings, alerts, thresholds)
        │
        ▼
  Anomaly Detection Engine
  ├── Static Threshold Breach
  ├── Z-Score Outlier Detection
  ├── Rolling Average Deviation
  └── % Change Spike Detection
        │
        ▼
  Alert Manager
  ├── Console / Log Output
  ├── SQL Alert Logging (audit trail)
  └── Email Digest (SMTP)
        │
        ▼
  Dashboard (Power BI / Streamlit) [Optional]

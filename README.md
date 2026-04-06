# Real_Time_Kpi_Alert_System
Automated KPI monitoring system with anomaly detection and alerting (Python + SQL)
# Clone and setup
git clone (https://github.com/Nivpatel23/Real_Time_Kpi_Alert_System)
cd real-time-kpi-alert-system

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt

# Create environment file
cp .env.example .env
# Edit .env with your actual credentials

# Create necessary directories
mkdir -p data logs

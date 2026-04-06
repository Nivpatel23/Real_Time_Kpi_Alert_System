"""
Abstract base class for data sources.

WHY: The Strategy Pattern decouples 'how we get data' from 'what we do with it'.
Adding Alpha Vantage, a REST API, or Kafka later requires only a new class
that implements this interface — zero changes to detection or alerting code.
"""

from abc import ABC, abstractmethod
import pandas as pd


class BaseDataSource(ABC):
    """Interface that all data sources must implement."""

    @abstractmethod
    def fetch_data(self) -> pd.DataFrame:
        """
        Fetch KPI data and return a standardized DataFrame.

        Required columns:
            - kpi_name (str): Name of the KPI metric
            - value (float): The metric value
            - timestamp (str/datetime): When the value was recorded
            - source (str): Data source identifier
            - symbol (str, optional): Ticker/identifier for the entity

        Returns:
            pd.DataFrame with standardized columns
        """
        pass

    @abstractmethod
    def validate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean the fetched data."""
        pass

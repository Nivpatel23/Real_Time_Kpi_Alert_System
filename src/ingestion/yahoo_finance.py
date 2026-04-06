"""
Yahoo Finance data ingestion for real market KPIs.

WHY: Real data makes the portfolio project credible. Yahoo Finance
provides OHLCV data without API keys. We extract multiple KPIs
(close price, volume, daily return, volatility) from a single fetch.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Optional

from src.ingestion.base import BaseDataSource
from config.settings import settings
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    logger.warning("yfinance not installed. Yahoo Finance source unavailable.")


class YahooFinanceDataSource(BaseDataSource):
    """Fetches stock market data and derives multiple KPIs per symbol."""

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        period: str = None
    ):
        """
        Args:
            symbols: List of ticker symbols (default from settings)
            period: yfinance period string: '1mo', '3mo', '6mo', '1y', etc.
        """
        if not YFINANCE_AVAILABLE:
            raise ImportError("yfinance is required. Install with: pip install yfinance")

        self.symbols = symbols or settings.pipeline.yahoo_symbols
        self.period = period or settings.pipeline.yahoo_period

    def fetch_data(self) -> pd.DataFrame:
        """
        Fetch OHLCV data for all symbols and derive KPI metrics.

        Derived KPIs per symbol:
          - stock_close: Closing price (absolute level)
          - stock_volume: Trading volume (activity indicator)
          - stock_daily_return: Day-over-day % change (momentum)
          - stock_volatility: 5-day rolling std of returns (risk metric)
        """
        logger.info(f"Fetching Yahoo Finance data for {self.symbols} (period={self.period})...")

        all_data = []

        for symbol in self.symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=self.period)

                if hist.empty:
                    logger.warning(f"No data returned for {symbol}. Skipping.")
                    continue

                # Derive KPIs from raw OHLCV
                kpi_frames = self._derive_kpis(hist, symbol)
                all_data.extend(kpi_frames)

                logger.info(f"  {symbol}: {len(hist)} trading days fetched.")

            except Exception as e:
                logger.error(f"Failed to fetch {symbol}: {e}")
                continue

        if not all_data:
            logger.error("No data fetched from any symbol.")
            return pd.DataFrame()

        result = pd.concat(all_data, ignore_index=True)
        result = self.validate(result)

        logger.info(f"Total: {len(result)} KPI data points from Yahoo Finance.")
        return result

    def _derive_kpis(self, hist: pd.DataFrame, symbol: str) -> List[pd.DataFrame]:
        """
        Transform raw OHLCV into multiple KPI series.

        WHY multiple KPIs: A single stock ticker yields several meaningful
        metrics. This demonstrates the system handles multi-KPI monitoring,
        which is realistic for production dashboards.
        """
        frames = []
        timestamps = hist.index.strftime("%Y-%m-%d %H:%M:%S").tolist()

        # KPI: Close Price
        frames.append(pd.DataFrame({
            "kpi_name": "stock_close",
            "value": hist["Close"].values,
            "timestamp": timestamps,
            "source": "yahoo_finance",
            "symbol": symbol,
        }))

        # KPI: Volume
        frames.append(pd.DataFrame({
            "kpi_name": "stock_volume",
            "value": hist["Volume"].values,
            "timestamp": timestamps,
            "source": "yahoo_finance",
            "symbol": symbol,
        }))

        # KPI: Daily Return (% change)
        daily_returns = hist["Close"].pct_change().fillna(0)
        frames.append(pd.DataFrame({
            "kpi_name": "stock_daily_return",
            "value": daily_returns.values,
            "timestamp": timestamps,
            "source": "yahoo_finance",
            "symbol": symbol,
        }))

        # KPI: 5-day Rolling Volatility
        volatility = daily_returns.rolling(window=5).std().fillna(0)
        frames.append(pd.DataFrame({
            "kpi_name": "stock_volatility",
            "value": volatility.values,
            "timestamp": timestamps,
            "source": "yahoo_finance",
            "symbol": symbol,
        }))

        return frames

    def validate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate Yahoo Finance data — handle missing values, infinities."""
        initial_len = len(data)

        # Replace infinities (can occur in pct_change if previous close was 0)
        data["value"] = data["value"].replace([np.inf, -np.inf], np.nan)

        # Drop NaN values
        data = data.dropna(subset=["kpi_name", "value", "timestamp"])

        # Ensure correct types
        data["value"] = pd.to_numeric(data["value"], errors="coerce")
        data = data.dropna(subset=["value"])

        data["timestamp"] = data["timestamp"].astype(str)

        dropped = initial_len - len(data)
        if dropped > 0:
            logger.warning(f"Yahoo Finance validation dropped {dropped} rows.")

        return data.reset_index(drop=True)

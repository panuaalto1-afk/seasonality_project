"""
Data Loader - Efficient data loading and caching
Handles price data, universe, and regime indicators
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Loads and caches data for backtest.
    
    Responsibilities:
    - Load universe (constituents)
    - Load stock price data
    - Load regime indicator prices
    - Build trading calendar
    - Data validation
    """
    
    def __init__(self):
        """Initialize data loader."""
        self.price_cache = {}
        
    def load_universe(self, constituents_path: Path) -> pd.DataFrame:
        """
        Load universe of stocks from constituents CSV.
        
        Expected columns:
        - ticker (required)
        - sector (optional)
        - industry (optional)
        """
        logger.info(f"Loading universe from {constituents_path}")
        
        if not constituents_path.exists():
            raise FileNotFoundError(f"Constituents file not found: {constituents_path}")
        
        df = pd.read_csv(constituents_path)
        
        # Normalize column names
        df.columns = [c.lower().strip() for c in df.columns]
        
        # Ensure ticker column exists
        if 'ticker' not in df.columns:
            # Try common alternatives
            for alt in ['symbol', 'code', 'tkr']:
                if alt in df.columns:
                    df['ticker'] = df[alt]
                    break
        
        if 'ticker' not in df.columns:
            raise ValueError("Constituents CSV must have 'ticker' column")
        
        # Ensure sector column (fill Unknown if missing)
        if 'sector' not in df.columns:
            df['sector'] = 'Unknown'
        
        # Clean tickers
        df['ticker'] = df['ticker'].str.upper().str.strip()
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['ticker'])
        
        logger.info(f"Loaded {len(df)} tickers")
        
        return df
    
    def read_price_csv(self, file_path: Path) -> Optional[pd.DataFrame]:
        """
        Read a single price CSV file.
        
        Expected columns (flexible):
        - Date/Datetime
        - Open
        - High
        - Low
        - Close/Adj Close
        """
        try:
            # Try different encodings
            for encoding in ['utf-8', 'utf-8-sig', 'cp1252']:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    break
                except:
                    continue
            else:
                return None
            
            # Normalize column names
            df.columns = [c.lower().strip().replace(' ', '_') for c in df.columns]
            
            # Find date column
            date_col = None
            for col in ['date', 'datetime', 'timestamp']:
                if col in df.columns:
                    date_col = col
                    break
            
            if date_col is None:
                return None
            
            # Parse dates
            df['date'] = pd.to_datetime(df[date_col], errors='coerce')
            df = df.dropna(subset=['date'])
            
            # Find OHLC columns
            rename_map = {}
            
            for target, candidates in [
                ('open', ['open']),
                ('high', ['high']),
                ('low', ['low']),
                ('close', ['close', 'adj_close', 'adjclose', 'adj close']),
            ]:
                for cand in candidates:
                    if cand in df.columns:
                        rename_map[cand] = target
                        break
            
            df = df.rename(columns=rename_map)
            
            # Ensure we have OHLC
            required = ['open', 'high', 'low', 'close']
            if not all(col in df.columns for col in required):
                return None
            
            # Select and convert
            df = df[['date', 'open', 'high', 'low', 'close']].copy()
            
            for col in ['open', 'high', 'low', 'close']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df = df.dropna()
            
            if df.empty:
                return None
            
            # Sort by date
            df = df.sort_values('date').reset_index(drop=True)
            
            # Remove duplicates (keep last)
            df = df.drop_duplicates(subset=['date'], keep='last')
            
            return df
            
        except Exception as e:
            logger.debug(f"Error reading {file_path.name}: {str(e)}")
            return None
    
    def load_price_data(
        self,
        tickers: List[str],
        price_cache_dir: Path,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, pd.DataFrame]:
        """
        Load price data for all tickers.
        
        Returns dict: {ticker: DataFrame with OHLC}
        """
        logger.info(f"Loading price data from {price_cache_dir}")
        
        price_data = {}
        failed_tickers = []
        
        for ticker in tickers:
            # Try different file name formats
            possible_names = [
                f"{ticker}.csv",
                f"{ticker.replace('-', '.')}.csv",
                f"{ticker.replace('.', '-')}.csv",
            ]
            
            df = None
            for name in possible_names:
                file_path = price_cache_dir / name
                if file_path.exists():
                    df = self.read_price_csv(file_path)
                    if df is not None:
                        break
            
            if df is None:
                failed_tickers.append(ticker)
                continue
            
            # Filter to date range
            df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
            
            if len(df) < 60:  # Require at least 60 days
                failed_tickers.append(ticker)
                continue
            
            price_data[ticker] = df
        
        if failed_tickers:
            logger.warning(f"Failed to load {len(failed_tickers)} tickers")
            if len(failed_tickers) <= 10:
                logger.warning(f"Failed tickers: {', '.join(failed_tickers)}")
        
        logger.info(f"Successfully loaded {len(price_data)} tickers")
        
        return price_data
    
    def load_regime_prices(
        self,
        regime_cache_dir: Path,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, pd.DataFrame]:
        """
        Load price data for regime indicators.
        
        Required tickers:
        - SPY, QQQ, IWM (equity)
        - ^SPX, ^VIX (market)
        - GLD, TLT (safe havens)
        - HYG, LQD (credit)
        """
        logger.info(f"Loading regime prices from {regime_cache_dir}")
        
        regime_tickers = [
            'SPY', 'QQQ', 'IWM',
            '^SPX', '^VIX',
            'GLD', 'TLT',
            'HYG', 'LQD',
        ]
        
        regime_prices = {}
        
        for ticker in regime_tickers:
            # Try different file formats
            possible_names = [
                f"{ticker}.csv",
                f"{ticker.replace('^', '')}.csv",
            ]
            
            df = None
            for name in possible_names:
                file_path = regime_cache_dir / name
                if file_path.exists():
                    df = self.read_price_csv(file_path)
                    if df is not None:
                        break
            
            if df is None:
                logger.warning(f"Could not load regime ticker: {ticker}")
                continue
            
            # Filter to date range (with buffer for calculations)
            buffer_start = start_date - timedelta(days=365)
            df = df[(df['date'] >= buffer_start) & (df['date'] <= end_date)]
            
            regime_prices[ticker] = df
        
        logger.info(f"Loaded {len(regime_prices)} regime indicators")
        
        return regime_prices
    
    def build_trading_calendar(
        self,
        price_data: Dict[str, pd.DataFrame],
        start_date: datetime,
        end_date: datetime
    ) -> List[datetime]:
        """
        Build trading calendar from actual trading days in data.
        
        Uses union of all dates from price data.
        """
        all_dates = set()
        
        for df in price_data.values():
            dates = df['date'].dt.normalize()
            all_dates.update(dates)
        
        # Filter to range
        trading_days = sorted([
            d for d in all_dates
            if start_date <= d <= end_date
        ])
        
        return trading_days
    
    def validate_data(self, price_data: Dict[str, pd.DataFrame]) -> Dict[str, List[str]]:
        """
        Validate loaded price data.
        
        Returns dict with issues found.
        """
        issues = {
            'missing_data': [],
            'short_history': [],
            'price_errors': [],
        }
        
        for ticker, df in price_data.items():
            # Check for gaps
            date_diffs = df['date'].diff()
            large_gaps = date_diffs[date_diffs > timedelta(days=14)]
            if len(large_gaps) > 5:
                issues['missing_data'].append(ticker)
            
            # Check history length
            if len(df) < 252:  # Less than 1 year
                issues['short_history'].append(ticker)
            
            # Check for price errors (e.g., zeros, negatives)
            if (df[['open', 'high', 'low', 'close']] <= 0).any().any():
                issues['price_errors'].append(ticker)
            
            # Check for unrealistic moves (>50% in one day)
            daily_return = df['close'].pct_change()
            if (daily_return.abs() > 0.5).any():
                issues['price_errors'].append(ticker)
        
        return issues
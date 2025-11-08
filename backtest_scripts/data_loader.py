# backtest_scripts/data_loader.py
"""
Data Loader for Backtesting
Loads historical stock prices, macro ETF prices, and vintage seasonality data
"""

import os
import glob
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import date, datetime, timedelta

class BacktestDataLoader:
    """
    Centralized data loader for backtest
    Handles stock prices, macro ETFs, and vintage seasonality data
    """
    
    def __init__(self, 
                 stock_price_cache: str,
                 macro_price_cache: str,
                 vintage_dir: str):
        """
        Initialize data loader
        
        Args:
            stock_price_cache: Path to stock prices (e.g., runs/.../price_cache)
            macro_price_cache: Path to macro ETF prices
            vintage_dir: Path to vintage seasonality data
        """
        self.stock_price_cache = stock_price_cache
        self.macro_price_cache = macro_price_cache
        self.vintage_dir = vintage_dir
        
        # Cache loaded data
        self._stock_prices_cache = {}
        self._macro_prices_cache = {}
        self._vintage_cache = {}
        
        print(f"[DataLoader] Initialized")
        print(f"  Stock prices: {stock_price_cache}")
        print(f"  Macro prices: {macro_price_cache}")
        print(f"  Vintage data: {vintage_dir}")
    
    def load_universe(self, universe_csv: str, max_size: Optional[int] = None) -> List[str]:
        """
        Load stock universe from CSV
        
        Args:
            universe_csv: Path to constituents CSV
            max_size: Optional limit on universe size
        
        Returns:
            List of ticker symbols
        """
        print(f"[DataLoader] Loading universe from: {universe_csv}")
        
        df = pd.read_csv(universe_csv)
        
        # Find ticker column
        cols = {c.lower(): c for c in df.columns}
        ticker_col = None
        for k in ["ticker", "symbol", "ric"]:
            if k in cols:
                ticker_col = cols[k]
                break
        
        if ticker_col is None:
            ticker_col = df.columns[0]
        
        tickers = df[ticker_col].astype(str).str.strip().str.upper().tolist()
        tickers = [t for t in tickers if t and t != 'NAN']
        
        if max_size:
            tickers = tickers[:max_size]
        
        print(f"[DataLoader] Loaded {len(tickers)} tickers")
        return tickers
    
    def load_stock_prices(self, ticker: str, 
                          start_date: Optional[date] = None,
                          end_date: Optional[date] = None) -> Optional[pd.DataFrame]:
        """
        Load historical stock prices
        
        Args:
            ticker: Stock ticker
            start_date: Optional start date filter
            end_date: Optional end date filter
        
        Returns:
            DataFrame with columns: date, open, high, low, close, volume (if available)
        """
        # Check cache
        cache_key = f"{ticker}_{start_date}_{end_date}"
        if cache_key in self._stock_prices_cache:
            return self._stock_prices_cache[cache_key]
        
        # Find CSV file
        patterns = [
            os.path.join(self.stock_price_cache, f"{ticker}.csv"),
            os.path.join(self.stock_price_cache, f"{ticker.upper()}.csv"),
            os.path.join(self.stock_price_cache, f"{ticker.lower()}.csv"),
        ]
        
        path = None
        for p in patterns:
            if os.path.isfile(p):
                path = p
                break
        
        if path is None:
            return None
        
        try:
            df = pd.read_csv(path)
            
            # Normalize column names
            cols = {c.lower(): c for c in df.columns}
            
            # Find date column
            date_col = None
            for k in ['date', 'time', 'datetime', 'timestamp']:
                if k in cols:
                    date_col = cols[k]
                    break
            
            if date_col is None:
                return None
            
            # Find price columns
            open_col = cols.get('open')
            high_col = cols.get('high')
            low_col = cols.get('low')
            
            close_col = None
            for k in ['adj close', 'adj_close', 'close', 'last', 'price']:
                if k in cols:
                    close_col = cols[k]
                    break
            
            volume_col = cols.get('volume')
            
            if close_col is None:
                return None
            
            # Build output DataFrame
            out_cols = {'date': date_col, 'close': close_col}
            if open_col:
                out_cols['open'] = open_col
            if high_col:
                out_cols['high'] = high_col
            if low_col:
                out_cols['low'] = low_col
            if volume_col:
                out_cols['volume'] = volume_col
            
            df = df[list(out_cols.values())].rename(columns={v: k for k, v in out_cols.items()})
            
            # Parse date
            df['date'] = pd.to_datetime(df['date']).dt.date
            
            # Convert to numeric
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Remove NaNs
            df = df.dropna(subset=['close'])
            
            # Sort by date
            df = df.sort_values('date').reset_index(drop=True)
            
            # Fill missing OHLC if needed
            if 'open' not in df.columns:
                df['open'] = df['close']
            if 'high' not in df.columns:
                df['high'] = df['close']
            if 'low' not in df.columns:
                df['low'] = df['close']
            
            # Filter by date range
            if start_date:
                df = df[df['date'] >= start_date]
            if end_date:
                df = df[df['date'] <= end_date]
            
            # Cache
            self._stock_prices_cache[cache_key] = df
            
            return df
        
        except Exception as e:
            print(f"[WARN] Failed to load {ticker}: {e}")
            return None
    
    def load_macro_prices(self, symbol: str,
                         start_date: Optional[date] = None,
                         end_date: Optional[date] = None) -> Optional[pd.DataFrame]:
        """
        Load macro ETF prices (for regime detection)
        
        Args:
            symbol: ETF symbol (SPY, QQQ, VIX, etc.)
            start_date: Optional start date
            end_date: Optional end date
        
        Returns:
            DataFrame with columns: date, close
        """
        cache_key = f"macro_{symbol}_{start_date}_{end_date}"
        if cache_key in self._macro_prices_cache:
            return self._macro_prices_cache[cache_key]
        
        path = os.path.join(self.macro_price_cache, f"{symbol}.csv")
        
        if not os.path.isfile(path):
            return None
        
        try:
            df = pd.read_csv(path)
            
            cols = {c.lower(): c for c in df.columns}
            date_col = cols.get('date')
            
            close_col = None
            for k in ['adj close', 'adj_close', 'close', 'price']:
                if k in cols:
                    close_col = cols[k]
                    break
            
            if date_col is None or close_col is None:
                return None
            
            df = df[[date_col, close_col]].rename(columns={date_col: 'date', close_col: 'close'})
            df['date'] = pd.to_datetime(df['date']).dt.date
            df['close'] = pd.to_numeric(df['close'], errors='coerce')
            df = df.dropna().sort_values('date').reset_index(drop=True)
            
            if start_date:
                df = df[df['date'] >= start_date]
            if end_date:
                df = df[df['date'] <= end_date]
            
            self._macro_prices_cache[cache_key] = df
            return df
        
        except Exception as e:
            print(f"[WARN] Failed to load macro {symbol}: {e}")
            return None
    
    def load_vintage_seasonality(self, ticker: str) -> Dict[str, pd.DataFrame]:
        """
        Load vintage seasonality data for ticker
        
        Args:
            ticker: Stock ticker
        
        Returns:
            Dict with keys: 'segments_up', 'segments_down', 'vintage_10y', 'seasonality_week'
        """
        if ticker in self._vintage_cache:
            return self._vintage_cache[ticker]
        
        result = {}
        
        # Load segments_up
        path_up = os.path.join(self.vintage_dir, f"{ticker}_segments_up.csv")
        if os.path.isfile(path_up):
            try:
                result['segments_up'] = pd.read_csv(path_up)
            except:
                pass
        
        # Load segments_down
        path_down = os.path.join(self.vintage_dir, f"{ticker}_segments_down.csv")
        if os.path.isfile(path_down):
            try:
                result['segments_down'] = pd.read_csv(path_down)
            except:
                pass
        
        # Load vintage_10y
        path_vintage = os.path.join(self.vintage_dir, f"{ticker}_vintage_10y.csv")
        if os.path.isfile(path_vintage):
            try:
                result['vintage_10y'] = pd.read_csv(path_vintage)
            except:
                pass
        
        # Load seasonality_week
        path_week = os.path.join(self.vintage_dir, f"{ticker}_seasonality_week.csv")
        if os.path.isfile(path_week):
            try:
                result['seasonality_week'] = pd.read_csv(path_week)
            except:
                pass
        
        self._vintage_cache[ticker] = result
        return result
    
    def preload_all_stock_prices(self, tickers: List[str], 
                                 start_date: date, 
                                 end_date: date) -> Dict[str, pd.DataFrame]:
        """
        Preload all stock prices into memory (for speed)
        
        Args:
            tickers: List of tickers
            start_date: Backtest start
            end_date: Backtest end
        
        Returns:
            Dict mapping ticker → DataFrame
        """
        print(f"[DataLoader] Preloading {len(tickers)} stock prices...")
        
        prices = {}
        loaded = 0
        
        for i, ticker in enumerate(tickers):
            if (i + 1) % 50 == 0:
                print(f"  Progress: {i+1}/{len(tickers)}")
            
            df = self.load_stock_prices(ticker, start_date, end_date)
            if df is not None and len(df) > 0:
                prices[ticker] = df
                loaded += 1
        
        print(f"[DataLoader] Preloaded {loaded}/{len(tickers)} successfully")
        return prices
    
    def preload_all_macro_prices(self, symbols: List[str],
                                 start_date: date,
                                 end_date: date) -> Dict[str, pd.DataFrame]:
        """
        Preload all macro ETF prices
        
        Args:
            symbols: List of macro symbols (SPY, QQQ, etc.)
            start_date: Backtest start
            end_date: Backtest end
        
        Returns:
            Dict mapping symbol → DataFrame
        """
        print(f"[DataLoader] Preloading {len(symbols)} macro prices...")
        
        prices = {}
        for symbol in symbols:
            df = self.load_macro_prices(symbol, start_date, end_date)
            if df is not None:
                prices[symbol] = df
        
        print(f"[DataLoader] Preloaded {len(prices)}/{len(symbols)} macro prices")
        return prices
    
    def get_trading_days(self, start_date: date, end_date: date, 
                        reference_ticker: str = 'SPY') -> List[date]:
        """
        Get list of trading days (using SPY as reference)
        
        Args:
            start_date: Start date
            end_date: End date
            reference_ticker: Ticker to use as trading calendar (default SPY)
        
        Returns:
            List of trading dates
        """
        # Load reference prices (from macro cache, as SPY is there)
        df = self.load_macro_prices(reference_ticker, start_date, end_date)
        
        if df is None or df.empty:
            # Fallback: all weekdays
            print(f"[WARN] Could not load {reference_ticker} for trading calendar, using weekdays")
            dates = []
            current = start_date
            while current <= end_date:
                if current.weekday() < 5:  # Monday-Friday
                    dates.append(current)
                current += timedelta(days=1)
            return dates
        
        return df['date'].tolist()
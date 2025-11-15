# backtest_scripts/seasonality_calculator.py
"""
Seasonality Calculator for Backtesting
Walk-forward safe seasonality feature calculation
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from datetime import date, timedelta

class SeasonalityCalculator:
    """
    Calculate seasonality features for backtest
    Walk-forward safe: Only uses data BEFORE target date
    """
    
    def __init__(self, lookback_years: int = 10):
        """
        Initialize seasonality calculator
        
        Args:
            lookback_years: Years of historical data to use (default 10)
        """
        self.lookback_years = lookback_years
        print(f"[SeasonalityCalculator] Initialized (lookback: {lookback_years} years)")
    
    def calculate_features(self, 
                          ticker: str, 
                          prices: pd.DataFrame, 
                          target_date: date) -> Dict:
        """
        Calculate seasonality features for a specific date
        
        WALK-FORWARD SAFE: Only uses data BEFORE target_date
        
        Args:
            ticker: Stock ticker
            prices: Historical prices (full history up to target_date)
            target_date: Date to calculate features for
        
        Returns:
            dict with seasonality features:
            {
                'season_week_avg': float,
                'season_week_hit_rate': float,
                'season_20d_avg': float,
                'season_20d_hit_rate': float,
                'season_month_avg': float,
                'season_quarter_avg': float,
                'in_bullish_segment': int (0 or 1),
                'in_bearish_segment': int (0 or 1),
                'days_into_segment': int,
                'segment_strength': float
            }
        """
        # Filter to data BEFORE target_date only
        prices_before = prices[prices['date'] < target_date].copy()
        
        if len(prices_before) < 252:  # Need at least 1 year
            return self._empty_features()
        
        # Ensure close is numeric
        prices_before['close'] = pd.to_numeric(prices_before['close'], errors='coerce')
        prices_before = prices_before.dropna(subset=['close'])
        
        if prices_before.empty or len(prices_before) < 252:
            return self._empty_features()
        
        # Calculate returns
        prices_before['return'] = prices_before['close'].pct_change()
        
        # Add time markers
        prices_before['date_dt'] = pd.to_datetime(prices_before['date'])
        prices_before['week'] = prices_before['date_dt'].dt.isocalendar().week
        prices_before['month'] = prices_before['date_dt'].dt.month
        prices_before['quarter'] = prices_before['date_dt'].dt.quarter
        prices_before['doy'] = prices_before['date_dt'].dt.dayofyear
        
        # Current time markers
        current_week = target_date.isocalendar()[1]
        current_month = target_date.month
        current_quarter = (target_date.month - 1) // 3 + 1
        current_doy = target_date.timetuple().tm_yday
        
        # Filter to lookback window
        cutoff_date = target_date - timedelta(days=365 * self.lookback_years)
        hist = prices_before[prices_before['date'] >= cutoff_date].copy()
        
        if hist.empty:
            return self._empty_features()
        
        # 1. Week-of-Year seasonality
        week_data = hist[hist['week'] == current_week]
        season_week_avg = float(week_data['return'].mean()) if not week_data.empty else 0.0
        season_week_hit_rate = float((week_data['return'] > 0).mean()) if not week_data.empty else 0.5
        
        # 2. Day-of-Year (±3 day window) → 20d forward return
        day_window = hist[
            (hist['doy'] >= current_doy - 3) & 
            (hist['doy'] <= current_doy + 3)
        ]
        
        forward_returns = []
        for idx in day_window.index:
            try:
                current_loc = hist.index.get_loc(idx)
                future_loc = current_loc + 20
                if future_loc < len(hist):
                    future_price = hist.iloc[future_loc]['close']
                    current_price = hist.iloc[current_loc]['close']
                    if current_price > 0:
                        fwd_ret = (future_price / current_price) - 1
                        forward_returns.append(fwd_ret)
            except:
                pass
        
        season_20d_avg = float(np.mean(forward_returns)) if forward_returns else 0.0
        season_20d_hit_rate = float(np.mean([1 if r > 0 else 0 for r in forward_returns])) if forward_returns else 0.5
        
        # 3. Month-of-Year
        month_data = hist[hist['month'] == current_month]
        season_month_avg = float(month_data['return'].mean()) if not month_data.empty else 0.0
        
        # 4. Quarter-of-Year
        quarter_data = hist[hist['quarter'] == current_quarter]
        season_quarter_avg = float(quarter_data['return'].mean()) if not quarter_data.empty else 0.0
        
        # 5. Segment detection (simplified - no vintage data dependency)
        # Look at recent 60d momentum to detect if in bullish/bearish segment
        recent_60d = prices_before.tail(60)
        if len(recent_60d) >= 60:
            recent_mom = (recent_60d.iloc[-1]['close'] / recent_60d.iloc[0]['close']) - 1
            
            in_bullish = 1 if recent_mom > 0.05 else 0
            in_bearish = 1 if recent_mom < -0.05 else 0
            segment_strength = abs(recent_mom)
            
            # Days into segment (count consecutive positive/negative days)
            returns_tail = recent_60d['return'].tail(30).dropna()
            days_into = 0
            if in_bullish:
                for r in reversed(returns_tail.tolist()):
                    if r > 0:
                        days_into += 1
                    else:
                        break
            elif in_bearish:
                for r in reversed(returns_tail.tolist()):
                    if r < 0:
                        days_into += 1
                    else:
                        break
        else:
            in_bullish = 0
            in_bearish = 0
            days_into = 0
            segment_strength = 0.0
        
        return {
            'season_week_avg': season_week_avg,
            'season_week_hit_rate': season_week_hit_rate,
            'season_20d_avg': season_20d_avg,
            'season_20d_hit_rate': season_20d_hit_rate,
            'season_month_avg': season_month_avg,
            'season_quarter_avg': season_quarter_avg,
            'in_bullish_segment': in_bullish,
            'in_bearish_segment': in_bearish,
            'days_into_segment': days_into,
            'segment_strength': segment_strength,
        }
    
    def _empty_features(self) -> Dict:
        """Return empty features dict"""
        return {
            'season_week_avg': 0.0,
            'season_week_hit_rate': 0.5,
            'season_20d_avg': 0.0,
            'season_20d_hit_rate': 0.5,
            'season_month_avg': 0.0,
            'season_quarter_avg': 0.0,
            'in_bullish_segment': 0,
            'in_bearish_segment': 0,
            'days_into_segment': 0,
            'segment_strength': 0.0,
        }
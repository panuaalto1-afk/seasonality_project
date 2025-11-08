# backtest_scripts/ml_signal_generator.py
"""
ML Signal Generator for Backtesting
Replicates ml_unified_pipeline.py logic in walk-forward manner
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import date, timedelta

from .seasonality_calculator import SeasonalityCalculator

class MLSignalGenerator:
    """
    Generate ML signals (candidates) for backtest
    Replicates ml_unified_pipeline.py v2.0 logic
    """
    
    # Regime-specific SL/TP multipliers (from TradingLevelsCalculator)
    REGIME_STRATEGIES = {
        'BULL_STRONG': {
            'stop_multiplier': 0.8,
            'tp_multiplier': 2.5,
        },
        'BULL_WEAK': {
            'stop_multiplier': 1.0,
            'tp_multiplier': 2.0,
        },
        'NEUTRAL_BULLISH': {
            'stop_multiplier': 1.0,
            'tp_multiplier': 1.5,
        },
        'NEUTRAL_BEARISH': {
            'stop_multiplier': 1.2,
            'tp_multiplier': 1.2,
        },
        'BEAR_WEAK': {
            'stop_multiplier': 1.5,
            'tp_multiplier': 1.0,
        },
        'BEAR_STRONG': {
            'stop_multiplier': 1.8,
            'tp_multiplier': 0.8,
        },
        'CRISIS': {
            'stop_multiplier': 2.0,
            'tp_multiplier': 0.5,
        },
    }
    
    def __init__(self, seasonality_calculator: SeasonalityCalculator):
        """
        Initialize ML signal generator
        
        Args:
            seasonality_calculator: Seasonality calculator instance
        """
        self.seasonality_calc = seasonality_calculator
        print(f"[MLSignalGenerator] Initialized")
    
    def generate_signals(self,
                        target_date: date,
                        stock_prices: Dict[str, pd.DataFrame],
                        regime: str,
                        gate_alpha: float = 0.10) -> pd.DataFrame:
        """
        Generate trading signals (candidates) for a specific date
        
        WALK-FORWARD SAFE: Only uses data BEFORE target_date
        
        Args:
            target_date: Date to generate signals for
            stock_prices: Dict mapping ticker → full price history
            regime: Current market regime
            gate_alpha: Minimum score threshold
        
        Returns:
            DataFrame with columns matching ml_unified_pipeline output:
            ticker, mom5, mom20, mom60, vol20,
            season_*, entry_price, stop_loss, take_profit, atr_14,
            score_long, score_short, ml_expected_return
        """
        rows = []
        
        for ticker, prices in stock_prices.items():
            # Filter to data BEFORE target_date
            prices_before = prices[prices['date'] < target_date].copy()
            
            if len(prices_before) < 60:
                continue
            
            # 1. Momentum features
            mom_features = self._calculate_momentum_features(prices_before)
            if mom_features is None:
                continue
            
            # 2. Seasonality features
            season_features = self.seasonality_calc.calculate_features(ticker, prices, target_date)
            
            # 3. Trading levels (Entry/SL/TP)
            trading_levels = self._calculate_trading_levels(prices_before, regime)
            
            # 4. ML scoring (momentum + seasonality blend)
            score = self._calculate_score(mom_features, season_features)
            
            # Combine all features
            row = {'ticker': ticker}
            row.update(mom_features)
            row.update(season_features)
            row.update(trading_levels)
            row.update(score)
            
            rows.append(row)
        
        if not rows:
            return pd.DataFrame()
        
        df = pd.DataFrame(rows)
        
        # Rank to 0-1 score
        if len(df) > 1:
            df['score_long'] = df['ml_expected_return'].rank(pct=True)
            df['score_short'] = 1.0 - df['score_long']
        else:
            df['score_long'] = 0.5
            df['score_short'] = 0.5
        
        # Filter by gate_alpha
        df = df[df['score_long'] >= gate_alpha].copy()
        
        # Sort by score_long descending
        df = df.sort_values('score_long', ascending=False).reset_index(drop=True)
        
        return df
    
    def _calculate_momentum_features(self, prices: pd.DataFrame) -> Optional[Dict]:
        """
        Calculate momentum features (mom5, mom20, mom60, vol20)
        
        Args:
            prices: Price DataFrame (filtered to BEFORE target_date)
        
        Returns:
            Dict with momentum features or None
        """
        if len(prices) < 60:
            return None
        
        # Ensure close is numeric
        prices = prices.copy()
        prices['close'] = pd.to_numeric(prices['close'], errors='coerce')
        prices = prices.dropna(subset=['close'])
        
        if len(prices) < 60:
            return None
        
        close = prices['close']
        returns = close.pct_change()
        
        # Momentum
        mom5 = (close.iloc[-1] / close.iloc[-6]) - 1 if len(close) >= 6 else 0.0
        mom20 = (close.iloc[-1] / close.iloc[-21]) - 1 if len(close) >= 21 else 0.0
        mom60 = (close.iloc[-1] / close.iloc[-61]) - 1 if len(close) >= 61 else 0.0
        
        # Volatility (20d rolling std of returns)
        vol20 = returns.tail(20).std() if len(returns) >= 20 else 0.0
        
        return {
            'mom5': float(mom5) if not pd.isna(mom5) else 0.0,
            'mom20': float(mom20) if not pd.isna(mom20) else 0.0,
            'mom60': float(mom60) if not pd.isna(mom60) else 0.0,
            'vol20': float(vol20) if not pd.isna(vol20) else 0.0,
        }
    
    def _calculate_trading_levels(self, prices: pd.DataFrame, regime: str) -> Dict:
        """
        Calculate entry price, stop loss, take profit
        
        Args:
            prices: Price DataFrame (filtered to BEFORE target_date)
            regime: Current regime
        
        Returns:
            Dict with entry_price, stop_loss, take_profit, atr_14, sl_distance_pct, tp_distance_pct
        """
        if len(prices) < 14:
            return self._empty_trading_levels()
        
        # Entry price = last close (T-1)
        entry_price = float(prices.iloc[-1]['close'])
        
        if entry_price <= 0:
            return self._empty_trading_levels()
        
        # Calculate ATR-14
        atr = self._calculate_atr(prices, period=14)
        
        if atr <= 0:
            # Fallback: use 2% of entry price
            atr = entry_price * 0.02
        
        # Get regime strategy
        strategy = self.REGIME_STRATEGIES.get(regime, self.REGIME_STRATEGIES['NEUTRAL_BULLISH'])
        
        # Calculate SL/TP
        stop_loss = entry_price - (atr * strategy['stop_multiplier'])
        take_profit = entry_price + (atr * strategy['tp_multiplier'])
        
        # Ensure positive
        stop_loss = max(stop_loss, entry_price * 0.5)
        take_profit = max(take_profit, entry_price * 1.01)
        
        # Distance percentages
        sl_distance_pct = ((entry_price - stop_loss) / entry_price) * 100
        tp_distance_pct = ((take_profit - entry_price) / entry_price) * 100
        
        return {
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'atr_14': atr,
            'sl_distance_pct': sl_distance_pct,
            'tp_distance_pct': tp_distance_pct,
        }
    
    def _calculate_atr(self, prices: pd.DataFrame, period: int = 14) -> float:
        """
        Calculate Average True Range
        
        Args:
            prices: Price DataFrame
            period: ATR period (default 14)
        
        Returns:
            ATR value
        """
        if len(prices) < period:
            return 0.0
        
        prices = prices.copy()
        
        # Ensure numeric
        for col in ['high', 'low', 'close']:
            if col in prices.columns:
                prices[col] = pd.to_numeric(prices[col], errors='coerce')
        
        # If we have OHLC, use true range
        if 'high' in prices.columns and 'low' in prices.columns:
            high = prices['high']
            low = prices['low']
            close = prices['close']
            
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(period).mean().iloc[-1]
        else:
            # Fallback: close-to-close volatility
            prices['close'] = pd.to_numeric(prices['close'], errors='coerce')
            returns = prices['close'].pct_change()
            volatility = returns.rolling(period).std().iloc[-1]
            atr = volatility * prices['close'].iloc[-1]
        
        return float(atr) if not pd.isna(atr) else 0.0
    
    def _calculate_score(self, mom_features: Dict, season_features: Dict) -> Dict:
        """
        Calculate ML score (momentum + seasonality blend)
        
        For now: Simple blend (50% momentum, 50% seasonality)
        Future: Train actual ML model
        
        Args:
            mom_features: Momentum features
            season_features: Seasonality features
        
        Returns:
            Dict with ml_expected_return
        """
        # Momentum score (weighted average)
        mom_score = (0.6 * mom_features['mom5'] + 0.4 * mom_features['mom20'])
        
        # Seasonality score (average of week and 20d)
        season_score = (season_features['season_week_avg'] + season_features['season_20d_avg']) / 2
        
        # Combined score (50/50 blend)
        combined = 0.5 * mom_score + 0.5 * season_score
        
        return {
            'ml_expected_return': combined
        }
    
    def _empty_trading_levels(self) -> Dict:
        """Return empty trading levels"""
        return {
            'entry_price': 0.0,
            'stop_loss': 0.0,
            'take_profit': 0.0,
            'atr_14': 0.0,
            'sl_distance_pct': 0.0,
            'tp_distance_pct': 0.0,
        }
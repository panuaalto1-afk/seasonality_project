# backtest_scripts/regime_calculator.py
"""
Regime Calculator for Backtesting
Replicates regime_detector.py logic for historical dates
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from datetime import date, timedelta

class RegimeCalculator:
    """
    Calculate historical market regimes
    Replicates regime_detector.py logic
    """
    
    # Regime thresholds (from regime_detector.py)
    REGIME_THRESHOLDS = {
        'CRISIS': -1.0,
        'BEAR_STRONG': -0.50,
        'BEAR_WEAK': -0.25,
        'NEUTRAL_BEARISH': -0.05,
        'NEUTRAL_BULLISH': 0.05,
        'BULL_WEAK': 0.25,
        'BULL_STRONG': 0.50,
    }
    
    # Component weights (from regime_detector.py)
    COMPONENT_WEIGHTS = {
        'equity': 0.35,
        'volatility': 0.20,
        'credit': 0.20,
        'safe_haven': 0.15,
        'breadth': 0.10,
    }
    
    def __init__(self, macro_prices: Dict[str, pd.DataFrame]):
        """
        Initialize regime calculator
        
        Args:
            macro_prices: Dict mapping symbol → DataFrame (preloaded macro prices)
                         Expected symbols: SPY, QQQ, IWM, GLD, TLT, HYG, LQD, VIX/VIXM
        """
        self.macro_prices = macro_prices
        
        # Verify required symbols
        required = ['SPY', 'QQQ', 'IWM', 'GLD', 'TLT', 'HYG', 'LQD']
        missing = [s for s in required if s not in macro_prices]
        
        if missing:
            print(f"[WARN] Missing macro prices: {missing}")
        
        print(f"[RegimeCalculator] Initialized with {len(macro_prices)} macro symbols")
    
    def calculate_regime(self, target_date: date) -> Dict:
        """
        Calculate market regime for a specific date
        
        Args:
            target_date: Date to calculate regime for
        
        Returns:
            dict: {
                'regime': str (e.g., 'BULL_STRONG'),
                'composite_score': float,
                'components': dict,
                'confidence': float
            }
        """
        # Calculate all components
        components = {
            'equity': self._equity_signal(target_date),
            'volatility': self._volatility_signal(target_date),
            'credit': self._credit_signal(target_date),
            'safe_haven': self._safe_haven_signal(target_date),
            'breadth': self._breadth_signal(target_date),
        }
        
        # Composite score (weighted average)
        composite = sum(
            components[k] * self.COMPONENT_WEIGHTS[k]
            for k in components.keys()
        )
        
        # Determine regime from composite score
        regime = 'NEUTRAL_BULLISH'  # Default
        for regime_name, threshold in sorted(self.REGIME_THRESHOLDS.items(), key=lambda x: x[1], reverse=True):
            if composite >= threshold:
                regime = regime_name
                break
        
        # Calculate confidence (distance from nearest threshold)
        confidence = abs(composite) * 100  # Simple heuristic
        
        return {
            'regime': regime,
            'composite_score': float(composite),
            'components': components,
            'confidence': float(confidence),
            'date': target_date
        }
    
    def _get_price_at_date(self, symbol: str, target_date: date, lookback_days: int = 0) -> Optional[float]:
        """
        Get price for symbol at specific date
        
        Args:
            symbol: Macro symbol
            target_date: Target date
            lookback_days: If 0, get exact date. If > 0, get price N days before
        
        Returns:
            Close price or None
        """
        if symbol not in self.macro_prices:
            return None
        
        df = self.macro_prices[symbol]
        
        if lookback_days > 0:
            target_date = target_date - timedelta(days=lookback_days)
        
        # Find closest date <= target_date
        df_filtered = df[df['date'] <= target_date]
        
        if df_filtered.empty:
            return None
        
        return float(df_filtered.iloc[-1]['close'])
    
    def _calculate_momentum(self, symbol: str, target_date: date, period: int = 60) -> float:
        """
        Calculate momentum (% change over period)
        
        Args:
            symbol: Macro symbol
            target_date: Target date
            period: Lookback period in days
        
        Returns:
            Momentum (fraction)
        """
        current_price = self._get_price_at_date(symbol, target_date)
        past_price = self._get_price_at_date(symbol, target_date, lookback_days=period)
        
        if current_price is None or past_price is None or past_price == 0:
            return 0.0
        
        return (current_price / past_price) - 1
    
    def _equity_signal(self, target_date: date) -> float:
        """
        Equity signal: SPY, QQQ, IWM momentum
        Positive = bullish, negative = bearish
        """
        symbols = ['SPY', 'QQQ', 'IWM']
        momentums = []
        
        for sym in symbols:
            mom = self._calculate_momentum(sym, target_date, period=60)
            momentums.append(mom)
        
        if not momentums:
            return 0.0
        
        # Average momentum, normalized to ~[-1, 1]
        avg_mom = np.mean(momentums)
        return float(np.clip(avg_mom * 5, -1, 1))  # Scale to [-1, 1]
    
    def _volatility_signal(self, target_date: date) -> float:
        """
        Volatility signal: VIX level
        High VIX = bearish (negative signal)
        """
        # Try VIX or VIXM
        vix_symbols = ['VIX', 'VIXM']
        
        current_vix = None
        for sym in vix_symbols:
            current_vix = self._get_price_at_date(sym, target_date)
            if current_vix is not None:
                break
        
        if current_vix is None:
            return 0.0
        
        # VIX normalization:
        # 10-15: Low vol (bullish) → +0.5
        # 15-20: Normal → 0
        # 20-30: Elevated (bearish) → -0.5
        # 30+: High (very bearish) → -1.0
        
        if current_vix < 15:
            return 0.5
        elif current_vix < 20:
            return 0.0
        elif current_vix < 30:
            return -0.5
        else:
            return -1.0
    
    def _credit_signal(self, target_date: date) -> float:
        """
        Credit signal: HYG/LQD spread
        Widening spread = bearish (negative)
        """
        hyg_mom = self._calculate_momentum('HYG', target_date, period=60)
        lqd_mom = self._calculate_momentum('LQD', target_date, period=60)
        
        spread = hyg_mom - lqd_mom
        
        # Positive spread (HYG outperforming) = bullish
        return float(np.clip(spread * 10, -1, 1))
    
    def _safe_haven_signal(self, target_date: date) -> float:
        """
        Safe haven signal: GLD, TLT
        Strong safe haven = bearish equities (negative)
        """
        gld_mom = self._calculate_momentum('GLD', target_date, period=60)
        tlt_mom = self._calculate_momentum('TLT', target_date, period=60)
        
        avg_mom = (gld_mom + tlt_mom) / 2
        
        # Strong safe haven momentum = bearish equities (inverted)
        return float(np.clip(-avg_mom * 5, -1, 1))
    
    def _breadth_signal(self, target_date: date) -> float:
        """
        Breadth signal: Compare large cap (SPY) vs small cap (IWM)
        Positive breadth: IWM outperforming (broad rally)
        """
        spy_mom = self._calculate_momentum('SPY', target_date, period=60)
        iwm_mom = self._calculate_momentum('IWM', target_date, period=60)
        
        breadth = iwm_mom - spy_mom
        return float(np.clip(breadth * 10, -1, 1))
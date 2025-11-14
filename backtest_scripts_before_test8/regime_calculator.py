# backtest_scripts/regime_calculator.py
"""
Market Regime Detection
Analyzes macro indicators to determine current market regime

UPDATED: 2025-11-13 20:10 UTC - Original working version from GitHub
NOTE: This version takes DATE parameter, not dict
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from datetime import date, timedelta

class RegimeCalculator:
    """
    Calculate market regime based on multiple macro indicators
    
    Regimes:
    - BULL_STRONG: Strong uptrend, low volatility
    - BULL_WEAK: Moderate uptrend
    - NEUTRAL_BULLISH: Sideways with bullish bias
    - NEUTRAL_BEARISH: Sideways with bearish bias
    - BEAR_WEAK: Moderate downtrend
    - BEAR_STRONG: Strong downtrend
    - CRISIS: Market stress/panic
    """
    
    def __init__(self, macro_prices: Dict[str, pd.DataFrame]):
        """
        Initialize regime calculator
        
        Args:
            macro_prices: Dict mapping symbol → price DataFrame
        """
        self.macro_prices = macro_prices
        self.available_symbols = list(macro_prices.keys())
        
        print(f"[RegimeCalculator] Initialized with {len(self.available_symbols)} macro symbols")
    
    def calculate_regime(self, target_date: date) -> Dict:
        """
        Calculate market regime for a specific date
        
        Args:
            target_date: Date to calculate regime for
        
        Returns:
            Dict with regime info: {'date', 'regime', 'score', 'components'}
        """
        # Calculate component scores
        equity_score = self._calculate_equity_component(target_date)
        volatility_score = self._calculate_volatility_component(target_date)
        credit_score = self._calculate_credit_component(target_date)
        safe_haven_score = self._calculate_safe_haven_component(target_date)
        
        # Weighted composite score
        from .config import REGIME_COMPONENT_WEIGHTS
        
        composite_score = (
            equity_score * REGIME_COMPONENT_WEIGHTS['equity'] +
            volatility_score * REGIME_COMPONENT_WEIGHTS['volatility'] +
            credit_score * REGIME_COMPONENT_WEIGHTS['credit'] +
            safe_haven_score * REGIME_COMPONENT_WEIGHTS['safe_haven']
        )
        
        # Determine regime from score
        regime = self._score_to_regime(composite_score)
        
        return {
            'date': target_date,
            'regime': regime,
            'score': composite_score,
            'components': {
                'equity': equity_score,
                'volatility': volatility_score,
                'credit': credit_score,
                'safe_haven': safe_haven_score
            }
        }
    
    def _score_to_regime(self, score: float) -> str:
        """Map composite score to regime"""
        from .config import REGIME_THRESHOLDS
        
        if score >= REGIME_THRESHOLDS['BULL_STRONG']:
            return 'BULL_STRONG'
        elif score >= REGIME_THRESHOLDS['BULL_WEAK']:
            return 'BULL_WEAK'
        elif score >= REGIME_THRESHOLDS['NEUTRAL_BULLISH']:
            return 'NEUTRAL_BULLISH'
        elif score >= REGIME_THRESHOLDS['NEUTRAL_BEARISH']:
            return 'NEUTRAL_BEARISH'
        elif score >= REGIME_THRESHOLDS['BEAR_WEAK']:
            return 'BEAR_WEAK'
        elif score >= REGIME_THRESHOLDS['BEAR_STRONG']:
            return 'BEAR_STRONG'
        else:
            return 'CRISIS'
    
    def _calculate_equity_component(self, target_date: date) -> float:
        """Calculate equity market component score"""
        
        score = 0.0
        count = 0
        
        # Check SPY (S&P 500)
        if 'SPY' in self.macro_prices:
            spy_mom = self._calculate_momentum('SPY', target_date, periods=[20, 60])
            score += spy_mom
            count += 1
        
        # Check QQQ (NASDAQ)
        if 'QQQ' in self.macro_prices:
            qqq_mom = self._calculate_momentum('QQQ', target_date, periods=[20, 60])
            score += qqq_mom * 0.8  # Slightly less weight (more volatile)
            count += 0.8
        
        # Check IWM (Russell 2000)
        if 'IWM' in self.macro_prices:
            iwm_mom = self._calculate_momentum('IWM', target_date, periods=[20, 60])
            score += iwm_mom * 0.5  # Less weight (risk-on indicator)
            count += 0.5
        
        return score / count if count > 0 else 0.0
    
    def _calculate_volatility_component(self, target_date: date) -> float:
        """Calculate volatility component score (inverted - high vol = negative)"""
        
        score = 0.0
        count = 0
        
        # Check for VIX (try both ^VIX and VIX)
        vix_symbol = '^VIX' if '^VIX' in self.macro_prices else 'VIX'
        
        if vix_symbol in self.macro_prices:
            vix_level = self._get_indicator_level(vix_symbol, target_date)
            
            if vix_level is not None:
                # Normalize VIX: 10-15 = calm, 20+ = elevated, 30+ = panic
                if vix_level < 15:
                    vix_score = 1.0  # Very bullish
                elif vix_level < 20:
                    vix_score = 0.5  # Neutral-bullish
                elif vix_level < 30:
                    vix_score = -0.5  # Bearish
                else:
                    vix_score = -1.0  # Very bearish
                
                score += vix_score
                count += 1
        
        return score / count if count > 0 else 0.0
    
    def _calculate_credit_component(self, target_date: date) -> float:
        """Calculate credit market component score"""
        
        score = 0.0
        count = 0
        
        # High Yield (HYG)
        if 'HYG' in self.macro_prices:
            hyg_mom = self._calculate_momentum('HYG', target_date, periods=[20, 60])
            score += hyg_mom
            count += 1
        
        # Investment Grade (LQD)
        if 'LQD' in self.macro_prices:
            lqd_mom = self._calculate_momentum('LQD', target_date, periods=[20, 60])
            score += lqd_mom * 0.7  # Less weight (less risky)
            count += 0.7
        
        return score / count if count > 0 else 0.0
    
    def _calculate_safe_haven_component(self, target_date: date) -> float:
        """Calculate safe haven component score (inverted - strong safe haven = bearish)"""
        
        score = 0.0
        count = 0
        
        # Gold (GLD) - inverted
        if 'GLD' in self.macro_prices:
            gld_mom = self._calculate_momentum('GLD', target_date, periods=[20, 60])
            score -= gld_mom * 0.5  # Inverted and moderate weight
            count += 0.5
        
        # Treasuries (TLT) - inverted
        if 'TLT' in self.macro_prices:
            tlt_mom = self._calculate_momentum('TLT', target_date, periods=[20, 60])
            score -= tlt_mom * 0.5  # Inverted and moderate weight
            count += 0.5
        
        return score / count if count > 0 else 0.0
    
    def _calculate_momentum(self, symbol: str, target_date: date, periods: list = [20, 60]) -> float:
        """
        Calculate momentum score for a symbol
        
        Returns:
            float: Score between -1 (strong downtrend) and +1 (strong uptrend)
        """
        if symbol not in self.macro_prices:
            return 0.0
        
        prices = self.macro_prices[symbol]
        
        # Get price at target date
        target_row = prices[prices['date'] == target_date]
        if target_row.empty:
            return 0.0
        
        current_price = target_row.iloc[0]['close']
        
        scores = []
        
        for period in periods:
            past_date = target_date - timedelta(days=period)
            past_row = prices[prices['date'] <= past_date].tail(1)
            
            if not past_row.empty:
                past_price = past_row.iloc[0]['close']
                momentum = (current_price - past_price) / past_price
                
                # Normalize to -1 to +1 range
                # ±10% move = ±1 score
                normalized = np.clip(momentum * 10, -1, 1)
                scores.append(normalized)
        
        return np.mean(scores) if scores else 0.0
    
    def _get_indicator_level(self, symbol: str, target_date: date) -> Optional[float]:
        """Get indicator level (e.g., VIX level) at target date"""
        
        if symbol not in self.macro_prices:
            return None
        
        prices = self.macro_prices[symbol]
        target_row = prices[prices['date'] == target_date]
        
        if target_row.empty:
            return None
        
        return target_row.iloc[0]['close']
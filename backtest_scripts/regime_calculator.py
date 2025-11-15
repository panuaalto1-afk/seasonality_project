"""
Regime Calculator - Historical Regime Detection
Matches live regime_detector.py logic for backtesting
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class RegimeCalculator:
    """
    Calculates market regime using historical data.
    
    Matches live regime_detector.py:
    - 5 components (Equity, Volatility, Credit, Safe Haven, Breadth)
    - 7 regime states
    - Composite score calculation
    
    Key difference from live:
    - Uses walk-forward approach (only data up to target_date)
    - No future leak
    """
    
    def __init__(self):
        """Initialize regime calculator."""
        self.lookback_short = 20   # Short-term (1 month)
        self.lookback_medium = 60  # Medium-term (3 months)
        self.lookback_long = 252   # Long-term (1 year)
        
        # Component weights
        self.weights = {
            'equity': 0.35,
            'volatility': 0.20,
            'credit': 0.20,
            'safe_haven': 0.15,
            'breadth': 0.10,
        }
    
    def get_price_data(
        self,
        ticker: str,
        regime_prices: Dict[str, pd.DataFrame],
        target_date: datetime,
        lookback_days: int = 252
    ) -> Optional[pd.DataFrame]:
        """Get price data up to target_date with lookback."""
        if ticker not in regime_prices:
            return None
        
        df = regime_prices[ticker]
        
        # Filter up to target_date
        cutoff_date = target_date - timedelta(days=lookback_days)
        df = df[(df['date'] >= cutoff_date) & (df['date'] <= target_date)]
        
        if len(df) < lookback_days // 2:
            return None
        
        return df
    
    def calculate_momentum(self, df: pd.DataFrame, periods: List[int]) -> float:
        """
        Calculate momentum score from returns.
        
        Returns normalized score: -1 (bearish) to +1 (bullish)
        """
        if len(df) < max(periods):
            return 0.0
        
        closes = df['close'].values
        scores = []
        
        for period in periods:
            if len(closes) < period:
                continue
            ret = (closes[-1] / closes[-period] - 1)
            scores.append(ret)
        
        # Average return
        avg_return = np.mean(scores) if scores else 0
        
        # Normalize to -1 to +1
        # Typical range: -30% to +30%
        normalized = np.clip(avg_return * 3, -1, 1)
        
        return normalized
    
    def calculate_equity_component(
        self,
        regime_prices: Dict[str, pd.DataFrame],
        target_date: datetime
    ) -> float:
        """
        Calculate equity momentum component.
        
        Uses: SPY (35%), QQQ (35%), IWM (30%)
        """
        tickers = ['SPY', 'QQQ', 'IWM']
        weights = [0.35, 0.35, 0.30]
        scores = []
        
        for ticker, weight in zip(tickers, weights):
            df = self.get_price_data(ticker, regime_prices, target_date)
            if df is None:
                continue
            
            # Calculate momentum over multiple periods
            mom = self.calculate_momentum(
                df,
                periods=[self.lookback_short, self.lookback_medium]
            )
            
            scores.append(mom * weight)
        
        return sum(scores) if scores else 0.0
    
    def calculate_volatility_component(
        self,
        regime_prices: Dict[str, pd.DataFrame],
        target_date: datetime
    ) -> float:
        """
        Calculate volatility component.
        
        Uses: SPY realized volatility
        High vol = bearish, Low vol = bullish
        """
        df = self.get_price_data('SPY', regime_prices, target_date)
        if df is None:
            return 0.0
        
        # Calculate 20-day realized volatility
        returns = df['close'].pct_change().dropna()
        
        if len(returns) < self.lookback_short:
            return 0.0
        
        recent_vol = returns.tail(self.lookback_short).std() * np.sqrt(252)
        long_vol = returns.tail(self.lookback_long).std() * np.sqrt(252) if len(returns) >= self.lookback_long else recent_vol
        
        # High vol (>30%) = bearish (-1)
        # Low vol (<15%) = bullish (+1)
        if recent_vol > 0.30:
            vol_score = -1.0
        elif recent_vol < 0.15:
            vol_score = 1.0
        else:
            # Linear interpolation
            vol_score = 1.0 - (recent_vol - 0.15) / 0.15
        
        # Compare to long-term vol (increasing vol = bearish)
        if long_vol > 0:
            vol_change = (recent_vol - long_vol) / long_vol
            vol_score -= vol_change * 0.5
        
        return np.clip(vol_score, -1, 1)
    
    def calculate_credit_component(
        self,
        regime_prices: Dict[str, pd.DataFrame],
        target_date: datetime
    ) -> float:
        """
        Calculate credit spread component.
        
        Uses: HYG (high yield) vs LQD (investment grade)
        Narrowing spreads = bullish, Widening = bearish
        """
        hyg_df = self.get_price_data('HYG', regime_prices, target_date)
        lqd_df = self.get_price_data('LQD', regime_prices, target_date)
        
        if hyg_df is None or lqd_df is None:
            return 0.0
        
        # Calculate relative performance
        hyg_ret = self.calculate_momentum(hyg_df, [self.lookback_short, self.lookback_medium])
        lqd_ret = self.calculate_momentum(lqd_df, [self.lookback_short, self.lookback_medium])
        
        # HYG outperforming LQD = narrowing spreads = bullish
        credit_score = (hyg_ret - lqd_ret) * 2  # Amplify difference
        
        return np.clip(credit_score, -1, 1)
    
    def calculate_safe_haven_component(
        self,
        regime_prices: Dict[str, pd.DataFrame],
        target_date: datetime
    ) -> float:
        """
        Calculate safe haven flows component.
        
        Uses: GLD (gold) and TLT (long-term treasuries)
        Strong safe haven demand = bearish for equities
        """
        gld_df = self.get_price_data('GLD', regime_prices, target_date)
        tlt_df = self.get_price_data('TLT', regime_prices, target_date)
        
        scores = []
        
        # Gold momentum (inverse relationship with equities)
        if gld_df is not None:
            gld_mom = self.calculate_momentum(gld_df, [self.lookback_short])
            scores.append(-gld_mom * 0.6)  # Inverse and weighted
        
        # TLT momentum (inverse relationship with equities)
        if tlt_df is not None:
            tlt_mom = self.calculate_momentum(tlt_df, [self.lookback_short])
            scores.append(-tlt_mom * 0.4)  # Inverse and weighted
        
        return sum(scores) if scores else 0.0
    
    def calculate_breadth_component(
        self,
        regime_prices: Dict[str, pd.DataFrame],
        target_date: datetime
    ) -> float:
        """
        Calculate market breadth component.
        
        Uses: SPY vs IWM correlation
        High correlation = good breadth = bullish
        Low/negative correlation = poor breadth = bearish
        """
        spy_df = self.get_price_data('SPY', regime_prices, target_date, lookback_days=120)
        iwm_df = self.get_price_data('IWM', regime_prices, target_date, lookback_days=120)
        
        if spy_df is None or iwm_df is None:
            return 0.0
        
        # Align dates
        merged = pd.merge(
            spy_df[['date', 'close']].rename(columns={'close': 'spy'}),
            iwm_df[['date', 'close']].rename(columns={'close': 'iwm'}),
            on='date'
        )
        
        if len(merged) < 30:
            return 0.0
        
        # Calculate returns
        merged['spy_ret'] = merged['spy'].pct_change()
        merged['iwm_ret'] = merged['iwm'].pct_change()
        merged = merged.dropna()
        
        # Calculate rolling correlation
        if len(merged) < 20:
            return 0.0
        
        corr = merged['spy_ret'].tail(20).corr(merged['iwm_ret'].tail(20))
        
        # High correlation (>0.7) = bullish
        # Low correlation (<0.3) = bearish
        if corr > 0.7:
            breadth_score = 1.0
        elif corr < 0.3:
            breadth_score = -1.0
        else:
            # Linear interpolation
            breadth_score = (corr - 0.3) / 0.4 * 2 - 1
        
        return np.clip(breadth_score, -1, 1)
    
    def calculate_regime(
        self,
        target_date: datetime,
        regime_prices: Dict[str, pd.DataFrame]
    ) -> Optional[Dict]:
        """
        Calculate market regime for target_date.
        
        Returns dict with:
        - regime: str (7 states)
        - score: float (composite)
        - components: dict (individual scores)
        """
        try:
            # Calculate all components
            equity_score = self.calculate_equity_component(regime_prices, target_date)
            vol_score = self.calculate_volatility_component(regime_prices, target_date)
            credit_score = self.calculate_credit_component(regime_prices, target_date)
            safe_haven_score = self.calculate_safe_haven_component(regime_prices, target_date)
            breadth_score = self.calculate_breadth_component(regime_prices, target_date)
            
            # Calculate composite score
            composite = (
                equity_score * self.weights['equity'] +
                vol_score * self.weights['volatility'] +
                credit_score * self.weights['credit'] +
                safe_haven_score * self.weights['safe_haven'] +
                breadth_score * self.weights['breadth']
            )
            
            # Map to regime
            if composite >= 0.50:
                regime = 'BULL_STRONG'
            elif composite >= 0.25:
                regime = 'BULL_WEAK'
            elif composite >= 0.0:
                regime = 'NEUTRAL_BULLISH'
            elif composite >= -0.25:
                regime = 'NEUTRAL_BEARISH'
            elif composite >= -0.50:
                regime = 'BEAR_WEAK'
            elif composite >= -0.75:
                regime = 'BEAR_STRONG'
            else:
                regime = 'CRISIS'
            
            return {
                'date': target_date,
                'regime': regime,
                'composite_score': composite,
                'components': {
                    'equity': equity_score,
                    'volatility': vol_score,
                    'credit': credit_score,
                    'safe_haven': safe_haven_score,
                    'breadth': breadth_score,
                }
            }
            
        except Exception as e:
            logger.error(f"Error calculating regime for {target_date}: {str(e)}")
            return None
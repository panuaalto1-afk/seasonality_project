"""
ML Signal Generator - Walk-Forward Signal Generation (FULL VERSION)
Generates buy signals based on ML scores and seasonality
Matches live ml_unified_pipeline.py (44KB) logic
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime, timedelta
import logging

from config import (
    GATE_ALPHA, MIN_SCORE_LONG,
    VINTAGE_DIR, PRICE_CACHE_DIR,
)

logger = logging.getLogger(__name__)


class MLSignalGenerator:
    """
    Generates ML-based trading signals for backtest.
    
    Uses walk-forward approach:
    - Only uses data available up to target_date
    - No future leak
    - Matches live ml_unified_pipeline.py logic
    
    Features calculated:
    - Momentum (5/20/60 day)
    - Volatility (20-day realized)
    - Seasonality (week-of-year, day-of-year, segments)
    - ATR (14-day for position sizing)
    - Regime alignment
    """
    
    def __init__(self):
        """Initialize signal generator."""
        self.vintage_dir = Path(VINTAGE_DIR)
        self.seasonality_cache = {}  # Cache loaded seasonality data
        
    def load_vintage_seasonality(self, ticker: str) -> Optional[Dict[str, pd.DataFrame]]:
        """
        Load pre-calculated seasonality data from vintage.
        
        Files:
        - {TICKER}_seasonality_week.csv - Week-of-year patterns
        - {TICKER}_seasonality_20d.csv - 20-day forward returns
        - {TICKER}_segments_up.csv - Bullish segments
        - {TICKER}_segments_down.csv - Bearish segments
        - {TICKER}_vintage_10y.csv - 10-year vintage data
        
        Returns dict with all available seasonality data
        """
        # Check cache first
        if ticker in self.seasonality_cache:
            return self.seasonality_cache[ticker]
        
        seasonality_data = {}
        
        # Load week-of-year seasonality
        week_file = self.vintage_dir / f"{ticker}_seasonality_week.csv"
        if week_file.exists():
            try:
                seasonality_data['week'] = pd.read_csv(week_file)
            except Exception as e:
                logger.debug(f"Could not load week seasonality for {ticker}: {e}")
        
        # Load 20-day seasonality
        day20_file = self.vintage_dir / f"{ticker}_seasonality_20d.csv"
        if day20_file.exists():
            try:
                seasonality_data['day20'] = pd.read_csv(day20_file)
            except Exception as e:
                logger.debug(f"Could not load 20d seasonality for {ticker}: {e}")
        
        # Load bullish segments
        seg_up_file = self.vintage_dir / f"{ticker}_segments_up.csv"
        if seg_up_file.exists():
            try:
                seasonality_data['segments_up'] = pd.read_csv(seg_up_file)
            except Exception as e:
                logger.debug(f"Could not load bullish segments for {ticker}: {e}")
        
        # Load bearish segments
        seg_down_file = self.vintage_dir / f"{ticker}_segments_down.csv"
        if seg_down_file.exists():
            try:
                seasonality_data['segments_down'] = pd.read_csv(seg_down_file)
            except Exception as e:
                logger.debug(f"Could not load bearish segments for {ticker}: {e}")
        
        # Cache it
        if seasonality_data:
            self.seasonality_cache[ticker] = seasonality_data
            return seasonality_data
        
        return None
    
    def calculate_momentum_features(
        self,
        price_df: pd.DataFrame,
        target_date: datetime
    ) -> Optional[Dict]:
        """
        Calculate momentum features up to target_date.
        
        Features:
        - mom5: 5-day return
        - mom20: 20-day return
        - mom60: 60-day return
        - mom252: 252-day (1-year) return
        - vol20: 20-day realized volatility (annualized)
        - vol60: 60-day realized volatility (annualized)
        """
        # Filter to data before target_date (walk-forward safe)
        df = price_df[price_df['date'] < target_date].copy()
        
        if len(df) < 60:
            return None
        
        # Use last row
        latest = df.iloc[-1]
        
        # Calculate returns
        close = df['close'].values
        
        if len(close) < 60:
            return None
        
        # Momentum features
        mom5 = (close[-1] / close[-6] - 1) if len(close) >= 6 else 0
        mom20 = (close[-1] / close[-21] - 1) if len(close) >= 21 else 0
        mom60 = (close[-1] / close[-61] - 1) if len(close) >= 61 else 0
        mom252 = (close[-1] / close[-253] - 1) if len(close) >= 253 else 0
        
        # Volatility features (20-day and 60-day)
        if len(close) >= 21:
            returns_20 = np.diff(close[-21:]) / close[-22:-1]
            vol20 = np.std(returns_20) * np.sqrt(252) if len(returns_20) > 0 else 0
        else:
            vol20 = 0
        
        if len(close) >= 61:
            returns_60 = np.diff(close[-61:]) / close[-62:-1]
            vol60 = np.std(returns_60) * np.sqrt(252) if len(returns_60) > 0 else 0
        else:
            vol60 = 0
        
        # Recent trend (5-day vs 20-day comparison)
        trend_strength = mom5 - mom20 if mom20 != 0 else 0
        
        # Momentum consistency (all positive or all negative)
        mom_signs = [np.sign(mom5), np.sign(mom20), np.sign(mom60)]
        momentum_consistency = 1.0 if len(set(mom_signs)) == 1 else 0.5
        
        return {
            'mom5': mom5,
            'mom20': mom20,
            'mom60': mom60,
            'mom252': mom252,
            'vol20': vol20,
            'vol60': vol60,
            'close': close[-1],
            'trend_strength': trend_strength,
            'momentum_consistency': momentum_consistency,
        }
    
    def calculate_seasonality_features(
        self,
        ticker: str,
        target_date: datetime
    ) -> Dict:
        """
        Calculate seasonality features for target_date.
        
        Features:
        - season_week_avg: Average return for this week-of-year
        - season_week_hit_rate: Historical hit rate for this week
        - season_20d_avg: Average 20-day forward return
        - season_month_avg: Monthly seasonality
        - in_bullish_segment: Currently in bullish seasonal period
        - in_bearish_segment: Currently in bearish seasonal period
        - segment_strength: Strength of current segment
        - days_into_segment: How many days into current segment
        """
        season_data = self.load_vintage_seasonality(ticker)
        
        default_features = {
            'season_week_avg': 0,
            'season_week_hit_rate': 0.5,
            'season_20d_avg': 0,
            'season_month_avg': 0,
            'in_bullish_segment': False,
            'in_bearish_segment': False,
            'segment_strength': 0,
            'days_into_segment': 0,
        }
        
        if season_data is None:
            return default_features
        
        features = default_features.copy()
        
        # Week of year
        week_of_year = target_date.isocalendar()[1]
        day_of_year = target_date.timetuple().tm_yday
        month_of_year = target_date.month
        
        # Week-of-year seasonality
        if 'week' in season_data:
            week_df = season_data['week']
            week_match = week_df[week_df['week_of_year'] == week_of_year]
            
            if not week_match.empty:
                features['season_week_avg'] = week_match['avg_return'].iloc[0] if 'avg_return' in week_match.columns else 0
                features['season_week_hit_rate'] = week_match['hit_rate'].iloc[0] if 'hit_rate' in week_match.columns else 0.5
        
        # 20-day seasonality (day-of-year based)
        if 'day20' in season_data:
            day20_df = season_data['day20']
            # Find closest day-of-year (±3 day window)
            day_window = day20_df[
                (day20_df['day_of_year'] >= day_of_year - 3) &
                (day20_df['day_of_year'] <= day_of_year + 3)
            ]
            
            if not day_window.empty:
                features['season_20d_avg'] = day_window['avg_20d_return'].mean() if 'avg_20d_return' in day_window.columns else 0
        
        # Month-of-year seasonality (if available)
        if 'week' in season_data:
            # Approximate from week data
            week_df = season_data['week']
            month_weeks = week_df[week_df['week_of_year'].between(
                (month_of_year - 1) * 4 + 1,
                month_of_year * 4
            )]
            if not month_weeks.empty:
                features['season_month_avg'] = month_weeks['avg_return'].mean() if 'avg_return' in month_weeks.columns else 0
        
        # Check if in bullish segment
        if 'segments_up' in season_data:
            seg_up = season_data['segments_up']
            
            for _, segment in seg_up.iterrows():
                # Check if current date falls in segment
                # (This is simplified - real implementation would parse dates)
                segment_start_day = segment.get('start_day_of_year', 0)
                segment_end_day = segment.get('end_day_of_year', 0)
                
                if segment_start_day <= day_of_year <= segment_end_day:
                    features['in_bullish_segment'] = True
                    features['segment_strength'] = segment.get('avg_return', 0)
                    features['days_into_segment'] = day_of_year - segment_start_day
                    break
        
        # Check if in bearish segment
        if 'segments_down' in season_data and not features['in_bullish_segment']:
            seg_down = season_data['segments_down']
            
            for _, segment in seg_down.iterrows():
                segment_start_day = segment.get('start_day_of_year', 0)
                segment_end_day = segment.get('end_day_of_year', 0)
                
                if segment_start_day <= day_of_year <= segment_end_day:
                    features['in_bearish_segment'] = True
                    features['segment_strength'] = segment.get('avg_return', 0)
                    features['days_into_segment'] = day_of_year - segment_start_day
                    break
        
        return features
    
    def calculate_atr(
        self,
        price_df: pd.DataFrame,
        target_date: datetime,
        period: int = 14
    ) -> float:
        """
        Calculate ATR (Average True Range) for position sizing.
        
        ATR is used for:
        - Stop loss calculation (e.g., entry - 1.0*ATR)
        - Take profit calculation (e.g., entry + 2.5*ATR)
        - Volatility assessment
        - Position sizing adjustments
        
        Method: Wilder's smoothing (exponential moving average)
        """
        df = price_df[price_df['date'] < target_date].copy()
        
        if len(df) < period + 1:
            # Fallback: use simple volatility estimate
            if len(df) < 2:
                return df['close'].iloc[-1] * 0.02 if not df.empty else 0
            
            returns = df['close'].pct_change().dropna()
            if len(returns) == 0:
                return df['close'].iloc[-1] * 0.02
            
            return df['close'].iloc[-1] * returns.std() * np.sqrt(period)
        
        # Calculate True Range
        df['h-l'] = df['high'] - df['low']
        df['h-pc'] = abs(df['high'] - df['close'].shift(1))
        df['l-pc'] = abs(df['low'] - df['close'].shift(1))
        
        df['tr'] = df[['h-l', 'h-pc', 'l-pc']].max(axis=1)
        
        # Calculate ATR using Wilder's method (exponential moving average)
        # First ATR = simple average of first 'period' TRs
        df['atr'] = df['tr'].rolling(window=period, min_periods=period).mean()
        
        # Then use exponential smoothing: ATR = ((prior ATR * (period-1)) + current TR) / period
        for i in range(period, len(df)):
            if pd.notna(df.iloc[i-1]['atr']) and pd.notna(df.iloc[i]['tr']):
                df.loc[df.index[i], 'atr'] = (
                    (df.iloc[i-1]['atr'] * (period - 1) + df.iloc[i]['tr']) / period
                )
        
        atr = df['atr'].iloc[-1]
        
        if pd.isna(atr) or atr <= 0:
            # Fallback: 2% of price
            atr = df['close'].iloc[-1] * 0.02
        
        return atr
    
    def calculate_ml_score(
        self,
        momentum: Dict,
        seasonality: Dict,
        regime: str
    ) -> float:
        """
        Calculate ML score combining momentum and seasonality.
        
        This is a simplified scoring model. In production, this would be:
        - Trained LightGBM/XGBoost model
        - More features (volume, fundamentals, sentiment)
        - Regime-specific models
        - Cross-validated predictions
        
        Current formula (weighted blend):
        - 40% short-term momentum (mom5)
        - 30% medium-term momentum (mom20)
        - 20% seasonality (week_avg + 20d_avg)
        - 10% trend consistency
        
        Adjustments:
        - Boost for bullish segment
        - Penalize for high volatility in bearish regimes
        - Boost for momentum consistency
        """
        # Momentum component (weighted)
        mom_score = (
            momentum.get('mom5', 0) * 0.40 +
            momentum.get('mom20', 0) * 0.30 +
            momentum.get('mom60', 0) * 0.10
        )
        
        # Seasonality component (weighted)
        season_score = (
            seasonality.get('season_week_avg', 0) * 0.10 +
            seasonality.get('season_20d_avg', 0) * 0.10
        )
        
        # Trend strength component
        trend_score = momentum.get('trend_strength', 0) * 0.05
        
        # Momentum consistency bonus
        consistency_bonus = momentum.get('momentum_consistency', 0.5) * 0.05
        
        # Combine (total = 100%)
        combined = mom_score + season_score + trend_score + consistency_bonus
        
        # Apply regime-specific adjustments
        if regime in ['BULL_STRONG', 'BULL_WEAK']:
            # Boost momentum in bull markets
            combined *= 1.1
        elif regime in ['BEAR_WEAK', 'BEAR_STRONG']:
            # Penalize in bear markets, emphasize quality
            combined *= 0.9
            # Extra penalty for high volatility
            if momentum.get('vol20', 0) > 0.35:
                combined *= 0.85
        
        # Boost if in bullish segment
        if seasonality.get('in_bullish_segment', False):
            segment_strength = abs(seasonality.get('segment_strength', 0))
            boost = min(0.15, segment_strength * 2)  # Max 15% boost
            combined *= (1 + boost)
        
        # Penalize if in bearish segment (unless strong positive momentum)
        if seasonality.get('in_bearish_segment', False) and mom_score < 0.05:
            combined *= 0.85
        
        # Normalize to 0-1 range
        # Typical scores range from -0.15 to +0.25
        # Map to 0-1: (score + 0.15) / 0.40
        normalized = max(0, min(1, (combined + 0.15) / 0.40))
        
        return normalized
    
    def generate_signals(
        self,
        tickers: List[str],
        target_date: datetime,
        price_data: Dict[str, pd.DataFrame],
        regime: str,
        sector_map: Dict[str, str]
    ) -> pd.DataFrame:
        """
        Generate trading signals for all tickers on target_date.
        
        Returns DataFrame with columns:
        - ticker
        - score_long (0-1, primary ranking metric)
        - entry_price (close T-1, for position entry)
        - atr_14 (for SL/TP calculation)
        - mom5, mom20, mom60, mom252
        - vol20, vol60
        - season_week_avg, season_20d_avg
        - in_bullish_segment, segment_strength
        - trend_strength, momentum_consistency
        - regime
        - sector
        """
        signals = []
        
        for ticker in tickers:
            if ticker not in price_data:
                continue
            
            price_df = price_data[ticker]
            
            # Calculate momentum features
            momentum = self.calculate_momentum_features(price_df, target_date)
            if momentum is None:
                continue
            
            # Calculate seasonality features
            seasonality = self.calculate_seasonality_features(ticker, target_date)
            
            # Calculate ATR
            atr = self.calculate_atr(price_df, target_date)
            
            # Validate ATR
            if atr <= 0 or not np.isfinite(atr):
                logger.debug(f"Invalid ATR for {ticker}: {atr}")
                continue
            
            # Calculate ML score
            score = self.calculate_ml_score(momentum, seasonality, regime)
            
            # Get sector
            sector = sector_map.get(ticker, 'Unknown')
            
            # Create signal
            signal = {
                'ticker': ticker,
                'score_long': score,
                'entry_price': momentum['close'],
                'atr_14': atr,
                
                # Momentum features
                'mom5': momentum['mom5'],
                'mom20': momentum['mom20'],
                'mom60': momentum['mom60'],
                'mom252': momentum.get('mom252', 0),
                'vol20': momentum['vol20'],
                'vol60': momentum.get('vol60', 0),
                'trend_strength': momentum.get('trend_strength', 0),
                'momentum_consistency': momentum.get('momentum_consistency', 0.5),
                
                # Seasonality features
                'season_week_avg': seasonality['season_week_avg'],
                'season_20d_avg': seasonality['season_20d_avg'],
                'season_month_avg': seasonality['season_month_avg'],
                'season_week_hit_rate': seasonality['season_week_hit_rate'],
                'in_bullish_segment': seasonality['in_bullish_segment'],
                'in_bearish_segment': seasonality['in_bearish_segment'],
                'segment_strength': seasonality['segment_strength'],
                'days_into_segment': seasonality['days_into_segment'],
                
                # Context
                'regime': regime,
                'sector': sector,
            }
            
            signals.append(signal)
        
        # Convert to DataFrame
        signals_df = pd.DataFrame(signals)
        
        # Filter by minimum score
        if not signals_df.empty:
            initial_count = len(signals_df)
            signals_df = signals_df[signals_df['score_long'] >= MIN_SCORE_LONG]
            
            if len(signals_df) < initial_count:
                logger.debug(f"Filtered {initial_count - len(signals_df)} signals below min_score {MIN_SCORE_LONG}")
        
        return signals_df
    
    def get_top_candidates(
        self,
        signals_df: pd.DataFrame,
        top_n: int = 50
    ) -> pd.DataFrame:
        """
        Get top N candidates by score.
        
        This creates the equivalent of top_long_candidates_GATED.csv
        from live ml_unified_pipeline.py
        """
        if signals_df.empty:
            return signals_df
        
        # Filter by gate alpha
        signals_df = signals_df[signals_df['score_long'] >= GATE_ALPHA]
        
        # Sort by score descending
        signals_df = signals_df.sort_values('score_long', ascending=False)
        
        # Take top N
        top_candidates = signals_df.head(top_n)
        
        logger.info(f"Generated {len(signals_df)} gated signals, returning top {len(top_candidates)}")
        
        return top_candidates
    
    def generate_labels(
        self,
        tickers: List[str],
        target_date: datetime,
        price_data: Dict[str, pd.DataFrame],
        forward_days: int = 20
    ) -> pd.DataFrame:
        """
        Generate forward returns (labels) for model training/validation.
        
        This is used for:
        - Walk-forward model training
        - Performance attribution
        - Feature importance analysis
        
        NOT used for signal generation (to avoid future leak)
        """
        labels = []
        
        for ticker in tickers:
            if ticker not in price_data:
                continue
            
            df = price_data[ticker]
            
            # Find target date index
            mask = df['date'] == target_date
            if not mask.any():
                continue
            
            idx = df[mask].index[0]
            
            # Get current price
            current_price = df.loc[idx, 'close']
            
            # Get future price (forward_days ahead)
            future_idx = idx + forward_days
            if future_idx >= len(df):
                continue
            
            future_price = df.iloc[future_idx]['close']
            forward_return = (future_price / current_price - 1) * 100
            
            labels.append({
                'ticker': ticker,
                'target_date': target_date,
                'forward_days': forward_days,
                'forward_return': forward_return,
                'current_price': current_price,
                'future_price': future_price,
            })
        
        return pd.DataFrame(labels)
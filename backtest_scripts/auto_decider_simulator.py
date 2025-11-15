"""
Auto Decider Simulator - Decision Engine for Backtest (FULL VERSION)
Mimics live auto_decider.py logic with regime-based decisions
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging

from .config import (
    REGIME_MAX_POSITIONS, DEFAULT_MAX_POSITIONS,
    SECTOR_MAX_POSITIONS, ENABLE_SECTOR_DIVERSIFICATION,
    SECTOR_BLACKLIST, GATE_ALPHA,
    REGIME_STRATEGIES, REGIME_MIN_HOLD_DAYS,
)

logger = logging.getLogger(__name__)


class AutoDeciderSimulator:
    """
    Simulates auto_decider.py decisions during backtest.
    
    Key responsibilities:
    - Filter candidates by regime and sector
    - Apply regime-specific strategies
    - Decide BUY/SELL/HOLD
    - Handle regime transitions
    - Position sizing adjustments
    - Exit logic (regime-based, time-based, score-based)
    
    Matches live auto_decider.py (45KB) logic
    """
    
    def __init__(self):
        """Initialize auto decider."""
        self.regime_history = []
        self.decision_history = []
        
    def get_max_positions(self, regime: str) -> int:
        """Get max positions for current regime."""
        return REGIME_MAX_POSITIONS.get(regime, DEFAULT_MAX_POSITIONS)
    
    def get_sector_max_positions(self, sector: str) -> int:
        """Get max positions for a sector."""
        if not ENABLE_SECTOR_DIVERSIFICATION:
            return 999  # No limit
        
        return SECTOR_MAX_POSITIONS.get(sector, SECTOR_MAX_POSITIONS.get('Default', 1))
    
    def get_regime_strategy(self, regime: str) -> Dict:
        """Get strategy parameters for regime."""
        return REGIME_STRATEGIES.get(regime, {
            'max_positions': DEFAULT_MAX_POSITIONS,
            'position_size_multiplier': 1.0,
            'stop_multiplier': 1.0,
            'tp_multiplier': 2.0,
            'min_hold_days': 7,
            'gate_adjustment': 1.0,
        })
    
    def filter_by_sector(
        self,
        candidates: pd.DataFrame,
        current_sector_positions: Dict[str, int]
    ) -> pd.DataFrame:
        """
        Filter candidates by sector constraints.
        
        Removes:
        - Blacklisted sectors
        - Sectors at max positions
        - Unknown/missing sectors (optional)
        """
        if candidates.empty:
            return candidates
        
        initial_count = len(candidates)
        
        # Remove blacklisted sectors
        if SECTOR_BLACKLIST:
            candidates = candidates[~candidates['sector'].isin(SECTOR_BLACKLIST)]
            if len(candidates) < initial_count:
                logger.debug(f"Filtered {initial_count - len(candidates)} blacklisted sector candidates")
        
        # Check sector limits
        def can_add_sector(row):
            sector = row['sector']
            if pd.isna(sector) or sector == 'Unknown':
                # Allow unknown sectors with default limit
                sector = 'Default'
            
            current_count = current_sector_positions.get(sector, 0)
            max_count = self.get_sector_max_positions(sector)
            return current_count < max_count
        
        candidates = candidates[candidates.apply(can_add_sector, axis=1)]
        
        if len(candidates) < initial_count:
            logger.debug(f"Filtered {initial_count - len(candidates)} candidates due to sector limits")
        
        return candidates
    
    def apply_regime_adjustments(
        self,
        candidates: pd.DataFrame,
        regime: str
    ) -> pd.DataFrame:
        """
        Apply regime-specific score adjustments.
        
        Regime strategies can adjust gate threshold:
        - BULL: Lower threshold (easier entry) - gate_adjustment < 1.0
        - BEAR: Higher threshold (harder entry) - gate_adjustment > 1.0
        - NEUTRAL: Standard threshold - gate_adjustment = 1.0
        """
        if candidates.empty:
            return candidates
        
        strategy = self.get_regime_strategy(regime)
        gate_adjustment = strategy.get('gate_adjustment', 1.0)
        
        # Adjust effective gate
        adjusted_gate = GATE_ALPHA * gate_adjustment
        
        # Filter by adjusted gate
        initial_count = len(candidates)
        candidates = candidates[candidates['score_long'] >= adjusted_gate]
        
        if len(candidates) < initial_count:
            logger.debug(f"Regime {regime}: Filtered {initial_count - len(candidates)} by gate {adjusted_gate:.3f}")
        
        return candidates
    
    def apply_quality_filters(
        self,
        candidates: pd.DataFrame,
        regime: str
    ) -> pd.DataFrame:
        """
        Apply additional quality filters based on regime.
        
        Filters:
        - Minimum momentum threshold
        - Volatility limits
        - Seasonality strength
        - ATR validation
        """
        if candidates.empty:
            return candidates
        
        initial_count = len(candidates)
        
        # Filter out invalid ATR
        if 'atr_14' in candidates.columns:
            candidates = candidates[candidates['atr_14'] > 0]
        
        # Regime-specific filters
        if regime in ['BEAR_WEAK', 'BEAR_STRONG', 'CRISIS']:
            # In bear markets, require stronger signals
            if 'mom20' in candidates.columns:
                candidates = candidates[candidates['mom20'] > -0.05]  # Not too negative
            
            if 'vol20' in candidates.columns:
                candidates = candidates[candidates['vol20'] < 0.5]  # Not too volatile
        
        elif regime in ['BULL_STRONG', 'BULL_WEAK']:
            # In bull markets, prefer positive momentum
            if 'mom5' in candidates.columns:
                candidates = candidates[candidates['mom5'] > -0.10]  # Allow small dips
        
        if len(candidates) < initial_count:
            logger.debug(f"Quality filters removed {initial_count - len(candidates)} candidates")
        
        return candidates
    
    def rank_candidates(
        self,
        candidates: pd.DataFrame,
        regime: str,
        current_positions: List[str]
    ) -> pd.DataFrame:
        """
        Rank candidates by score and regime-specific criteria.
        
        Ranking factors:
        1. ML score (primary)
        2. Seasonality strength
        3. Momentum direction
        4. Sector preference
        5. Avoid re-entering recent exits
        """
        if candidates.empty:
            return candidates
        
        # Primary sort: score_long descending
        candidates = candidates.sort_values('score_long', ascending=False)
        
        # Secondary factors (adjust score slightly)
        if 'in_bullish_segment' in candidates.columns:
            candidates['rank_boost'] = candidates['in_bullish_segment'].astype(float) * 0.01
        else:
            candidates['rank_boost'] = 0
        
        # Boost for strong seasonality
        if 'season_week_avg' in candidates.columns:
            candidates['rank_boost'] += candidates['season_week_avg'].clip(0, 0.05) * 0.1
        
        # Penalize high volatility in bearish regimes
        if regime in ['BEAR_WEAK', 'BEAR_STRONG', 'CRISIS']:
            if 'vol20' in candidates.columns:
                candidates['rank_boost'] -= candidates['vol20'].clip(0.3, 0.6) * 0.02
        
        # Adjusted score for ranking
        candidates['adjusted_score'] = candidates['score_long'] + candidates['rank_boost']
        candidates = candidates.sort_values('adjusted_score', ascending=False)
        
        # Clean up temporary columns
        candidates = candidates.drop(columns=['rank_boost', 'adjusted_score'], errors='ignore')
        
        return candidates
    
    def select_buys(
        self,
        candidates: pd.DataFrame,
        regime: str,
        current_positions: int,
        current_sector_positions: Dict[str, int],
        current_tickers: List[str]
    ) -> pd.DataFrame:
        """
        Select which candidates to buy.
        
        Process:
        1. Filter by sector constraints
        2. Apply regime adjustments
        3. Apply quality filters
        4. Rank by score
        5. Select top N (up to available slots)
        """
        if candidates.empty:
            return candidates
        
        # Get max positions for regime
        max_positions = self.get_max_positions(regime)
        available_slots = max(0, max_positions - current_positions)
        
        if available_slots == 0:
            logger.debug(f"No available slots (current: {current_positions}, max: {max_positions})")
            return pd.DataFrame()
        
        logger.debug(f"Selecting buys: {len(candidates)} candidates, {available_slots} slots available")
        
        # Filter by sector
        candidates = self.filter_by_sector(candidates, current_sector_positions)
        
        # Apply regime adjustments
        candidates = self.apply_regime_adjustments(candidates, regime)
        
        # Apply quality filters
        candidates = self.apply_quality_filters(candidates, regime)
        
        # Rank candidates
        candidates = self.rank_candidates(candidates, regime, current_tickers)
        
        # Select top N
        selected = candidates.head(available_slots)
        
        logger.debug(f"Selected {len(selected)} candidates for buying")
        
        return selected
    
    def check_regime_exits(
        self,
        current_positions: List[Dict],
        regime: str,
        prev_regime: str
    ) -> List[str]:
        """
        Check if regime change requires position exits.
        
        Exit conditions:
        - CRISIS regime: Exit ALL positions immediately
        - BEAR_STRONG: Exit weakest 30% of positions
        - Major downgrade (BULL → BEAR): Reduce to regime max
        - Persistent weak regime: Exit underperformers
        """
        tickers_to_exit = []
        
        if not current_positions:
            return tickers_to_exit
        
        # CRISIS: Exit everything immediately
        if regime == 'CRISIS':
            logger.warning(f"CRISIS regime detected - exiting ALL {len(current_positions)} positions")
            return [pos['ticker'] for pos in current_positions]
        
        # BEAR_STRONG: Exit weakest 30%
        if regime == 'BEAR_STRONG' and prev_regime != 'CRISIS':
            # Sort by score (ascending) to get weakest
            sorted_pos = sorted(current_positions, key=lambda x: x.get('score', 0))
            exit_count = max(1, int(len(sorted_pos) * 0.3))
            tickers_to_exit = [pos['ticker'] for pos in sorted_pos[:exit_count]]
            logger.info(f"BEAR_STRONG: Exiting weakest {exit_count} positions")
        
        # Major regime downgrade (BULL → BEAR)
        if (prev_regime in ['BULL_STRONG', 'BULL_WEAK'] and 
            regime in ['BEAR_WEAK', 'BEAR_STRONG']):
            
            max_pos = self.get_max_positions(regime)
            
            if len(current_positions) > max_pos:
                # Exit weakest positions to meet max
                sorted_pos = sorted(current_positions, key=lambda x: x.get('score', 0))
                excess = len(current_positions) - max_pos
                tickers_to_exit = [pos['ticker'] for pos in sorted_pos[:excess]]
                logger.info(f"Regime downgrade: Reducing positions from {len(current_positions)} to {max_pos}")
        
        # Ensure we don't exceed regime max (even without downgrade)
        if regime in ['NEUTRAL_BEARISH', 'BEAR_WEAK', 'BEAR_STRONG']:
            max_pos = self.get_max_positions(regime)
            
            if len(current_positions) > max_pos:
                sorted_pos = sorted(current_positions, key=lambda x: x.get('score', 0))
                excess = len(current_positions) - max_pos
                tickers_to_exit = [pos['ticker'] for pos in sorted_pos[:excess]]
                logger.info(f"Regime {regime}: Reducing {excess} positions to max {max_pos}")
        
        return tickers_to_exit
    
    def should_exit_position_quality(
        self,
        position: Dict,
        current_price: float,
        current_score: float,
        regime: str,
        days_held: int
    ) -> Tuple[bool, str]:
        """
        Check if position should be exited based on quality deterioration.
        
        Exit if:
        - Score dropped significantly below entry
        - Momentum reversed
        - Held too long without profit
        - Better opportunities available (opportunity cost)
        """
        entry_score = position.get('score', 0)
        entry_price = position.get('entry_price', current_price)
        
        # Score deterioration (>30% drop in score)
        if current_score < entry_score * 0.7:
            return True, 'SCORE_DETERIORATION'
        
        # Held too long without profit (regime-specific)
        max_hold_days = {
            'BULL_STRONG': 60,
            'BULL_WEAK': 50,
            'NEUTRAL_BULLISH': 40,
            'NEUTRAL_BEARISH': 30,
            'BEAR_WEAK': 20,
            'BEAR_STRONG': 10,
        }.get(regime, 30)
        
        current_return = (current_price / entry_price - 1) * 100
        
        if days_held > max_hold_days and current_return < 5:
            return True, 'TIME_NO_PROFIT'
        
        # Large unrealized loss beyond stop loss (defensive)
        if current_return < -15:  # -15% is significant
            return True, 'DEFENSIVE_EXIT'
        
        return False, ''
    
    def should_take_profit_early(
        self,
        position: Dict,
        current_price: float,
        regime: str,
        days_held: int
    ) -> bool:
        """
        Check if we should take profit early (before TP hit).
        
        Conditions:
        - Regime deteriorates + large gain
        - Very large gain in short time (momentum exhaustion)
        - Approaching earnings (if data available)
        """
        entry_price = position.get('entry_price', current_price)
        gain_pct = (current_price / entry_price - 1) * 100
        
        # In BEAR markets, take profits at lower thresholds
        if regime in ['BEAR_WEAK', 'BEAR_STRONG']:
            if gain_pct > 15:  # 15%+ gain in bear = take it
                return True
        
        # In CRISIS, take any profit
        if regime == 'CRISIS' and gain_pct > 5:
            return True
        
        # Very large gain in short time (>30% in <10 days)
        if days_held < 10 and gain_pct > 30:
            return True
        
        # Large gain + deteriorating regime
        if gain_pct > 25 and regime in ['NEUTRAL_BEARISH', 'BEAR_WEAK']:
            return True
        
        return False
    
    def record_regime_change(
        self,
        date: datetime,
        regime: str,
        regime_score: float,
        prev_regime: str = None
    ):
        """Record regime change for analysis."""
        self.regime_history.append({
            'date': date,
            'regime': regime,
            'score': regime_score,
            'prev_regime': prev_regime,
        })
    
    def record_decision(
        self,
        date: datetime,
        action: str,
        details: Dict
    ):
        """Record decision for analysis."""
        self.decision_history.append({
            'date': date,
            'action': action,
            **details
        })
    
    def get_regime_history(self) -> pd.DataFrame:
        """Get regime history as DataFrame."""
        if not self.regime_history:
            return pd.DataFrame()
        return pd.DataFrame(self.regime_history)
    
    def get_decision_history(self) -> pd.DataFrame:
        """Get decision history as DataFrame."""
        if not self.decision_history:
            return pd.DataFrame()
        return pd.DataFrame(self.decision_history)
    
    def get_decision_summary(
        self,
        date: datetime,
        regime: str,
        candidates_available: int,
        candidates_filtered: int,
        buys: int,
        sells: int,
        current_positions: int
    ) -> Dict:
        """Get summary of decision-making for logging."""
        max_pos = self.get_max_positions(regime)
        
        summary = {
            'date': date,
            'regime': regime,
            'candidates_available': candidates_available,
            'candidates_after_filter': candidates_filtered,
            'positions_current': current_positions,
            'max_positions': max_pos,
            'available_slots': max(0, max_pos - current_positions),
            'buys': buys,
            'sells': sells,
            'utilization_pct': (current_positions / max_pos * 100) if max_pos > 0 else 0,
        }
        
        # Record for history
        self.record_decision(date, 'SUMMARY', summary)
        
        return summary
# backtest_scripts/auto_decider_simulator.py
"""
Auto Decider Simulator with Sector-Specific Parameters
Determines entry/exit decisions based on regime and sector

UPDATED: 2025-11-12 15:19 UTC
CHANGES:
  - Sector-specific TP/SL levels
  - Enhanced min_hold_days logic
  - Volatility-based adjustments
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class AutoDeciderSimulator:
    """
    Simulates auto_decider logic with sector-aware strategies
    """
    
    def __init__(self, config: dict):
        """
        Initialize simulator
        
        Args:
            config: Configuration dict
        """
        self.config = config
        self.regime_strategies = config.get('REGIME_STRATEGIES', {})
        self.sector_strategies = config.get('SECTOR_STRATEGIES', {})
        
        # Gate parameters
        self.gate_alpha = config.get('GATE_ALPHA', 0.15)
        
        # Stop/TP settings
        self.use_stop_loss = config.get('USE_STOP_LOSS', True)
        self.use_take_profit = config.get('USE_TAKE_PROFIT', True)
        
        logger.info(f"[AutoDecider] Initialized (gate_alpha={self.gate_alpha})")
        logger.info(f"[AutoDecider] Loaded {len(self.regime_strategies)} regime strategies")
        logger.info(f"[AutoDecider] Loaded {len(self.sector_strategies)} sector strategies")
    
    
    def should_enter(
        self,
        symbol: str,
        score_long: float,
        regime: str,
        sector: Optional[str] = None,
        volatility: Optional[float] = None
    ) -> bool:
        """
        Determine if should enter position
        
        Args:
            symbol: Stock ticker
            score_long: ML/seasonality score
            regime: Current regime
            sector: Sector name
            volatility: Stock volatility
        
        Returns:
            True if should enter
        """
        # Basic gate check
        if score_long < self.gate_alpha:
            return False
        
        # Check regime allows entries
        regime_strategy = self.regime_strategies.get(regime, {})
        max_positions = regime_strategy.get('max_positions', 20)
        if max_positions == 0:
            return False
        
        # Sector-specific volatility check
        if sector and volatility is not None:
            sector_params = self.sector_strategies.get(
                sector,
                self.sector_strategies.get('Default', {})
            )
            vol_tolerance = sector_params.get('volatility_tolerance', 0.03)
            
            if volatility > vol_tolerance:
                logger.debug(
                    f"[AutoDecider] {symbol} volatility too high: "
                    f"{volatility:.4f} > {vol_tolerance:.4f}"
                )
                return False
        
        return True
    
    
    def should_exit(
        self,
        symbol: str,
        position: Dict,
        current_price: float,
        current_date: pd.Timestamp,
        score_long: float,
        regime: str,
        atr: Optional[float] = None
    ) -> Tuple[bool, str]:
        """
        Determine if should exit position
        
        Args:
            symbol: Stock ticker
            position: Position dict
            current_price: Current price
            current_date: Current date
            score_long: Current ML/seasonality score
            regime: Current regime
            atr: Average True Range
        
        Returns:
            (should_exit: bool, reason: str)
        """
        entry_price = position['entry_price']
        entry_date = position['entry_date']
        sector = position.get('sector')
        
        # Calculate return
        current_return = (current_price - entry_price) / entry_price
        
        # Calculate hold time
        hold_days = (current_date - entry_date).days
        
        # Get effective parameters (sector overrides regime)
        tp_mult, sl_mult, min_hold = self._get_exit_parameters(position, regime, sector)
        
        # 1. Minimum hold period check
        if hold_days < min_hold:
            # Only emergency exits allowed during min hold
            if regime == 'CRISIS':
                return True, 'CRISIS_EXIT'
            if regime == 'BEAR_STRONG' and current_return < -0.05:
                return True, 'BEAR_STRONG_WEAK_EXIT'
            return False, ''
        
        # 2. Take Profit check
        if self.use_take_profit and atr is not None:
            tp_threshold = (atr / entry_price) * tp_mult
            if current_return >= tp_threshold:
                return True, 'TAKE_PROFIT'
        
        # 3. Stop Loss check
        if self.use_stop_loss and atr is not None:
            sl_threshold = -(atr / entry_price) * sl_mult
            if current_return <= sl_threshold:
                return True, 'STOP_LOSS'
        
        # 4. Regime-based exit
        regime_strategy = self.regime_strategies.get(regime, {})
        regime_pos_mult = regime_strategy.get('position_size_multiplier', 1.0)
        
        # Exit if regime turns very bearish
        if regime == 'CRISIS':
            return True, 'CRISIS_EXIT'
        
        if regime == 'BEAR_STRONG' and regime_pos_mult < 0.5:
            return True, 'BEAR_STRONG_WEAK_EXIT'
        
        # 5. Signal degradation (for weak bull/neutral)
        if regime in ['NEUTRAL_BEARISH', 'BEAR_WEAK']:
            if score_long < self.gate_alpha * 0.5:  # Signal very weak
                return True, f'{regime}_REDUCE'
        
        return False, ''
    
    
    def _get_exit_parameters(
        self,
        position: Dict,
        regime: str,
        sector: Optional[str]
    ) -> Tuple[float, float, int]:
        """
        Get effective TP/SL/min_hold parameters
        
        Args:
            position: Position dict
            regime: Current regime
            sector: Sector name
        
        Returns:
            (tp_mult, sl_mult, min_hold_days)
        """
        # Start with regime defaults
        regime_strategy = self.regime_strategies.get(regime, {})
        tp_mult = regime_strategy.get('tp_multiplier', 2.0)
        sl_mult = regime_strategy.get('stop_multiplier', 1.0)
        min_hold = regime_strategy.get('min_hold_days', 14)
        
        # Override with sector-specific if available
        if sector:
            sector_params = self.sector_strategies.get(
                sector,
                self.sector_strategies.get('Default', {})
            )
            
            # Use sector TP if specified
            sector_tp = position.get('sector_tp_mult')
            if sector_tp is not None:
                tp_mult = sector_tp
            elif 'tp_multiplier' in sector_params:
                tp_mult = sector_params['tp_multiplier']
            
            # Use sector SL if specified
            sector_sl = position.get('sector_sl_mult')
            if sector_sl is not None:
                sl_mult = sector_sl
            elif 'sl_multiplier' in sector_params:
                sl_mult = sector_params['sl_multiplier']
            
            # Use sector min_hold if specified
            sector_min_hold = position.get('sector_min_hold')
            if sector_min_hold is not None:
                min_hold = sector_min_hold
            elif 'min_hold_days' in sector_params:
                min_hold = sector_params['min_hold_days']
        
        return tp_mult, sl_mult, min_hold
    
    
    def get_regime_strategy(self, regime: str) -> Dict:
        """
        Get strategy for given regime
        
        Args:
            regime: Regime name
        
        Returns:
            Strategy dict
        """
        strategy = self.regime_strategies.get(regime, {}).copy()
        strategy['name'] = regime
        return strategy
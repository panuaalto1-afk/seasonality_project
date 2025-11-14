# backtest_scripts/portfolio.py
"""
Portfolio Management with Dynamic Position Sizing
Handles position tracking, entry/exit, and P/L calculation

UPDATED: 2025-11-12 15:19 UTC
CHANGES:
  - Dynamic position sizing based on portfolio value
  - Sector-specific position limits
  - Adaptive position sizing (drawdown/Sharpe/volatility)
  - Enhanced position tracking
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class Portfolio:
    """
    Portfolio manager with dynamic position sizing
    """
    
    def __init__(
        self,
        initial_cash: float,
        config: dict,
        sector_mapping: Optional[Dict[str, str]] = None
    ):
        """
        Initialize portfolio
        
        Args:
            initial_cash: Starting capital
            config: Configuration dictionary
            sector_mapping: Dict mapping ticker -> sector
        """
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.config = config
        self.sector_mapping = sector_mapping or {}
        
        # Portfolio state
        self.positions = {}  # symbol -> position dict
        self.equity_curve = []
        self.trades_history = []
        
        # Performance tracking
        self.total_value = initial_cash
        self.peak_value = initial_cash
        self.current_drawdown = 0.0
        
        # Rolling metrics for adaptive sizing
        self.recent_returns = []  # Last 60 days
        self.rolling_sharpe = 0.0
        self.rolling_volatility = 0.0
        
        # Position sizing config
        self.position_method = config.get('POSITION_SIZE_METHOD', 'percentage')
        self.position_pct = config.get('POSITION_SIZE_PCT', 0.05)
        self.position_fixed = config.get('POSITION_SIZE_FIXED', 5000.0)
        self.min_position = config.get('MIN_POSITION_SIZE', 1000.0)
        self.max_position = config.get('MAX_POSITION_SIZE', 50000.0)
        self.max_position_pct = config.get('MAX_POSITION_PCT', 0.10)
        
        # Adaptive sizing config
        self.adaptive_config = config.get('ADAPTIVE_POSITION_SIZING', {})
        
        # Sector limits
        self.sector_max = config.get('SECTOR_MAX_POSITIONS', {})
        self.sector_strategies = config.get('SECTOR_STRATEGIES', {})
        
        # Slippage
        self.slippage = config.get('SLIPPAGE_PCT', 0.001)
        
        logger.info(f"[Portfolio] Initialized with ${initial_cash:,.2f}")
        if self.position_method == 'percentage':
            logger.info(f"[Portfolio] Using DYNAMIC position sizing: {self.position_pct*100:.1f}% per position")
        else:
            logger.info(f"[Portfolio] Using FIXED position sizing: ${self.position_fixed:,.2f} per position")
        
        if sector_mapping:
            logger.info(f"[Portfolio] Loaded sector mapping for {len(sector_mapping)} tickers")
    
    
    def calculate_position_size(
        self,
        symbol: str,
        regime_multiplier: float = 1.0,
        sector: Optional[str] = None
    ) -> float:
        """
        Calculate dynamic position size
        
        Args:
            symbol: Stock ticker
            regime_multiplier: Multiplier from regime strategy
            sector: Sector name (optional)
        
        Returns:
            Position size in dollars
        """
        # Base position size
        if self.position_method == 'percentage':
            base_size = self.total_value * self.position_pct
        else:
            base_size = self.position_fixed
        
        # Apply regime multiplier
        position_size = base_size * regime_multiplier
        
        # Apply sector-specific boost
        if sector and sector in self.sector_strategies:
            sector_boost = self.sector_strategies[sector].get('position_size_boost', 1.0)
            position_size *= sector_boost
        
        # Apply adaptive adjustments
        if self.adaptive_config.get('enabled', False):
            position_size = self._apply_adaptive_sizing(position_size)
        
        # Enforce limits
        position_size = max(self.min_position, position_size)
        position_size = min(self.max_position, position_size)
        position_size = min(self.total_value * self.max_position_pct, position_size)
        
        return position_size
    
    
    def _apply_adaptive_sizing(self, position_size: float) -> float:
        """
        Apply adaptive position sizing adjustments
        
        Args:
            position_size: Base position size
        
        Returns:
            Adjusted position size
        """
        multiplier = 1.0
        
        # 1. Drawdown reduction
        dd_config = self.adaptive_config.get('drawdown_reduction', {})
        if dd_config.get('enabled', False):
            for dd_threshold, dd_mult in sorted(dd_config.get('thresholds', {}).items()):
                if abs(self.current_drawdown) >= dd_threshold:
                    multiplier = min(multiplier, dd_mult)
        
        # 2. Sharpe boost
        sharpe_config = self.adaptive_config.get('sharpe_boost', {})
        if sharpe_config.get('enabled', False) and self.rolling_sharpe > 0:
            for sharpe_threshold, sharpe_mult in sorted(sharpe_config.get('thresholds', {}).items(), reverse=True):
                if self.rolling_sharpe >= sharpe_threshold:
                    multiplier = max(multiplier, sharpe_mult)
                    break
        
        # 3. Volatility reduction
        vol_config = self.adaptive_config.get('volatility_reduction', {})
        if vol_config.get('enabled', False):
            vol_threshold = vol_config.get('threshold', 0.04)
            if self.rolling_volatility > vol_threshold:
                multiplier *= vol_config.get('multiplier', 0.8)
        
        return position_size * multiplier
    
    
    def can_open_position(self, sector: Optional[str] = None) -> bool:
        """
        Check if we can open a new position
        
        Args:
            sector: Sector name (for sector limits)
        
        Returns:
            True if position can be opened
        """
        # Check total position limit
        max_pos = self.config.get('MAX_POSITIONS', 20)
        if len(self.positions) >= max_pos:
            return False
        
        # Check sector-specific limit
        if sector and self.config.get('ENABLE_SECTOR_DIVERSIFICATION', False):
            sector_max = self.sector_max.get(sector, self.sector_max.get('Default', 4))
            current_sector_count = sum(
                1 for pos in self.positions.values() 
                if pos.get('sector') == sector
            )
            if current_sector_count >= sector_max:
                return False
        
        return True
    
    
    def open_position(
        self,
        symbol: str,
        date: pd.Timestamp,
        price: float,
        regime_strategy: dict,
        sector: Optional[str] = None,
        entry_reason: str = "NEW_CANDIDATE"
    ) -> bool:
        """
        Open a new position with dynamic sizing
        
        Args:
            symbol: Stock ticker
            date: Entry date
            price: Entry price
            regime_strategy: Current regime strategy dict
            sector: Sector name
            entry_reason: Reason for entry
        
        Returns:
            True if position opened successfully
        """
        if not self.can_open_position(sector):
            return False
        
        if symbol in self.positions:
            logger.warning(f"Position for {symbol} already exists")
            return False
        
        # Calculate position size
        regime_mult = regime_strategy.get('position_size_multiplier', 1.0)
        target_value = self.calculate_position_size(symbol, regime_mult, sector)
        
        # Apply slippage to entry price
        entry_price = price * (1 + self.slippage)
        
        # Calculate shares
        shares = int(target_value / entry_price)
        
        if shares == 0:
            logger.warning(f"Cannot open position for {symbol}: insufficient capital")
            return False
        
        # Actual cost
        actual_cost = shares * entry_price
        
        if actual_cost > self.cash:
            logger.warning(f"Cannot open position for {symbol}: insufficient cash")
            return False
        
        # Deduct cash
        self.cash -= actual_cost
        
        # Get sector-specific parameters
        sector_params = self.sector_strategies.get(sector, self.sector_strategies.get('Default', {}))
        
        # Store position
        self.positions[symbol] = {
            'symbol': symbol,
            'sector': sector,
            'entry_date': date,
            'entry_price': entry_price,
            'shares': shares,
            'cost_basis': actual_cost,
            'entry_reason': entry_reason,
            'regime': regime_strategy.get('name', 'UNKNOWN'),
            'stop_multiplier': regime_strategy.get('stop_multiplier', 1.0),
            'tp_multiplier': regime_strategy.get('tp_multiplier', 2.0),
            'min_hold_days': regime_strategy.get('min_hold_days', 14),
            # Sector-specific overrides
            'sector_tp_mult': sector_params.get('tp_multiplier'),
            'sector_sl_mult': sector_params.get('sl_multiplier'),
            'sector_min_hold': sector_params.get('min_hold_days'),
        }
        
        logger.debug(
            f"[Portfolio] OPEN {symbol} @ ${entry_price:.2f} "
            f"x {shares} shares = ${actual_cost:.2f} "
            f"({sector or 'N/A'}) [{entry_reason}]"
        )
        
        return True
    
    
    def close_position(
        self,
        symbol: str,
        date: pd.Timestamp,
        price: float,
        reason: str = "SIGNAL_EXIT"
    ) -> Optional[Dict]:
        """
        Close an existing position
        
        Args:
            symbol: Stock ticker
            date: Exit date
            price: Exit price
            reason: Exit reason
        
        Returns:
            Trade result dict or None
        """
        if symbol not in self.positions:
            logger.warning(f"Cannot close {symbol}: position not found")
            return None
        
        pos = self.positions[symbol]
        
        # Apply slippage to exit price
        exit_price = price * (1 - self.slippage)
        
        # Calculate P/L
        proceeds = pos['shares'] * exit_price
        cost = pos['cost_basis']
        pl = proceeds - cost
        pl_pct = (pl / cost) * 100 if cost > 0 else 0
        
        # Add to cash
        self.cash += proceeds
        
        # Calculate hold time
        hold_days = (date - pos['entry_date']).days
        
        # Record trade
        trade = {
            'ticker': symbol,
            'sector': pos['sector'],
            'entry_date': pos['entry_date'],
            'exit_date': date,
            'entry_price': pos['entry_price'],
            'exit_price': exit_price,
            'shares': pos['shares'],
            'pl': pl,
            'pl_pct': pl_pct,
            'hold_days': hold_days,
            'reason': reason,
            'entry_reason': pos['entry_reason'],
            'action': 'SELL',
            'date': date,
        }
        
        self.trades_history.append(trade)
        
        # Remove position
        del self.positions[symbol]
        
        logger.debug(
            f"[Portfolio] CLOSE {symbol} @ ${exit_price:.2f} "
            f"P/L: ${pl:.2f} ({pl_pct:+.2f}%) "
            f"[{reason}]"
        )
        
        return trade
    
    
    def update_market_value(self, date: pd.Timestamp, prices: Dict[str, float]):
        """
        Update portfolio value with current prices
        
        Args:
            date: Current date
            prices: Dict of symbol -> price
        """
        # Calculate position values
        position_value = 0.0
        for symbol, pos in self.positions.items():
            if symbol in prices:
                position_value += pos['shares'] * prices[symbol]
        
        # Total value
        self.total_value = self.cash + position_value
        
        # Update drawdown
        if self.total_value > self.peak_value:
            self.peak_value = self.total_value
            self.current_drawdown = 0.0
        else:
            self.current_drawdown = (self.total_value - self.peak_value) / self.peak_value
        
        # Update rolling metrics
        self._update_rolling_metrics()
        
        # Record equity curve
        self.equity_curve.append({
            'date': date,
            'cash': self.cash,
            'position_value': position_value,
            'total_value': self.total_value,
            'num_positions': len(self.positions),
            'drawdown': self.current_drawdown,
        })
    
    
    def _update_rolling_metrics(self):
        """
        Update rolling performance metrics for adaptive sizing
        """
        if len(self.equity_curve) < 2:
            return
        
        # Calculate daily return
        prev_value = self.equity_curve[-2]['total_value']
        curr_value = self.total_value
        daily_return = (curr_value - prev_value) / prev_value if prev_value > 0 else 0
        
        # Store recent returns (60-day window)
        self.recent_returns.append(daily_return)
        if len(self.recent_returns) > 60:
            self.recent_returns.pop(0)
        
        # Calculate rolling Sharpe (if enough data)
        if len(self.recent_returns) >= 30:
            returns = np.array(self.recent_returns)
            self.rolling_volatility = np.std(returns) * np.sqrt(252)
            
            if self.rolling_volatility > 0:
                mean_return = np.mean(returns) * 252
                self.rolling_sharpe = mean_return / self.rolling_volatility
            else:
                self.rolling_sharpe = 0.0
    
    
    def get_equity_curve(self) -> pd.DataFrame:
        """Get equity curve as DataFrame"""
        return pd.DataFrame(self.equity_curve)
    
    
    def get_trades_history(self) -> pd.DataFrame:
        """Get trades history as DataFrame"""
        return pd.DataFrame(self.trades_history)
    
    
    def get_current_positions(self) -> Dict:
        """Get current positions"""
        return self.positions.copy()
    
    
    def get_summary(self) -> Dict:
        """Get portfolio summary"""
        return {
            'total_value': self.total_value,
            'cash': self.cash,
            'num_positions': len(self.positions),
            'current_drawdown': self.current_drawdown,
            'rolling_sharpe': self.rolling_sharpe,
            'rolling_volatility': self.rolling_volatility,
        }
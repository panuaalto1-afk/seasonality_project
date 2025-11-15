"""
Portfolio Management - Aggressive 6% Dynamic Position Sizing (FULL VERSION)
Handles position tracking, risk management, and execution
Matches live system portfolio logic with enhanced features
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging

from .config import (
    POSITION_SIZE_METHOD, POSITION_SIZE_PCT, POSITION_SIZE_FIXED,
    MIN_POSITION_SIZE, MAX_POSITION_SIZE, MAX_POSITION_PCT,
    USE_STOP_LOSS, USE_TAKE_PROFIT,
    REGIME_STOP_LOSS_MULTIPLIER, REGIME_TAKE_PROFIT_MULTIPLIER,
    DEFAULT_STOP_MULTIPLIER, DEFAULT_TP_MULTIPLIER,
    SLIPPAGE_PCT, COMMISSION_PCT, TOTAL_COST_PER_SIDE,
    REGIME_MIN_HOLD_DAYS, DEFAULT_MIN_HOLD_DAYS,
)

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """
    Represents an open position.
    
    Tracks:
    - Entry details (date, price, shares)
    - Risk management (SL, TP)
    - Performance tracking (highest/lowest, unrealized P&L)
    - Context (regime, sector, score)
    """
    ticker: str
    entry_date: datetime
    entry_price: float
    shares: int
    stop_loss: float
    take_profit: float
    regime: str
    sector: str
    atr: float
    score: float
    days_held: int = 0
    highest_price: float = 0.0
    lowest_price: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    max_unrealized_gain: float = 0.0
    max_unrealized_loss: float = 0.0
    
    def __post_init__(self):
        """Initialize tracking fields."""
        self.highest_price = self.entry_price
        self.lowest_price = self.entry_price
    
    def update_tracking(self, current_price: float):
        """Update position tracking with current price."""
        self.highest_price = max(self.highest_price, current_price)
        self.lowest_price = min(self.lowest_price, current_price)
        
        # Unrealized P&L
        self.unrealized_pnl = (current_price - self.entry_price) * self.shares
        self.unrealized_pnl_pct = (current_price / self.entry_price - 1) * 100
        
        # Track max gains/losses
        gain = (self.highest_price / self.entry_price - 1) * 100
        loss = (self.lowest_price / self.entry_price - 1) * 100
        
        self.max_unrealized_gain = max(self.max_unrealized_gain, gain)
        self.max_unrealized_loss = min(self.max_unrealized_loss, loss)
    
    def get_position_value(self, current_price: float) -> float:
        """Get current position value."""
        return self.shares * current_price
    
    def get_cost_basis(self) -> float:
        """Get original cost basis."""
        return self.shares * self.entry_price
    
    def distance_to_stop_loss(self, current_price: float) -> float:
        """Get distance to stop loss in percentage."""
        return (current_price / self.stop_loss - 1) * 100 if self.stop_loss > 0 else 999
    
    def distance_to_take_profit(self, current_price: float) -> float:
        """Get distance to take profit in percentage."""
        return (self.take_profit / current_price - 1) * 100 if current_price > 0 else 999


class Portfolio:
    """
    Portfolio manager with aggressive 6% dynamic position sizing.
    
    Key Features:
    - 6% position sizing (grows with portfolio)
    - Regime-based SL/TP (tight stops, high targets)
    - Intraday SL/TP checks
    - Transaction costs (0.6% per side)
    - Position tracking and analytics
    - Risk metrics calculation
    
    Matches live system with backtest-specific enhancements
    """
    
    def __init__(self, initial_cash: float):
        """Initialize portfolio."""
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: Dict[str, Position] = {}
        self.closed_positions: List[Dict] = []
        self.equity_history: List[Dict] = []
        
        # Performance tracking
        self.peak_value = initial_cash
        self.total_trades = 0
        self.winning_trades = 0
        self.total_commissions = 0.0
        self.total_slippage_cost = 0.0
        
    def get_total_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate total portfolio value (cash + positions)."""
        position_value = sum(
            pos.shares * current_prices.get(pos.ticker, pos.entry_price)
            for pos in self.positions.values()
        )
        return self.cash + position_value
    
    def get_position_value(self, ticker: str, current_price: float) -> float:
        """Get current value of a specific position."""
        if ticker not in self.positions:
            return 0.0
        pos = self.positions[ticker]
        return pos.get_position_value(current_price)
    
    def get_positions_value(self, current_prices: Dict[str, float]) -> float:
        """Get total value of all positions."""
        return sum(
            pos.shares * current_prices.get(pos.ticker, pos.entry_price)
            for pos in self.positions.values()
        )
    
    def get_exposure_pct(self, current_prices: Dict[str, float]) -> float:
        """Get portfolio exposure as percentage."""
        total_value = self.get_total_value(current_prices)
        if total_value == 0:
            return 0
        positions_value = self.get_positions_value(current_prices)
        return (positions_value / total_value) * 100
    
    def calculate_position_size(
        self,
        current_portfolio_value: float,
        regime: str,
        sector: str,
        score: float
    ) -> float:
        """
        Calculate position size - AGGRESSIVE 6% DYNAMIC.
        
        Formula: 6% of current portfolio value
        Caps: $1k minimum, $50k maximum, 10% of portfolio maximum
        
        This ensures:
        - Position size grows with portfolio (compound effect)
        - Risk per trade stays constant (6%)
        - Prevents over-concentration (10% max)
        """
        if POSITION_SIZE_METHOD == 'fixed':
            base_size = POSITION_SIZE_FIXED
        else:  # 'percentage'
            base_size = current_portfolio_value * POSITION_SIZE_PCT
        
        # Apply minimum
        base_size = max(base_size, MIN_POSITION_SIZE)
        
        # Apply maximum (both absolute and percentage)
        max_size_abs = MAX_POSITION_SIZE
        max_size_pct = current_portfolio_value * MAX_POSITION_PCT
        base_size = min(base_size, max_size_abs, max_size_pct)
        
        return base_size
    
    def calculate_stop_loss(
        self,
        entry_price: float,
        atr: float,
        regime: str
    ) -> float:
        """
        Calculate stop loss - AGGRESSIVE (TIGHT).
        
        Uses regime-specific multipliers:
        - BULL_STRONG: 0.5x ATR (very tight, let winners run)
        - BULL_WEAK: 0.6x ATR
        - NEUTRAL_BULLISH: 0.7x ATR
        - NEUTRAL_BEARISH: 0.8x ATR
        - BEAR_WEAK: 0.9x ATR
        - BEAR_STRONG: 1.0x ATR (wider in bear markets)
        
        Safety: Never allow SL below 30% loss
        """
        multiplier = REGIME_STOP_LOSS_MULTIPLIER.get(
            regime,
            DEFAULT_STOP_MULTIPLIER
        )
        
        stop_distance = atr * multiplier
        stop_loss = entry_price - stop_distance
        
        # Safety floor: never below 30% loss
        min_stop = entry_price * 0.70
        stop_loss = max(stop_loss, min_stop)
        
        # Ensure stop is below entry
        if stop_loss >= entry_price:
            stop_loss = entry_price * 0.95  # 5% stop as fallback
        
        return stop_loss
    
    def calculate_take_profit(
        self,
        entry_price: float,
        atr: float,
        regime: str
    ) -> float:
        """
        Calculate take profit - AGGRESSIVE (HIGH).
        
        Uses regime-specific multipliers:
        - BULL_STRONG: 4.0x ATR (very high, capture big moves)
        - BULL_WEAK: 3.5x ATR
        - NEUTRAL_BULLISH: 3.0x ATR
        - NEUTRAL_BEARISH: 2.5x ATR
        - BEAR_WEAK: 2.0x ATR
        - BEAR_STRONG: 1.5x ATR (take profits faster in bear)
        
        Minimum: 10% profit target
        """
        multiplier = REGIME_TAKE_PROFIT_MULTIPLIER.get(
            regime,
            DEFAULT_TP_MULTIPLIER
        )
        
        tp_distance = atr * multiplier
        take_profit = entry_price + tp_distance
        
        # Ensure minimum 10% profit target
        min_tp = entry_price * 1.10
        take_profit = max(take_profit, min_tp)
        
        return take_profit
    
    def can_open_position(
        self,
        ticker: str,
        max_positions: int,
        sector_positions: Dict[str, int],
        sector: str,
        sector_max: int
    ) -> bool:
        """Check if we can open a new position."""
        # Already have position in this ticker
        if ticker in self.positions:
            return False
        
        # At max total positions
        if len(self.positions) >= max_positions:
            return False
        
        # At max positions for this sector
        current_sector_count = sector_positions.get(sector, 0)
        if current_sector_count >= sector_max:
            return False
        
        return True
    
    def open_position(
        self,
        ticker: str,
        date: datetime,
        price: float,
        regime: str,
        sector: str,
        atr: float,
        score: float,
        current_prices: Dict[str, float]
    ) -> Optional[Dict]:
        """
        Open a new position with 6% dynamic sizing.
        
        Process:
        1. Calculate current portfolio value
        2. Calculate position size (6% of portfolio)
        3. Calculate shares to buy
        4. Apply slippage and commission
        5. Calculate SL/TP levels
        6. Create position object
        7. Deduct cash
        8. Return trade record
        
        Returns trade record if successful, None otherwise.
        """
        # Calculate current portfolio value
        portfolio_value = self.get_total_value(current_prices)
        
        # Calculate position size
        position_size = self.calculate_position_size(
            portfolio_value, regime, sector, score
        )
        
        # Calculate shares
        shares = int(position_size / price)
        if shares <= 0:
            logger.debug(f"Cannot open {ticker}: shares=0 (size=${position_size:.2f}, price=${price:.2f})")
            return None
        
        # Calculate actual cost with slippage
        actual_entry_price = price * (1 + SLIPPAGE_PCT)
        cost = shares * actual_entry_price
        commission = cost * COMMISSION_PCT
        total_cost = cost + commission
        
        # Check if we have enough cash
        if total_cost > self.cash:
            # Try with reduced shares to fit available cash
            max_shares = int((self.cash / (1 + COMMISSION_PCT)) / actual_entry_price)
            if max_shares <= 0:
                logger.debug(f"Cannot open {ticker}: insufficient cash (need ${total_cost:.2f}, have ${self.cash:.2f})")
                return None
            
            shares = max_shares
            cost = shares * actual_entry_price
            commission = cost * COMMISSION_PCT
            total_cost = cost + commission
        
        # Calculate SL/TP
        stop_loss = self.calculate_stop_loss(actual_entry_price, atr, regime)
        take_profit = self.calculate_take_profit(actual_entry_price, atr, regime)
        
        # Validate SL/TP
        if stop_loss >= actual_entry_price or take_profit <= actual_entry_price:
            logger.warning(f"Invalid SL/TP for {ticker}: SL={stop_loss:.2f}, Entry={actual_entry_price:.2f}, TP={take_profit:.2f}")
            return None
        
        # Create position
        position = Position(
            ticker=ticker,
            entry_date=date,
            entry_price=actual_entry_price,
            shares=shares,
            stop_loss=stop_loss,
            take_profit=take_profit,
            regime=regime,
            sector=sector,
            atr=atr,
            score=score,
        )
        
        # Deduct cash
        self.cash -= total_cost
        
        # Add to positions
        self.positions[ticker] = position
        
        # Update tracking
        self.total_commissions += commission
        self.total_slippage_cost += (actual_entry_price - price) * shares
        
        # Calculate position metrics
        risk_dollars = (actual_entry_price - stop_loss) * shares
        risk_pct = (actual_entry_price - stop_loss) / actual_entry_price * 100
        reward_dollars = (take_profit - actual_entry_price) * shares
        reward_pct = (take_profit - actual_entry_price) / actual_entry_price * 100
        risk_reward_ratio = reward_dollars / risk_dollars if risk_dollars > 0 else 0
        
        # Return trade record
        return {
            'date': date,
            'ticker': ticker,
            'action': 'BUY',
            'price': actual_entry_price,
            'shares': shares,
            'value': cost,
            'commission': commission,
            'slippage_cost': (actual_entry_price - price) * shares,
            'total_cost': total_cost,
            'regime': regime,
            'sector': sector,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'atr': atr,
            'score': score,
            'portfolio_value': portfolio_value,
            'cash_before': self.cash + total_cost,
            'cash_after': self.cash,
            'position_size_pct': (cost / portfolio_value * 100) if portfolio_value > 0 else 0,
            'risk_dollars': risk_dollars,
            'risk_pct': risk_pct,
            'reward_dollars': reward_dollars,
            'reward_pct': reward_pct,
            'risk_reward_ratio': risk_reward_ratio,
        }
    
    def check_stops_intraday(
        self,
        ticker: str,
        date: datetime,
        high: float,
        low: float,
        close: float
    ) -> Optional[Dict]:
        """
        Check if stop loss or take profit was hit intraday.
        
        Logic:
        1. Check if low <= stop_loss → STOP_LOSS exit
        2. Check if high >= take_profit → TAKE_PROFIT exit
        3. Update position tracking (highest/lowest prices)
        4. Increment days_held
        
        Note: SL checked before TP (conservative assumption)
        
        Returns trade record if position closed, None otherwise.
        """
        if ticker not in self.positions:
            return None
        
        pos = self.positions[ticker]
        
        # Update tracking
        pos.update_tracking(close)
        pos.days_held += 1
        
        exit_price = None
        exit_reason = None
        
        # Check stop loss first (use intraday low)
        if USE_STOP_LOSS and low <= pos.stop_loss:
            # Conservative: assume exit at stop loss price
            exit_price = pos.stop_loss
            exit_reason = 'STOP_LOSS'
            
            logger.debug(f"  SL triggered: {ticker} low={low:.2f} <= SL={pos.stop_loss:.2f}")
        
        # Check take profit (use intraday high)
        elif USE_TAKE_PROFIT and high >= pos.take_profit:
            # Conservative: assume exit at take profit price
            exit_price = pos.take_profit
            exit_reason = 'TAKE_PROFIT'
            
            logger.debug(f"  TP triggered: {ticker} high={high:.2f} >= TP={pos.take_profit:.2f}")
        
        if exit_price is not None:
            return self.close_position(
                ticker, date, exit_price, exit_reason
            )
        
        return None
    
    def close_position(
        self,
        ticker: str,
        date: datetime,
        price: float,
        reason: str
    ) -> Dict:
        """
        Close a position.
        
        Reasons: 
        - 'STOP_LOSS': Stop loss triggered
        - 'TAKE_PROFIT': Take profit triggered
        - 'TIME_EXIT': Maximum hold time reached
        - 'REGIME_EXIT': Regime change forced exit
        - 'QUALITY_EXIT': Score deterioration
        - 'EOD': End of backtest
        
        Returns complete trade record with P&L and metrics.
        """
        if ticker not in self.positions:
            raise ValueError(f"No position in {ticker}")
        
        pos = self.positions[ticker]
        
        # Calculate exit with slippage (negative for sells)
        actual_exit_price = price * (1 - SLIPPAGE_PCT)
        proceeds = pos.shares * actual_exit_price
        commission = proceeds * COMMISSION_PCT
        net_proceeds = proceeds - commission
        
        # Add to cash
        self.cash += net_proceeds
        
        # Calculate P&L
        cost_basis = pos.get_cost_basis()
        pnl = net_proceeds - cost_basis
        pnl_pct = (actual_exit_price / pos.entry_price - 1) * 100
        
        # Calculate metrics
        holding_period_return = pnl_pct
        r_multiple = pnl / ((pos.entry_price - pos.stop_loss) * pos.shares) if pos.stop_loss > 0 else 0
        
        # Update tracking
        self.total_trades += 1
        if pnl > 0:
            self.winning_trades += 1
        self.total_commissions += commission
        self.total_slippage_cost += abs(actual_exit_price - price) * pos.shares
        
        # Create comprehensive trade record
        trade = {
            'date': date,
            'ticker': ticker,
            'action': 'SELL',
            'price': actual_exit_price,
            'shares': pos.shares,
            'value': proceeds,
            'commission': commission,
            'slippage_cost': abs(actual_exit_price - price) * pos.shares,
            'net_proceeds': net_proceeds,
            
            # Entry details
            'entry_date': pos.entry_date,
            'entry_price': pos.entry_price,
            'cost_basis': cost_basis,
            
            # P&L
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'holding_period_return': holding_period_return,
            'r_multiple': r_multiple,
            
            # Risk metrics
            'days_held': pos.days_held,
            'reason': reason,
            'stop_loss': pos.stop_loss,
            'take_profit': pos.take_profit,
            'highest_price': pos.highest_price,
            'lowest_price': pos.lowest_price,
            'max_unrealized_gain': pos.max_unrealized_gain,
            'max_unrealized_loss': pos.max_unrealized_loss,
            
            # Context
            'regime': pos.regime,
            'sector': pos.sector,
            'entry_score': pos.score,
            'atr': pos.atr,
        }
        
        # Store closed position
        self.closed_positions.append(trade)
        
        # Remove from positions
        del self.positions[ticker]
        
        return trade
    
    def check_minimum_hold(self, ticker: str, regime: str) -> bool:
        """Check if position has been held for minimum days."""
        if ticker not in self.positions:
            return True
        
        pos = self.positions[ticker]
        min_days = REGIME_MIN_HOLD_DAYS.get(regime, DEFAULT_MIN_HOLD_DAYS)
        
        return pos.days_held >= min_days
    
    def get_sector_positions(self) -> Dict[str, int]:
        """Get count of positions per sector."""
        sector_counts = {}
        for pos in self.positions.values():
            sector = pos.sector or 'Unknown'
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
        return sector_counts
    
    def get_sector_exposure(self, current_prices: Dict[str, float]) -> Dict[str, float]:
        """Get dollar exposure per sector."""
        sector_exposure = {}
        for pos in self.positions.values():
            sector = pos.sector or 'Unknown'
            value = pos.get_position_value(current_prices.get(pos.ticker, pos.entry_price))
            sector_exposure[sector] = sector_exposure.get(sector, 0) + value
        return sector_exposure
    
    def record_daily_value(
        self,
        date: datetime,
        current_prices: Dict[str, float],
        regime: str
    ):
        """Record daily portfolio value and metrics."""
        total_value = self.get_total_value(current_prices)
        positions_value = self.get_positions_value(current_prices)
        
        # Update peak
        if total_value > self.peak_value:
            self.peak_value = total_value
        
        # Calculate drawdown
        drawdown = (total_value / self.peak_value - 1) * 100 if self.peak_value > 0 else 0
        
        # Calculate exposure
        exposure_pct = (positions_value / total_value * 100) if total_value > 0 else 0
        
        # Update position tracking
        for pos in self.positions.values():
            current_price = current_prices.get(pos.ticker, pos.entry_price)
            pos.update_tracking(current_price)
        
        self.equity_history.append({
            'date': date,
            'total_value': total_value,
            'cash': self.cash,
            'positions_value': positions_value,
            'num_positions': len(self.positions),
            'regime': regime,
            'peak_value': self.peak_value,
            'drawdown': drawdown,
            'exposure_pct': exposure_pct,
        })
    
    def get_equity_curve(self) -> pd.DataFrame:
        """Get equity curve as DataFrame."""
        return pd.DataFrame(self.equity_history)
    
    def get_trades_history(self) -> pd.DataFrame:
        """Get all closed trades as DataFrame."""
        if not self.closed_positions:
            return pd.DataFrame()
        return pd.DataFrame(self.closed_positions)
    
    def get_current_positions(self, current_prices: Dict[str, float] = None) -> pd.DataFrame:
        """Get current open positions as DataFrame."""
        if not self.positions:
            return pd.DataFrame()
        
        if current_prices is None:
            current_prices = {}
        
        positions_list = []
        for ticker, pos in self.positions.items():
            current_price = current_prices.get(ticker, pos.entry_price)
            pos.update_tracking(current_price)
            
            positions_list.append({
                'ticker': ticker,
                'entry_date': pos.entry_date,
                'entry_price': pos.entry_price,
                'current_price': current_price,
                'shares': pos.shares,
                'cost_basis': pos.get_cost_basis(),
                'current_value': pos.get_position_value(current_price),
                'unrealized_pnl': pos.unrealized_pnl,
                'unrealized_pnl_pct': pos.unrealized_pnl_pct,
                'stop_loss': pos.stop_loss,
                'take_profit': pos.take_profit,
                'distance_to_sl': pos.distance_to_stop_loss(current_price),
                'distance_to_tp': pos.distance_to_take_profit(current_price),
                'days_held': pos.days_held,
                'regime': pos.regime,
                'sector': pos.sector,
                'score': pos.score,
            })
        
        return pd.DataFrame(positions_list)
    
    def get_summary_stats(self) -> Dict:
        """Get portfolio summary statistics."""
        trades_df = self.get_trades_history()
        
        if trades_df.empty:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'avg_win_pct': 0.0,
                'avg_loss_pct': 0.0,
                'avg_win_dollars': 0.0,
                'avg_loss_dollars': 0.0,
                'profit_factor': 0.0,
                'avg_hold_days': 0.0,
                'total_commissions': self.total_commissions,
                'total_slippage': self.total_slippage_cost,
            }
        
        sell_trades = trades_df[trades_df['action'] == 'SELL']
        wins = sell_trades[sell_trades['pnl'] > 0]
        losses = sell_trades[sell_trades['pnl'] < 0]
        
        win_rate = len(wins) / len(sell_trades) * 100 if len(sell_trades) > 0 else 0
        avg_win_pct = wins['pnl_pct'].mean() if len(wins) > 0 else 0
        avg_loss_pct = losses['pnl_pct'].mean() if len(losses) > 0 else 0
        avg_win_dollars = wins['pnl'].mean() if len(wins) > 0 else 0
        avg_loss_dollars = losses['pnl'].mean() if len(losses) > 0 else 0
        
        total_wins = wins['pnl'].sum() if len(wins) > 0 else 0
        total_losses = abs(losses['pnl'].sum()) if len(losses) > 0 else 1
        profit_factor = total_wins / total_losses if total_losses > 0 else 0
        
        avg_hold_days = sell_trades['days_held'].mean() if len(sell_trades) > 0 else 0
        
        return {
            'total_trades': len(sell_trades),
            'winning_trades': len(wins),
            'losing_trades': len(losses),
            'win_rate': win_rate,
            'avg_win_pct': avg_win_pct,
            'avg_loss_pct': avg_loss_pct,
            'avg_win_dollars': avg_win_dollars,
            'avg_loss_dollars': avg_loss_dollars,
            'profit_factor': profit_factor,
            'avg_hold_days': avg_hold_days,
            'total_commissions': self.total_commissions,
            'total_slippage': self.total_slippage_cost,
        }
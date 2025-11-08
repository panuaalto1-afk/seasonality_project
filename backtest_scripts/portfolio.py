# backtest_scripts/portfolio.py
"""
Portfolio State Management for Backtesting
Tracks positions, cash, and handles trade execution
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import date

class Portfolio:
    """
    Portfolio state tracker for backtest
    Manages positions, cash, SL/TP triggers
    """
    
    def __init__(self, initial_cash: float):
        """
        Initialize portfolio
        
        Args:
            initial_cash: Starting cash
        """
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions = []  # List of open positions
        self.trade_history = []  # All executed trades
        self.daily_values = []  # Daily portfolio value
        
        print(f"[Portfolio] Initialized with ${initial_cash:,.2f}")
    
    def get_state(self) -> Dict:
        """
        Get current portfolio state
        
        Returns:
            dict: {
                'cash': float,
                'positions': List[dict],
                'total_value': float,
                'positions_value': float
            }
        """
        positions_value = sum(
            pos['shares'] * pos['current_price']
            for pos in self.positions
        )
        
        return {
            'cash': self.cash,
            'positions': self.positions.copy(),
            'total_value': self.cash + positions_value,
            'positions_value': positions_value
        }
    
    def buy(self, 
            ticker: str,
            date: date,
            entry_price: float,
            shares: int,
            stop_loss: float,
            take_profit: float,
            reason: str = 'NEW_CANDIDATE') -> bool:
        """
        Execute BUY order
        
        Args:
            ticker: Stock ticker
            date: Trade date
            entry_price: Entry price (with gap/slippage if configured)
            shares: Number of shares
            stop_loss: Stop loss price
            take_profit: Take profit price
            reason: Reason for trade
        
        Returns:
            True if executed, False if insufficient cash
        """
        cost = shares * entry_price
        
        if cost > self.cash:
            return False
        
        # Deduct cash
        self.cash -= cost
        
        # Add position
        position = {
            'ticker': ticker,
            'entry_date': date,
            'entry_price': entry_price,
            'shares': shares,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'current_price': entry_price,  # Updated daily
            'cost': cost,
            'reason': reason
        }
        
        self.positions.append(position)
        
        # Record trade
        self.trade_history.append({
            'date': date,
            'ticker': ticker,
            'action': 'BUY',
            'price': entry_price,
            'shares': shares,
            'value': cost,
            'reason': reason
        })
        
        return True
    
    def sell(self,
            ticker: str,
            date: date,
            exit_price: float,
            reason: str = 'MANUAL') -> Optional[Dict]:
        """
        Execute SELL order
        
        Args:
            ticker: Stock ticker
            date: Trade date
            exit_price: Exit price
            reason: Reason for exit (SL, TP, REGIME, etc.)
        
        Returns:
            Trade result dict or None if position not found
        """
        # Find position
        position = None
        for i, pos in enumerate(self.positions):
            if pos['ticker'] == ticker:
                position = self.positions.pop(i)
                break
        
        if position is None:
            return None
        
        # Calculate P/L
        proceeds = position['shares'] * exit_price
        cost = position['cost']
        pl = proceeds - cost
        pl_pct = (pl / cost) * 100 if cost > 0 else 0.0
        
        # Add cash
        self.cash += proceeds
        
        # Calculate hold time
        hold_days = (date - position['entry_date']).days
        
        # Record trade
        trade = {
            'date': date,
            'ticker': ticker,
            'action': 'SELL',
            'entry_date': position['entry_date'],
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'shares': position['shares'],
            'cost': cost,
            'proceeds': proceeds,
            'pl': pl,
            'pl_pct': pl_pct,
            'hold_days': hold_days,
            'reason': reason
        }
        
        self.trade_history.append(trade)
        
        return trade
    
    def update_prices(self, date: date, prices: Dict[str, float]):
        """
        Update current prices for all positions
        
        Args:
            date: Current date
            prices: Dict mapping ticker → current price
        """
        for pos in self.positions:
            ticker = pos['ticker']
            if ticker in prices:
                pos['current_price'] = prices[ticker]
    
    def check_exits(self, 
                   date: date,
                   intraday_prices: Dict[str, Dict[str, float]]) -> List[Dict]:
        """
        Check for SL/TP triggers
        
        Args:
            date: Current date
            intraday_prices: Dict mapping ticker → {'high': float, 'low': float}
        
        Returns:
            List of exit trades
        """
        exits = []
        
        positions_to_remove = []
        
        for i, pos in enumerate(self.positions):
            ticker = pos['ticker']
            
            if ticker not in intraday_prices:
                continue
            
            day_high = intraday_prices[ticker].get('high', pos['current_price'])
            day_low = intraday_prices[ticker].get('low', pos['current_price'])
            
            exit_price = None
            reason = None
            
            # Check stop loss (triggered by low)
            if day_low <= pos['stop_loss']:
                exit_price = pos['stop_loss']
                reason = 'STOP_LOSS'
            
            # Check take profit (triggered by high)
            elif day_high >= pos['take_profit']:
                exit_price = pos['take_profit']
                reason = 'TAKE_PROFIT'
            
            if exit_price and reason:
                # Execute sell
                trade = self.sell(ticker, date, exit_price, reason)
                if trade:
                    exits.append(trade)
                
                # Mark for removal (already removed by sell())
                # No need to append to positions_to_remove
        
        return exits
    
    def record_daily_value(self, date: date):
        """
        Record daily portfolio value
        
        Args:
            date: Current date
        """
        state = self.get_state()
        
        self.daily_values.append({
            'date': date,
            'cash': state['cash'],
            'positions_value': state['positions_value'],
            'total_value': state['total_value'],
            'positions_count': len(self.positions)
        })
    
    def get_equity_curve(self) -> pd.DataFrame:
        """
        Get equity curve DataFrame
        
        Returns:
            DataFrame with columns: date, total_value, cash, positions_value, positions_count
        """
        return pd.DataFrame(self.daily_values)
    
    def get_trades_history(self) -> pd.DataFrame:
        """
        Get trades history DataFrame
        
        Returns:
            DataFrame with all executed trades
        """
        return pd.DataFrame(self.trade_history)
    
    def get_performance_summary(self) -> Dict:
        """
        Calculate performance metrics
        
        Returns:
            dict with performance metrics
        """
        if not self.daily_values:
            return {}
        
        equity_curve = self.get_equity_curve()
        
        # Total return
        final_value = equity_curve['total_value'].iloc[-1]
        total_return = ((final_value - self.initial_cash) / self.initial_cash) * 100
        
        # Daily returns
        equity_curve['daily_return'] = equity_curve['total_value'].pct_change()
        
        # Sharpe ratio (annualized, assuming 252 trading days)
        daily_returns = equity_curve['daily_return'].dropna()
        if len(daily_returns) > 0:
            sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0.0
        else:
            sharpe = 0.0
        
        # Max drawdown
        cumulative = (1 + equity_curve['daily_return'].fillna(0)).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() * 100
        
        # Trade statistics
        trades = self.get_trades_history()
        sell_trades = trades[trades['action'] == 'SELL']
        
        if not sell_trades.empty:
            total_trades = len(sell_trades)
            winning_trades = len(sell_trades[sell_trades['pl'] > 0])
            win_rate = (winning_trades / total_trades) * 100
            avg_win = sell_trades[sell_trades['pl'] > 0]['pl_pct'].mean() if winning_trades > 0 else 0.0
            avg_loss = sell_trades[sell_trades['pl'] < 0]['pl_pct'].mean() if (total_trades - winning_trades) > 0 else 0.0
            avg_hold_days = sell_trades['hold_days'].mean()
            
            # Profit factor
            total_wins = sell_trades[sell_trades['pl'] > 0]['pl'].sum()
            total_losses = abs(sell_trades[sell_trades['pl'] < 0]['pl'].sum())
            profit_factor = total_wins / total_losses if total_losses > 0 else 0.0
        else:
            total_trades = 0
            win_rate = 0.0
            avg_win = 0.0
            avg_loss = 0.0
            avg_hold_days = 0.0
            profit_factor = 0.0
        
        return {
            'initial_cash': self.initial_cash,
            'final_value': final_value,
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_hold_days': avg_hold_days,
            'profit_factor': profit_factor,
        }
# backtest_scripts/portfolio.py
"""
Portfolio Management for Backtesting
Tracks positions, executes trades, handles SL/TP

UPDATED: 2025-11-11 19:33 UTC - Added sector tracking and enhanced position management
"""

import pandas as pd
from typing import Dict, List, Optional
from datetime import date as date_type

class Portfolio:
    """
    Portfolio simulator for backtest
    Tracks cash, positions, equity curve with sector information
    """
    
    def __init__(self, initial_cash: float, constituents: Optional[pd.DataFrame] = None):
        """
        Initialize portfolio
        
        Args:
            initial_cash: Starting cash
            constituents: Optional DataFrame with ticker-to-sector mapping
        """
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions = []
        self.trades_history = []
        self.equity_curve = []
        
        # NEW: Sector mapping
        self.constituents = constituents
        self.ticker_to_sector = {}
        
        if constituents is not None:
            # Build ticker → sector mapping
            ticker_col = None
            for col in ['ticker', 'Ticker', 'Symbol']:
                if col in constituents.columns:
                    ticker_col = col
                    break
            
            sector_col = None
            for col in ['Sector', 'GICS Sector', 'sector', 'gics_sector']:
                if col in constituents.columns:
                    sector_col = col
                    break
            
            if ticker_col and sector_col:
                self.ticker_to_sector = dict(zip(
                    constituents[ticker_col].astype(str).str.upper(),
                    constituents[sector_col]
                ))
                print(f"[Portfolio] Loaded sector mapping for {len(self.ticker_to_sector)} tickers")
        
        print(f"[Portfolio] Initialized with ${initial_cash:,.2f}")
    
    def get_sector(self, ticker: str) -> str:
        """Get sector for ticker"""
        return self.ticker_to_sector.get(ticker.upper(), 'Unknown')
    
    def get_sector_exposure(self) -> Dict[str, int]:
        """Get current number of positions per sector"""
        sector_counts = {}
        for pos in self.positions:
            sector = pos.get('sector', 'Unknown')
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
        return sector_counts
    
    def buy(self, ticker: str, date: date_type, entry_price: float, shares: int,
            stop_loss: float, take_profit: float, reason: str = "BUY"):
        """
        Buy a position
        
        Args:
            ticker: Stock ticker
            date: Entry date
            entry_price: Entry price
            shares: Number of shares
            stop_loss: Stop loss level
            take_profit: Take profit level
            reason: Reason for entry
        """
        cost = shares * entry_price
        
        if cost > self.cash:
            return  # Not enough cash
        
        self.cash -= cost
        
        # NEW: Get sector
        sector = self.get_sector(ticker)
        
        self.positions.append({
            'ticker': ticker,
            'entry_date': date,
            'entry_price': entry_price,
            'shares': shares,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'current_price': entry_price,
            'reason': reason,
            'sector': sector,  # NEW
        })
        
        # DEBUG: Log sector info
        # print(f"[Portfolio] Opened {ticker} ({sector}) | {shares} shares @ ${entry_price:.2f}")
    
    def sell(self, ticker: str, date: date_type, exit_price: float, reason: str = "SELL"):
        """
        Sell a position
        
        Args:
            ticker: Stock ticker
            date: Exit date
            exit_price: Exit price
            reason: Reason for exit
        """
        # Find position
        position = None
        for i, pos in enumerate(self.positions):
            if pos['ticker'] == ticker:
                position = self.positions.pop(i)
                break
        
        if position is None:
            return  # Position not found
        
        # Calculate P/L
        proceeds = position['shares'] * exit_price
        cost = position['shares'] * position['entry_price']
        pl = proceeds - cost
        pl_pct = (pl / cost) * 100 if cost > 0 else 0.0
        
        self.cash += proceeds
        
        # Calculate hold time
        hold_days = (date - position['entry_date']).days if isinstance(position['entry_date'], date_type) else 0
        
        # Record trade with sector info
        self.trades_history.append({
            'ticker': ticker,
            'sector': position.get('sector', 'Unknown'),  # NEW
            'entry_date': position['entry_date'],
            'exit_date': date,
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'shares': position['shares'],
            'pl': pl,
            'pl_pct': pl_pct,
            'hold_days': hold_days,
            'reason': reason,
            'entry_reason': position.get('reason', 'UNKNOWN'),
            'action': 'SELL',  # For compatibility with visualizer
            'date': date,  # For compatibility with visualizer
        })
    
    def check_exits(self, current_date: date_type, intraday_prices: Dict[str, Dict[str, float]], 
                    regime: str = 'NEUTRAL_BULLISH', min_hold_days: int = 0) -> List[Dict]:
        """
        Check for SL/TP triggers
        
        Args:
            current_date: Current date
            intraday_prices: Dict mapping ticker → {'high': float, 'low': float}
            regime: Current regime
            min_hold_days: Minimum hold days from regime strategy
        
        Returns:
            List of exits triggered
        """
        exits = []
        
        for pos in self.positions[:]:  # Iterate over copy
            ticker = pos['ticker']
            
            if ticker not in intraday_prices:
                continue
            
            high = intraday_prices[ticker]['high']
            low = intraday_prices[ticker]['low']
            
            # Calculate days held
            entry_date = pos.get('entry_date', current_date)
            days_held = (current_date - entry_date).days if isinstance(entry_date, date_type) else 0
            
            # Check Stop Loss (ALWAYS enforced, regardless of min_hold_days)
            if low <= pos['stop_loss']:
                self.sell(ticker, current_date, pos['stop_loss'], "STOP_LOSS")
                exits.append({
                    'ticker': ticker,
                    'reason': 'STOP_LOSS',
                    'price': pos['stop_loss'],
                    'days_held': days_held
                })
                continue
            
            # Check Take Profit (ONLY if min_hold_days met)
            if high >= pos['take_profit']:
                if days_held >= min_hold_days:
                    # Min hold days met → execute TP
                    self.sell(ticker, current_date, pos['take_profit'], "TAKE_PROFIT")
                    exits.append({
                        'ticker': ticker,
                        'reason': 'TAKE_PROFIT',
                        'price': pos['take_profit'],
                        'days_held': days_held
                    })
                else:
                    # Min hold days NOT met → skip TP, let it run
                    pass
        
        return exits
    
    def update_prices(self, current_date: date_type, eod_prices: Dict[str, float]):
        """
        Update current prices for all positions (end of day)
        
        Args:
            current_date: Current date
            eod_prices: Dict mapping ticker → close price
        """
        for pos in self.positions:
            ticker = pos['ticker']
            if ticker in eod_prices:
                pos['current_price'] = eod_prices[ticker]
    
    def record_daily_value(self, current_date: date_type):
        """
        Record daily portfolio value
        
        Args:
            current_date: Current date
        """
        # Calculate position values
        position_value = sum(
            pos['shares'] * pos['current_price']
            for pos in self.positions
        )
        
        total_value = self.cash + position_value
        
        # NEW: Track sector exposure in daily record
        sector_exposure = self.get_sector_exposure()
        
        self.equity_curve.append({
            'date': current_date,
            'cash': self.cash,
            'position_value': position_value,
            'total_value': total_value,
            'num_positions': len(self.positions),
            'sector_exposure': sector_exposure.copy(),  # NEW
        })
    
    def get_state(self) -> Dict:
        """
        Get current portfolio state
        
        Returns:
            Dict with portfolio state
        """
        return {
            'cash': self.cash,
            'positions': self.positions.copy(),
            'num_positions': len(self.positions),
            'sector_exposure': self.get_sector_exposure(),  # NEW
        }
    
    def get_equity_curve(self) -> pd.DataFrame:
        """
        Get equity curve as DataFrame
        
        Returns:
            DataFrame with equity curve
        """
        return pd.DataFrame(self.equity_curve)
    
    def get_trades_history(self) -> pd.DataFrame:
        """
        Get trades history as DataFrame
        
        Returns:
            DataFrame with all trades
        """
        return pd.DataFrame(self.trades_history)
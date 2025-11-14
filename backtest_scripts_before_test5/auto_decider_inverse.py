# backtest_scripts/auto_decider_inverse.py
"""
Auto Decider with Inverse ETF Support
Extends AutoDeciderSimulator with inverse ETF logic
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import date
from .auto_decider_simulator import AutoDeciderSimulator

class AutoDeciderInverse(AutoDeciderSimulator):
    """
    Enhanced Auto Decider with Inverse ETF support
    Buys SH/PSQ/RWM in bear/crisis regimes
    """
    
    def __init__(self, regime_strategies: Dict, inverse_etf_mapping: Dict, 
                 inverse_etf_params: Dict, stock_prices: Dict):
        """
        Initialize with inverse ETF support
        
        Args:
            regime_strategies: Enhanced regime strategies with inverse flags
            inverse_etf_mapping: Dict mapping index → inverse ETF
            inverse_etf_params: Trading params for inverse ETFs
            stock_prices: Preloaded stock prices (includes inverse ETFs)
        """
        super().__init__(regime_strategies)
        self.inverse_etf_mapping = inverse_etf_mapping
        self.inverse_etf_params = inverse_etf_params
        self.stock_prices = stock_prices
        print(f"[AutoDeciderInverse] Initialized with {len(inverse_etf_mapping)} inverse ETFs")
    
    def decide_trades(self, target_date: date, candidates_df: pd.DataFrame,
                     portfolio_state: Dict, regime: str, cash: float,
                     max_positions_override: Optional[int] = None,
                     position_size_override: Optional[float] = None) -> Dict:
        """
        Make trading decisions including inverse ETFs
        
        Returns:
            dict: {'buy': [], 'sell': [], 'hold': []}
        """
        # Get base decisions (regular stocks)
        base_decisions = super().decide_trades(
            target_date, candidates_df, portfolio_state, regime, cash,
            max_positions_override, position_size_override
        )
        
        # Get strategy for regime
        strategy = self.regime_strategies.get(regime, self.regime_strategies['NEUTRAL_BULLISH'])
        
        # ✅ CHECK IF INVERSE ETFs ALLOWED
        if not strategy.get('allow_inverse_etfs', False):
            return base_decisions
        
        # ✅ GENERATE INVERSE ETF POSITIONS
        inverse_allocation = strategy.get('inverse_etf_allocation', 0.0)
        
        if inverse_allocation > 0:
            inverse_buys = self._generate_inverse_etf_positions(
                target_date=target_date,
                regime=regime,
                cash=cash,
                allocation_pct=inverse_allocation,
                portfolio_state=portfolio_state
            )
            
            # Add inverse buys to base decisions
            base_decisions['buy'].extend(inverse_buys)
        
        return base_decisions
    
    def _generate_inverse_etf_positions(self, target_date: date, regime: str,
                                        cash: float, allocation_pct: float,
                                        portfolio_state: Dict) -> List[Dict]:
        """
        Generate inverse ETF buy decisions
        
        Args:
            target_date: Current date
            regime: Current regime
            cash: Available cash
            allocation_pct: % of cash to allocate to inverse ETFs
            portfolio_state: Current portfolio
        
        Returns:
            List of buy decisions for inverse ETFs
        """
        inverse_buys = []
        
        # Calculate cash allocation for inverse ETFs
        inverse_cash = cash * allocation_pct
        
        # Current inverse positions
        current_positions = portfolio_state.get('positions', [])
        current_inverse_tickers = [
            pos['ticker'] for pos in current_positions
            if pos['ticker'] in self.inverse_etf_mapping.values()
        ]
        
        # Available inverse ETFs (not already held)
        available_inverse = [
            etf for etf in self.inverse_etf_mapping.values()
            if etf not in current_inverse_tickers
        ]
        
        if not available_inverse or inverse_cash < 1000:
            return []
        
        # Distribute cash equally among available inverse ETFs
        position_size = inverse_cash / len(available_inverse)
        
        for inverse_ticker in available_inverse:
            # Get price data
            if inverse_ticker not in self.stock_prices:
                continue
            
            price_df = self.stock_prices[inverse_ticker]
            price_row = price_df[price_df['date'] == target_date]
            
            if price_row.empty:
                continue
            
            entry_price = float(price_row.iloc[0]['close'])
            
            if entry_price <= 0:
                continue
            
            # Calculate shares
            shares = int(position_size / entry_price)
            if shares <= 0:
                continue
            
            actual_cost = shares * entry_price
            
            # Calculate stop loss and take profit
            stop_loss = entry_price * (1 - self.inverse_etf_params['stop_loss_pct'])
            take_profit = entry_price * (1 + self.inverse_etf_params['take_profit_pct'])
            
            inverse_buys.append({
                'ticker': inverse_ticker,
                'entry': entry_price,
                'shares': shares,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'atr': 0.0,  # Not used for inverse ETFs
                'score_long': 1.0,  # Max score (defensive position)
                'reason': f'INVERSE_ETF_{regime}',
                'date': target_date,
                'cost': actual_cost
            })
        
        return inverse_buys
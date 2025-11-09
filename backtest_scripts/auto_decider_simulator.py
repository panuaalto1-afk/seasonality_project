# backtest_scripts/auto_decider_simulator.py
"""
Auto Decider Simulator for Backtesting
Replicates auto_decider.py decision logic
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import date

class AutoDeciderSimulator:
    """
    Simulates auto_decider.py logic for backtesting
    Makes BUY/SELL/HOLD decisions based on:
    - ML candidates (from ml_signal_generator)
    - Current regime (from regime_calculator)
    - Portfolio state
    - Regime strategies
    """
    
    def __init__(self, regime_strategies: Dict):
        """
        Initialize auto decider simulator
        
        Args:
            regime_strategies: Dict mapping regime → strategy params
        """
        self.regime_strategies = regime_strategies
        print(f"[AutoDeciderSimulator] Initialized with {len(regime_strategies)} regime strategies")
    
    def decide_trades(self,
                     target_date: date,
                     candidates_df: pd.DataFrame,
                     portfolio_state: Dict,
                     regime: str,
                     cash: float,
                     max_positions_override: Optional[int] = None,
                     position_size_override: Optional[float] = None) -> Dict:
        """
        Make trading decisions for a specific date
        
        Args:
            target_date: Date of decision
            candidates_df: ML candidates (from ml_signal_generator)
            portfolio_state: Current portfolio state
            regime: Current market regime
            cash: Available cash
            max_positions_override: Override max positions
            position_size_override: Override position size
        
        Returns:
            dict: {'buy': [], 'sell': [], 'hold': []}
        """
        # Check if candidates_df is empty or invalid
        if candidates_df is None or candidates_df.empty:
            current_positions = portfolio_state.get('positions', [])
            return {
                'buy': [],
                'sell': [],
                'hold': current_positions.copy()
            }
        
        # Get regime strategy
        strategy = self.regime_strategies.get(regime, self.regime_strategies['NEUTRAL_BULLISH'])
        
        max_positions = max_positions_override if max_positions_override else strategy['max_positions']
        position_size_mult = strategy['position_size_multiplier']
        base_position_size = position_size_override if position_size_override else 5000.0
        position_size = base_position_size * position_size_mult
        
        current_positions = portfolio_state.get('positions', [])
        current_tickers = [pos['ticker'] for pos in current_positions]
        
        # SELL DECISIONS (Regime-based exits)
        sell_decisions = []
        hold_decisions = []
        
        # CRISIS regime: Exit ALL positions
        if regime == 'CRISIS':
            sell_decisions = [
                {'ticker': pos['ticker'], 'reason': 'CRISIS_EXIT', 'date': target_date}
                for pos in current_positions
            ]
        
        # BEAR_STRONG: Exit weak positions
        elif regime == 'BEAR_STRONG' and len(current_positions) > 0:
            positions_with_pl = []
            for pos in current_positions:
                entry = pos.get('entry_price', 0)
                current_price = pos.get('current_price', entry)
                pl_pct = ((current_price - entry) / entry) * 100 if entry > 0 else 0.0
                positions_with_pl.append({'position': pos, 'pl_pct': pl_pct})
            
            positions_with_pl.sort(key=lambda x: x['pl_pct'])
            exit_count = max(1, int(len(positions_with_pl) * 0.3))
            
            for i in range(exit_count):
                sell_decisions.append({
                    'ticker': positions_with_pl[i]['position']['ticker'],
                    'reason': 'BEAR_STRONG_WEAK_EXIT',
                    'date': target_date
                })
            
            for i in range(exit_count, len(positions_with_pl)):
                hold_decisions.append(positions_with_pl[i]['position'])
        
        # BEAR_WEAK / NEUTRAL_BEARISH: Reduce if over max
        elif regime in ['BEAR_WEAK', 'NEUTRAL_BEARISH']:
            if len(current_positions) > max_positions:
                positions_sorted = sorted(current_positions, key=lambda x: x.get('entry_date', target_date))
                exit_count = len(current_positions) - max_positions
                
                for i in range(exit_count):
                    sell_decisions.append({
                        'ticker': positions_sorted[i]['ticker'],
                        'reason': f'{regime}_REDUCE',
                        'date': target_date
                    })
                
                hold_decisions = positions_sorted[exit_count:]
            else:
                hold_decisions = current_positions.copy()
        else:
            hold_decisions = current_positions.copy()
        
        # BUY DECISIONS
        buy_decisions = []
        
        positions_after_sells = len(current_positions) - len(sell_decisions)
        available_slots = max_positions - positions_after_sells
        
        if available_slots > 0 and cash >= position_size and position_size_mult > 0:
            sell_tickers = [s['ticker'] for s in sell_decisions]
            
            available_candidates = candidates_df[
                ~candidates_df['ticker'].isin(current_tickers) &
                ~candidates_df['ticker'].isin(sell_tickers)
            ].copy()
            
            if not available_candidates.empty:
                top_candidates = available_candidates.nlargest(available_slots, 'score_long')
                
                for _, row in top_candidates.iterrows():
                    if cash < position_size:
                        break
                    
                    entry_price = row['entry_price']
                    if entry_price <= 0:
                        continue
                    
                    shares = int(position_size / entry_price)
                    if shares <= 0:
                        continue
                    
                    actual_cost = shares * entry_price
                    if actual_cost > cash:
                        shares = int(cash / entry_price)
                        actual_cost = shares * entry_price
                        if shares <= 0:
                            continue
                    
                    buy_decisions.append({
                        'ticker': row['ticker'],
                        'entry': entry_price,
                        'shares': shares,
                        'stop_loss': row['stop_loss'],
                        'take_profit': row['take_profit'],
                        'atr': row.get('atr_14', 0.0),
                        'score_long': row['score_long'],
                        'reason': 'NEW_CANDIDATE',
                        'date': target_date,
                        'cost': actual_cost
                    })
                    
                    cash -= actual_cost
        
        return {
            'buy': buy_decisions,
            'sell': sell_decisions,
            'hold': hold_decisions
        }

# backtest_scripts/backtest_engine.py
"""
Backtest Engine - Main Orchestrator
Coordinates all backtest components

UPDATED: 2025-11-11 19:36 UTC - Enhanced 10-year analysis with sector diversification
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import date, timedelta, datetime
from tqdm import tqdm

from .config import *
from .data_loader import BacktestDataLoader
from .regime_calculator import RegimeCalculator
from .seasonality_calculator import SeasonalityCalculator
from .ml_signal_generator import MLSignalGenerator
from .auto_decider_simulator import AutoDeciderSimulator
from .portfolio import Portfolio
from .performance_analyzer import PerformanceAnalyzer
from .visualizer import BacktestVisualizer

class BacktestEngine:
    """
    Main backtest orchestrator
    Runs full backtest simulation with sector diversification
    """
    
    def __init__(self, 
                 config_overrides: Optional[Dict] = None,
                 constituents_path: Optional[str] = None):
        """
        Initialize backtest engine
        
        Args:
            config_overrides: Optional dict to override config.py settings
            constituents_path: Path to constituents.csv with sector data
        """
        self.config = self._load_config(config_overrides)
        
        print("=" * 80)
        print("BACKTEST ENGINE - Enhanced 10-Year Analysis with Sector Diversification")
        print("=" * 80)
        print(f"Period: {self.config['start_date']} to {self.config['end_date']}")
        print(f"Initial Capital: ${self.config['initial_cash']:,.2f}")
        print(f"Max Positions: {self.config['max_positions']}")
        print(f"Position Size: ${self.config['position_size']:,.2f}")
        print("=" * 80)
        print("")
        
        # Load constituents if provided
        self.constituents = None
        if constituents_path:
            try:
                self.constituents = pd.read_csv(constituents_path)
                print(f"[Engine] ✓ Loaded constituents: {len(self.constituents)} tickers")
                
                # Check for sector column
                sector_col = None
                for col in ['Sector', 'GICS Sector', 'sector', 'gics_sector']:
                    if col in self.constituents.columns:
                        sector_col = col
                        break
                
                if sector_col:
                    print(f"[Engine] ✓ Sector data available: {self.constituents[sector_col].nunique()} unique sectors")
                    if ENABLE_SECTOR_DIVERSIFICATION:
                        print(f"[Engine] ✓ Sector diversification enabled (max {MAX_POSITIONS_PER_SECTOR} per sector)")
                else:
                    print(f"[Engine] ⚠ No sector column found in constituents")
            except Exception as e:
                print(f"[Engine] ⚠ Could not load constituents: {e}")
        
        # Initialize components
        self._init_components()
    
    def _load_config(self, overrides: Optional[Dict]) -> Dict:
        """Load configuration"""
        cfg = {
            'start_date': BACKTEST_START,
            'end_date': BACKTEST_END,
            'initial_cash': INITIAL_CASH,
            'max_positions': MAX_POSITIONS,
            'position_size': POSITION_SIZE,
            'gate_alpha': GATE_ALPHA,
            'universe_csv': UNIVERSE_CSV,
            'stock_price_cache': STOCK_PRICE_CACHE,
            'macro_price_cache': MACRO_PRICE_CACHE,
            'vintage_dir': VINTAGE_DIR,
            'output_dir': OUTPUT_DIR,
            'regime_strategies': REGIME_STRATEGIES,
            'entry_method': ENTRY_METHOD,
            'slippage_pct': SLIPPAGE_PCT,
            'benchmarks': BENCHMARKS,
            'enable_sector_diversification': ENABLE_SECTOR_DIVERSIFICATION,
            'max_positions_per_sector': MAX_POSITIONS_PER_SECTOR,
        }
        
        if overrides:
            cfg.update(overrides)
        
        return cfg
    
    def _init_components(self):
        """Initialize all backtest components"""
        print("[1/7] Initializing data loader...")
        self.data_loader = BacktestDataLoader(
            stock_price_cache=self.config['stock_price_cache'],
            macro_price_cache=self.config['macro_price_cache'],
            vintage_dir=self.config['vintage_dir']
        )
        
        print("[2/7] Loading universe...")
        self.universe = self.data_loader.load_universe(self.config['universe_csv'])
        
        print("[3/7] Preloading stock prices...")
        # Load stock prices with 1 year lookback for momentum calculation
        lookback_start = self.config['start_date'] - timedelta(days=365)
        self.stock_prices = self.data_loader.preload_all_stock_prices(
            self.universe, 
            lookback_start,
            self.config['end_date']
        )
        
        print("[4/7] Preloading macro prices...")
        # Use ^VIX (Yahoo Finance format)
        macro_symbols = ['SPY', 'QQQ', 'IWM', 'GLD', 'TLT', 'HYG', 'LQD', '^VIX']
        self.macro_prices = self.data_loader.preload_all_macro_prices(
            macro_symbols,
            self.config['start_date'],
            self.config['end_date']
        )
        
        print("[5/7] Initializing calculators...")
        self.regime_calc = RegimeCalculator(self.macro_prices)
        self.seasonality_calc = SeasonalityCalculator(lookback_years=SEASONALITY_LOOKBACK_YEARS)
        self.ml_generator = MLSignalGenerator(self.seasonality_calc)
        self.auto_decider = AutoDeciderSimulator(self.config['regime_strategies'])
        
        print("[6/7] Initializing portfolio...")
        # Pass constituents to portfolio for sector tracking
        self.portfolio = Portfolio(
            self.config['initial_cash'],
            constituents=self.constituents
        )
        
        print("[7/7] Initializing analyzer...")
        # Pass constituents to analyzer for sector analysis
        self.analyzer = PerformanceAnalyzer(constituents=self.constituents)
        
        self.visualizer = BacktestVisualizer(
            output_dir=self.config['output_dir'],
            dpi=PLOT_DPI
        )
        
        print("[OK] All components initialized\n")
    
    def run(self) -> Dict:
        """
        Run full backtest
        
        Returns:
            dict with results
        """
        print("=" * 80)
        print("RUNNING BACKTEST")
        print("=" * 80)
        
        # Get trading days
        trading_days = self.data_loader.get_trading_days(
            self.config['start_date'],
            self.config['end_date']
        )
        
        print(f"Total trading days: {len(trading_days)}")
        print(f"Expected years: {len(trading_days) / 252:.1f}\n")
        
        # Track regime history
        regime_history = []
        
        # Run day-by-day simulation
        for i, current_date in enumerate(tqdm(trading_days, desc="Simulating")):
            
            # 1. Calculate regime
            regime_data = self.regime_calc.calculate_regime(current_date)
            regime_history.append(regime_data)
            
            regime = regime_data['regime']
            
            # 2. Get regime strategy (for min_hold_days)
            regime_strategy = self.config['regime_strategies'].get(
                regime, 
                self.config['regime_strategies']['NEUTRAL_BULLISH']
            )
            min_hold_days = regime_strategy.get('min_hold_days', 0)
            
            # 3. Generate ML signals (candidates)
            candidates = self.ml_generator.generate_signals(
                target_date=current_date,
                stock_prices=self.stock_prices,
                regime=regime,
                gate_alpha=self.config['gate_alpha']
            )
            
            # NEW: Apply sector diversification filter if enabled
            if self.config['enable_sector_diversification'] and not candidates.empty:
                candidates = self._apply_sector_filter(candidates)
            
            # 4. Make trading decisions (auto_decider logic)
            portfolio_state = self.portfolio.get_state()
            
            decisions = self.auto_decider.decide_trades(
                target_date=current_date,
                candidates_df=candidates,
                portfolio_state=portfolio_state,
                regime=regime,
                cash=self.portfolio.cash
            )
            
            # 5. Execute SELL orders first (from regime exits)
            for sell in decisions['sell']:
                ticker = sell['ticker']
                # Get exit price (close of current day + slippage)
                exit_price = self._get_exit_price(ticker, current_date)
                if exit_price:
                    self.portfolio.sell(ticker, current_date, exit_price, sell['reason'])
            
            # 6. Check SL/TP triggers (intraday)
            intraday_prices = self._get_intraday_prices(current_date)
            sl_tp_exits = self.portfolio.check_exits(
                current_date, 
                intraday_prices,
                regime=regime,
                min_hold_days=min_hold_days
            )
            
            # 7. Execute BUY orders
            for buy in decisions['buy']:
                entry_price = self._get_entry_price(buy, current_date)
                
                self.portfolio.buy(
                    ticker=buy['ticker'],
                    date=current_date,
                    entry_price=entry_price,
                    shares=buy['shares'],
                    stop_loss=buy['stop_loss'],
                    take_profit=buy['take_profit'],
                    reason=buy['reason']
                )
            
            # 8. Update position prices (end of day)
            eod_prices = self._get_eod_prices(current_date)
            self.portfolio.update_prices(current_date, eod_prices)
            
            # 9. Record daily value
            self.portfolio.record_daily_value(current_date)
        
        print("\n[OK] Simulation complete\n")
        
        # Analyze results
        print("Analyzing results...")
        
        equity_curve = self.portfolio.get_equity_curve()
        trades_history = self.portfolio.get_trades_history()
        regime_df = pd.DataFrame(regime_history)
        
        # Load benchmark prices
        benchmark_prices = {}
        for bench in self.config['benchmarks']:
            if bench in self.macro_prices:
                benchmark_prices[bench] = self.macro_prices[bench]
        
        analysis = self.analyzer.analyze(
            portfolio_equity_curve=equity_curve,
            trades_history=trades_history,
            regime_history=regime_df,
            benchmark_prices=benchmark_prices
        )
        
        print("[OK] Analysis complete\n")
        
        # Create visualizations
        if self.config.get('save_plots', True):
            print("Creating visualizations...")
            
            # Add timestamp to output folder (HHMMSS format)
            timestamp = datetime.now().strftime("%H%M%S")
            output_dir_full = os.path.join(
                self.config['output_dir'],
                f"{self.config['start_date'].strftime('%Y-%m-%d')}_{self.config['end_date'].strftime('%Y-%m-%d')}_{timestamp}"
            )
            
            viz = BacktestVisualizer(output_dir_full, dpi=150)
            viz.create_all_plots(
                equity_curve=equity_curve,
                trades_history=trades_history,
                regime_history=regime_df,
                benchmark_prices=benchmark_prices,
                analysis=analysis
            )
            print("[OK] Visualizations complete\n")
        
        # Save results
        self._save_results(equity_curve, trades_history, regime_df, analysis)
        
        return {
            'equity_curve': equity_curve,
            'trades_history': trades_history,
            'regime_history': regime_df,
            'analysis': analysis
        }
    
    def _apply_sector_filter(self, candidates: pd.DataFrame) -> pd.DataFrame:
        """
        NEW: Apply sector diversification filter to candidates
        
        Args:
            candidates: ML candidates DataFrame
        
        Returns:
            Filtered candidates
        """
        if candidates.empty:
            return candidates
        
        # Get current sector exposure
        sector_exposure = self.portfolio.get_sector_exposure()
        
        # Add sector to candidates
        candidates = candidates.copy()
        candidates['sector'] = candidates['ticker'].apply(lambda x: self.portfolio.get_sector(x))
        
        # Filter out sectors that are at max capacity
        max_per_sector = self.config['max_positions_per_sector']
        
        def can_add_sector(sector):
            current_count = sector_exposure.get(sector, 0)
            return current_count < max_per_sector
        
        candidates['can_add'] = candidates['sector'].apply(can_add_sector)
        filtered = candidates[candidates['can_add']].copy()
        
        # Drop helper columns
        filtered = filtered.drop(columns=['sector', 'can_add'])
        
        return filtered
    
    def _get_entry_price(self, buy_decision: Dict, current_date: date) -> float:
        """
        Get entry price with gap/slippage
        
        Args:
            buy_decision: Buy decision dict
            current_date: Current date
        
        Returns:
            Adjusted entry price
        """
        base_price = buy_decision['entry']
        
        if self.config['entry_method'] == 'T_open_with_gap':
            # Simulate realistic gap: ±1-2% random
            gap_pct = np.random.uniform(-0.02, 0.02)
            entry_price = base_price * (1 + gap_pct)
        elif self.config['entry_method'] == 'T_open':
            entry_price = base_price
        else:  # T-1_close
            entry_price = base_price
        
        # Add slippage
        entry_price *= (1 + self.config['slippage_pct'])
        
        return entry_price
    
    def _get_exit_price(self, ticker: str, current_date: date) -> Optional[float]:
        """Get exit price (close + slippage)"""
        if ticker not in self.stock_prices:
            return None
        
        prices = self.stock_prices[ticker]
        row = prices[prices['date'] == current_date]
        
        if row.empty:
            return None
        
        close = row.iloc[0]['close']
        
        # Subtract slippage on exit
        exit_price = close * (1 - self.config['slippage_pct'])
        
        return exit_price
    
    def _get_intraday_prices(self, current_date: date) -> Dict[str, Dict[str, float]]:
        """Get intraday high/low for SL/TP checks"""
        intraday = {}
        
        for pos in self.portfolio.positions:
            ticker = pos['ticker']
            
            if ticker not in self.stock_prices:
                continue
            
            prices = self.stock_prices[ticker]
            row = prices[prices['date'] == current_date]
            
            if row.empty:
                continue
            
            intraday[ticker] = {
                'high': row.iloc[0]['high'],
                'low': row.iloc[0]['low']
            }
        
        return intraday
    
    def _get_eod_prices(self, current_date: date) -> Dict[str, float]:
        """Get end-of-day prices for all positions"""
        eod = {}
        
        for pos in self.portfolio.positions:
            ticker = pos['ticker']
            
            if ticker not in self.stock_prices:
                continue
            
            prices = self.stock_prices[ticker]
            row = prices[prices['date'] == current_date]
            
            if not row.empty:
                eod[ticker] = row.iloc[0]['close']
        
        return eod
    
    def _save_results(self, equity_curve: pd.DataFrame, 
                     trades: pd.DataFrame,
                     regime_history: pd.DataFrame,
                     analysis: Dict):
        """Save all results to disk"""
        print("Saving results...")
        
        # Add timestamp to output folder (HHMMSS format)
        timestamp = datetime.now().strftime("%H%M%S")
        output_dir = os.path.join(
            self.config['output_dir'],
            f"{self.config['start_date'].strftime('%Y-%m-%d')}_{self.config['end_date'].strftime('%Y-%m-%d')}_{timestamp}"
        )
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save config
        config_copy = self.config.copy()
        config_copy['start_date'] = str(config_copy['start_date'])
        config_copy['end_date'] = str(config_copy['end_date'])
        
        with open(os.path.join(output_dir, 'config_used.json'), 'w') as f:
            json.dump(config_copy, f, indent=2)
        
        # Save CSVs
        equity_curve.to_csv(os.path.join(output_dir, 'equity_curve.csv'), index=False)
        trades.to_csv(os.path.join(output_dir, 'trades_history.csv'), index=False)
        regime_history.to_csv(os.path.join(output_dir, 'regime_history.csv'), index=False)
        
        # Save regime breakdown
        if 'regime_breakdown' in analysis and not analysis['regime_breakdown'].empty:
            analysis['regime_breakdown'].to_csv(os.path.join(output_dir, 'regime_breakdown.csv'), index=False)
        
        # Save sector breakdown
        if 'sector_breakdown' in analysis and not analysis['sector_breakdown'].empty:
            analysis['sector_breakdown'].to_csv(os.path.join(output_dir, 'sector_breakdown.csv'), index=False)
        
        # NEW: Save yearly breakdown
        if 'yearly_breakdown' in analysis and not analysis['yearly_breakdown'].empty:
            analysis['yearly_breakdown'].to_csv(os.path.join(output_dir, 'yearly_breakdown.csv'), index=False)
        
        # Save hold time analysis
        if 'hold_time_analysis' in analysis:
            hold_time_data = analysis['hold_time_analysis']
            if 'hold_time_by_regime_bucket' in hold_time_data and not hold_time_data['hold_time_by_regime_bucket'].empty:
                hold_time_data['hold_time_by_regime_bucket'].to_csv(
                    os.path.join(output_dir, 'hold_time_analysis.csv'), index=False
                )
        
        # Save regime transitions
        if 'regime_transitions' in analysis and not analysis['regime_transitions'].empty:
            analysis['regime_transitions'].to_csv(os.path.join(output_dir, 'regime_transitions.csv'), index=False)
        
        # Save monthly/yearly returns
        if 'monthly_returns' in analysis and not analysis['monthly_returns'].empty:
            analysis['monthly_returns'].to_csv(os.path.join(output_dir, 'monthly_returns.csv'), index=False)
        
        if 'yearly_returns' in analysis and not analysis['yearly_returns'].empty:
            analysis['yearly_returns'].to_csv(os.path.join(output_dir, 'yearly_returns.csv'), index=False)
        
        # Save summary text
        summary_text = self.analyzer.generate_summary_text(analysis)
        with open(os.path.join(output_dir, 'performance_summary.txt'), 'w') as f:
            f.write(summary_text)
        
        print(f"[OK] Results saved to: {output_dir}\n")
        print(summary_text)


# Main entry point
if __name__ == "__main__":
    engine = BacktestEngine()
    results = engine.run()
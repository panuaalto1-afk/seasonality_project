# backtest_scripts/backtest_engine.py
"""
Enhanced Backtest Engine with Dynamic Position Sizing
Main orchestrator for 10-year backtest analysis

UPDATED: 2025-11-14 16:25 UTC
CRITICAL FIX: Load 10 years of historical data for seasonality calculation
FIXES:
  - Now loads lookback_start = start_date - 10 years (was only loading backtest period)
  - Uses MLSignalGenerator.generate_signals() directly (not broken _calculate_score loop)
  - Proper data availability for seasonality and momentum calculations
"""

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional
import logging
from tqdm import tqdm
import os

from .config import *
from .data_loader import BacktestDataLoader
from .portfolio import Portfolio
from .regime_calculator import RegimeCalculator
from .seasonality_calculator import SeasonalityCalculator
from .ml_signal_generator import MLSignalGenerator
from .auto_decider_simulator import AutoDeciderSimulator
from .performance_analyzer import PerformanceAnalyzer
from .visualizer import BacktestVisualizer

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    Enhanced backtest engine with dynamic position sizing
    """
    
    def __init__(
        self,
        config_overrides: Optional[Dict] = None,
        constituents_path: Optional[str] = None
    ):
        """
        Initialize backtest engine
        
        Args:
            config_overrides: Optional config overrides
            constituents_path: Path to constituents CSV
        """
        # Merge config
        self.config = self._build_config(config_overrides)
        
        # Extract key config
        self.start_date = self.config.get('start_date', BACKTEST_START)
        self.end_date = self.config.get('end_date', BACKTEST_END)
        self.initial_cash = self.config.get('initial_cash', INITIAL_CASH)
        
        # Constituents
        self.constituents_path = constituents_path or UNIVERSE_CSV
        
        # Print header
        self._print_header()
        
        # Components (initialized in setup)
        self.data_loader = None
        self.portfolio = None
        self.regime_calc = None
        self.seasonality_calc = None
        self.ml_generator = None
        self.auto_decider = None
        self.analyzer = None
        self.visualizer = None
        
        # Universe storage
        self.universe_tickers = []
        self.universe_df = None
        
        # Preloaded data
        self.stock_prices_cache = {}
        self.macro_prices_cache = {}
        
        # Results
        self.results = None
    
    
    def _build_config(self, overrides: Optional[Dict]) -> Dict:
        """Build configuration from globals + overrides"""
        config = {
            'start_date': BACKTEST_START,
            'end_date': BACKTEST_END,
            'initial_cash': INITIAL_CASH,
            'POSITION_SIZE_METHOD': POSITION_SIZE_METHOD,
            'POSITION_SIZE_PCT': POSITION_SIZE_PCT,
            'POSITION_SIZE_FIXED': POSITION_SIZE_FIXED,
            'MIN_POSITION_SIZE': MIN_POSITION_SIZE,
            'MAX_POSITION_SIZE': MAX_POSITION_SIZE,
            'MAX_POSITION_PCT': MAX_POSITION_PCT,
            'MAX_POSITIONS': MAX_POSITIONS,
            'GATE_ALPHA': GATE_ALPHA,
            'REGIME_STRATEGIES': REGIME_STRATEGIES,
            'SECTOR_STRATEGIES': SECTOR_STRATEGIES,
            'SECTOR_MAX_POSITIONS': SECTOR_MAX_POSITIONS,
            'ADAPTIVE_POSITION_SIZING': ADAPTIVE_POSITION_SIZING,
            'ENABLE_SECTOR_DIVERSIFICATION': ENABLE_SECTOR_DIVERSIFICATION,
            'USE_STOP_LOSS': USE_STOP_LOSS,
            'USE_TAKE_PROFIT': USE_TAKE_PROFIT,
            'SLIPPAGE_PCT': SLIPPAGE_PCT,
            'save_plots': SAVE_PLOTS,
        }
        
        if overrides:
            config.update(overrides)
        
        return config
    
    
    def _print_header(self):
        """Print backtest header"""
        print("\n" + "="*80)
        print(f"BACKTEST ENGINE - Version {CONFIG_VERSION}")
        print(f"Enhanced with Dynamic Position Sizing & Sector Optimization")
        print("="*80)
        print(f"Period: {self.start_date} to {self.end_date}")
        print(f"Initial Capital: ${self.initial_cash:,.2f}")
        print(f"Max Positions: {self.config['MAX_POSITIONS']}")
        
        if self.config['POSITION_SIZE_METHOD'] == 'percentage':
            print(f"Position Sizing: DYNAMIC {self.config['POSITION_SIZE_PCT']*100:.1f}% of portfolio")
        else:
            print(f"Position Sizing: FIXED ${self.config['POSITION_SIZE_FIXED']:,.2f}")
        
        print(f"Gate Alpha: {self.config['GATE_ALPHA']:.2f}")
        
        if self.config.get('ADAPTIVE_POSITION_SIZING', {}).get('enabled'):
            print(f"Adaptive Sizing: ENABLED")
        
        print("="*80 + "\n")
    
    
    def setup(self):
        """Initialize all components"""
        logger.info("[Engine] Initializing components...")
        
        # 1. Data Loader
        logger.info("[1/7] Initializing data loader...")
        self.data_loader = BacktestDataLoader(
            stock_price_cache=STOCK_PRICE_CACHE,
            macro_price_cache=MACRO_PRICE_CACHE,
            vintage_dir=VINTAGE_DIR
        )
        
        # 2. Load universe
        logger.info("[2/7] Loading universe...")
        self.universe_tickers = self.data_loader.load_universe(self.constituents_path)
        
        # Load universe DF for sector mapping
        import pandas as pd
        self.universe_df = pd.read_csv(self.constituents_path)
        
        # Extract sector mapping
        sector_mapping = {}
        if 'Sector' in self.universe_df.columns:
            ticker_col = None
            for col in ['Ticker', 'Symbol', 'ticker', 'symbol']:
                if col in self.universe_df.columns:
                    ticker_col = col
                    break
            
            if ticker_col:
                sector_mapping = dict(zip(
                    self.universe_df[ticker_col],
                    self.universe_df['Sector']
                ))
                logger.info(f"[Engine] ✓ Loaded sector mapping for {len(sector_mapping)} tickers")
        
        # 3. Preload stock data
        # CRITICAL FIX: Need 10 years of history for seasonality calculation!
        logger.info("[3/7] Preloading stock prices...")
        
        # Calculate lookback start date (10 years + buffer for momentum)
        lookback_years = SEASONALITY_LOOKBACK_YEARS  # 10 years from config
        lookback_days = lookback_years * 365 + 90  # +90 days buffer
        lookback_start = self.start_date - timedelta(days=lookback_days)
        
        logger.info(f"     Date range: {lookback_start} to {self.end_date}")
        logger.info(f"     (Need {lookback_years} years for seasonality calculation)")
        
        self.stock_prices_cache = self.data_loader.preload_all_stock_prices(
            self.universe_tickers,
            lookback_start,  # ← 10 YEARS BACK!
            self.end_date
        )
        
        # 4. Preload macro data
        logger.info("[4/7] Preloading macro prices...")
        macro_symbols = ['SPY', 'QQQ', 'TLT', 'GLD', 'VIX', 'HYG', 'LQD', 'DIA']
        
        # Macro also needs history for regime calculation
        macro_lookback = self.start_date - timedelta(days=365)  # 1 year for regime
        
        self.macro_prices_cache = self.data_loader.preload_all_macro_prices(
            macro_symbols,
            macro_lookback,
            self.end_date
        )
        
        # 5. Initialize calculators
        logger.info("[5/7] Initializing calculators...")
        
        # 5a. Regime calculator (needs macro_prices)
        self.regime_calc = RegimeCalculator(self.macro_prices_cache)
        
        # 5b. Seasonality calculator (standalone)
        self.seasonality_calc = SeasonalityCalculator(
            lookback_years=SEASONALITY_LOOKBACK_YEARS
        )
        
        # 5c. ML generator (needs seasonality_calc)
        self.ml_generator = MLSignalGenerator(self.seasonality_calc)
        
        # 5d. Auto decider (needs config)
        self.auto_decider = AutoDeciderSimulator(self.config)
        
        # 6. Initialize portfolio
        logger.info("[6/7] Initializing portfolio...")
        self.portfolio = Portfolio(
            initial_cash=self.initial_cash,
            config=self.config,
            sector_mapping=sector_mapping
        )
        
        # 7. Initialize analyzer
        logger.info("[7/7] Initializing analyzer...")
        self.analyzer = PerformanceAnalyzer()
        
        # 8. Visualizer (lazy init)
        output_dir = self.config.get('output_dir', OUTPUT_DIR)
        plots_dir = os.path.join(output_dir, 'plots')
        self.visualizer = BacktestVisualizer(plots_dir)
        
        logger.info("[OK] All components initialized\n")
    
    
    def run(self) -> Dict:
        """
        Run the backtest
        
        Returns:
            Results dictionary
        """
        logger.info("\n" + "="*80)
        logger.info("RUNNING BACKTEST")
        logger.info("="*80)
        
        # Get trading days
        trading_days = self.data_loader.get_trading_days(
            self.start_date,
            self.end_date
        )
        
        total_days = len(trading_days)
        years = (self.end_date - self.start_date).days / 365.25
        
        logger.info(f"Total trading days: {total_days}")
        logger.info(f"Expected years: {years:.1f}\n")
        
        # Progress bar
        pbar = tqdm(
            trading_days,
            desc="Simulating",
            unit="day",
            disable=not SHOW_PROGRESS_BAR
        )
        
        # Main simulation loop
        for current_date in pbar:
            self._simulate_day(current_date)
        
        pbar.close()
        logger.info("\n[OK] Simulation complete\n")
        
        # Analyze results
        logger.info("Analyzing results...")
        self.results = self._analyze_results()
        logger.info("[OK] Analysis complete\n")
        
        # Create visualizations
        if self.config.get('save_plots', True):
            logger.info("Creating visualizations...")
            self._create_visualizations()
            logger.info("[OK] Visualizations complete\n")
        
        # Save results
        logger.info("Saving results...")
        output_path = self._save_results()
        logger.info(f"[OK] Results saved to: {output_path}\n")
        
        # Print summary
        self._print_summary()
        
        return self.results
    
    
    def _simulate_day(self, current_date: pd.Timestamp):
        """
        Simulate one trading day
        
        FIXED: Now uses MLSignalGenerator.generate_signals() directly
        Same logic as debug_entry_conditions.py which WORKS!
        
        Args:
            current_date: Current date
        """
        # Convert to date object if needed
        if isinstance(current_date, pd.Timestamp):
            current_date_obj = current_date.date()
        else:
            current_date_obj = current_date
        
        # Get current prices for all positions
        current_prices = {}
        for symbol in list(self.portfolio.positions.keys()):
            price_data = self.get_stock_price(symbol, current_date_obj)
            if price_data is not None and 'close' in price_data:
                current_prices[symbol] = price_data['close']
        
        # Update portfolio value
        self.portfolio.update_market_value(pd.Timestamp(current_date_obj), current_prices)
        
        # Calculate regime
        regime_result = self.regime_calc.calculate_regime(current_date_obj)
        regime = regime_result.get('regime', 'NEUTRAL_BULLISH')
        regime_strategy = self.auto_decider.get_regime_strategy(regime)
        max_positions = regime_strategy.get('max_positions', 20)
        
        # ========================================================================
        # EXIT LOGIC - Check existing positions
        # ========================================================================
        for symbol in list(self.portfolio.positions.keys()):
            if symbol not in current_prices:
                continue
            
            position = self.portfolio.positions[symbol]
            current_price = current_prices[symbol]
            
            # Get ATR for stop/TP calculation
            price_history = self.get_stock_price_history(
                symbol,
                current_date_obj,
                lookback=20
            )
            atr = self._calculate_atr(price_history) if price_history is not None else None
            
            # Calculate current score (for signal degradation check)
            # Use simple momentum as proxy
            score_long = 0.5  # Neutral default
            if price_history is not None and len(price_history) >= 20:
                returns = price_history['close'].pct_change()
                if len(returns) > 0:
                    momentum = returns.tail(20).mean()
                    score_long = 0.5 + (momentum * 10)  # Rough conversion
                    score_long = max(0.0, min(1.0, score_long))
            
            # Check if should exit
            should_exit, reason = self.auto_decider.should_exit(
                symbol,
                position,
                current_price,
                pd.Timestamp(current_date_obj),
                score_long,
                regime,
                atr
            )
            
            if should_exit:
                self.portfolio.close_position(
                    symbol,
                    pd.Timestamp(current_date_obj),
                    current_price,
                    reason
                )
        
        # ========================================================================
        # ENTRY LOGIC - Use MLSignalGenerator (FIXED!)
        # ========================================================================
        
        # Skip if no capacity
        current_position_count = len(self.portfolio.positions)
        if current_position_count >= max_positions:
            return
        
        # Generate signals using MLSignalGenerator
        # (Same as debug_entry_conditions.py which WORKS!)
        try:
            signals = self.ml_generator.generate_signals(
                target_date=current_date_obj,
                stock_prices=self.stock_prices_cache,
                regime=regime,
                gate_alpha=self.config['GATE_ALPHA']
            )
        except Exception as e:
            logger.debug(f"Error generating signals: {e}")
            return
        
        if signals.empty:
            return
        
        # Process top candidates
        slots_available = max_positions - current_position_count
        
        for _, row in signals.head(slots_available * 2).iterrows():  # Check 2x candidates
            ticker = row['ticker']
            score_long = row['score_long']
            entry_price = row['entry_price']
            
            # Skip if already have position
            if ticker in self.portfolio.positions:
                continue
            
            # Get sector
            sector = self.portfolio.sector_mapping.get(ticker)
            
            # Check if can open position (sector limits)
            if not self.portfolio.can_open_position(sector):
                continue
            
            # Calculate volatility for auto_decider check
            volatility = None
            if ticker in self.stock_prices_cache:
                prices = self.stock_prices_cache[ticker]
                prices_before = prices[prices['date'] < current_date_obj]
                if len(prices_before) >= 20:
                    returns = prices_before['close'].pct_change().dropna()
                    if len(returns) > 0:
                        volatility = returns.tail(20).std()
            
            # Final check with auto_decider
            should_enter = self.auto_decider.should_enter(
                ticker,
                score_long,
                regime,
                sector,
                volatility
            )
            
            if not should_enter:
                continue
            
            # Open position!
            success = self.portfolio.open_position(
                ticker,
                pd.Timestamp(current_date_obj),
                entry_price,
                regime_strategy,
                sector,
                entry_reason="ML_SIGNAL"
            )
            
            if success:
                # Check if we're full
                if len(self.portfolio.positions) >= max_positions:
                    break
    
    
    def get_stock_price(self, symbol: str, date: date) -> Optional[Dict]:
        """Get stock price for a specific date"""
        if symbol not in self.stock_prices_cache:
            return None
        
        df = self.stock_prices_cache[symbol]
        
        # Find row for this date
        matching = df[df['date'] == date]
        
        if matching.empty:
            return None
        
        row = matching.iloc[0]
        return {
            'close': row['close'],
            'open': row.get('open', row['close']),
            'high': row.get('high', row['close']),
            'low': row.get('low', row['close']),
        }
    
    
    def get_stock_price_history(
        self,
        symbol: str,
        end_date: date,
        lookback: int = 252
    ) -> Optional[pd.DataFrame]:
        """Get stock price history"""
        if symbol not in self.stock_prices_cache:
            return None
        
        df = self.stock_prices_cache[symbol].copy()
        
        # Filter up to end_date
        df = df[df['date'] <= end_date]
        
        # Get last N rows
        if len(df) > lookback:
            df = df.iloc[-lookback:]
        
        return df if not df.empty else None
    
    
    def _calculate_atr(self, price_history: pd.DataFrame, period: int = 14) -> Optional[float]:
        """Calculate Average True Range"""
        try:
            if len(price_history) < period + 1:
                return None
            
            high = price_history['high']
            low = price_history['low']
            close = price_history['close']
            
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(period).mean().iloc[-1]
            
            return atr if not pd.isna(atr) else None
        
        except:
            return None
    
    
    def _analyze_results(self) -> Dict:
        """Analyze backtest results"""
        equity_curve = self.portfolio.get_equity_curve()
        trades_history = self.portfolio.get_trades_history()
        
        # Get benchmark data
        benchmark_data = {}
        for benchmark in BENCHMARKS:
            if benchmark in self.macro_prices_cache:
                benchmark_data[benchmark] = self.macro_prices_cache[benchmark]
        
        # Run analysis
        analysis = self.analyzer.analyze(
            equity_curve,
            trades_history,
            benchmark_data,
            self.config
        )
        
        return {
            'equity_curve': equity_curve,
            'trades_history': trades_history,
            'analysis': analysis,
            'config': self.config,
        }
    
    
    def _create_visualizations(self):
        """Create all visualizations"""
        if not self.results:
            return
        
        equity_curve = self.results['equity_curve']
        trades_history = self.results['trades_history']
        analysis = self.results['analysis']
        
        self.visualizer.create_all_plots(
            equity_curve,
            trades_history,
            analysis
        )
    
    
    def _save_results(self) -> str:
        """Save all results to disk"""
        # Create output directory
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        output_dir = os.path.join(
            OUTPUT_DIR,
            f"{self.start_date}_{self.end_date}_{timestamp}"
        )
        os.makedirs(output_dir, exist_ok=True)
        
        # Save equity curve
        equity_path = os.path.join(output_dir, 'equity_curve.csv')
        self.results['equity_curve'].to_csv(equity_path, index=False)
        
        # Save trades
        trades_path = os.path.join(output_dir, 'trades_history.csv')
        self.results['trades_history'].to_csv(trades_path, index=False)
        
        # Save analysis results
        analysis = self.results['analysis']
        
        # Yearly breakdown
        if 'yearly_breakdown' in analysis and analysis['yearly_breakdown']:
            yearly_path = os.path.join(output_dir, 'yearly_breakdown.csv')
            pd.DataFrame(analysis['yearly_breakdown']).to_csv(yearly_path, index=False)
        
        # Sector breakdown
        if 'sector_breakdown' in analysis and analysis['sector_breakdown']:
            sector_path = os.path.join(output_dir, 'sector_breakdown.csv')
            pd.DataFrame(analysis['sector_breakdown']).to_csv(sector_path, index=False)
        
        # Regime breakdown
        if 'regime_breakdown' in analysis and analysis['regime_breakdown']:
            regime_path = os.path.join(output_dir, 'regime_breakdown.csv')
            pd.DataFrame(analysis['regime_breakdown']).to_csv(regime_path, index=False)
        
        # Monthly returns
        if 'monthly_returns' in analysis and not analysis['monthly_returns'].empty:
            monthly_path = os.path.join(output_dir, 'monthly_returns.csv')
            analysis['monthly_returns'].to_csv(monthly_path)
        
        # Performance summary (text)
        summary_path = os.path.join(output_dir, 'performance_summary.txt')
        with open(summary_path, 'w') as f:
            f.write(self._generate_summary_text())
        
        # Save config
        config_path = os.path.join(output_dir, 'config.txt')
        with open(config_path, 'w') as f:
            f.write(f"Backtest Configuration - Version {CONFIG_VERSION}\n")
            f.write(f"Date: {CONFIG_DATE}\n\n")
            for key, value in self.config.items():
                if not key.startswith('_'):
                    f.write(f"{key}: {value}\n")
        
        return output_dir
    
    
    def _generate_summary_text(self) -> str:
        """Generate performance summary text"""
        analysis = self.results['analysis']
        pm = analysis.get('portfolio_metrics', {})
        
        lines = []
        lines.append("="*80)
        lines.append(f"BACKTEST PERFORMANCE SUMMARY - Version {CONFIG_VERSION}")
        lines.append("="*80)
        lines.append("")
        
        # Portfolio performance
        lines.append("PORTFOLIO PERFORMANCE:")
        ec = self.results['equity_curve']
        lines.append(f"  Initial Value:       ${ec.iloc[0]['total_value']:,.2f}")
        lines.append(f"  Final Value:         ${ec.iloc[-1]['total_value']:,.2f}")
        lines.append(f"  Total Return:        {pm.get('total_return', 0):.2f}%")
        lines.append(f"  CAGR:                {pm.get('cagr', 0):.2f}%")
        lines.append(f"  Sharpe Ratio:        {pm.get('sharpe_ratio', 0):.3f}")
        lines.append(f"  Sortino Ratio:       {pm.get('sortino_ratio', 0):.3f}")
        lines.append(f"  Calmar Ratio:        {pm.get('calmar_ratio', 0):.3f}")
        lines.append(f"  Max Drawdown:        {pm.get('max_drawdown', 0):.2f}%")
        lines.append(f"  Max DD Duration:     {pm.get('max_dd_duration', 0)} days")
        lines.append(f"  Volatility (Ann):    {pm.get('volatility', 0)*100:.2f}%")
        lines.append("")
        
        # Trade statistics
        lines.append("TRADE STATISTICS:")
        tm = analysis.get('trade_metrics', {})
        lines.append(f"  Total Trades:        {tm.get('total_trades', 0)}")
        lines.append(f"  Winning Trades:      {tm.get('winning_trades', 0)}")
        lines.append(f"  Win Rate:            {tm.get('win_rate', 0):.2f}%")
        lines.append(f"  Avg Win:             {tm.get('avg_win', 0):.2f}%")
        lines.append(f"  Avg Loss:            {tm.get('avg_loss', 0):.2f}%")
        lines.append(f"  Profit Factor:       {tm.get('profit_factor', 0):.3f}")
        lines.append(f"  Avg Hold Time:       {tm.get('avg_hold_time', 0):.1f} days")
        lines.append("")
        
        # Benchmark comparison
        if 'benchmark_comparison' in analysis:
            lines.append("BENCHMARK COMPARISON:")
            for benchmark, metrics in analysis['benchmark_comparison'].items():
                lines.append(f"  vs {benchmark}:")
                lines.append(f"    Benchmark Return:    {metrics.get('benchmark_return', 0):.2f}%")
                lines.append(f"    Portfolio Return:    {metrics.get('portfolio_return', 0):.2f}%")
                lines.append(f"    Outperformance:      {metrics.get('outperformance', 0):+.2f}%")
                lines.append(f"    Alpha:               {metrics.get('alpha', 0):.2f}%")
                lines.append(f"    Beta:                {metrics.get('beta', 0):.3f}")
            lines.append("")
        
        # Yearly breakdown
        if 'yearly_breakdown' in analysis and analysis['yearly_breakdown']:
            lines.append("YEARLY PERFORMANCE:")
            lines.append("-"*80)
            lines.append(f"{'Year':<8} {'Return %':<12} {'Sharpe':<10} {'Max DD %':<12} {'Trades'}")
            lines.append("-"*80)
            for year_data in analysis['yearly_breakdown']:
                lines.append(
                    f"{year_data['year']:<8} "
                    f"{year_data['return_pct']:>10.2f}% "
                    f"{year_data['sharpe']:>9.3f} "
                    f"{year_data['max_dd']:>10.2f}% "
                    f"{year_data['num_trades']:>7}"
                )
            lines.append("")
        
        # Sector breakdown (top 5)
        if 'sector_breakdown' in analysis and analysis['sector_breakdown']:
            lines.append("Top 5 Sectors by Performance:")
            lines.append("-"*80)
            sector_df = pd.DataFrame(analysis['sector_breakdown'])
            if not sector_df.empty:
                top_sectors = sector_df.nlargest(5, 'total_return_pct')
                for _, row in top_sectors.iterrows():
                    lines.append(
                        f"  {row['sector']:<30} "
                        f"Return: {row['total_return_pct']:>8.2f}%  "
                        f"Trades: {row['num_trades']:>4}  "
                        f"Win Rate: {row['win_rate']:>6.2f}%"
                    )
            lines.append("")
        
        lines.append("="*80)
        
        return "\n".join(lines)
    
    
    def _print_summary(self):
        """Print summary to console"""
        print("\n" + self._generate_summary_text())
"""
Backtest Engine - Main Orchestrator (OPTIMIZED v3)
Runs complete backtest simulation with optimized data loading

FIXED v3 - 2025-11-15 08:49 UTC:
- Optimized data loading (progress bar, batch processing)
- Fixed Unicode logging errors (removed special characters)
- Double header row handling
- Date type compatibility
- Robust error handling

Author: panuaalto1-afk
Last Updated: 2025-11-15 08:49:56 UTC
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from config import (
    BACKTEST_START, BACKTEST_END, INITIAL_CASH,
    PRICE_CACHE_DIR, REGIME_PRICE_CACHE, CONSTITUENTS_CSV,
    BACKTEST_RESULTS_DIR, SHOW_PROGRESS_BAR, VERBOSE,
    ENTRY_METHOD, EXIT_METHOD, GAP_MIN, GAP_MAX,
    SAVE_PLOTS, SAVE_TRADE_HISTORY, SAVE_REGIME_HISTORY,
    REGIME_MAX_POSITIONS, SECTOR_BLACKLIST,
    POSITION_SIZE_PCT, POSITION_SIZE_METHOD,
    GATE_ALPHA, SLIPPAGE_PCT, COMMISSION_PCT, 
)

from data_loader import DataLoader
from regime_calculator import RegimeCalculator
from ml_signal_generator import MLSignalGenerator
from auto_decider_simulator import AutoDeciderSimulator
from portfolio import Portfolio
from performance_analyzer import PerformanceAnalyzer
from visualizer import BacktestVisualizer


# Setup logging (ASCII only to avoid Unicode errors)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/backtest_engine.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    Main backtest engine coordinating all components.
    
    Architecture (matches live system):
    ========================================
    
    Daily Loop:
    1. Calculate regime (regime_calculator.py)
    2. Generate ML signals (ml_signal_generator.py)
    3. Auto decider makes decisions (auto_decider_simulator.py)
    4. Execute exits (regime-based, SL/TP)
    5. Execute entries (new positions)
    6. Update portfolio tracking
    7. Record daily metrics
    
    Components:
    - DataLoader: Load prices, universe, regime data
    - RegimeCalculator: Calculate 7-state market regime
    - MLSignalGenerator: Generate ML-based signals
    - AutoDeciderSimulator: Make buy/sell decisions
    - Portfolio: Track positions, execute trades
    - PerformanceAnalyzer: Calculate metrics
    - BacktestVisualizer: Create plots
    
    Key Features:
    - Walk-forward validation (no future leak)
    - Realistic execution (slippage, commissions)
    - Regime-adaptive position sizing
    - Intraday SL/TP checks
    - Comprehensive logging
    - Optimized data loading with progress bars
    """
    
    def __init__(self):
        """Initialize backtest engine and all components."""
        logger.info("=" * 80)
        logger.info("INITIALIZING BACKTEST ENGINE")
        logger.info("=" * 80)
        
        # Initialize components
        self.data_loader = DataLoader()
        self.regime_calculator = RegimeCalculator()
        self.signal_generator = MLSignalGenerator()
        self.auto_decider = AutoDeciderSimulator()
        self.portfolio = Portfolio(INITIAL_CASH)
        
        # State tracking
        self.current_date = None
        self.current_regime = 'NEUTRAL_BULLISH'
        self.prev_regime = 'NEUTRAL_BULLISH'
        
        # Data containers
        self.price_data: Dict[str, pd.DataFrame] = {}
        self.regime_prices: Dict[str, pd.DataFrame] = {}
        self.sector_map: Dict[str, str] = {}
        self.trading_days: List[datetime] = []
        self.tickers: List[str] = []
        
        # Performance tracking
        self.daily_summaries = []
        self.regime_changes = []
        
        logger.info("Backtest Engine initialized successfully")
    
    def load_data(self):
        """
        Load all required data for backtest.
        
        OPTIMIZED v3: Fast loading with progress bars
        """
        logger.info("\n" + "=" * 80)
        logger.info("LOADING DATA")
        logger.info("=" * 80)
        
        # 1. Load universe
        logger.info(f"Loading universe from {CONSTITUENTS_CSV}")
        universe_df = self.data_loader.load_universe(CONSTITUENTS_CSV)
        self.tickers = universe_df['ticker'].tolist()
        self.sector_map = dict(zip(universe_df['ticker'], universe_df['sector']))
        
        logger.info(f"[OK] Universe loaded: {len(self.tickers)} tickers")
        
        # Log sector distribution
        sector_counts = universe_df['sector'].value_counts()
        logger.info(f"  Sector distribution:")
        for sector, count in sector_counts.head(10).items():
            logger.info(f"    {sector:30s}: {count:3d} tickers")
        
        # 2. Load price data with progress bar
        start_ts = pd.Timestamp(BACKTEST_START)
        end_ts = pd.Timestamp(BACKTEST_END)
        
        logger.info(f"\nLoading price data from {PRICE_CACHE_DIR}")
        logger.info(f"Date range: {start_ts.date()} to {end_ts.date()}")
        logger.info("This may take 1-2 minutes for 500+ tickers...")
        
        loaded_count = 0
        failed_count = 0
        
        # Use tqdm for progress bar
        for ticker in tqdm(self.tickers, desc="Loading stock prices", unit="ticker"):
            try:
                ticker_file = Path(PRICE_CACHE_DIR) / f"{ticker}.csv"
                if not ticker_file.exists():
                    failed_count += 1
                    continue
                
                # Read CSV
                df = pd.read_csv(ticker_file)
                
                # Skip duplicate header if exists (ticker names row)
                if len(df) > 0:
                    first_row = df.iloc[0]
                    if first_row.astype(str).str.contains(ticker, case=False).sum() >= 3:
                        df = df.iloc[1:].reset_index(drop=True)
                
                # Normalize columns
                df.columns = df.columns.str.lower().str.strip()
                
                # Handle adj close -> close
                if 'adj close' in df.columns:
                    df = df.rename(columns={'adj close': 'close'})
                
                # Validate required columns
                required_cols = ['date', 'close']
                if not all(col in df.columns for col in required_cols):
                    failed_count += 1
                    continue
                
                # Process dates
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                df = df.dropna(subset=['date'])
                
                # Process prices
                for col in ['open', 'high', 'low', 'close']:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                df = df.dropna(subset=['close'])
                
                # Filter date range
                df = df[(df['date'] >= start_ts) & (df['date'] <= end_ts)]
                
                if df.empty:
                    failed_count += 1
                    continue
                
                # Store
                self.price_data[ticker] = df.sort_values('date').reset_index(drop=True)
                loaded_count += 1
                
            except Exception as e:
                failed_count += 1
                continue
        
        logger.info(f"[OK] Price data loaded for {loaded_count} tickers")
        if failed_count > 0:
            logger.warning(f"  [WARNING] Failed to load {failed_count} tickers")
        
        # Calculate data coverage
        coverage_pct = loaded_count / len(self.tickers) * 100
        logger.info(f"  Data coverage: {coverage_pct:.1f}%")
        
        # 3. Load regime price data
        logger.info(f"\nLoading regime indicators from {REGIME_PRICE_CACHE}")
        
        regime_symbols = ['SPY', 'QQQ', 'VIX', 'TLT', 'GLD', 'HYG', 'LQD', 'IWM']
        for symbol in regime_symbols:
            try:
                regime_file = Path(REGIME_PRICE_CACHE) / f"{symbol}.csv"
                if not regime_file.exists():
                    logger.warning(f"  [SKIP] Regime file not found: {symbol}")
                    continue
                
                df = pd.read_csv(regime_file)
                
                # Normalize columns
                df.columns = df.columns.str.lower().str.strip()
                
                # Skip duplicate header if exists
                if len(df) > 0:
                    first_row = df.iloc[0]
                    if first_row.astype(str).str.contains(symbol, case=False).sum() >= 2:
                        df = df.iloc[1:].reset_index(drop=True)
                
                # Handle adj close
                if 'adj close' in df.columns:
                    df = df.rename(columns={'adj close': 'close'})
                
                # Validate columns
                if 'date' not in df.columns or 'close' not in df.columns:
                    logger.warning(f"  [SKIP] Missing columns in {symbol}")
                    continue
                
                # Process data
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                df = df.dropna(subset=['date'])
                
                df['close'] = pd.to_numeric(df['close'], errors='coerce')
                df = df.dropna(subset=['close'])
                
                # Filter date range
                df = df[(df['date'] >= start_ts) & (df['date'] <= end_ts)]
                
                if not df.empty:
                    self.regime_prices[symbol] = df.sort_values('date').reset_index(drop=True)
                    
            except Exception as e:
                logger.warning(f"  [ERROR] Could not load {symbol}: {str(e)}")
        
        logger.info(f"[OK] Regime data loaded for {len(self.regime_prices)} indicators")
        
        # 4. Build trading calendar
        logger.info("\nBuilding trading calendar")
        
        # Use SPY as reference for trading days
        if 'SPY' in self.regime_prices and not self.regime_prices['SPY'].empty:
            self.trading_days = self.regime_prices['SPY']['date'].tolist()
        elif self.price_data:
            # Fallback: use union of all available dates
            all_dates = set()
            for df in self.price_data.values():
                all_dates.update(df['date'].tolist())
            self.trading_days = sorted(list(all_dates))
        else:
            raise ValueError("No price data available to build trading calendar!")
        
        logger.info(f"[OK] Trading calendar built: {len(self.trading_days)} days")
        logger.info(f"  Start: {self.trading_days[0].date()}")
        logger.info(f"  End:   {self.trading_days[-1].date()}")
        
        # 5. Data quality validation
        logger.info("\nData quality summary:")
        logger.info(f"  Tickers loaded:       {loaded_count}/{len(self.tickers)}")
        logger.info(f"  Regime indicators:    {len(self.regime_prices)}/8")
        logger.info(f"  Trading days:         {len(self.trading_days)}")
        
        expected_days = (end_ts - start_ts).days * 5 / 7
        logger.info(f"  Expected days (~2y):  ~{int(expected_days)}")
        
        if loaded_count < len(self.tickers) * 0.8:
            logger.warning(f"  [WARNING] Low data coverage: {coverage_pct:.1f}%")
        
        logger.info("\n" + "=" * 80)
        logger.info("DATA LOADING COMPLETE")
        logger.info("=" * 80)
    
    def calculate_regime(self, date: datetime) -> str:
        """Calculate market regime for given date."""
        regime_result = self.regime_calculator.calculate_regime(
            date,
            self.regime_prices
        )
        
        if regime_result is None:
            logger.warning(f"  Regime calculation failed for {date.date()}, keeping {self.current_regime}")
            return self.current_regime
        
        return regime_result['regime']
    
    def generate_signals(self, date: datetime) -> pd.DataFrame:
        """Generate trading signals for given date."""
        signals_df = self.signal_generator.generate_signals(
            tickers=self.tickers,
            target_date=date,
            price_data=self.price_data,
            regime=self.current_regime,
            sector_map=self.sector_map
        )
        
        return signals_df
    
    def get_current_prices(self, date: datetime) -> Dict[str, float]:
        """Get current closing prices for all tickers."""
        prices = {}
        
        for ticker, df in self.price_data.items():
            mask = df['date'] <= date
            if mask.any():
                price_row = df[mask].iloc[-1]
                prices[ticker] = float(price_row['close'])
        
        return prices
    
    def get_ohlc_data(self, ticker: str, date: datetime) -> Optional[Dict]:
        """Get OHLC data for ticker on specific date."""
        if ticker not in self.price_data:
            return None
        
        df = self.price_data[ticker]
        mask = df['date'] == date
        
        if not mask.any():
            return None
        
        row = df[mask].iloc[0]
        
        return {
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'close': float(row['close']),
            'date': row['date'],
        }
    
    def process_regime_exits(self, date: datetime) -> List[Dict]:
        """Process regime-based exits."""
        trades = []
        
        if not self.portfolio.positions:
            return trades
        
        # Get current positions info
        current_positions_info = []
        for ticker, pos in self.portfolio.positions.items():
            current_positions_info.append({
                'ticker': ticker,
                'score': pos.score,
                'entry_price': pos.entry_price,
                'days_held': pos.days_held,
            })
        
        # Check which positions should be exited
        tickers_to_exit = self.auto_decider.check_regime_exits(
            current_positions_info,
            self.current_regime,
            self.prev_regime
        )
        
        if not tickers_to_exit:
            return trades
        
        # Execute exits
        current_prices = self.get_current_prices(date)
        
        for ticker in tickers_to_exit:
            if ticker in self.portfolio.positions:
                exit_price = current_prices.get(ticker)
                if exit_price is not None and exit_price > 0:
                    try:
                        trade = self.portfolio.close_position(
                            ticker, date, exit_price, 'REGIME_EXIT'
                        )
                        trades.append(trade)
                        
                        if VERBOSE:
                            logger.info(f"    REGIME EXIT: {ticker} at ${exit_price:.2f} "
                                      f"(P&L: {trade['pnl_pct']:+.2f}%)")
                    except Exception as e:
                        logger.error(f"    Error closing {ticker}: {str(e)}")
        
        return trades
    
    def process_intraday_stops(self, date: datetime) -> List[Dict]:
        """Process intraday stop loss and take profit checks."""
        trades = []
        
        for ticker in list(self.portfolio.positions.keys()):
            ohlc = self.get_ohlc_data(ticker, date)
            
            if ohlc is None:
                continue
            
            try:
                trade = self.portfolio.check_stops_intraday(
                    ticker, date,
                    ohlc['high'], ohlc['low'], ohlc['close']
                )
                
                if trade is not None:
                    trades.append(trade)
                    
                    if VERBOSE:
                        reason_label = "SL" if trade['reason'] == 'STOP_LOSS' else "TP"
                        logger.info(f"    {reason_label}: {ticker} at ${trade['price']:.2f} "
                                  f"(P&L: {trade['pnl_pct']:+.2f}%, {trade['days_held']}d)")
            except Exception as e:
                logger.error(f"    Error checking stops for {ticker}: {str(e)}")
        
        return trades
    
    def process_buys(self, date: datetime, candidates: pd.DataFrame) -> List[Dict]:
        """Process buy decisions."""
        trades = []
        
        if candidates.empty:
            return trades
        
        # Get current state
        current_positions = len(self.portfolio.positions)
        current_tickers = list(self.portfolio.positions.keys())
        sector_positions = self.portfolio.get_sector_positions()
        
        # Select candidates to buy
        buys_df = self.auto_decider.select_buys(
            candidates,
            self.current_regime,
            current_positions,
            sector_positions,
            current_tickers
        )
        
        if buys_df.empty:
            return trades
        
        if VERBOSE:
            logger.info(f"    Buy candidates: {len(buys_df)}")
        
        # Get current prices
        current_prices = self.get_current_prices(date)
        
        # Execute buys
        for _, candidate in buys_df.iterrows():
            ticker = candidate['ticker']
            
            # Check if we can open position
            sector = candidate.get('sector', 'Unknown')
            sector_max = self.auto_decider.get_sector_max_positions(sector)
            
            if not self.portfolio.can_open_position(
                ticker,
                self.auto_decider.get_max_positions(self.current_regime),
                sector_positions,
                sector,
                sector_max
            ):
                continue
            
            # Get entry price
            entry_price = candidate['entry_price']
            
            # Apply gap if method is open_with_gap
            if ENTRY_METHOD == 'open_with_gap':
                gap = np.random.uniform(GAP_MIN, GAP_MAX)
                entry_price = entry_price * (1 + gap)
            
            # Validate price
            if entry_price <= 0 or not np.isfinite(entry_price):
                logger.warning(f"    Invalid entry price for {ticker}: {entry_price}")
                continue
            
            # Open position
            try:
                trade = self.portfolio.open_position(
                    ticker=ticker,
                    date=date,
                    price=entry_price,
                    regime=self.current_regime,
                    sector=sector,
                    atr=candidate['atr_14'],
                    score=candidate['score_long'],
                    current_prices=current_prices
                )
                
                if trade is not None:
                    trades.append(trade)
                    sector_positions[sector] = sector_positions.get(sector, 0) + 1
                    
                    if VERBOSE:
                        logger.info(f"    BUY: {ticker} at ${entry_price:.2f} "
                                  f"({trade['shares']} shares, score={candidate['score_long']:.3f}, "
                                  f"SL=${trade['stop_loss']:.2f}, TP=${trade['take_profit']:.2f})")
            except Exception as e:
                logger.error(f"    Error opening position in {ticker}: {str(e)}")
        
        return trades
    
    def run_single_day(self, date: datetime, day_idx: int, total_days: int) -> Dict:
        """Run backtest for a single trading day."""
        self.current_date = date
        
        # 1. Update regime
        self.prev_regime = self.current_regime
        self.current_regime = self.calculate_regime(date)
        
        regime_changed = self.current_regime != self.prev_regime
        
        # 2. Log progress
        if VERBOSE or (day_idx % 20 == 0):
            portfolio_value = self.portfolio.get_total_value(
                self.get_current_prices(date)
            )
            positions_count = len(self.portfolio.positions)
            exposure = self.portfolio.get_exposure_pct(self.get_current_prices(date))
            
            logger.info(f"\n[{day_idx+1}/{total_days}] {date.date()} "
                       f"| Regime: {self.current_regime:18s} "
                       f"| Portfolio: ${portfolio_value:>12,.0f} "
                       f"| Positions: {positions_count:2d} "
                       f"| Exposure: {exposure:5.1f}%")
        
        # 3. Record regime change
        if regime_changed:
            self.auto_decider.record_regime_change(
                date, self.current_regime, 0.0, self.prev_regime
            )
            self.regime_changes.append({
                'date': date,
                'from': self.prev_regime,
                'to': self.current_regime
            })
            logger.info(f"    [!] REGIME CHANGE: {self.prev_regime} -> {self.current_regime}")
        
        all_trades = []
        
        # 4. Process regime exits
        if regime_changed:
            regime_exit_trades = self.process_regime_exits(date)
            all_trades.extend(regime_exit_trades)
            if regime_exit_trades:
                logger.info(f"    Regime exits: {len(regime_exit_trades)}")
        
        # 5. Process intraday stops
        stop_trades = self.process_intraday_stops(date)
        all_trades.extend(stop_trades)
        
        # 6. Generate signals
        signals_df = self.generate_signals(date)
        
        # 7. Process buys
        buy_trades = self.process_buys(date, signals_df)
        all_trades.extend(buy_trades)
        
        # 8. Record daily portfolio value
        current_prices = self.get_current_prices(date)
        self.portfolio.record_daily_value(date, current_prices, self.current_regime)
        
        # 9. Create daily summary
        summary = {
            'date': date,
            'regime': self.current_regime,
            'regime_changed': regime_changed,
            'signals_generated': len(signals_df),
            'buy_orders': len(buy_trades),
            'sell_orders': len(stop_trades) + len(regime_exit_trades) if regime_changed else len(stop_trades),
            'positions_open': len(self.portfolio.positions),
            'portfolio_value': self.portfolio.get_total_value(current_prices),
            'cash': self.portfolio.cash,
            'exposure_pct': self.portfolio.get_exposure_pct(current_prices),
        }
        
        self.daily_summaries.append(summary)
        
        return summary

    def run(self) -> Dict:
        """Run complete backtest."""
        logger.info("\n" + "=" * 80)
        logger.info("STARTING BACKTEST")
        logger.info("=" * 80)
        logger.info(f"Period: {BACKTEST_START} to {BACKTEST_END}")
        logger.info(f"Initial Capital: ${INITIAL_CASH:,.2f}")
        logger.info(f"Position Size: {POSITION_SIZE_PCT*100:.1f}% dynamic")
        logger.info(f"Gate Alpha: {GATE_ALPHA}")
        logger.info(f"Transaction Costs: {(SLIPPAGE_PCT + COMMISSION_PCT)*100:.2f}% per side")
        logger.info(f"Excluded Sectors: {len(SECTOR_BLACKLIST)}")
        
        # Load data
        self.load_data()
        
        # Run simulation
        logger.info("\n" + "=" * 80)
        logger.info("RUNNING SIMULATION")
        logger.info("=" * 80)
        logger.info(f"Trading days: {len(self.trading_days)}")
        logger.info(f"Tickers in universe: {len(self.tickers)}")
        logger.info(f"Tickers with data: {len(self.price_data)}")
        
        # Progress bar
        iterator = enumerate(self.trading_days)
        if SHOW_PROGRESS_BAR:
            iterator = tqdm(
                iterator, 
                total=len(self.trading_days), 
                desc="Backtest Progress",
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
            )
        
        for day_idx, date in iterator:
            try:
                summary = self.run_single_day(date, day_idx, len(self.trading_days))
            except Exception as e:
                logger.error(f"Error on {date.date()}: {str(e)}")
                if VERBOSE:
                    import traceback
                    traceback.print_exc()
                continue
        
        # Close remaining positions
        logger.info("\n" + "=" * 80)
        logger.info("CLOSING REMAINING POSITIONS")
        logger.info("=" * 80)
        
        if self.portfolio.positions:
            final_date = self.trading_days[-1]
            final_prices = self.get_current_prices(final_date)
            
            logger.info(f"Closing {len(self.portfolio.positions)} positions at EOD")
            
            for ticker in list(self.portfolio.positions.keys()):
                exit_price = final_prices.get(ticker)
                if exit_price is not None and exit_price > 0:
                    try:
                        trade = self.portfolio.close_position(
                            ticker, final_date, exit_price, 'EOD'
                        )
                        logger.info(f"  CLOSE: {ticker} at ${exit_price:.2f} "
                                  f"(P&L: {trade['pnl_pct']:+.2f}%)")
                    except Exception as e:
                        logger.error(f"  Error closing {ticker}: {str(e)}")
        else:
            logger.info("No positions to close")
        
        # Gather results
        logger.info("\n" + "=" * 80)
        logger.info("ANALYZING RESULTS")
        logger.info("=" * 80)
        
        results = {
            'equity_curve': self.portfolio.get_equity_curve(),
            'trades_history': self.portfolio.get_trades_history(),
            'regime_history': self.auto_decider.get_regime_history(),
            'daily_summaries': pd.DataFrame(self.daily_summaries),
        }
        
        # Validate results
        if results['equity_curve'].empty:
            logger.error("ERROR: Empty equity curve!")
            return results
        
        if results['trades_history'].empty:
            logger.warning("WARNING: No trades executed!")
        
        # Analyze performance
        try:
            analyzer = PerformanceAnalyzer(results)
            performance = analyzer.analyze()
            results['performance'] = performance
        except Exception as e:
            logger.error(f"Error analyzing performance: {str(e)}")
            import traceback
            traceback.print_exc()
            results['performance'] = {}
        
        # Save results
        try:
            self.save_results(results, results.get('performance', {}))
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            import traceback
            traceback.print_exc()
        
        # Generate visualizations
        if SAVE_PLOTS and hasattr(self, 'output_dir'):
            logger.info("\n" + "=" * 80)
            logger.info("GENERATING VISUALIZATIONS")
            logger.info("=" * 80)
            
            try:
                visualizer = BacktestVisualizer(results, self.output_dir)
                visualizer.create_all_plots()
            except Exception as e:
                logger.error(f"Error creating visualizations: {str(e)}")
                import traceback
                traceback.print_exc()
        
        logger.info("\n" + "=" * 80)
        logger.info("BACKTEST COMPLETE")
        logger.info("=" * 80)
        
        return results
    
    def save_results(self, results: Dict, performance: Dict):
        """Save backtest results to disk."""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        folder_name = f"{BACKTEST_START}_{BACKTEST_END}_{timestamp}"
        self.output_dir = BACKTEST_RESULTS_DIR / folder_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"\nSaving results to: {self.output_dir}")
        
        # Save equity curve
        if not results['equity_curve'].empty:
            results['equity_curve'].to_csv(
                self.output_dir / 'equity_curve.csv', index=False
            )
            logger.info("  [OK] equity_curve.csv")
        
        # Save trades history
        if SAVE_TRADE_HISTORY and not results['trades_history'].empty:
            results['trades_history'].to_csv(
                self.output_dir / 'trades_history.csv', index=False
            )
            logger.info("  [OK] trades_history.csv")
        
        # Save regime history
        if SAVE_REGIME_HISTORY and not results['regime_history'].empty:
            results['regime_history'].to_csv(
                self.output_dir / 'regime_history.csv', index=False
            )
            logger.info("  [OK] regime_history.csv")
        
        # Save daily summaries
        if not results['daily_summaries'].empty:
            results['daily_summaries'].to_csv(
                self.output_dir / 'daily_summaries.csv', index=False
            )
            logger.info("  [OK] daily_summaries.csv")
        
        # Save performance breakdowns
        if performance:
            if 'regime_breakdown' in performance and not performance['regime_breakdown'].empty:
                performance['regime_breakdown'].to_csv(
                    self.output_dir / 'regime_breakdown.csv', index=False
                )
                logger.info("  [OK] regime_breakdown.csv")
            
            if 'sector_breakdown' in performance and not performance['sector_breakdown'].empty:
                performance['sector_breakdown'].to_csv(
                    self.output_dir / 'sector_breakdown.csv', index=False
                )
                logger.info("  [OK] sector_breakdown.csv")
            
            if 'monthly_returns' in performance and not performance['monthly_returns'].empty:
                performance['monthly_returns'].to_csv(
                    self.output_dir / 'monthly_returns.csv', index=False
                )
                logger.info("  [OK] monthly_returns.csv")
            
            if 'yearly_breakdown' in performance and not performance['yearly_breakdown'].empty:
                performance['yearly_breakdown'].to_csv(
                    self.output_dir / 'yearly_breakdown.csv', index=False
                )
                logger.info("  [OK] yearly_breakdown.csv")
        
        # Save performance summary
        self.save_performance_summary(performance)
        logger.info("  [OK] performance_summary.txt")
        
        # Save configuration
        self.save_config()
        logger.info("  [OK] config.txt")
        
        logger.info("\n[SUCCESS] Results saved successfully")
    
    def save_performance_summary(self, performance: Dict):
        """Save human-readable performance summary."""
        summary_path = self.output_dir / 'performance_summary.txt'
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("BACKTEST PERFORMANCE SUMMARY - Aggressive v3.0 (OPTIMIZED)\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("PORTFOLIO PERFORMANCE:\n")
            f.write(f"  Initial Value:       ${performance.get('initial_value', 0):,.2f}\n")
            f.write(f"  Final Value:         ${performance.get('final_value', 0):,.2f}\n")
            f.write(f"  Total Return:        {performance.get('total_return', 0):.2f}%\n")
            f.write(f"  CAGR:                {performance.get('cagr', 0):.2f}%\n")
            f.write(f"  Sharpe Ratio:        {performance.get('sharpe', 0):.3f}\n")
            f.write(f"  Sortino Ratio:       {performance.get('sortino', 0):.3f}\n")
            f.write(f"  Calmar Ratio:        {performance.get('calmar', 0):.3f}\n")
            f.write(f"  Max Drawdown:        {performance.get('max_drawdown', 0):.2f}%\n")
            f.write(f"  Max DD Duration:     {performance.get('max_dd_duration', 0):.0f} days\n")
            f.write(f"  Volatility (Ann):    {performance.get('volatility', 0):.2f}%\n\n")
            
            f.write("TRADE STATISTICS:\n")
            f.write(f"  Total Trades:        {performance.get('total_trades', 0)}\n")
            f.write(f"  Winning Trades:      {performance.get('winning_trades', 0)}\n")
            f.write(f"  Losing Trades:       {performance.get('losing_trades', 0)}\n")
            f.write(f"  Win Rate:            {performance.get('win_rate', 0):.2f}%\n")
            f.write(f"  Avg Win:             {performance.get('avg_win', 0):.2f}%\n")
            f.write(f"  Avg Loss:            {performance.get('avg_loss', 0):.2f}%\n")
            f.write(f"  Profit Factor:       {performance.get('profit_factor', 0):.3f}\n")
            f.write(f"  Expectancy:          {performance.get('expectancy_pct', 0):.2f}%\n")
            f.write(f"  Avg Hold Time:       {performance.get('avg_hold_time', 0):.1f} days\n\n")
            
            # Yearly breakdown
            if 'yearly_breakdown' in performance and not performance['yearly_breakdown'].empty:
                yearly_df = performance['yearly_breakdown']
                f.write("YEARLY PERFORMANCE:\n")
                f.write("-" * 80 + "\n")
                f.write(f"{'Year':6s} {'Return %':>10s} {'Sharpe':>8s} {'Max DD %':>10s} {'Trades':>8s}\n")
                f.write("-" * 80 + "\n")
                
                for _, row in yearly_df.iterrows():
                    f.write(f"{int(row['year']):6d} "
                           f"{row['return_pct']:10.2f}% "
                           f"{row['sharpe']:8.3f} "
                           f"{row['max_dd']:10.2f}% "
                           f"{int(row['num_trades']):8d}\n")
                f.write("\n")
            
            # Top sectors
            if 'sector_breakdown' in performance and not performance['sector_breakdown'].empty:
                sector_df = performance['sector_breakdown']
                top_sectors = sector_df.nlargest(5, 'total_return')
                
                f.write("Top 5 Sectors by Performance:\n")
                f.write("-" * 80 + "\n")
                for _, row in top_sectors.iterrows():
                    f.write(f"  {row['sector']:30s} "
                           f"Return: {row['total_return']:>10,.2f}  "
                           f"Trades: {int(row['num_trades']):>4d}  "
                           f"Win Rate: {row['win_rate']:>6.2f}%\n")
                f.write("\n")
            
            f.write("=" * 80 + "\n")
    
    def save_config(self):
        """Save configuration used for this backtest."""
        from config import (
            POSITION_SIZE_METHOD, POSITION_SIZE_PCT, POSITION_SIZE_FIXED,
            MIN_POSITION_SIZE, MAX_POSITION_SIZE, MAX_POSITION_PCT,
            REGIME_MAX_POSITIONS, GATE_ALPHA, MIN_SCORE_LONG,
            SECTOR_BLACKLIST, SECTOR_MAX_POSITIONS,
            REGIME_STOP_LOSS_MULTIPLIER, REGIME_TAKE_PROFIT_MULTIPLIER,
            SLIPPAGE_PCT, COMMISSION_PCT,
        )
        
        config_path = self.output_dir / 'config.txt'
        
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write("Backtest Configuration - Aggressive v3.0 (OPTIMIZED)\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("=" * 60 + "\n")
            f.write("BACKTEST PERIOD\n")
            f.write("=" * 60 + "\n")
            f.write(f"start_date: {BACKTEST_START}\n")
            f.write(f"end_date: {BACKTEST_END}\n")
            f.write(f"initial_cash: {INITIAL_CASH}\n\n")
            
            f.write("=" * 60 + "\n")
            f.write("POSITION SIZING - AGGRESSIVE (6% Dynamic)\n")
            f.write("=" * 60 + "\n")
            f.write(f"POSITION_SIZE_METHOD: {POSITION_SIZE_METHOD}\n")
            f.write(f"POSITION_SIZE_PCT: {POSITION_SIZE_PCT} (6%)\n")
            f.write(f"POSITION_SIZE_FIXED: {POSITION_SIZE_FIXED}\n")
            f.write(f"MIN_POSITION_SIZE: {MIN_POSITION_SIZE}\n")
            f.write(f"MAX_POSITION_SIZE: {MAX_POSITION_SIZE}\n")
            f.write(f"MAX_POSITION_PCT: {MAX_POSITION_PCT}\n\n")
            
            f.write("=" * 60 + "\n")
            f.write("PORTFOLIO LIMITS - REGIME-BASED\n")
            f.write("=" * 60 + "\n")
            for regime, max_pos in REGIME_MAX_POSITIONS.items():
                f.write(f"{regime:20s}: {max_pos} positions\n")
            f.write("\n")
            
            f.write("=" * 60 + "\n")
            f.write("SIGNAL FILTERING\n")
            f.write("=" * 60 + "\n")
            f.write(f"GATE_ALPHA: {GATE_ALPHA}\n")
            f.write(f"MIN_SCORE_LONG: {MIN_SCORE_LONG}\n\n")
            
            f.write("=" * 60 + "\n")
            f.write("SECTOR FILTERING\n")
            f.write("=" * 60 + "\n")
            f.write("BLACKLISTED SECTORS:\n")
            for sector in SECTOR_BLACKLIST:
                f.write(f"  - {sector}\n")
            f.write("\nSECTOR MAX POSITIONS:\n")
            for sector, max_pos in SECTOR_MAX_POSITIONS.items():
                f.write(f"  {sector:30s}: {max_pos}\n")
            f.write("\n")
            
            f.write("=" * 60 + "\n")
            f.write("RISK MANAGEMENT\n")
            f.write("=" * 60 + "\n")
            f.write("Stop Loss Multipliers (ATR-based):\n")
            for regime, mult in REGIME_STOP_LOSS_MULTIPLIER.items():
                f.write(f"  {regime:20s}: {mult}x ATR\n")
            f.write("\nTake Profit Multipliers (ATR-based):\n")
            for regime, mult in REGIME_TAKE_PROFIT_MULTIPLIER.items():
                f.write(f"  {regime:20s}: {mult}x ATR\n")
            f.write("\n")
            
            f.write("=" * 60 + "\n")
            f.write("TRANSACTION COSTS\n")
            f.write("=" * 60 + "\n")
            f.write(f"SLIPPAGE_PCT: {SLIPPAGE_PCT} (0.5%)\n")
            f.write(f"COMMISSION_PCT: {COMMISSION_PCT} (0.1%)\n")
            f.write(f"TOTAL_COST_PER_SIDE: {SLIPPAGE_PCT + COMMISSION_PCT} (0.6%)\n\n")
            
            f.write("=" * 60 + "\n")
            f.write("EXECUTION SETTINGS\n")
            f.write("=" * 60 + "\n")
            f.write(f"ENTRY_METHOD: {ENTRY_METHOD}\n")
            f.write(f"EXIT_METHOD: {EXIT_METHOD}\n")
            if ENTRY_METHOD == 'open_with_gap':
                f.write(f"GAP_RANGE: {GAP_MIN*100:.1f}% to {GAP_MAX*100:.1f}%\n")
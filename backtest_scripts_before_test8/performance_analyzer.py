# backtest_scripts/performance_analyzer.py
"""
Enhanced Performance Analyzer with Position Sizing Analysis
Calculates comprehensive performance metrics

UPDATED: 2025-11-12 15:24 UTC
CHANGES:
  - Position sizing analysis
  - Enhanced sector breakdown
  - Rolling metrics tracking
  - Adaptive sizing impact analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class PerformanceAnalyzer:
    """
    Comprehensive performance analysis with position sizing metrics
    """
    
    def __init__(self):
        """Initialize analyzer"""
        logger.info("[PerformanceAnalyzer] Initialized")
    
    
    def analyze(
        self,
        equity_curve: pd.DataFrame,
        trades_history: pd.DataFrame,
        benchmark_data: Dict[str, pd.DataFrame],
        config: Dict
    ) -> Dict:
        """
        Perform comprehensive analysis
        
        Args:
            equity_curve: Portfolio equity curve
            trades_history: All trades
            benchmark_data: Benchmark price data
            config: Configuration dict
        
        Returns:
            Analysis results dictionary
        """
        logger.info("[PerformanceAnalyzer] Starting analysis...")
        
        results = {}
        
        # 1. Portfolio metrics
        results['portfolio_metrics'] = self._calculate_portfolio_metrics(equity_curve)
        
        # 2. Trade metrics
        results['trade_metrics'] = self._calculate_trade_metrics(trades_history)
        
        # 3. Benchmark comparison
        if benchmark_data:
            results['benchmark_comparison'] = self._calculate_benchmark_comparison(
                equity_curve,
                benchmark_data
            )
        
        # 4. Yearly breakdown
        results['yearly_breakdown'] = self._calculate_yearly_breakdown(
            equity_curve,
            trades_history
        )
        
        # 5. Monthly breakdown
        results['monthly_returns'] = self._calculate_monthly_returns(equity_curve)
        
        # 6. Sector analysis
        if not trades_history.empty and 'sector' in trades_history.columns:
            results['sector_breakdown'] = self._calculate_sector_breakdown(trades_history)
        
        # 7. Regime analysis
        if not trades_history.empty and 'entry_reason' in trades_history.columns:
            results['regime_breakdown'] = self._calculate_regime_breakdown(trades_history)
        
        # 8. Rolling metrics
        results['rolling_metrics'] = self._calculate_rolling_metrics(equity_curve)
        
        # 9. NEW: Position sizing analysis
        results['position_sizing_analysis'] = self._analyze_position_sizing(
            equity_curve,
            trades_history
        )
        
        # 10. Drawdown analysis
        results['drawdown_analysis'] = self._analyze_drawdowns(equity_curve)
        
        logger.info("[PerformanceAnalyzer] Analysis complete")
        
        return results
    
    
    def _calculate_portfolio_metrics(self, equity_curve: pd.DataFrame) -> Dict:
        """Calculate portfolio-level metrics"""
        if equity_curve.empty:
            return {}
        
        # Convert to series
        values = equity_curve['total_value'].values
        dates = pd.to_datetime(equity_curve['date'])
        
        # Basic returns
        initial_value = values[0]
        final_value = values[-1]
        total_return = ((final_value - initial_value) / initial_value) * 100
        
        # Time period
        days = (dates.iloc[-1] - dates.iloc[0]).days
        years = days / 365.25
        
        # CAGR
        cagr = (((final_value / initial_value) ** (1 / years)) - 1) * 100 if years > 0 else 0
        
        # Daily returns
        daily_returns = pd.Series(values).pct_change().dropna()
        
        # Volatility (annualized)
        volatility = daily_returns.std() * np.sqrt(252)
        
        # Sharpe Ratio
        risk_free_rate = 0.02  # 2% annual
        excess_returns = daily_returns - (risk_free_rate / 252)
        sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252) if excess_returns.std() > 0 else 0
        
        # Sortino Ratio
        negative_returns = daily_returns[daily_returns < 0]
        downside_std = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else 0
        sortino = (excess_returns.mean() / downside_std) * np.sqrt(252) if downside_std > 0 else 0
        
        # Drawdown
        cummax = pd.Series(values).cummax()
        drawdown = (pd.Series(values) - cummax) / cummax
        max_drawdown = drawdown.min() * 100
        
        # Calmar Ratio
        calmar = abs(cagr / max_drawdown) if max_drawdown != 0 else 0
        
        # Max drawdown duration
        dd_duration = self._calculate_max_dd_duration(equity_curve)
        
        return {
            'initial_value': initial_value,
            'final_value': final_value,
            'total_return': total_return,
            'cagr': cagr,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            'max_drawdown': max_drawdown,
            'max_dd_duration': dd_duration,
            'trading_days': len(values),
            'years': years,
        }
    
    
    def _calculate_trade_metrics(self, trades: pd.DataFrame) -> Dict:
        """Calculate trade-level metrics"""
        if trades.empty:
            return {}
        
        total_trades = len(trades)
        winning_trades = len(trades[trades['pl'] > 0])
        losing_trades = len(trades[trades['pl'] < 0])
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # Average win/loss
        wins = trades[trades['pl'] > 0]['pl_pct']
        losses = trades[trades['pl'] < 0]['pl_pct']
        
        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = losses.mean() if len(losses) > 0 else 0
        
        # Profit factor
        total_wins = trades[trades['pl'] > 0]['pl'].sum()
        total_losses = abs(trades[trades['pl'] < 0]['pl'].sum())
        profit_factor = total_wins / total_losses if total_losses > 0 else 0
        
        # Hold time
        avg_hold_time = trades['hold_days'].mean() if 'hold_days' in trades.columns else 0
        
        # Best/worst trades
        best_trade = trades['pl_pct'].max() if not trades.empty else 0
        worst_trade = trades['pl_pct'].min() if not trades.empty else 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'avg_hold_time': avg_hold_time,
            'best_trade': best_trade,
            'worst_trade': worst_trade,
        }
    
    
    def _calculate_benchmark_comparison(
        self,
        equity_curve: pd.DataFrame,
        benchmark_data: Dict[str, pd.DataFrame]
    ) -> Dict:
        """Compare portfolio to benchmarks"""
        results = {}
        
        if equity_curve.empty:
            return results
        
        # Portfolio returns
        portfolio_values = equity_curve['total_value'].values
        portfolio_returns = pd.Series(portfolio_values).pct_change().dropna()
        
        portfolio_total_return = ((portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]) * 100
        
        for benchmark_name, benchmark_prices in benchmark_data.items():
            if benchmark_prices.empty or 'close' not in benchmark_prices.columns:
                continue
            
            # Align dates
            benchmark_prices = benchmark_prices.copy()
            benchmark_prices['date'] = pd.to_datetime(benchmark_prices.index)
            
            # Calculate benchmark returns
            bench_values = benchmark_prices['close'].values
            bench_returns = pd.Series(bench_values).pct_change().dropna()
            
            benchmark_total_return = ((bench_values[-1] - bench_values[0]) / bench_values[0]) * 100
            
            # Outperformance
            outperformance = portfolio_total_return - benchmark_total_return
            
            # Alpha & Beta (simplified)
            # Align portfolio and benchmark returns
            min_len = min(len(portfolio_returns), len(bench_returns))
            if min_len > 30:
                port_ret = portfolio_returns.iloc[:min_len]
                bench_ret = bench_returns.iloc[:min_len]
                
                # Beta
                covariance = np.cov(port_ret, bench_ret)[0, 1]
                benchmark_variance = np.var(bench_ret)
                beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
                
                # Alpha (annualized)
                portfolio_annual = port_ret.mean() * 252
                benchmark_annual = bench_ret.mean() * 252
                alpha = (portfolio_annual - benchmark_annual) * 100
            else:
                beta = 0
                alpha = 0
            
            results[benchmark_name] = {
                'benchmark_return': benchmark_total_return,
                'portfolio_return': portfolio_total_return,
                'outperformance': outperformance,
                'alpha': alpha,
                'beta': beta,
            }
        
        return results
    
    
    def _calculate_yearly_breakdown(
        self,
        equity_curve: pd.DataFrame,
        trades: pd.DataFrame
    ) -> List[Dict]:
        """Calculate performance by year"""
        if equity_curve.empty:
            return []
        
        equity_curve = equity_curve.copy()
        equity_curve['date'] = pd.to_datetime(equity_curve['date'])
        equity_curve['year'] = equity_curve['date'].dt.year
        
        yearly_results = []
        
        for year in sorted(equity_curve['year'].unique()):
            year_data = equity_curve[equity_curve['year'] == year]
            
            if len(year_data) < 2:
                continue
            
            # Returns
            start_value = year_data.iloc[0]['total_value']
            end_value = year_data.iloc[-1]['total_value']
            year_return = ((end_value - start_value) / start_value) * 100 if start_value > 0 else 0
            
            # Sharpe
            daily_returns = year_data['total_value'].pct_change().dropna()
            sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0
            
            # Drawdown
            cummax = year_data['total_value'].cummax()
            drawdown = ((year_data['total_value'] - cummax) / cummax) * 100
            max_dd = drawdown.min()
            
            # Trades - KORJATTU: tarkista onko trades tyhjä
            num_trades = 0
            if not trades.empty and 'exit_date' in trades.columns:
                trades_copy = trades.copy()
                trades_copy['exit_date'] = pd.to_datetime(trades_copy['exit_date'])
                year_trades = trades_copy[trades_copy['exit_date'].dt.year == year]
                num_trades = len(year_trades)
            
            yearly_results.append({
                'year': year,
                'return_pct': year_return,
                'sharpe': sharpe,
                'max_dd': max_dd,
                'num_trades': num_trades,
            })
        
        return yearly_results
    
    
    def _calculate_monthly_returns(self, equity_curve: pd.DataFrame) -> pd.DataFrame:
        """Calculate monthly returns as heatmap data"""
        if equity_curve.empty:
            return pd.DataFrame()
        
        ec = equity_curve.copy()
        ec['date'] = pd.to_datetime(ec['date'])
        ec = ec.set_index('date')
        
        # Resample to month-end
        monthly = ec['total_value'].resample('M').last()
        monthly_returns = monthly.pct_change() * 100
        
        # Pivot to year x month
        monthly_returns = monthly_returns.to_frame('return')
        monthly_returns['year'] = monthly_returns.index.year
        monthly_returns['month'] = monthly_returns.index.month
        
        pivot = monthly_returns.pivot(index='year', columns='month', values='return')
        
        return pivot
    
    
    def _calculate_sector_breakdown(self, trades: pd.DataFrame) -> List[Dict]:
        """Analyze performance by sector"""
        if trades.empty or 'sector' not in trades.columns:
            return []
        
        sector_results = []
        
        for sector in trades['sector'].unique():
            if pd.isna(sector):
                continue
            
            sector_trades = trades[trades['sector'] == sector]
            
            total_return = sector_trades['pl'].sum()
            num_trades = len(sector_trades)
            winning = len(sector_trades[sector_trades['pl'] > 0])
            win_rate = (winning / num_trades * 100) if num_trades > 0 else 0
            
            avg_return_pct = sector_trades['pl_pct'].mean()
            total_return_pct = ((sector_trades['pl'].sum() / (num_trades * 5000)) * 100) if num_trades > 0 else 0
            
            sector_results.append({
                'sector': sector,
                'num_trades': num_trades,
                'total_return': total_return,
                'total_return_pct': total_return_pct,
                'win_rate': win_rate,
                'avg_return_pct': avg_return_pct,
            })
        
        # Sort by total return
        sector_results = sorted(sector_results, key=lambda x: x['total_return_pct'], reverse=True)
        
        return sector_results
    
    
    def _calculate_regime_breakdown(self, trades: pd.DataFrame) -> List[Dict]:
        """Analyze performance by regime"""
        if trades.empty:
            return []
        
        # Try to infer regime from exit reason
        regime_trades = {}
        
        for _, trade in trades.iterrows():
            reason = trade.get('reason', '')
            
            # Map reason to regime
            if 'BULL' in reason:
                regime = 'BULL'
            elif 'BEAR' in reason:
                regime = 'BEAR'
            elif 'NEUTRAL' in reason:
                regime = 'NEUTRAL'
            elif 'CRISIS' in reason:
                regime = 'CRISIS'
            else:
                regime = 'OTHER'
            
            if regime not in regime_trades:
                regime_trades[regime] = []
            
            regime_trades[regime].append(trade)
        
        # Calculate metrics per regime
        regime_results = []
        
        for regime, regime_trade_list in regime_trades.items():
            regime_df = pd.DataFrame(regime_trade_list)
            
            num_trades = len(regime_df)
            total_return = regime_df['pl'].sum()
            winning = len(regime_df[regime_df['pl'] > 0])
            win_rate = (winning / num_trades * 100) if num_trades > 0 else 0
            
            regime_results.append({
                'regime': regime,
                'num_trades': num_trades,
                'total_return': total_return,
                'win_rate': win_rate,
            })
        
        return regime_results
    
    
    def _calculate_rolling_metrics(
        self,
        equity_curve: pd.DataFrame,
        window: int = 252
    ) -> Dict:
        """Calculate rolling performance metrics"""
        if equity_curve.empty or len(equity_curve) < window:
            return {}
        
        values = equity_curve['total_value']
        returns = values.pct_change().dropna()
        
        # Rolling Sharpe
        rolling_sharpe = (
            returns.rolling(window).mean() / returns.rolling(window).std()
        ) * np.sqrt(252)
        
        # Rolling volatility
        rolling_vol = returns.rolling(window).std() * np.sqrt(252)
        
        # Rolling max drawdown
        rolling_max = values.rolling(window).max()
        rolling_dd = ((values - rolling_max) / rolling_max) * 100
        
        return {
            'rolling_sharpe': rolling_sharpe.dropna().tolist(),
            'rolling_volatility': rolling_vol.dropna().tolist(),
            'rolling_drawdown': rolling_dd.dropna().tolist(),
        }
    
    
    def _analyze_position_sizing(
        self,
        equity_curve: pd.DataFrame,
        trades: pd.DataFrame
    ) -> Dict:
        """Analyze position sizing effectiveness (NEW!)"""
        if trades.empty:
            return {}
        
        # Calculate actual position sizes from trades
        trades = trades.copy()
        trades['position_size'] = trades['shares'] * trades['entry_price']
        
        # Position size statistics
        avg_position_size = trades['position_size'].mean()
        min_position_size = trades['position_size'].min()
        max_position_size = trades['position_size'].max()
        std_position_size = trades['position_size'].std()
        
        # Position size vs return correlation
        correlation = trades['position_size'].corr(trades['pl_pct'])
        
        # Position size over time
        trades['entry_date'] = pd.to_datetime(trades['entry_date'])
        trades = trades.sort_values('entry_date')
        
        # Group by year
        trades['year'] = trades['entry_date'].dt.year
        yearly_avg_size = trades.groupby('year')['position_size'].mean().to_dict()
        
        # Adaptive sizing impact (if drawdown column exists)
        if 'drawdown' in equity_curve.columns:
            # Analyze position sizes during drawdowns
            ec = equity_curve.copy()
            ec['date'] = pd.to_datetime(ec['date'])
            
            high_dd_periods = ec[ec['drawdown'] < -0.03]  # > 3% drawdown
            
            if not high_dd_periods.empty:
                # Find trades during high DD
                high_dd_dates = set(high_dd_periods['date'])
                trades['in_drawdown'] = trades['entry_date'].isin(high_dd_dates)
                
                dd_trades = trades[trades['in_drawdown']]
                normal_trades = trades[~trades['in_drawdown']]
                
                avg_size_in_dd = dd_trades['position_size'].mean() if not dd_trades.empty else 0
                avg_size_normal = normal_trades['position_size'].mean() if not normal_trades.empty else 0
                
                dd_reduction = ((avg_size_normal - avg_size_in_dd) / avg_size_normal * 100) if avg_size_normal > 0 else 0
            else:
                dd_reduction = 0
        else:
            dd_reduction = 0
        
        return {
            'avg_position_size': avg_position_size,
            'min_position_size': min_position_size,
            'max_position_size': max_position_size,
            'std_position_size': std_position_size,
            'size_return_correlation': correlation,
            'yearly_avg_sizes': yearly_avg_size,
            'drawdown_size_reduction_pct': dd_reduction,
        }
    
    
    def _analyze_drawdowns(self, equity_curve: pd.DataFrame) -> Dict:
        """Detailed drawdown analysis"""
        if equity_curve.empty:
            return {}
        
        values = equity_curve['total_value'].values
        dates = pd.to_datetime(equity_curve['date'])
        
        # Calculate drawdown series
        cummax = pd.Series(values).cummax()
        drawdown = ((pd.Series(values) - cummax) / cummax) * 100
        
        # Find all drawdown periods
        in_dd = drawdown < 0
        dd_starts = in_dd & ~in_dd.shift(1, fill_value=False)
        dd_ends = ~in_dd & in_dd.shift(1, fill_value=False)
        
        drawdown_periods = []
        start_idx = None
        
        for i in range(len(in_dd)):
            if dd_starts.iloc[i]:
                start_idx = i
            elif dd_ends.iloc[i] and start_idx is not None:
                dd_depth = drawdown.iloc[start_idx:i].min()
                dd_duration = (dates.iloc[i] - dates.iloc[start_idx]).days
                
                drawdown_periods.append({
                    'start_date': dates.iloc[start_idx],
                    'end_date': dates.iloc[i],
                    'depth': dd_depth,
                    'duration_days': dd_duration,
                })
                
                start_idx = None
        
        # Handle ongoing drawdown
        if start_idx is not None:
            dd_depth = drawdown.iloc[start_idx:].min()
            dd_duration = (dates.iloc[-1] - dates.iloc[start_idx]).days
            
            drawdown_periods.append({
                'start_date': dates.iloc[start_idx],
                'end_date': dates.iloc[-1],
                'depth': dd_depth,
                'duration_days': dd_duration,
                'ongoing': True,
            })
        
        # Find max drawdown
        if drawdown_periods:
            max_dd_period = min(drawdown_periods, key=lambda x: x['depth'])
        else:
            max_dd_period = None
        
        return {
            'num_drawdowns': len(drawdown_periods),
            'avg_drawdown_depth': np.mean([dd['depth'] for dd in drawdown_periods]) if drawdown_periods else 0,
            'avg_drawdown_duration': np.mean([dd['duration_days'] for dd in drawdown_periods]) if drawdown_periods else 0,
            'max_drawdown_period': max_dd_period,
            'all_drawdowns': drawdown_periods[:10],  # Top 10
        }
    
    
    def _calculate_max_dd_duration(self, equity_curve: pd.DataFrame) -> int:
        """Calculate maximum drawdown duration in days"""
        if equity_curve.empty:
            return 0
        
        values = equity_curve['total_value'].values
        cummax = pd.Series(values).cummax()
        drawdown = (pd.Series(values) - cummax) / cummax
        
        # Find drawdown periods
        in_dd = drawdown < 0
        dd_periods = (in_dd != in_dd.shift()).cumsum()
        
        # Count duration of each period
        dd_durations = in_dd.groupby(dd_periods).sum()
        
        return int(dd_durations.max()) if len(dd_durations) > 0 else 0
# backtest_scripts/performance_analyzer.py
"""
Performance Analysis Module
Calculates metrics, benchmarks, and generates reports
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import date

class PerformanceAnalyzer:
    """
    Analyzes backtest performance
    Calculates metrics, compares to benchmarks
    """
    
    def __init__(self, constituents: Optional[pd.DataFrame] = None):
        """
        Initialize performance analyzer
        
        Args:
            constituents: Optional DataFrame with ticker and Sector columns
        """
        self.constituents = constituents
        print("[PerformanceAnalyzer] Initialized")
    
    def analyze(self,
                portfolio_equity_curve: pd.DataFrame,
                trades_history: pd.DataFrame,
                regime_history: pd.DataFrame,
                benchmark_prices: Dict[str, pd.DataFrame]) -> Dict:
        """
        Perform full performance analysis
        
        Args:
            portfolio_equity_curve: Portfolio value over time
            trades_history: All trades executed
            regime_history: Market regime over time
            benchmark_prices: Dict mapping benchmark → price DataFrame
        
        Returns:
            Dict with comprehensive analysis
        """
        print("[PerformanceAnalyzer] Starting analysis...")
        
        analysis = {}
        
        # Portfolio metrics
        analysis['portfolio_metrics'] = self._calculate_portfolio_metrics(portfolio_equity_curve)
        
        # Trade statistics
        analysis['trade_stats'] = self._calculate_trade_stats(trades_history)
        
        # Benchmark comparison
        analysis['benchmark_comparison'] = self._compare_benchmarks(
            portfolio_equity_curve, benchmark_prices
        )
        
        # Regime breakdown
        analysis['regime_breakdown'] = self._analyze_by_regime(
            trades_history, regime_history
        )
        
        # Monthly/Yearly returns
        analysis['monthly_returns'] = self._calculate_monthly_returns(portfolio_equity_curve)
        analysis['yearly_returns'] = self._calculate_yearly_returns(portfolio_equity_curve)
        
        # Sector breakdown (if constituents available)
        if self.constituents is not None:
            analysis['sector_breakdown'] = self._analyze_by_sector(trades_history)
        
        # Hold time analysis
        analysis['hold_time_analysis'] = self._analyze_hold_times(trades_history, regime_history)
        
        # Regime transitions
        analysis['regime_transitions'] = self._analyze_regime_transitions(regime_history)
        
        print("[PerformanceAnalyzer] Analysis complete")
        
        return analysis
    
    def _calculate_portfolio_metrics(self, equity_curve: pd.DataFrame) -> Dict:
        """Calculate portfolio performance metrics"""
        
        if equity_curve.empty:
            return {}
        
        initial_value = equity_curve.iloc[0]['total_value']
        final_value = equity_curve.iloc[-1]['total_value']
        
        total_return = ((final_value - initial_value) / initial_value) * 100
        
        # Calculate returns
        equity_curve = equity_curve.copy()
        equity_curve['returns'] = equity_curve['total_value'].pct_change()
        
        # Annual return
        days = (equity_curve.iloc[-1]['date'] - equity_curve.iloc[0]['date']).days
        years = days / 365.25
        annual_return = ((final_value / initial_value) ** (1 / years) - 1) * 100 if years > 0 else 0
        
        # Volatility (annualized)
        volatility = equity_curve['returns'].std() * np.sqrt(252) * 100
        
        # Sharpe ratio (assuming 0% risk-free rate)
        sharpe = (annual_return / volatility) if volatility > 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = equity_curve[equity_curve['returns'] < 0]['returns']
        downside_vol = downside_returns.std() * np.sqrt(252) * 100 if len(downside_returns) > 0 else 0.001
        sortino = (annual_return / downside_vol) if downside_vol > 0 else 0
        
        # Max drawdown
        equity_curve['cummax'] = equity_curve['total_value'].cummax()
        equity_curve['drawdown'] = (equity_curve['total_value'] - equity_curve['cummax']) / equity_curve['cummax'] * 100
        max_drawdown = equity_curve['drawdown'].min()
        
        # Calmar ratio
        calmar = (annual_return / abs(max_drawdown)) if max_drawdown != 0 else 0
        
        return {
            'initial_value': initial_value,
            'final_value': final_value,
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar
        }
    
    def _calculate_trade_stats(self, trades: pd.DataFrame) -> Dict:
        """
        Calculate detailed trade statistics
        
        Args:
            trades: Trades history DataFrame
        
        Returns:
            Dict with trade statistics
        """
        if trades.empty:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'avg_hold_time': 0.0,
                'profit_factor': 0.0,
                'max_win': 0.0,
                'max_loss': 0.0
            }
        
        # FIXED: trades DataFrame already contains all exits (no 'action' column)
        # Just work with trades directly
        winning = trades[trades['pl_pct'] > 0]
        losing = trades[trades['pl_pct'] <= 0]
        
        total_trades = len(trades)
        winning_trades = len(winning)
        losing_trades = len(losing)
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0
        
        avg_win = winning['pl_pct'].mean() if len(winning) > 0 else 0.0
        avg_loss = losing['pl_pct'].mean() if len(losing) > 0 else 0.0
        
        avg_hold_time = trades['hold_days'].mean() if 'hold_days' in trades.columns else 0.0
        
        # Profit factor
        gross_profit = winning['pl'].sum() if len(winning) > 0 else 0.0
        gross_loss = abs(losing['pl'].sum()) if len(losing) > 0 else 0.0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0.0
        
        max_win = winning['pl_pct'].max() if len(winning) > 0 else 0.0
        max_loss = losing['pl_pct'].min() if len(losing) > 0 else 0.0
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_hold_time': avg_hold_time,
            'profit_factor': profit_factor,
            'max_win': max_win,
            'max_loss': max_loss
        }
    
    def _compare_benchmarks(self,
                           equity_curve: pd.DataFrame,
                           benchmark_prices: Dict[str, pd.DataFrame]) -> Dict:
        """Compare portfolio to benchmarks"""
        
        comparison = {}
        
        for bench_name, bench_df in benchmark_prices.items():
            if bench_df.empty:
                continue
            
            # Align dates
            merged = equity_curve.merge(
                bench_df[['date', 'close']],
                on='date',
                how='inner'
            )
            
            if merged.empty:
                continue
            
            # Calculate benchmark return
            bench_start = merged.iloc[0]['close']
            bench_end = merged.iloc[-1]['close']
            bench_return = ((bench_end - bench_start) / bench_start) * 100
            
            # Portfolio return over same period
            port_start = merged.iloc[0]['total_value']
            port_end = merged.iloc[-1]['total_value']
            port_return = ((port_end - port_start) / port_start) * 100
            
            outperformance = port_return - bench_return
            
            # Calculate alpha and beta
            merged['port_returns'] = merged['total_value'].pct_change()
            merged['bench_returns'] = merged['close'].pct_change()
            merged = merged.dropna()
            
            if len(merged) > 1:
                covariance = merged['port_returns'].cov(merged['bench_returns'])
                bench_variance = merged['bench_returns'].var()
                beta = covariance / bench_variance if bench_variance > 0 else 0
                
                # Alpha (annualized)
                port_annual = merged['port_returns'].mean() * 252
                bench_annual = merged['bench_returns'].mean() * 252
                alpha = (port_annual - beta * bench_annual) * 100
            else:
                beta = 0
                alpha = 0
            
            comparison[bench_name] = {
                'benchmark_return': bench_return,
                'portfolio_return': port_return,
                'outperformance': outperformance,
                'alpha': alpha,
                'beta': beta
            }
        
        return comparison
    
    def _analyze_by_regime(self,
                          trades: pd.DataFrame,
                          regime_history: pd.DataFrame) -> pd.DataFrame:
        """Analyze performance by market regime"""
        
        if trades.empty or regime_history.empty:
            return pd.DataFrame()
        
        # Merge trades with regime
        trades_with_regime = trades.copy()
        
        # Map entry_date to regime
        regime_dict = dict(zip(regime_history['date'], regime_history['regime']))
        trades_with_regime['regime'] = trades_with_regime['entry_date'].map(regime_dict)
        
        # Group by regime
        regime_stats = trades_with_regime.groupby('regime').agg({
            'pl_pct': ['count', 'mean', 'sum'],
            'hold_days': 'mean'
        }).reset_index()
        
        regime_stats.columns = ['regime', 'num_trades', 'avg_return', 'total_return', 'avg_hold_days']
        
        # Calculate win rate per regime
        win_rates = trades_with_regime.groupby('regime').apply(
            lambda x: (x['pl_pct'] > 0).sum() / len(x) * 100
        ).reset_index(name='win_rate')
        
        regime_stats = regime_stats.merge(win_rates, on='regime')
        
        return regime_stats
    
    def _calculate_monthly_returns(self, equity_curve: pd.DataFrame) -> pd.DataFrame:
        """Calculate monthly returns"""
        
        if equity_curve.empty:
            return pd.DataFrame()
        
        equity_curve = equity_curve.copy()
        equity_curve['year_month'] = pd.to_datetime(equity_curve['date']).dt.to_period('M')
        
        monthly = equity_curve.groupby('year_month').agg({
            'total_value': ['first', 'last']
        }).reset_index()
        
        monthly.columns = ['year_month', 'start_value', 'end_value']
        monthly['return'] = ((monthly['end_value'] - monthly['start_value']) / monthly['start_value']) * 100
        
        return monthly
    
    def _calculate_yearly_returns(self, equity_curve: pd.DataFrame) -> pd.DataFrame:
        """Calculate yearly returns"""
        
        if equity_curve.empty:
            return pd.DataFrame()
        
        equity_curve = equity_curve.copy()
        equity_curve['year'] = pd.to_datetime(equity_curve['date']).dt.year
        
        yearly = equity_curve.groupby('year').agg({
            'total_value': ['first', 'last']
        }).reset_index()
        
        yearly.columns = ['year', 'start_value', 'end_value']
        yearly['return'] = ((yearly['end_value'] - yearly['start_value']) / yearly['start_value']) * 100
        
        return yearly
    
    def _analyze_by_sector(self, trades: pd.DataFrame) -> pd.DataFrame:
        """Analyze performance by sector"""
        
        if trades.empty or self.constituents is None or 'Sector' not in self.constituents.columns:
            return pd.DataFrame()
        
        # Merge trades with sector data
        trades_with_sector = trades.merge(
            self.constituents[['ticker', 'Sector']],
            on='ticker',
            how='left'
        )
        
        # Group by sector
        sector_stats = trades_with_sector.groupby('Sector').agg({
            'pl_pct': ['count', 'mean', 'sum'],
            'hold_days': 'mean'
        }).reset_index()
        
        sector_stats.columns = ['sector', 'num_trades', 'avg_return', 'total_return', 'avg_hold_days']
        
        # Calculate win rate per sector
        win_rates = trades_with_sector.groupby('Sector').apply(
            lambda x: (x['pl_pct'] > 0).sum() / len(x) * 100 if len(x) > 0 else 0
        ).reset_index(name='win_rate')
        
        sector_stats = sector_stats.merge(win_rates, left_on='sector', right_on='Sector', how='left')
        sector_stats = sector_stats.drop('Sector', axis=1)
        
        return sector_stats
    
    def _analyze_hold_times(self, trades: pd.DataFrame, regime_history: pd.DataFrame) -> Dict:
        """Analyze hold times and their relationship to returns"""
        
        if trades.empty:
            return {}
        
        # Overall hold time distribution
        hold_buckets = pd.cut(trades['hold_days'], bins=[0, 7, 14, 30, 60, 999], labels=['0-7d', '7-14d', '14-30d', '30-60d', '60+d'])
        
        hold_analysis = trades.groupby(hold_buckets).agg({
            'pl_pct': ['count', 'mean'],
        }).reset_index()
        
        hold_analysis.columns = ['hold_bucket', 'num_trades', 'avg_return']
        
        # Merge with regime
        if not regime_history.empty:
            trades_with_regime = trades.copy()
            regime_dict = dict(zip(regime_history['date'], regime_history['regime']))
            trades_with_regime['regime'] = trades_with_regime['entry_date'].map(regime_dict)
            trades_with_regime['hold_bucket'] = pd.cut(trades_with_regime['hold_days'], bins=[0, 7, 14, 30, 60, 999], labels=['0-7d', '7-14d', '14-30d', '30-60d', '60+d'])
            
            regime_hold_analysis = trades_with_regime.groupby(['regime', 'hold_bucket']).agg({
                'pl_pct': ['count', 'mean']
            }).reset_index()
            
            regime_hold_analysis.columns = ['regime', 'hold_bucket', 'num_trades', 'avg_return']
        else:
            regime_hold_analysis = pd.DataFrame()
        
        return {
            'hold_time_distribution': hold_analysis,
            'hold_time_by_regime_bucket': regime_hold_analysis
        }
    
    def _analyze_regime_transitions(self, regime_history: pd.DataFrame) -> pd.DataFrame:
        """Analyze regime transitions"""
        
        if regime_history.empty:
            return pd.DataFrame()
        
        regime_history = regime_history.copy()
        regime_history['prev_regime'] = regime_history['regime'].shift(1)
        
        transitions = regime_history[regime_history['regime'] != regime_history['prev_regime']].copy()
        
        return transitions[['date', 'prev_regime', 'regime', 'score']]
    
    def generate_summary_text(self, analysis: Dict) -> str:
        """Generate human-readable summary"""
        
        pm = analysis.get('portfolio_metrics', {})
        ts = analysis.get('trade_stats', {})
        bc = analysis.get('benchmark_comparison', {})
        
        summary = []
        summary.append("=" * 80)
        summary.append("BACKTEST PERFORMANCE SUMMARY")
        summary.append("=" * 80)
        summary.append("")
        
        summary.append("PORTFOLIO PERFORMANCE:")
        summary.append(f"  Initial Value:     ${pm.get('initial_value', 0):,.2f}")
        summary.append(f"  Final Value:       ${pm.get('final_value', 0):,.2f}")
        summary.append(f"  Total Return:      {pm.get('total_return', 0):.2f}%")
        summary.append(f"  Annual Return:     {pm.get('annual_return', 0):.2f}%")
        summary.append(f"  Sharpe Ratio:      {pm.get('sharpe_ratio', 0):.3f}")
        summary.append(f"  Sortino Ratio:     {pm.get('sortino_ratio', 0):.3f}")
        summary.append(f"  Max Drawdown:      {pm.get('max_drawdown', 0):.2f}%")
        summary.append(f"  Volatility (Ann):  {pm.get('volatility', 0):.2f}%")
        summary.append(f"  Calmar Ratio:      {pm.get('calmar_ratio', 0):.3f}")
        summary.append("")
        
        summary.append("TRADE STATISTICS:")
        summary.append(f"  Total Trades:      {ts.get('total_trades', 0)}")
        summary.append(f"  Winning Trades:    {ts.get('winning_trades', 0)}")
        summary.append(f"  Win Rate:          {ts.get('win_rate', 0):.2f}%")
        summary.append(f"  Avg Win:           {ts.get('avg_win', 0):.2f}%")
        summary.append(f"  Avg Loss:          {ts.get('avg_loss', 0):.2f}%")
        summary.append(f"  Profit Factor:     {ts.get('profit_factor', 0):.3f}")
        summary.append(f"  Avg Hold Time:     {ts.get('avg_hold_time', 0):.1f} days")
        summary.append("")
        
        if bc:
            summary.append("BENCHMARK COMPARISON:")
            for bench_name, bench_stats in bc.items():
                summary.append(f"  vs {bench_name}:")
                summary.append(f"    Benchmark Return:  {bench_stats.get('benchmark_return', 0):.2f}%")
                summary.append(f"    Outperformance:    {bench_stats.get('outperformance', 0):.2f}%")
                summary.append(f"    Alpha:             {bench_stats.get('alpha', 0):.2f}%")
                summary.append(f"    Beta:              {bench_stats.get('beta', 0):.3f}")
        
        summary.append("")
        summary.append("=" * 80)
        summary.append("")
        
        return "\n".join(summary)
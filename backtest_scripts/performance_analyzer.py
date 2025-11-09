# backtest_scripts2/performance_analyzer.py
"""
Performance Analyzer for Backtesting
Calculates metrics, generates reports, regime breakdown

UPDATED: 2025-11-09 - Added sector analysis, hold time analysis, regime transitions
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import date

class PerformanceAnalyzer:
    """
    Analyze backtest performance
    Calculate metrics, regime breakdown, generate reports
    """
    
    def __init__(self, constituents: Optional[pd.DataFrame] = None):
        """
        Initialize performance analyzer
        
        Args:
            constituents: Optional DataFrame with ticker→sector mapping (Ticker, Sector columns)
        """
        self.constituents = constituents
        print(f"[PerformanceAnalyzer] Initialized")
        if constituents is not None:
            print(f"  Sector data available: {len(constituents)} tickers")
    
    def analyze(self,
                portfolio_equity_curve: pd.DataFrame,
                trades_history: pd.DataFrame,
                regime_history: pd.DataFrame,
                benchmark_prices: Dict[str, pd.DataFrame]) -> Dict:
        """
        Full performance analysis
        
        Args:
            portfolio_equity_curve: Daily portfolio values
            trades_history: All executed trades
            regime_history: Daily regime data
            benchmark_prices: Dict mapping benchmark → price DataFrame
        
        Returns:
            dict with comprehensive analysis
        """
        analysis = {}
        
        # 1. Portfolio metrics (EXISTING)
        analysis['portfolio_metrics'] = self._calculate_portfolio_metrics(portfolio_equity_curve)
        
        # 2. Trade statistics (EXISTING)
        analysis['trade_stats'] = self._calculate_trade_stats(trades_history)
        
        # 3. Regime breakdown (EXISTING)
        analysis['regime_breakdown'] = self._calculate_regime_breakdown(
            portfolio_equity_curve, trades_history, regime_history
        )
        
        # 4. Benchmark comparison (EXISTING)
        analysis['benchmark_comparison'] = self._compare_benchmarks(
            portfolio_equity_curve, benchmark_prices
        )
        
        # 5. Monthly/Yearly returns (EXISTING)
        analysis['monthly_returns'] = self._calculate_monthly_returns(portfolio_equity_curve)
        analysis['yearly_returns'] = self._calculate_yearly_returns(portfolio_equity_curve)
        
        # 6. Risk metrics (EXISTING)
        analysis['risk_metrics'] = self._calculate_risk_metrics(portfolio_equity_curve)
        
        # === NEW ANALYSES (only if constituents available) ===
        if self.constituents is not None and not trades_history.empty:
            print("[PerformanceAnalyzer] Running enhanced analyses (sector, hold time, transitions)...")
            
            analysis['sector_breakdown'] = self._calculate_sector_breakdown(
                trades_history, regime_history
            )
            
            analysis['hold_time_analysis'] = self._calculate_hold_time_analysis(
                trades_history, regime_history
            )
            
            analysis['regime_transitions'] = self._calculate_regime_transitions(
                regime_history, portfolio_equity_curve
            )
        
        return analysis
    
    def _calculate_portfolio_metrics(self, equity_curve: pd.DataFrame) -> Dict:
        """Calculate portfolio-level metrics"""
        if equity_curve.empty:
            return {}
        
        initial = equity_curve['total_value'].iloc[0]
        final = equity_curve['total_value'].iloc[-1]
        
        # Total return
        total_return = ((final - initial) / initial) * 100
        
        # Daily returns
        equity_curve = equity_curve.copy()
        equity_curve['daily_return'] = equity_curve['total_value'].pct_change()
        daily_returns = equity_curve['daily_return'].dropna()
        
        # Annualized return
        days = len(equity_curve)
        years = days / 252
        annual_return = ((final / initial) ** (1 / years) - 1) * 100 if years > 0 else 0.0
        
        # Sharpe ratio (annualized)
        if len(daily_returns) > 0 and daily_returns.std() > 0:
            sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
        else:
            sharpe = 0.0
        
        # Sortino ratio (annualized, downside deviation)
        negative_returns = daily_returns[daily_returns < 0]
        if len(negative_returns) > 0 and negative_returns.std() > 0:
            sortino = (daily_returns.mean() / negative_returns.std()) * np.sqrt(252)
        else:
            sortino = 0.0
        
        # Max drawdown
        cumulative = (1 + daily_returns.fillna(0)).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() * 100
        
        # Max drawdown duration (days)
        in_drawdown = drawdown < 0
        drawdown_periods = []
        current_period = 0
        for val in in_drawdown:
            if val:
                current_period += 1
            else:
                if current_period > 0:
                    drawdown_periods.append(current_period)
                current_period = 0
        if current_period > 0:
            drawdown_periods.append(current_period)
        
        max_dd_duration = max(drawdown_periods) if drawdown_periods else 0
        
        # Volatility (annualized)
        volatility = daily_returns.std() * np.sqrt(252) * 100
        
        # Calmar ratio (annual return / max drawdown)
        calmar = abs(annual_return / max_drawdown) if max_drawdown != 0 else 0.0
        
        return {
            'initial_value': initial,
            'final_value': final,
            'total_return_pct': total_return,
            'annual_return_pct': annual_return,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown_pct': max_drawdown,
            'max_dd_duration_days': max_dd_duration,
            'volatility_annual_pct': volatility,
            'calmar_ratio': calmar,
            'total_days': days,
            'total_years': years,
        }
    
    def _calculate_trade_stats(self, trades: pd.DataFrame) -> Dict:
        """Calculate trade-level statistics"""
        if trades.empty:
            return {}
        
        sell_trades = trades[trades['action'] == 'SELL'].copy()
        
        if sell_trades.empty:
            return {'total_trades': 0}
        
        total_trades = len(sell_trades)
        winning_trades = len(sell_trades[sell_trades['pl'] > 0])
        losing_trades = total_trades - winning_trades
        
        win_rate = (winning_trades / total_trades) * 100
        
        wins = sell_trades[sell_trades['pl'] > 0]
        losses = sell_trades[sell_trades['pl'] < 0]
        
        avg_win_pct = wins['pl_pct'].mean() if not wins.empty else 0.0
        avg_loss_pct = losses['pl_pct'].mean() if not losses.empty else 0.0
        
        max_win_pct = wins['pl_pct'].max() if not wins.empty else 0.0
        max_loss_pct = losses['pl_pct'].min() if not losses.empty else 0.0
        
        avg_hold_days = sell_trades['hold_days'].mean()
        
        # Profit factor
        total_wins = wins['pl'].sum() if not wins.empty else 0.0
        total_losses = abs(losses['pl'].sum()) if not losses.empty else 0.0
        profit_factor = total_wins / total_losses if total_losses > 0 else 0.0
        
        # Expectancy (average P/L per trade)
        expectancy = sell_trades['pl'].mean()
        
        # Exit reason breakdown
        exit_reasons = sell_trades['reason'].value_counts().to_dict()
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate_pct': win_rate,
            'avg_win_pct': avg_win_pct,
            'avg_loss_pct': avg_loss_pct,
            'max_win_pct': max_win_pct,
            'max_loss_pct': max_loss_pct,
            'avg_hold_days': avg_hold_days,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'exit_reasons': exit_reasons,
        }
    
    def _calculate_regime_breakdown(self,
                                    equity_curve: pd.DataFrame,
                                    trades: pd.DataFrame,
                                    regime_history: pd.DataFrame) -> pd.DataFrame:
        """Calculate performance breakdown by regime"""
        if equity_curve.empty or regime_history.empty:
            return pd.DataFrame()
        
        # Merge equity curve with regime
        merged = equity_curve.merge(regime_history[['date', 'regime']], on='date', how='left')
        merged['daily_return'] = merged['total_value'].pct_change()
        
        # Group by regime
        regime_stats = []
        
        for regime in merged['regime'].dropna().unique():
            regime_data = merged[merged['regime'] == regime]
            regime_trades = trades[trades['action'] == 'SELL']
            
            # Match trades to regime by date
            regime_trade_dates = regime_data['date'].values
            regime_trades_filtered = regime_trades[regime_trades['date'].isin(regime_trade_dates)]
            
            days = len(regime_data)
            total_return = ((regime_data['total_value'].iloc[-1] / regime_data['total_value'].iloc[0]) - 1) * 100 if days > 0 else 0.0
            
            daily_ret = regime_data['daily_return'].dropna()
            avg_daily_ret = daily_ret.mean() * 100
            sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252) if daily_ret.std() > 0 else 0.0
            
            # Max drawdown in this regime
            cumulative = (1 + daily_ret.fillna(0)).cumprod()
            running_max = cumulative.cummax()
            drawdown = (cumulative - running_max) / running_max
            max_dd = drawdown.min() * 100
            
            # Trade stats in this regime
            if not regime_trades_filtered.empty:
                trades_count = len(regime_trades_filtered)
                win_rate = (len(regime_trades_filtered[regime_trades_filtered['pl'] > 0]) / trades_count) * 100
            else:
                trades_count = 0
                win_rate = 0.0
            
            regime_stats.append({
                'regime': regime,
                'days': days,
                'total_return_pct': total_return,
                'avg_daily_return_pct': avg_daily_ret,
                'sharpe_ratio': sharpe,
                'max_drawdown_pct': max_dd,
                'trades_count': trades_count,
                'win_rate_pct': win_rate,
            })
        
        return pd.DataFrame(regime_stats).sort_values('total_return_pct', ascending=False)
    
    def _compare_benchmarks(self,
                           equity_curve: pd.DataFrame,
                           benchmark_prices: Dict[str, pd.DataFrame]) -> Dict:
        """Compare portfolio to benchmarks"""
        comparison = {}
        
        if equity_curve.empty:
            return comparison
        
        start_date = equity_curve['date'].iloc[0]
        end_date = equity_curve['date'].iloc[-1]
        
        portfolio_return = ((equity_curve['total_value'].iloc[-1] / equity_curve['total_value'].iloc[0]) - 1) * 100
        
        for name, prices in benchmark_prices.items():
            # Filter to same date range
            bench = prices[(prices['date'] >= start_date) & (prices['date'] <= end_date)].copy()
            
            if bench.empty or len(bench) < 2:
                continue
            
            bench_return = ((bench['close'].iloc[-1] / bench['close'].iloc[0]) - 1) * 100
            
            # Align dates with portfolio
            bench['date'] = pd.to_datetime(bench['date']).dt.date
            equity_curve_copy = equity_curve.copy()
            equity_curve_copy['date'] = pd.to_datetime(equity_curve_copy['date']).dt.date
            
            merged = equity_curve_copy.merge(bench[['date', 'close']], on='date', how='inner', suffixes=('', '_bench'))
            
            if len(merged) > 1:
                merged['port_return'] = merged['total_value'].pct_change()
                merged['bench_return'] = merged['close'].pct_change()
                
                # Correlation
                corr = merged[['port_return', 'bench_return']].corr().iloc[0, 1]
                
                # Beta
                cov = merged[['port_return', 'bench_return']].cov().iloc[0, 1]
                bench_var = merged['bench_return'].var()
                beta = cov / bench_var if bench_var > 0 else 0.0
                
                # Alpha (annualized excess return)
                port_annual = ((merged['total_value'].iloc[-1] / merged['total_value'].iloc[0]) ** (252 / len(merged)) - 1) * 100
                bench_annual = ((merged['close'].iloc[-1] / merged['close'].iloc[0]) ** (252 / len(merged)) - 1) * 100
                alpha = port_annual - (beta * bench_annual)
            else:
                corr = 0.0
                beta = 0.0
                alpha = 0.0
            
            comparison[name] = {
                'benchmark_return_pct': bench_return,
                'portfolio_return_pct': portfolio_return,
                'outperformance_pct': portfolio_return - bench_return,
                'correlation': corr,
                'beta': beta,
                'alpha': alpha,
            }
        
        return comparison
    
    def _calculate_monthly_returns(self, equity_curve: pd.DataFrame) -> pd.DataFrame:
        """Calculate monthly returns"""
        if equity_curve.empty:
            return pd.DataFrame()
        
        ec = equity_curve.copy()
        ec['date'] = pd.to_datetime(ec['date'])
        ec['year'] = ec['date'].dt.year
        ec['month'] = ec['date'].dt.month
        
        # Get first and last value of each month
        monthly = ec.groupby(['year', 'month']).agg({
            'total_value': ['first', 'last']
        }).reset_index()
        
        monthly.columns = ['year', 'month', 'start_value', 'end_value']
        monthly['return_pct'] = ((monthly['end_value'] / monthly['start_value']) - 1) * 100
        
        return monthly[['year', 'month', 'return_pct']]
    
    def _calculate_yearly_returns(self, equity_curve: pd.DataFrame) -> pd.DataFrame:
        """Calculate yearly returns"""
        if equity_curve.empty:
            return pd.DataFrame()
        
        ec = equity_curve.copy()
        ec['date'] = pd.to_datetime(ec['date'])
        ec['year'] = ec['date'].dt.year
        
        yearly = ec.groupby('year').agg({
            'total_value': ['first', 'last']
        }).reset_index()
        
        yearly.columns = ['year', 'start_value', 'end_value']
        yearly['return_pct'] = ((yearly['end_value'] / yearly['start_value']) - 1) * 100
        
        return yearly[['year', 'return_pct']]
    
    def _calculate_risk_metrics(self, equity_curve: pd.DataFrame) -> Dict:
        """Calculate additional risk metrics"""
        if equity_curve.empty:
            return {}
        
        ec = equity_curve.copy()
        ec['daily_return'] = ec['total_value'].pct_change()
        returns = ec['daily_return'].dropna()
        
        # Value at Risk (95% and 99%)
        var_95 = np.percentile(returns, 5) * 100
        var_99 = np.percentile(returns, 1) * 100
        
        # Conditional VaR (expected shortfall)
        cvar_95 = returns[returns <= np.percentile(returns, 5)].mean() * 100
        cvar_99 = returns[returns <= np.percentile(returns, 1)].mean() * 100
        
        # Skewness and Kurtosis
        skew = returns.skew()
        kurt = returns.kurtosis()
        
        return {
            'var_95_pct': var_95,
            'var_99_pct': var_99,
            'cvar_95_pct': cvar_95,
            'cvar_99_pct': cvar_99,
            'skewness': skew,
            'kurtosis': kurt,
        }
    
    # =====================================================================
    # NEW ANALYSIS METHODS - Added 2025-11-09
    # =====================================================================
    
    def _calculate_sector_breakdown(self, 
                                    trades: pd.DataFrame,
                                    regime_history: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate performance breakdown by sector and regime
        
        Args:
            trades: Trade history
            regime_history: Regime history
        
        Returns:
            DataFrame with sector performance by regime
        """
        if self.constituents is None:
            return pd.DataFrame()
        
        # Merge trades with sector info
        trades_with_sector = trades.merge(
            self.constituents[['Ticker', 'Sector']],
            left_on='ticker',
            right_on='Ticker',
            how='left'
        )
        
        # Filter to completed trades only
        sell_trades = trades_with_sector[trades_with_sector['action'] == 'SELL'].copy()
        
        if sell_trades.empty:
            return pd.DataFrame()
        
        # Merge with regime at trade date
        sell_trades['date'] = pd.to_datetime(sell_trades['date']).dt.date
        regime_history_copy = regime_history.copy()
        regime_history_copy['date'] = pd.to_datetime(regime_history_copy['date']).dt.date
        
        trades_with_regime = sell_trades.merge(
            regime_history_copy[['date', 'regime']],
            on='date',
            how='left'
        )
        
        # Group by sector + regime
        sector_stats = []
        
        for (sector, regime), group in trades_with_regime.groupby(['Sector', 'regime']):
            if pd.isna(sector):
                sector = 'Unknown'
            
            total_trades = len(group)
            winning_trades = len(group[group['pl'] > 0])
            win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0.0
            
            avg_pl_pct = group['pl_pct'].mean()
            total_pl = group['pl'].sum()
            
            sector_stats.append({
                'sector': sector,
                'regime': regime,
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'win_rate_pct': win_rate,
                'avg_pl_pct': avg_pl_pct,
                'total_pl': total_pl,
            })
        
        df = pd.DataFrame(sector_stats)
        
        if not df.empty:
            df = df.sort_values(['regime', 'avg_pl_pct'], ascending=[True, False])
        
        return df
    
    def _calculate_hold_time_analysis(self,
                                      trades: pd.DataFrame,
                                      regime_history: pd.DataFrame) -> Dict:
        """
        Analyze optimal hold time per regime
        
        Args:
            trades: Trade history
            regime_history: Regime history
        
        Returns:
            Dict with hold time statistics
        """
        sell_trades = trades[trades['action'] == 'SELL'].copy()
        
        if sell_trades.empty:
            return {}
        
        # Merge with regime
        sell_trades['date'] = pd.to_datetime(sell_trades['date']).dt.date
        regime_history_copy = regime_history.copy()
        regime_history_copy['date'] = pd.to_datetime(regime_history_copy['date']).dt.date
        
        trades_with_regime = sell_trades.merge(
            regime_history_copy[['date', 'regime']],
            on='date',
            how='left'
        )
        
        # Define hold time buckets
        def bucket_hold_time(days):
            if days <= 3:
                return '1-3d'
            elif days <= 7:
                return '4-7d'
            elif days <= 14:
                return '8-14d'
            elif days <= 30:
                return '15-30d'
            else:
                return '30d+'
        
        trades_with_regime['hold_bucket'] = trades_with_regime['hold_days'].apply(bucket_hold_time)
        
        # Calculate stats per regime per bucket
        hold_time_stats = []
        
        for (regime, bucket), group in trades_with_regime.groupby(['regime', 'hold_bucket']):
            if pd.isna(regime):
                continue
            
            total_trades = len(group)
            avg_pl_pct = group['pl_pct'].mean()
            win_rate = (len(group[group['pl'] > 0]) / total_trades) * 100
            
            hold_time_stats.append({
                'regime': regime,
                'hold_bucket': bucket,
                'total_trades': total_trades,
                'avg_pl_pct': avg_pl_pct,
                'win_rate_pct': win_rate,
            })
        
        df_hold_time = pd.DataFrame(hold_time_stats)
        
        # Find optimal hold time per regime (highest avg P/L)
        optimal_per_regime = {}
        
        if not df_hold_time.empty:
            for regime in df_hold_time['regime'].unique():
                regime_data = df_hold_time[df_hold_time['regime'] == regime]
                best_bucket = regime_data.loc[regime_data['avg_pl_pct'].idxmax()]
                optimal_per_regime[regime] = {
                    'optimal_hold_bucket': best_bucket['hold_bucket'],
                    'avg_pl_pct': best_bucket['avg_pl_pct'],
                    'win_rate_pct': best_bucket['win_rate_pct'],
                }
        
        return {
            'hold_time_by_regime_bucket': df_hold_time.to_dict('records') if not df_hold_time.empty else [],
            'optimal_per_regime': optimal_per_regime,
        }
    
    def _calculate_regime_transitions(self,
                                      regime_history: pd.DataFrame,
                                      equity_curve: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze performance around regime transitions
        
        Args:
            regime_history: Daily regime data
            equity_curve: Portfolio equity curve
        
        Returns:
            DataFrame with transition analysis
        """
        if regime_history.empty or equity_curve.empty:
            return pd.DataFrame()
        
        regime_history = regime_history.copy()
        regime_history['date'] = pd.to_datetime(regime_history['date']).dt.date
        
        # Detect regime changes
        regime_history['prev_regime'] = regime_history['regime'].shift(1)
        transitions = regime_history[regime_history['regime'] != regime_history['prev_regime']].copy()
        
        if len(transitions) < 2:
            return pd.DataFrame()
        
        # Merge with equity curve
        equity_curve_copy = equity_curve.copy()
        equity_curve_copy['date'] = pd.to_datetime(equity_curve_copy['date']).dt.date
        equity_curve_copy['daily_return'] = equity_curve_copy['total_value'].pct_change()
        
        transition_stats = []
        
        for idx, row in transitions.iterrows():
            transition_date = row['date']
            from_regime = row['prev_regime']
            to_regime = row['regime']
            
            # Get 5 days before transition
            before_data = equity_curve_copy[
                (equity_curve_copy['date'] < transition_date)
            ].tail(5)
            
            # Get 5 days after transition
            after_data = equity_curve_copy[
                (equity_curve_copy['date'] >= transition_date)
            ].head(5)
            
            if len(before_data) >= 3 and len(after_data) >= 3:
                pl_before_5d = ((before_data['total_value'].iloc[-1] / before_data['total_value'].iloc[0]) - 1) * 100
                pl_after_5d = ((after_data['total_value'].iloc[-1] / after_data['total_value'].iloc[0]) - 1) * 100
                
                transition_stats.append({
                    'date': transition_date,
                    'from_regime': from_regime,
                    'to_regime': to_regime,
                    'pl_before_5d_pct': pl_before_5d,
                    'pl_after_5d_pct': pl_after_5d,
                    'pl_delta_pct': pl_after_5d - pl_before_5d,
                })
        
        df = pd.DataFrame(transition_stats)
        
        if not df.empty:
            df = df.sort_values('pl_delta_pct', ascending=False)
        
        return df
    
    def generate_summary_text(self, analysis: Dict) -> str:
        """Generate text summary"""
        lines = []
        lines.append("=" * 80)
        lines.append("BACKTEST PERFORMANCE SUMMARY")
        lines.append("=" * 80)
        lines.append("")
        
        # Portfolio metrics
        pm = analysis.get('portfolio_metrics', {})
        lines.append("PORTFOLIO PERFORMANCE:")
        lines.append(f"  Initial Value:     ${pm.get('initial_value', 0):,.2f}")
        lines.append(f"  Final Value:       ${pm.get('final_value', 0):,.2f}")
        lines.append(f"  Total Return:      {pm.get('total_return_pct', 0):.2f}%")
        lines.append(f"  Annual Return:     {pm.get('annual_return_pct', 0):.2f}%")
        lines.append(f"  Sharpe Ratio:      {pm.get('sharpe_ratio', 0):.3f}")
        lines.append(f"  Sortino Ratio:     {pm.get('sortino_ratio', 0):.3f}")
        lines.append(f"  Max Drawdown:      {pm.get('max_drawdown_pct', 0):.2f}%")
        lines.append(f"  Volatility (Ann):  {pm.get('volatility_annual_pct', 0):.2f}%")
        lines.append(f"  Calmar Ratio:      {pm.get('calmar_ratio', 0):.3f}")
        lines.append("")
        
        # Trade stats
        ts = analysis.get('trade_stats', {})
        lines.append("TRADE STATISTICS:")
        lines.append(f"  Total Trades:      {ts.get('total_trades', 0)}")
        lines.append(f"  Winning Trades:    {ts.get('winning_trades', 0)}")
        lines.append(f"  Win Rate:          {ts.get('win_rate_pct', 0):.2f}%")
        lines.append(f"  Avg Win:           {ts.get('avg_win_pct', 0):.2f}%")
        lines.append(f"  Avg Loss:          {ts.get('avg_loss_pct', 0):.2f}%")
        lines.append(f"  Profit Factor:     {ts.get('profit_factor', 0):.3f}")
        lines.append(f"  Avg Hold Time:     {ts.get('avg_hold_days', 0):.1f} days")
        lines.append("")
        
        # Benchmark comparison
        bc = analysis.get('benchmark_comparison', {})
        if bc:
            lines.append("BENCHMARK COMPARISON:")
            for name, metrics in bc.items():
                lines.append(f"  vs {name}:")
                lines.append(f"    Benchmark Return:  {metrics.get('benchmark_return_pct', 0):.2f}%")
                lines.append(f"    Outperformance:    {metrics.get('outperformance_pct', 0):.2f}%")
                lines.append(f"    Alpha:             {metrics.get('alpha', 0):.2f}%")
                lines.append(f"    Beta:              {metrics.get('beta', 0):.3f}")
            lines.append("")
        
        lines.append("=" * 80)
        
        return "\n".join(lines)
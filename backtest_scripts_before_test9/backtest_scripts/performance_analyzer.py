"""
Performance Analyzer - Calculates backtest metrics (FULL VERSION)
Analyzes returns, risk, and regime-specific performance
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class PerformanceAnalyzer:
    """
    Analyzes backtest performance and generates comprehensive metrics.
    
    Metrics calculated:
    - Returns (total, CAGR, monthly, yearly, rolling)
    - Risk (volatility, Sharpe, Sortino, Calmar, max DD, recovery time)
    - Trade statistics (win rate, profit factor, expectancy, etc.)
    - Regime-specific performance
    - Sector breakdown
    - Time-based analysis
    - Benchmark comparison
    """
    
    def __init__(self, results: Dict):
        """Initialize with backtest results."""
        self.equity_curve = results['equity_curve']
        self.trades_history = results['trades_history']
        self.regime_history = results.get('regime_history', pd.DataFrame())
        self.daily_summaries = results.get('daily_summaries', pd.DataFrame())
        
    def calculate_returns_metrics(self) -> Dict:
        """Calculate comprehensive return-based metrics."""
        if self.equity_curve.empty:
            return {}
        
        initial = self.equity_curve['total_value'].iloc[0]
        final = self.equity_curve['total_value'].iloc[-1]
        
        # Total return
        total_return = (final / initial - 1) * 100
        
        # Date calculations
        start_date = self.equity_curve['date'].iloc[0]
        end_date = self.equity_curve['date'].iloc[-1]
        days = (end_date - start_date).days
        years = days / 365.25
        months = days / 30.44
        
        # CAGR
        cagr = ((final / initial) ** (1 / years) - 1) * 100 if years > 0 else 0
        
        # Monthly return (annualized)
        monthly_return = ((final / initial) ** (1 / months) - 1) * 12 * 100 if months > 0 else 0
        
        # Best/worst day
        self.equity_curve['daily_return'] = self.equity_curve['total_value'].pct_change()
        best_day = self.equity_curve['daily_return'].max() * 100 if len(self.equity_curve) > 1 else 0
        worst_day = self.equity_curve['daily_return'].min() * 100 if len(self.equity_curve) > 1 else 0
        
        # Best/worst month
        monthly_returns = self.calculate_monthly_returns()
        if not monthly_returns.empty:
            best_month = monthly_returns['return_pct'].max()
            worst_month = monthly_returns['return_pct'].min()
        else:
            best_month = 0
            worst_month = 0
        
        # Profit days
        positive_days = (self.equity_curve['daily_return'] > 0).sum()
        total_days = len(self.equity_curve) - 1
        profit_days_pct = (positive_days / total_days * 100) if total_days > 0 else 0
        
        return {
            'initial_value': initial,
            'final_value': final,
            'total_return': total_return,
            'cagr': cagr,
            'monthly_return_ann': monthly_return,
            'years': years,
            'days': days,
            'best_day': best_day,
            'worst_day': worst_day,
            'best_month': best_month,
            'worst_month': worst_month,
            'profit_days_pct': profit_days_pct,
        }
    
    def calculate_risk_metrics(self) -> Dict:
        """Calculate comprehensive risk-adjusted metrics."""
        if len(self.equity_curve) < 2:
            return {}
        
        # Calculate daily returns
        if 'daily_return' not in self.equity_curve.columns:
            self.equity_curve['daily_return'] = self.equity_curve['total_value'].pct_change()
        
        returns = self.equity_curve['daily_return'].dropna()
        
        # Volatility (annualized)
        volatility = returns.std() * np.sqrt(252) * 100
        
        # Sharpe ratio (assume 0% risk-free rate)
        sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
        
        # Sortino ratio (downside deviation only)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else returns.std()
        sortino = (returns.mean() / downside_std * np.sqrt(252)) if downside_std > 0 else 0
        
        # Max drawdown and related metrics
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        
        max_dd = drawdown.min() * 100
        max_dd_date = self.equity_curve.loc[drawdown.idxmin(), 'date'] if len(drawdown) > 0 else None
        
        # Drawdown duration
        dd_series = drawdown < -0.01  # Consider DD > 1%
        if dd_series.any():
            dd_lengths = []
            current_length = 0
            for is_dd in dd_series:
                if is_dd:
                    current_length += 1
                else:
                    if current_length > 0:
                        dd_lengths.append(current_length)
                    current_length = 0
            if current_length > 0:
                dd_lengths.append(current_length)
            
            max_dd_duration = max(dd_lengths) if dd_lengths else 0
            avg_dd_duration = np.mean(dd_lengths) if dd_lengths else 0
        else:
            max_dd_duration = 0
            avg_dd_duration = 0
        
        # Calmar ratio (CAGR / abs(max_dd))
        returns_metrics = self.calculate_returns_metrics()
        cagr = returns_metrics.get('cagr', 0)
        calmar = abs(cagr / max_dd) if max_dd != 0 else 0
        
        # Ulcer Index (measure of downside volatility)
        ulcer_index = np.sqrt(np.mean(drawdown ** 2)) * 100
        
        # Recovery factor (net profit / max DD)
        net_profit = returns_metrics.get('final_value', 0) - returns_metrics.get('initial_value', 0)
        max_dd_dollars = abs(max_dd / 100 * returns_metrics.get('initial_value', 1))
        recovery_factor = net_profit / max_dd_dollars if max_dd_dollars > 0 else 0
        
        # Value at Risk (VaR) - 95% and 99%
        var_95 = np.percentile(returns, 5) * 100
        var_99 = np.percentile(returns, 1) * 100
        
        # Conditional Value at Risk (CVaR) - average of worst 5%
        cvar_95 = returns[returns <= np.percentile(returns, 5)].mean() * 100
        
        # Skewness and Kurtosis
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        
        return {
            'volatility': volatility,
            'sharpe': sharpe,
            'sortino': sortino,
            'max_drawdown': max_dd,
            'max_dd_date': max_dd_date,
            'max_dd_duration': max_dd_duration,
            'avg_dd_duration': avg_dd_duration,
            'calmar': calmar,
            'ulcer_index': ulcer_index,
            'recovery_factor': recovery_factor,
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'skewness': skewness,
            'kurtosis': kurtosis,
        }
    
    def calculate_trade_statistics(self) -> Dict:
        """Calculate comprehensive trade-level statistics."""
        if self.trades_history.empty:
            return {}
        
        # Filter sell trades only
        sells = self.trades_history[self.trades_history['action'] == 'SELL'].copy()
        
        if sells.empty:
            return {}
        
        # Win/loss statistics
        wins = sells[sells['pnl'] > 0]
        losses = sells[sells['pnl'] < 0]
        breakeven = sells[sells['pnl'] == 0]
        
        total_trades = len(sells)
        winning_trades = len(wins)
        losing_trades = len(losses)
        breakeven_trades = len(breakeven)
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # Average win/loss (percentage)
        avg_win = wins['pnl_pct'].mean() if len(wins) > 0 else 0
        avg_loss = losses['pnl_pct'].mean() if len(losses) > 0 else 0
        
        # Average win/loss (dollars)
        avg_win_dollars = wins['pnl'].mean() if len(wins) > 0 else 0
        avg_loss_dollars = losses['pnl'].mean() if len(losses) > 0 else 0
        
        # Largest win/loss
        largest_win = wins['pnl_pct'].max() if len(wins) > 0 else 0
        largest_loss = losses['pnl_pct'].min() if len(losses) > 0 else 0
        
        largest_win_dollars = wins['pnl'].max() if len(wins) > 0 else 0
        largest_loss_dollars = losses['pnl'].min() if len(losses) > 0 else 0
        
        # Profit factor
        total_wins = wins['pnl'].sum() if len(wins) > 0 else 0
        total_losses = abs(losses['pnl'].sum()) if len(losses) > 0 else 1
        profit_factor = total_wins / total_losses if total_losses > 0 else 0
        
        # Expectancy (average P&L per trade)
        expectancy = sells['pnl'].mean()
        expectancy_pct = sells['pnl_pct'].mean()
        
        # Hold time statistics
        if 'days_held' in sells.columns:
            avg_hold_time = sells['days_held'].mean()
            avg_hold_time_wins = wins['days_held'].mean() if len(wins) > 0 else 0
            avg_hold_time_losses = losses['days_held'].mean() if len(losses) > 0 else 0
            max_hold_time = sells['days_held'].max()
            min_hold_time = sells['days_held'].min()
        else:
            avg_hold_time = 0
            avg_hold_time_wins = 0
            avg_hold_time_losses = 0
            max_hold_time = 0
            min_hold_time = 0
        
        # Consecutive wins/losses
        win_loss_series = (sells['pnl'] > 0).astype(int)
        consecutive_wins = self._max_consecutive(win_loss_series, 1)
        consecutive_losses = self._max_consecutive(win_loss_series, 0)
        
        # Exit reasons breakdown
        if 'reason' in sells.columns:
            exit_reasons = sells['reason'].value_counts().to_dict()
            stop_loss_count = exit_reasons.get('STOP_LOSS', 0)
            take_profit_count = exit_reasons.get('TAKE_PROFIT', 0)
            time_exit_count = exit_reasons.get('TIME_EXIT', 0)
            regime_exit_count = exit_reasons.get('REGIME_EXIT', 0)
        else:
            exit_reasons = {}
            stop_loss_count = 0
            take_profit_count = 0
            time_exit_count = 0
            regime_exit_count = 0
        
        # Win/loss streaks
        current_streak = 0
        max_win_streak = 0
        max_loss_streak = 0
        
        for result in win_loss_series:
            if result == 1:
                if current_streak >= 0:
                    current_streak += 1
                else:
                    current_streak = 1
                max_win_streak = max(max_win_streak, current_streak)
            else:
                if current_streak <= 0:
                    current_streak -= 1
                else:
                    current_streak = -1
                max_loss_streak = max(max_loss_streak, abs(current_streak))
        
        # Kelly Criterion (for position sizing)
        if avg_loss != 0:
            kelly_pct = (win_rate/100 - (1-win_rate/100) / abs(avg_win/avg_loss)) * 100
        else:
            kelly_pct = 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'breakeven_trades': breakeven_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_win_dollars': avg_win_dollars,
            'avg_loss_dollars': avg_loss_dollars,
            'largest_win': largest_win,
            'largest_loss': largest_loss,
            'largest_win_dollars': largest_win_dollars,
            'largest_loss_dollars': largest_loss_dollars,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'expectancy_pct': expectancy_pct,
            'avg_hold_time': avg_hold_time,
            'avg_hold_time_wins': avg_hold_time_wins,
            'avg_hold_time_losses': avg_hold_time_losses,
            'max_hold_time': max_hold_time,
            'min_hold_time': min_hold_time,
            'consecutive_wins': consecutive_wins,
            'consecutive_losses': consecutive_losses,
            'max_win_streak': max_win_streak,
            'max_loss_streak': max_loss_streak,
            'exit_reasons': exit_reasons,
            'stop_loss_exits': stop_loss_count,
            'take_profit_exits': take_profit_count,
            'time_exits': time_exit_count,
            'regime_exits': regime_exit_count,
            'kelly_criterion': kelly_pct,
        }
    
    def _max_consecutive(self, series, value):
        """Helper to find max consecutive occurrences of a value."""
        max_count = 0
        current_count = 0
        
        for v in series:
            if v == value:
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 0
        
        return max_count
    
    def calculate_regime_breakdown(self) -> pd.DataFrame:
        """Calculate detailed performance by regime."""
        if self.trades_history.empty or 'regime' not in self.trades_history.columns:
            return pd.DataFrame()
        
        sells = self.trades_history[self.trades_history['action'] == 'SELL']
        
        if sells.empty:
            return pd.DataFrame()
        
        regime_stats = []
        
        for regime in sells['regime'].unique():
            if pd.isna(regime):
                continue
            
            regime_trades = sells[sells['regime'] == regime]
            wins = regime_trades[regime_trades['pnl'] > 0]
            losses = regime_trades[regime_trades['pnl'] < 0]
            
            # Calculate metrics
            win_rate = len(wins) / len(regime_trades) * 100 if len(regime_trades) > 0 else 0
            avg_return = regime_trades['pnl_pct'].mean()
            total_pnl = regime_trades['pnl'].sum()
            avg_win = wins['pnl_pct'].mean() if len(wins) > 0 else 0
            avg_loss = losses['pnl_pct'].mean() if len(losses) > 0 else 0
            
            # Profit factor
            total_wins = wins['pnl'].sum() if len(wins) > 0 else 0
            total_losses = abs(losses['pnl'].sum()) if len(losses) > 0 else 1
            profit_factor = total_wins / total_losses if total_losses > 0 else 0
            
            regime_stats.append({
                'regime': regime,
                'num_trades': len(regime_trades),
                'winning_trades': len(wins),
                'losing_trades': len(losses),
                'win_rate': win_rate,
                'avg_return': avg_return,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'total_pnl': total_pnl,
                'profit_factor': profit_factor,
                'avg_hold_days': regime_trades['days_held'].mean() if 'days_held' in regime_trades.columns else 0,
            })
        
        df = pd.DataFrame(regime_stats)
        if not df.empty:
            df = df.sort_values('total_pnl', ascending=False)
        
        return df
    
    def calculate_sector_breakdown(self) -> pd.DataFrame:
        """Calculate detailed performance by sector."""
        if self.trades_history.empty or 'sector' not in self.trades_history.columns:
            return pd.DataFrame()
        
        sells = self.trades_history[self.trades_history['action'] == 'SELL']
        
        if sells.empty:
            return pd.DataFrame()
        
        sector_stats = []
        
        for sector in sells['sector'].unique():
            if pd.isna(sector):
                continue
            
            sector_trades = sells[sells['sector'] == sector]
            wins = sector_trades[sector_trades['pnl'] > 0]
            losses = sector_trades[sector_trades['pnl'] < 0]
            
            # Calculate metrics
            win_rate = len(wins) / len(sector_trades) * 100 if len(sector_trades) > 0 else 0
            avg_return_pct = sector_trades['pnl_pct'].mean()
            total_return = sector_trades['pnl'].sum()
            
            # Calculate total return percentage (cumulative)
            cumulative_return = 1.0
            for ret in sector_trades['pnl_pct']:
                cumulative_return *= (1 + ret/100)
            total_return_pct = (cumulative_return - 1) * 100
            
            sector_stats.append({
                'sector': sector,
                'num_trades': len(sector_trades),
                'winning_trades': len(wins),
                'losing_trades': len(losses),
                'win_rate': win_rate,
                'avg_return_pct': avg_return_pct,
                'total_return': total_return,
                'total_return_pct': total_return_pct,
            })
        
        df = pd.DataFrame(sector_stats)
        if not df.empty:
            df = df.sort_values('total_return', ascending=False)
        
        return df
    
    def calculate_monthly_returns(self) -> pd.DataFrame:
        """Calculate monthly returns with statistics."""
        if self.equity_curve.empty:
            return pd.DataFrame()
        
        df = self.equity_curve.copy()
        df['year_month'] = df['date'].dt.to_period('M')
        
        monthly = df.groupby('year_month').agg({
            'total_value': ['first', 'last', 'min', 'max']
        }).reset_index()
        
        monthly.columns = ['year_month', 'start_value', 'end_value', 'min_value', 'max_value']
        monthly['return_pct'] = (monthly['end_value'] / monthly['start_value'] - 1) * 100
        monthly['max_dd_month'] = (monthly['min_value'] / monthly['max_value'] - 1) * 100
        
        return monthly
    
    def calculate_yearly_breakdown(self) -> pd.DataFrame:
        """Calculate detailed yearly performance breakdown."""
        if self.equity_curve.empty:
            return pd.DataFrame()
        
        df = self.equity_curve.copy()
        df['year'] = df['date'].dt.year
        
        # Calculate daily returns if not present
        if 'daily_return' not in df.columns:
            df['daily_return'] = df['total_value'].pct_change()
        
        yearly_stats = []
        
        for year in sorted(df['year'].unique()):
            year_data = df[df['year'] == year].copy()
            
            if len(year_data) < 2:
                continue
            
            # Returns
            start_val = year_data['total_value'].iloc[0]
            end_val = year_data['total_value'].iloc[-1]
            year_return = (end_val / start_val - 1) * 100
            
            # Risk metrics
            year_returns = year_data['daily_return'].dropna()
            
            if len(year_returns) > 0:
                year_vol = year_returns.std() * np.sqrt(252) * 100
                year_sharpe = (year_returns.mean() / year_returns.std() * np.sqrt(252)) if year_returns.std() > 0 else 0
                
                # Max drawdown
                cumulative = (1 + year_returns).cumprod()
                running_max = cumulative.cummax()
                drawdown = (cumulative - running_max) / running_max
                year_max_dd = drawdown.min() * 100
                
                # Best/worst day
                best_day = year_returns.max() * 100
                worst_day = year_returns.min() * 100
            else:
                year_vol = 0
                year_sharpe = 0
                year_max_dd = 0
                best_day = 0
                worst_day = 0
            
            # Trades
            if not self.trades_history.empty and 'date' in self.trades_history.columns:
                year_trades = self.trades_history[
                    pd.to_datetime(self.trades_history['date']).dt.year == year
                ]
                sell_trades = year_trades[year_trades['action'] == 'SELL']
                num_trades = len(sell_trades)
                
                if len(sell_trades) > 0:
                    wins = sell_trades[sell_trades['pnl'] > 0]
                    year_win_rate = len(wins) / len(sell_trades) * 100
                else:
                    year_win_rate = 0
            else:
                num_trades = 0
                year_win_rate = 0
            
            yearly_stats.append({
                'year': int(year),
                'return_pct': year_return,
                'volatility': year_vol,
                'sharpe': year_sharpe,
                'max_dd': year_max_dd,
                'best_day': best_day,
                'worst_day': worst_day,
                'num_trades': num_trades,
                'win_rate': year_win_rate,
            })
        
        return pd.DataFrame(yearly_stats)
    
    def calculate_rolling_metrics(self, window_days: int = 60) -> pd.DataFrame:
        """Calculate rolling performance metrics."""
        if len(self.equity_curve) < window_days:
            return pd.DataFrame()
        
        df = self.equity_curve.copy()
        
        if 'daily_return' not in df.columns:
            df['daily_return'] = df['total_value'].pct_change()
        
        # Rolling Sharpe
        df['rolling_sharpe'] = (
            df['daily_return'].rolling(window_days).mean() /
            df['daily_return'].rolling(window_days).std() * np.sqrt(252)
        )
        
        # Rolling volatility
        df['rolling_vol'] = df['daily_return'].rolling(window_days).std() * np.sqrt(252) * 100
        
        # Rolling return
        df['rolling_return'] = (
            df['total_value'] / df['total_value'].shift(window_days) - 1
        ) * 100
        
        return df[['date', 'rolling_sharpe', 'rolling_vol', 'rolling_return']].dropna()
    
    def analyze(self) -> Dict:
        """Run complete performance analysis with all metrics."""
        logger.info("Running comprehensive performance analysis...")
        
        # Calculate all metrics
        returns = self.calculate_returns_metrics()
        risk = self.calculate_risk_metrics()
        trades = self.calculate_trade_statistics()
        
        # Combine core metrics
        performance = {
            **returns,
            **risk,
            **trades,
        }
        
        # Breakdowns
        performance['regime_breakdown'] = self.calculate_regime_breakdown()
        performance['sector_breakdown'] = self.calculate_sector_breakdown()
        performance['monthly_returns'] = self.calculate_monthly_returns()
        performance['yearly_breakdown'] = self.calculate_yearly_breakdown()
        performance['rolling_metrics'] = self.calculate_rolling_metrics()
        
        logger.info("Performance analysis complete")
        
        # Log key metrics
        logger.info(f"\n{'='*60}")
        logger.info("KEY PERFORMANCE METRICS")
        logger.info(f"{'='*60}")
        logger.info(f"Total Return:    {performance.get('total_return', 0):.2f}%")
        logger.info(f"CAGR:            {performance.get('cagr', 0):.2f}%")
        logger.info(f"Sharpe Ratio:    {performance.get('sharpe', 0):.3f}")
        logger.info(f"Max Drawdown:    {performance.get('max_drawdown', 0):.2f}%")
        logger.info(f"Win Rate:        {performance.get('win_rate', 0):.2f}%")
        logger.info(f"Profit Factor:   {performance.get('profit_factor', 0):.3f}")
        logger.info(f"Total Trades:    {performance.get('total_trades', 0)}")
        logger.info(f"{'='*60}\n")
        
        return performance
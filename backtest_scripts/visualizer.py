# backtest_scripts/visualizer.py
"""
Enhanced Backtest Visualizer with Position Sizing Plots
Creates comprehensive visualizations

UPDATED: 2025-11-12 15:24 UTC
CHANGES:
  - Position sizing over time plot
  - Sector allocation tracking
  - Adaptive sizing impact visualization
  - Enhanced styling
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class BacktestVisualizer:
    """
    Comprehensive visualization suite
    """
    
    def __init__(self, plots_dir: str = "plots"):
        """
        Initialize visualizer
        
        Args:
            plots_dir: Directory to save plots
        """
        self.plots_dir = plots_dir
        os.makedirs(plots_dir, exist_ok=True)
        
        logger.info(f"[Visualizer] Initialized (plots dir: {plots_dir})")
    
    
    def create_all_plots(
        self,
        equity_curve: pd.DataFrame,
        trades_history: pd.DataFrame,
        analysis: Dict
    ):
        """
        Create all visualization plots
        
        Args:
            equity_curve: Equity curve DataFrame
            trades_history: Trades history DataFrame
            analysis: Analysis results dict
        """
        logger.info("\n[Visualizer] Creating plots...")
        
        plots_created = []
        
        try:
            # 1. Equity curve
            self._plot_equity_curve(equity_curve, analysis)
            plots_created.append("equity_curve.png")
            
            # 2. Drawdown
            self._plot_drawdown(equity_curve)
            plots_created.append("drawdown.png")
            
            # 3. Monthly returns heatmap
            if 'monthly_returns' in analysis:
                self._plot_monthly_heatmap(analysis['monthly_returns'])
                plots_created.append("monthly_returns_heatmap.png")
            
            # 4. Regime performance
            if 'regime_breakdown' in analysis:
                self._plot_regime_performance(analysis['regime_breakdown'])
                plots_created.append("regime_performance.png")
            
            # 5. Trade distribution
            if not trades_history.empty:
                self._plot_trade_distribution(trades_history)
                plots_created.append("trade_distribution.png")
            
            # 6. Rolling Sharpe
            if 'rolling_metrics' in analysis:
                self._plot_rolling_sharpe(analysis['rolling_metrics'])
                plots_created.append("rolling_sharpe.png")
            
            # 7. Underwater plot
            self._plot_underwater(equity_curve)
            plots_created.append("underwater.png")
            
            # 8. Trade timeline
            if not trades_history.empty:
                self._plot_trade_timeline(trades_history, equity_curve)
                plots_created.append("trade_timeline.png")
            
            # 9. Yearly breakdown
            if 'yearly_breakdown' in analysis:
                self._plot_yearly_breakdown(analysis['yearly_breakdown'])
                plots_created.append("yearly_breakdown.png")
            
            # 10. Sector heatmap
            if 'sector_breakdown' in analysis:
                self._plot_sector_performance(analysis['sector_breakdown'])
                plots_created.append("sector_performance.png")
            
            # 11. Hold time vs return
            if not trades_history.empty:
                self._plot_hold_time_scatter(trades_history)
                plots_created.append("hold_time_scatter.png")
            
            # 12. Regime transitions
            if 'regime_breakdown' in analysis:
                self._plot_regime_transitions(trades_history)
                plots_created.append("regime_transitions.png")
            
            # 13. NEW: Position sizing over time
            if not trades_history.empty:
                self._plot_position_sizing_over_time(trades_history, equity_curve)
                plots_created.append("position_sizing_over_time.png")
            
            # 14. NEW: Sector allocation over time
            if not trades_history.empty and 'sector' in trades_history.columns:
                self._plot_sector_allocation_over_time(trades_history, equity_curve)
                plots_created.append("sector_allocation.png")
            
            # 15. NEW: Adaptive sizing impact
            if 'position_sizing_analysis' in analysis:
                self._plot_adaptive_sizing_impact(
                    trades_history,
                    equity_curve,
                    analysis['position_sizing_analysis']
                )
                plots_created.append("adaptive_sizing_impact.png")
            
        except Exception as e:
            logger.error(f"Error creating plots: {e}")
        
        # Print summary
        logger.info(f"[Visualizer] ✓ Created {len(plots_created)} plots in {self.plots_dir}/")
        for plot in plots_created:
            logger.info(f"  ✓ {plot}")
    
    
    def _plot_equity_curve(self, equity_curve: pd.DataFrame, analysis: Dict):
        """Plot portfolio equity curve with benchmarks"""
        fig, ax = plt.subplots(figsize=(14, 7))
        
        dates = pd.to_datetime(equity_curve['date'])
        values = equity_curve['total_value']
        
        # Plot portfolio
        ax.plot(dates, values, label='Portfolio', linewidth=2, color='#2E86AB')
        ax.fill_between(dates, values, alpha=0.3, color='#2E86AB')
        
        # Plot benchmarks if available
        if 'benchmark_comparison' in analysis:
            # This would require benchmark data - simplified for now
            pass
        
        # Formatting
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax.set_title('Portfolio Equity Curve', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Format y-axis as currency
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'equity_curve.png'), dpi=150)
        plt.close()
    
    
    def _plot_drawdown(self, equity_curve: pd.DataFrame):
        """Plot drawdown chart"""
        fig, ax = plt.subplots(figsize=(14, 5))
        
        dates = pd.to_datetime(equity_curve['date'])
        
        # Calculate drawdown if not in dataframe
        if 'drawdown' in equity_curve.columns:
            drawdown = equity_curve['drawdown'] * 100
        else:
            values = equity_curve['total_value']
            cummax = values.cummax()
            drawdown = ((values - cummax) / cummax) * 100
        
        # Plot
        ax.fill_between(dates, drawdown, 0, alpha=0.5, color='#A23B72', label='Drawdown')
        ax.plot(dates, drawdown, color='#A23B72', linewidth=1.5)
        
        # Formatting
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Drawdown (%)', fontsize=12)
        ax.set_title('Portfolio Drawdown', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'drawdown.png'), dpi=150)
        plt.close()
    
    
    def _plot_monthly_heatmap(self, monthly_returns: pd.DataFrame):
        """Plot monthly returns as heatmap"""
        if monthly_returns.empty:
            return
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Month names
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        # Create heatmap
        sns.heatmap(
            monthly_returns,
            annot=True,
            fmt='.1f',
            cmap='RdYlGn',
            center=0,
            cbar_kws={'label': 'Return (%)'},
            xticklabels=month_names,
            ax=ax,
            linewidths=0.5
        )
        
        ax.set_title('Monthly Returns Heatmap (%)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Month', fontsize=12)
        ax.set_ylabel('Year', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'monthly_returns_heatmap.png'), dpi=150)
        plt.close()
    
    
    def _plot_regime_performance(self, regime_breakdown: List[Dict]):
        """Plot performance by regime"""
        if not regime_breakdown:
            return
        
        df = pd.DataFrame(regime_breakdown)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Total return by regime
        ax1.bar(df['regime'], df['total_return'], color='#2E86AB', alpha=0.7)
        ax1.set_title('Total Return by Regime', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Regime')
        ax1.set_ylabel('Total Return ($)')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Win rate by regime
        ax2.bar(df['regime'], df['win_rate'], color='#F18F01', alpha=0.7)
        ax2.set_title('Win Rate by Regime', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Regime')
        ax2.set_ylabel('Win Rate (%)')
        ax2.tick_params(axis='x', rotation=45)
        ax2.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50%')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'regime_performance.png'), dpi=150)
        plt.close()
    
    
    def _plot_trade_distribution(self, trades: pd.DataFrame):
        """Plot trade P/L distribution"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram of returns
        returns = trades['pl_pct']
        
        ax1.hist(returns, bins=50, alpha=0.7, color='#2E86AB', edgecolor='black')
        ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Break-even')
        ax1.axvline(x=returns.mean(), color='green', linestyle='--', linewidth=2, label=f'Mean: {returns.mean():.2f}%')
        ax1.set_xlabel('Return (%)', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Trade Return Distribution', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Cumulative P/L
        cumulative_pl = trades.sort_values('exit_date')['pl'].cumsum()
        
        ax2.plot(range(len(cumulative_pl)), cumulative_pl, color='#2E86AB', linewidth=2)
        ax2.fill_between(range(len(cumulative_pl)), cumulative_pl, alpha=0.3, color='#2E86AB')
        ax2.set_xlabel('Trade Number', fontsize=12)
        ax2.set_ylabel('Cumulative P/L ($)', fontsize=12)
        ax2.set_title('Cumulative P/L Over Trades', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'trade_distribution.png'), dpi=150)
        plt.close()
    # ... (jatkoa visualizer.py:lle)
    
    def _plot_rolling_sharpe(self, rolling_metrics: Dict):
        """Plot rolling Sharpe ratio"""
        if 'rolling_sharpe' not in rolling_metrics:
            return
        
        fig, ax = plt.subplots(figsize=(14, 5))
        
        sharpe_values = rolling_metrics['rolling_sharpe']
        
        ax.plot(sharpe_values, color='#2E86AB', linewidth=2)
        ax.axhline(y=1.0, color='orange', linestyle='--', alpha=0.7, label='Sharpe = 1.0')
        ax.axhline(y=2.0, color='green', linestyle='--', alpha=0.7, label='Sharpe = 2.0')
        ax.fill_between(range(len(sharpe_values)), sharpe_values, alpha=0.3, color='#2E86AB')
        
        ax.set_xlabel('Trading Days', fontsize=12)
        ax.set_ylabel('Rolling Sharpe Ratio (252d)', fontsize=12)
        ax.set_title('Rolling 252-Day Sharpe Ratio', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'rolling_sharpe.png'), dpi=150)
        plt.close()
    
    
    def _plot_underwater(self, equity_curve: pd.DataFrame):
        """Plot underwater (drawdown) chart"""
        fig, ax = plt.subplots(figsize=(14, 5))
        
        dates = pd.to_datetime(equity_curve['date'])
        values = equity_curve['total_value']
        
        # Calculate drawdown
        cummax = values.cummax()
        drawdown = ((values - cummax) / cummax) * 100
        
        # Plot
        ax.fill_between(dates, drawdown, 0, where=(drawdown < 0), 
                        alpha=0.5, color='#A23B72', label='Underwater')
        ax.plot(dates, drawdown, color='#A23B72', linewidth=1)
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Drawdown (%)', fontsize=12)
        ax.set_title('Underwater Plot (% off peak)', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'underwater.png'), dpi=150)
        plt.close()
    
    
    def _plot_trade_timeline(self, trades: pd.DataFrame, equity_curve: pd.DataFrame):
        """Plot trades on equity curve"""
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Plot equity curve
        dates = pd.to_datetime(equity_curve['date'])
        values = equity_curve['total_value']
        ax.plot(dates, values, color='#2E86AB', linewidth=2, label='Portfolio Value')
        
        # Plot trades
        trades = trades.copy()
        trades['exit_date'] = pd.to_datetime(trades['exit_date'])
        
        # Winning trades
        wins = trades[trades['pl'] > 0]
        if not wins.empty:
            ax.scatter(wins['exit_date'], 
                      [values[dates.searchsorted(d)] for d in wins['exit_date'] if d in dates.values],
                      color='green', marker='^', s=50, alpha=0.6, label='Wins')
        
        # Losing trades
        losses = trades[trades['pl'] < 0]
        if not losses.empty:
            ax.scatter(losses['exit_date'],
                      [values[dates.searchsorted(d)] for d in losses['exit_date'] if d in dates.values],
                      color='red', marker='v', s=50, alpha=0.6, label='Losses')
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax.set_title('Trade Timeline', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'trade_timeline.png'), dpi=150)
        plt.close()
    
    
    def _plot_yearly_breakdown(self, yearly_breakdown: List[Dict]):
        """Plot yearly performance breakdown"""
        if not yearly_breakdown:
            return
        
        df = pd.DataFrame(yearly_breakdown)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        # Yearly returns
        colors = ['green' if x > 0 else 'red' for x in df['return_pct']]
        ax1.bar(df['year'], df['return_pct'], color=colors, alpha=0.7)
        ax1.set_title('Yearly Returns', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Return (%)')
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Sharpe ratio
        ax2.plot(df['year'], df['sharpe'], marker='o', linewidth=2, color='#2E86AB')
        ax2.set_title('Yearly Sharpe Ratio', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Year')
        ax2.set_ylabel('Sharpe Ratio')
        ax2.axhline(y=1.0, color='orange', linestyle='--', alpha=0.5)
        ax2.axhline(y=2.0, color='green', linestyle='--', alpha=0.5)
        ax2.grid(True, alpha=0.3)
        
        # Max drawdown
        ax3.bar(df['year'], df['max_dd'], color='#A23B72', alpha=0.7)
        ax3.set_title('Yearly Max Drawdown', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Year')
        ax3.set_ylabel('Max Drawdown (%)')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Number of trades
        ax4.bar(df['year'], df['num_trades'], color='#F18F01', alpha=0.7)
        ax4.set_title('Yearly Trade Count', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Year')
        ax4.set_ylabel('Number of Trades')
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'yearly_breakdown.png'), dpi=150)
        plt.close()
    
    
    def _plot_sector_performance(self, sector_breakdown: List[Dict]):
        """Plot sector performance"""
        if not sector_breakdown:
            return
        
        df = pd.DataFrame(sector_breakdown)
        df = df.sort_values('total_return_pct', ascending=True)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
        
        # Total return by sector (horizontal bar)
        colors = ['green' if x > 0 else 'red' for x in df['total_return_pct']]
        ax1.barh(df['sector'], df['total_return_pct'], color=colors, alpha=0.7)
        ax1.set_xlabel('Total Return (%)', fontsize=12)
        ax1.set_title('Sector Performance', fontsize=12, fontweight='bold')
        ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Win rate by sector
        ax2.barh(df['sector'], df['win_rate'], color='#2E86AB', alpha=0.7)
        ax2.set_xlabel('Win Rate (%)', fontsize=12)
        ax2.set_title('Win Rate by Sector', fontsize=12, fontweight='bold')
        ax2.axvline(x=50, color='red', linestyle='--', linewidth=1, alpha=0.5, label='50%')
        ax2.grid(True, alpha=0.3, axis='x')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'sector_performance.png'), dpi=150)
        plt.close()
    
    
    def _plot_hold_time_scatter(self, trades: pd.DataFrame):
        """Scatter plot of hold time vs return"""
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Separate wins and losses
        wins = trades[trades['pl'] > 0]
        losses = trades[trades['pl'] < 0]
        
        # Plot
        ax.scatter(wins['hold_days'], wins['pl_pct'], 
                  alpha=0.5, s=50, color='green', label='Wins')
        ax.scatter(losses['hold_days'], losses['pl_pct'], 
                  alpha=0.5, s=50, color='red', label='Losses')
        
        # Add trend line
        z = np.polyfit(trades['hold_days'], trades['pl_pct'], 1)
        p = np.poly1d(z)
        ax.plot(trades['hold_days'].sort_values(), 
               p(trades['hold_days'].sort_values()), 
               "b--", alpha=0.5, linewidth=2, label='Trend')
        
        ax.set_xlabel('Hold Time (days)', fontsize=12)
        ax.set_ylabel('Return (%)', fontsize=12)
        ax.set_title('Hold Time vs Return', fontsize=14, fontweight='bold')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'hold_time_scatter.png'), dpi=150)
        plt.close()
    
    
    def _plot_regime_transitions(self, trades: pd.DataFrame):
        """Plot regime transitions from trade reasons"""
        if trades.empty or 'reason' not in trades.columns:
            return
        
        # Count exit reasons
        reason_counts = trades['reason'].value_counts().head(10)
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        ax.barh(reason_counts.index, reason_counts.values, color='#2E86AB', alpha=0.7)
        ax.set_xlabel('Count', fontsize=12)
        ax.set_title('Top 10 Exit Reasons', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'regime_transitions.png'), dpi=150)
        plt.close()
    
    
    def _plot_position_sizing_over_time(self, trades: pd.DataFrame, equity_curve: pd.DataFrame):
        """Plot position sizes over time (NEW!)"""
        trades = trades.copy()
        trades['entry_date'] = pd.to_datetime(trades['entry_date'])
        trades['position_size'] = trades['shares'] * trades['entry_price']
        trades = trades.sort_values('entry_date')
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        
        # Portfolio value
        dates = pd.to_datetime(equity_curve['date'])
        values = equity_curve['total_value']
        ax1.plot(dates, values, color='#2E86AB', linewidth=2, label='Portfolio Value')
        ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax1.set_title('Portfolio Value & Position Sizing Over Time', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Position sizes
        ax2.scatter(trades['entry_date'], trades['position_size'], 
                   alpha=0.5, s=30, color='#F18F01', label='Position Size')
        
        # Rolling average
        trades['rolling_avg_size'] = trades['position_size'].rolling(window=20, min_periods=1).mean()
        ax2.plot(trades['entry_date'], trades['rolling_avg_size'], 
                color='red', linewidth=2, label='20-Trade Moving Avg')
        
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Position Size ($)', fontsize=12)
        ax2.set_title('Position Sizing Evolution', fontsize=12, fontweight='bold')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'position_sizing_over_time.png'), dpi=150)
        plt.close()
    
    
    def _plot_sector_allocation_over_time(self, trades: pd.DataFrame, equity_curve: pd.DataFrame):
        """Plot sector allocation over time (NEW!)"""
        trades = trades.copy()
        trades['entry_date'] = pd.to_datetime(trades['entry_date'])
        trades['exit_date'] = pd.to_datetime(trades['exit_date'])
        
        # Get all unique dates
        all_dates = pd.date_range(trades['entry_date'].min(), trades['exit_date'].max(), freq='W')
        
        # Count positions per sector at each date
        sector_counts = []
        
        for date in all_dates:
            active_trades = trades[
                (trades['entry_date'] <= date) & 
                (trades['exit_date'] >= date)
            ]
            
            if not active_trades.empty and 'sector' in active_trades.columns:
                sector_count = active_trades['sector'].value_counts().to_dict()
            else:
                sector_count = {}
            
            sector_counts.append({'date': date, **sector_count})
        
        df = pd.DataFrame(sector_counts).fillna(0)
        
        if len(df.columns) <= 1:  # Only date column
            return
        
        # Plot stacked area
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Get sector columns (exclude date)
        sector_cols = [col for col in df.columns if col != 'date']
        
        # Create stacked area plot
        ax.stackplot(df['date'], 
                    *[df[col] for col in sector_cols],
                    labels=sector_cols,
                    alpha=0.7)
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Number of Positions', fontsize=12)
        ax.set_title('Sector Allocation Over Time', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=1)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'sector_allocation.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    
    def _plot_adaptive_sizing_impact(
        self, 
        trades: pd.DataFrame, 
        equity_curve: pd.DataFrame,
        sizing_analysis: Dict
    ):
        """Plot adaptive sizing impact (NEW!)"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        trades = trades.copy()
        trades['entry_date'] = pd.to_datetime(trades['entry_date'])
        trades['position_size'] = trades['shares'] * trades['entry_price']
        
        # 1. Position size distribution
        ax1.hist(trades['position_size'], bins=50, alpha=0.7, color='#2E86AB', edgecolor='black')
        ax1.axvline(sizing_analysis.get('avg_position_size', 0), 
                   color='red', linestyle='--', linewidth=2, 
                   label=f"Avg: ${sizing_analysis.get('avg_position_size', 0):,.0f}")
        ax1.set_xlabel('Position Size ($)', fontsize=11)
        ax1.set_ylabel('Frequency', fontsize=11)
        ax1.set_title('Position Size Distribution', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Position size vs return correlation
        ax2.scatter(trades['position_size'], trades['pl_pct'], alpha=0.5, s=30, color='#F18F01')
        
        # Add trend line
        if len(trades) > 1:
            z = np.polyfit(trades['position_size'], trades['pl_pct'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(trades['position_size'].min(), trades['position_size'].max(), 100)
            ax2.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, 
                    label=f"Corr: {sizing_analysis.get('size_return_correlation', 0):.3f}")
        
        ax2.set_xlabel('Position Size ($)', fontsize=11)
        ax2.set_ylabel('Return (%)', fontsize=11)
        ax2.set_title('Position Size vs Return', fontsize=12, fontweight='bold')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Average position size by year
        yearly_sizes = sizing_analysis.get('yearly_avg_sizes', {})
        if yearly_sizes:
            years = sorted(yearly_sizes.keys())
            sizes = [yearly_sizes[y] for y in years]
            
            ax3.plot(years, sizes, marker='o', linewidth=2, markersize=8, color='#2E86AB')
            ax3.fill_between(years, sizes, alpha=0.3, color='#2E86AB')
            ax3.set_xlabel('Year', fontsize=11)
            ax3.set_ylabel('Avg Position Size ($)', fontsize=11)
            ax3.set_title('Average Position Size by Year', fontsize=12, fontweight='bold')
            ax3.grid(True, alpha=0.3)
            ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # 4. Drawdown impact
        dd_reduction = sizing_analysis.get('drawdown_size_reduction_pct', 0)
        
        metrics = ['Normal Periods', 'Drawdown Periods']
        reductions = [100, 100 - dd_reduction]
        colors = ['#2E86AB', '#A23B72']
        
        ax4.bar(metrics, reductions, color=colors, alpha=0.7)
        ax4.set_ylabel('Position Size (% of Normal)', fontsize=11)
        ax4.set_title(f'Adaptive Sizing During Drawdowns\n({dd_reduction:.1f}% reduction)', 
                     fontsize=12, fontweight='bold')
        ax4.axhline(y=100, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add values on bars
        for i, v in enumerate(reductions):
            ax4.text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'adaptive_sizing_impact.png'), dpi=150)
        plt.close()
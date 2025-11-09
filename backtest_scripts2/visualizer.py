# backtest_scripts/visualizer.py
"""
Visualization Module for Backtesting
Creates plots and charts for backtest results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import seaborn as sns
from typing import Dict, Optional
import os

class BacktestVisualizer:
    """
    Create visualizations for backtest results
    """
    
    def __init__(self, output_dir: str, dpi: int = 150):
        """
        Initialize visualizer
        
        Args:
            output_dir: Directory to save plots
            dpi: Plot resolution
        """
        self.output_dir = output_dir
        self.dpi = dpi
        self.plots_dir = os.path.join(output_dir, 'plots')
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        print(f"[Visualizer] Initialized (plots dir: {self.plots_dir})")
    
    def create_all_plots(self,
                        equity_curve: pd.DataFrame,
                        trades_history: pd.DataFrame,
                        regime_history: pd.DataFrame,
                        benchmark_prices: Dict[str, pd.DataFrame],
                        analysis: Dict):
        """
        Create all visualization plots
        
        Args:
            equity_curve: Portfolio daily values
            trades_history: All trades
            regime_history: Daily regime data
            benchmark_prices: Benchmark price data
            analysis: Performance analysis results
        """
        print("\n[Visualizer] Creating plots...")
        
        # 1. Equity Curve
        self.plot_equity_curve(equity_curve, benchmark_prices)
        
        # 2. Drawdown Chart
        self.plot_drawdown(equity_curve)
        
        # 3. Monthly Returns Heatmap
        self.plot_monthly_returns_heatmap(analysis.get('monthly_returns'))
        
        # 4. Regime Performance
        self.plot_regime_performance(analysis.get('regime_breakdown'))
        
        # 5. Trade Distribution
        self.plot_trade_distribution(trades_history)
        
        # 6. Rolling Sharpe
        self.plot_rolling_sharpe(equity_curve)
        
        # 7. Underwater Plot
        self.plot_underwater(equity_curve)
        
        # 8. Trade Timeline
        self.plot_trade_timeline(trades_history, equity_curve)
        
        print(f"[Visualizer] All plots saved to: {self.plots_dir}")
    
    def plot_equity_curve(self, equity_curve: pd.DataFrame, 
                         benchmark_prices: Dict[str, pd.DataFrame]):
        """Plot equity curve vs benchmarks"""
        fig, ax = plt.subplots(figsize=(14, 7))
        
        equity_curve['date'] = pd.to_datetime(equity_curve['date'])
        
        # Normalize to 100 at start
        start_value = equity_curve['total_value'].iloc[0]
        normalized_portfolio = (equity_curve['total_value'] / start_value) * 100
        
        ax.plot(equity_curve['date'], normalized_portfolio, 
                linewidth=2.5, label='Portfolio', color='#2E86AB')
        
        # Plot benchmarks
        colors = ['#A23B72', '#F18F01']
        for i, (name, prices) in enumerate(benchmark_prices.items()):
            prices = prices.copy()
            prices['date'] = pd.to_datetime(prices['date'])
            
            # Align with portfolio dates
            merged = equity_curve[['date']].merge(prices[['date', 'close']], 
                                                   on='date', how='left')
            merged['close'] = merged['close'].fillna(method='ffill')
            
            start_bench = merged['close'].iloc[0]
            normalized_bench = (merged['close'] / start_bench) * 100
            
            ax.plot(equity_curve['date'], normalized_bench, 
                   linewidth=2, label=name, color=colors[i % len(colors)], alpha=0.7)
        
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Value (Normalized to 100)', fontsize=12, fontweight='bold')
        ax.set_title('Equity Curve vs Benchmarks', fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'equity_curve.png'), 
                   dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print("  ✓ equity_curve.png")
    
    def plot_drawdown(self, equity_curve: pd.DataFrame):
        """Plot drawdown chart"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), 
                                       gridspec_kw={'height_ratios': [2, 1]})
        
        equity_curve['date'] = pd.to_datetime(equity_curve['date'])
        equity_curve['daily_return'] = equity_curve['total_value'].pct_change()
        
        # Cumulative returns
        cumulative = (1 + equity_curve['daily_return'].fillna(0)).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        
        # Top plot: Equity curve
        ax1.plot(equity_curve['date'], equity_curve['total_value'], 
                linewidth=2, color='#2E86AB')
        ax1.fill_between(equity_curve['date'], equity_curve['total_value'], 
                         alpha=0.3, color='#2E86AB')
        ax1.set_ylabel('Portfolio Value (\$)', fontsize=12, fontweight='bold')
        ax1.set_title('Portfolio Value & Drawdown', fontsize=16, fontweight='bold', pad=20)
        ax1.grid(True, alpha=0.3)
        
        # Bottom plot: Drawdown
        ax2.fill_between(equity_curve['date'], drawdown * 100, 0, 
                        alpha=0.5, color='#C1121F', label='Drawdown')
        ax2.plot(equity_curve['date'], drawdown * 100, 
                linewidth=1.5, color='#780000')
        ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Drawdown (%)', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Format x-axis
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'drawdown.png'), 
                   dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print("  ✓ drawdown.png")
    
    def plot_monthly_returns_heatmap(self, monthly_returns: Optional[pd.DataFrame]):
        """Plot monthly returns heatmap"""
        if monthly_returns is None or monthly_returns.empty:
            return
        
        # Skip if less than 12 months
        if len(monthly_returns) < 12:
            print("  ⊘ monthly_returns_heatmap.png (skipped: < 12 months)")
            return
        
        # Pivot to year x month matrix
        pivot = monthly_returns.pivot(index='year', columns='month', values='return_pct')
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(14, 8))
        
        sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
                   linewidths=0.5, cbar_kws={'label': 'Return (%)'}, ax=ax)
        
        ax.set_title('Monthly Returns Heatmap', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Month', fontsize=12, fontweight='bold')
        ax.set_ylabel('Year', fontsize=12, fontweight='bold')
        
        # Month labels
        month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        ax.set_xticklabels(month_labels)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'monthly_returns_heatmap.png'), 
                   dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print("  ✓ monthly_returns_heatmap.png")
    
    def plot_regime_performance(self, regime_breakdown: Optional[pd.DataFrame]):
        """Plot performance by regime"""
        if regime_breakdown is None or regime_breakdown.empty:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Total return by regime
        ax1 = axes[0, 0]
        colors = ['#06A77D' if x > 0 else '#C1121F' 
                 for x in regime_breakdown['total_return_pct']]
        regime_breakdown.plot(x='regime', y='total_return_pct', kind='bar', 
                             ax=ax1, color=colors, legend=False)
        ax1.set_title('Total Return by Regime', fontsize=13, fontweight='bold')
        ax1.set_xlabel('Regime', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Return (%)', fontsize=11, fontweight='bold')
        ax1.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax1.grid(True, alpha=0.3, axis='y')
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 2. Sharpe ratio by regime
        ax2 = axes[0, 1]
        regime_breakdown.plot(x='regime', y='sharpe_ratio', kind='bar', 
                             ax=ax2, color='#2E86AB', legend=False)
        ax2.set_title('Sharpe Ratio by Regime', fontsize=13, fontweight='bold')
        ax2.set_xlabel('Regime', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Sharpe Ratio', fontsize=11, fontweight='bold')
        ax2.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax2.grid(True, alpha=0.3, axis='y')
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 3. Win rate by regime
        ax3 = axes[1, 0]
        regime_breakdown.plot(x='regime', y='win_rate_pct', kind='bar', 
                             ax=ax3, color='#F18F01', legend=False)
        ax3.set_title('Win Rate by Regime', fontsize=13, fontweight='bold')
        ax3.set_xlabel('Regime', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Win Rate (%)', fontsize=11, fontweight='bold')
        ax3.axhline(50, color='black', linewidth=0.8, linestyle='--')
        ax3.grid(True, alpha=0.3, axis='y')
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 4. Number of trades by regime
        ax4 = axes[1, 1]
        regime_breakdown.plot(x='regime', y='trades_count', kind='bar', 
                             ax=ax4, color='#A23B72', legend=False)
        ax4.set_title('Trades Count by Regime', fontsize=13, fontweight='bold')
        ax4.set_xlabel('Regime', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Number of Trades', fontsize=11, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.suptitle('Performance Breakdown by Market Regime', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'regime_performance.png'), 
                   dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print("  ✓ regime_performance.png")
    
    def plot_trade_distribution(self, trades: pd.DataFrame):
        """Plot trade distribution statistics"""
        if trades.empty:
            return
        
        sell_trades = trades[trades['action'] == 'SELL'].copy()
        
        if sell_trades.empty:
            return
        
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 3, figure=fig)
        
        # 1. P/L distribution
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.hist(sell_trades['pl_pct'], bins=50, color='#2E86AB', alpha=0.7, edgecolor='black')
        ax1.axvline(0, color='red', linewidth=2, linestyle='--')
        ax1.set_xlabel('P/L (%)', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax1.set_title('P/L Distribution', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # 2. Hold time distribution
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.hist(sell_trades['hold_days'], bins=50, color='#F18F01', alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Hold Time (days)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax2.set_title('Hold Time Distribution', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 3. Exit reason breakdown
        ax3 = fig.add_subplot(gs[0, 2])
        exit_counts = sell_trades['reason'].value_counts()
        ax3.pie(exit_counts.values, labels=exit_counts.index, autopct='%1.1f%%',
               startangle=90, colors=sns.color_palette("husl", len(exit_counts)))
        ax3.set_title('Exit Reasons', fontsize=13, fontweight='bold')
        
        # 4. Wins vs Losses over time
        ax4 = fig.add_subplot(gs[1, :])
        sell_trades['date'] = pd.to_datetime(sell_trades['date'])
        sell_trades = sell_trades.sort_values('date')
        sell_trades['cumulative_pl'] = sell_trades['pl'].cumsum()
        
        wins = sell_trades[sell_trades['pl'] > 0]
        losses = sell_trades[sell_trades['pl'] < 0]
        
        ax4.scatter(wins['date'], wins['cumulative_pl'], 
                   color='green', alpha=0.6, s=50, label='Wins')
        ax4.scatter(losses['date'], losses['cumulative_pl'], 
                   color='red', alpha=0.6, s=50, label='Losses')
        ax4.plot(sell_trades['date'], sell_trades['cumulative_pl'], 
                color='blue', linewidth=2, label='Cumulative P/L')
        
        ax4.set_xlabel('Date', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Cumulative P/L (\$)', fontsize=11, fontweight='bold')
        ax4.set_title('Cumulative P/L Over Time', fontsize=13, fontweight='bold')
        ax4.legend(loc='upper left', fontsize=10)
        ax4.grid(True, alpha=0.3)
        ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.suptitle('Trade Distribution Analysis', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'trade_distribution.png'), 
                   dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print("  ✓ trade_distribution.png")
    
    def plot_rolling_sharpe(self, equity_curve: pd.DataFrame, window: int = 60):
        """Plot rolling Sharpe ratio"""
        equity_curve = equity_curve.copy()
        equity_curve['date'] = pd.to_datetime(equity_curve['date'])
        equity_curve['daily_return'] = equity_curve['total_value'].pct_change()
        
        # Rolling Sharpe (60-day window, annualized)
        rolling_mean = equity_curve['daily_return'].rolling(window).mean()
        rolling_std = equity_curve['daily_return'].rolling(window).std()
        rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(252)
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        ax.plot(equity_curve['date'], rolling_sharpe, linewidth=2, color='#2E86AB')
        ax.axhline(0, color='black', linewidth=1, linestyle='--')
        ax.axhline(1, color='green', linewidth=1, linestyle=':', alpha=0.5, label='Sharpe = 1')
        ax.axhline(2, color='darkgreen', linewidth=1, linestyle=':', alpha=0.5, label='Sharpe = 2')
        
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'{window}-Day Rolling Sharpe Ratio', fontsize=12, fontweight='bold')
        ax.set_title(f'Rolling Sharpe Ratio ({window}-Day Window)', fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'rolling_sharpe.png'), 
                   dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print("  ✓ rolling_sharpe.png")
    
    def plot_underwater(self, equity_curve: pd.DataFrame):
        """Plot underwater (drawdown) chart"""
        equity_curve = equity_curve.copy()
        equity_curve['date'] = pd.to_datetime(equity_curve['date'])
        equity_curve['daily_return'] = equity_curve['total_value'].pct_change()
        
        cumulative = (1 + equity_curve['daily_return'].fillna(0)).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        ax.fill_between(equity_curve['date'], drawdown * 100, 0, 
                       alpha=0.5, color='#C1121F')
        ax.plot(equity_curve['date'], drawdown * 100, 
               linewidth=1.5, color='#780000')
        
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Drawdown (%)', fontsize=12, fontweight='bold')
        ax.set_title('Underwater Plot (Drawdown from Peak)', fontsize=16, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'underwater.png'), 
                   dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print("  ✓ underwater.png")
    
    def plot_trade_timeline(self, trades: pd.DataFrame, equity_curve: pd.DataFrame):
        """Plot trade markers on equity curve"""
        if trades.empty:
            return
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        equity_curve = equity_curve.copy()
        equity_curve['date'] = pd.to_datetime(equity_curve['date'])
        
        # Plot equity curve
        ax.plot(equity_curve['date'], equity_curve['total_value'], 
               linewidth=2, color='#2E86AB', label='Portfolio Value')
        
        # Mark trades
        buy_trades = trades[trades['action'] == 'BUY'].copy()
        sell_trades = trades[trades['action'] == 'SELL'].copy()
        
        if not buy_trades.empty:
            buy_trades['date'] = pd.to_datetime(buy_trades['date'])
            # Get portfolio value at buy dates
            buy_with_value = buy_trades.merge(
                equity_curve[['date', 'total_value']], 
                on='date', how='left'
            )
            ax.scatter(buy_with_value['date'], buy_with_value['total_value'], 
                      color='green', marker='^', s=100, alpha=0.6, label='Buy', zorder=5)
        
        if not sell_trades.empty:
            sell_trades['date'] = pd.to_datetime(sell_trades['date'])
            sell_with_value = sell_trades.merge(
                equity_curve[['date', 'total_value']], 
                on='date', how='left'
            )
            # Color by profit/loss
            wins = sell_with_value[sell_with_value['pl'] > 0]
            losses = sell_with_value[sell_with_value['pl'] < 0]
            
            if not wins.empty:
                ax.scatter(wins['date'], wins['total_value'], 
                          color='darkgreen', marker='v', s=100, alpha=0.8, label='Sell (Win)', zorder=5)
            if not losses.empty:
                ax.scatter(losses['date'], losses['total_value'], 
                          color='darkred', marker='v', s=100, alpha=0.8, label='Sell (Loss)', zorder=5)
        
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Portfolio Value (\$)', fontsize=12, fontweight='bold')
        ax.set_title('Trade Timeline on Equity Curve', fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'trade_timeline.png'), 
                   dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print("  ✓ trade_timeline.png")


"""
Utility Functions for Advanced Backtesting Framework
Helper functions for technical indicators, regime detection, and statistics

Author: @panuaalto1-afk
Date: 2025-01-04
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf


# ═══════════════════════════════════════════════════════════════
# TECHNICAL INDICATORS
# ═══════════════════════════════════════════════════════════════

def calculate_atr(df, period=14):
    """
    Calculate Average True Range
    
    Args:
        df: DataFrame with High, Low, Close columns
        period: ATR period (default 14)
    
    Returns:
        Series with ATR values
    """
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    return atr

def calculate_rsi(df, period=14):
    """
    Calculate Relative Strength Index
    
    Args:
        df: DataFrame with Close column
        period: RSI period (default 14)
    
    Returns:
        Series with RSI values
    """
    close = df['Close']
    delta = close.diff()
    
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_macd(df, fast=12, slow=26, signal=9):
    """
    Calculate MACD
    
    Args:
        df: DataFrame with Close column
        fast: Fast EMA period
        slow: Slow EMA period
        signal: Signal line period
    
    Returns:
        Dict with macd, signal, histogram
    """
    close = df['Close']
    
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    
    return {
        'macd': macd,
        'signal': signal_line,
        'histogram': histogram
    }

def calculate_bollinger_bands(df, period=20, std_dev=2):
    """
    Calculate Bollinger Bands
    
    Args:
        df: DataFrame with Close column
        period: Moving average period
        std_dev: Standard deviation multiplier
    
    Returns:
        Dict with upper, middle, lower bands
    """
    close = df['Close']
    
    middle = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    
    return {
        'upper': upper,
        'middle': middle,
        'lower': lower
    }

def calculate_technical_indicators(df):
    """
    Calculate all technical indicators at once
    
    Args:
        df: DataFrame with OHLCV data
    
    Returns:
        DataFrame with all indicators added
    """
    df = df.copy()
    
    # ATR
    df['atr'] = calculate_atr(df)
    
    # RSI
    df['rsi'] = calculate_rsi(df)
    
    # MACD
    macd_data = calculate_macd(df)
    df['macd'] = macd_data['macd']
    df['macd_signal'] = macd_data['signal']
    df['macd_histogram'] = macd_data['histogram']
    
    # Bollinger Bands
    bb_data = calculate_bollinger_bands(df)
    df['bb_upper'] = bb_data['upper']
    df['bb_middle'] = bb_data['middle']
    df['bb_lower'] = bb_data['lower']
    
    # Moving Averages
    df['sma_20'] = df['Close'].rolling(window=20).mean()
    df['sma_50'] = df['Close'].rolling(window=50).mean()
    df['sma_200'] = df['Close'].rolling(window=200).mean()
    
    # Volume MA
    df['volume_ma_20'] = df['Volume'].rolling(window=20).mean()
    return df

# ═══════════════════════════════════════════════════════════════
# REGIME DETECTION
# ═══════════════════════════════════════════════════════════════

def get_vix_data(start_date, end_date):
    """
    Fetch VIX data from Yahoo Finance
    
    Args:
        start_date: Start date (datetime or string)
        end_date: End date (datetime or string)
    
    Returns:
        DataFrame with VIX data
    """
    try:
        vix = yf.download('^VIX', start=start_date, end=end_date, progress=False)
        return vix
    except Exception as e:
        print(f"⚠️ Error fetching VIX data: {e}")
        return pd.DataFrame()

def classify_regime(vix_value):
    """
    Classify market regime based on VIX level
    
    Args:
        vix_value: VIX closing price
    
    Returns:
        String: 'low_volatility', 'normal', or 'high_volatility'
    """
    if pd.isna(vix_value):
        return 'unknown'
    
    if vix_value < 15:
        return 'low_volatility'
    elif vix_value > 25:
        return 'high_volatility'
    else:
        return 'normal'

def get_regime_for_date(date, vix_df):
    """
    Get market regime for a specific date
    
    Args:
        date: Date to check
        vix_df: DataFrame with VIX data
    
    Returns:
        String: regime classification
    """
    try:
        date = pd.to_datetime(date)
        
        # Find closest date in VIX data
        if date in vix_df.index:
            vix_value = vix_df.loc[date, 'Close']
        else:
            # Get closest previous date
            closest_date = vix_df.index[vix_df.index <= date][-1]
            vix_value = vix_df.loc[closest_date, 'Close']
        
        return classify_regime(vix_value)
    except:
        return 'unknown'

# ═══════════════════════════════════════════════════════════════
# POSITION SIZING & RISK MANAGEMENT
# ═══════════════════════════════════════════════════════════════

def calculate_kelly_criterion(win_rate, avg_win, avg_loss):
    """
    Calculate Kelly Criterion for optimal position sizing
    
    Args:
        win_rate: Probability of winning (0-1)
        avg_win: Average win amount
        avg_loss: Average loss amount (positive number)
    
    Returns:
        Float: Kelly percentage (capped at 25%)
    """
    if avg_loss == 0 or win_rate == 0 or win_rate == 1:
        return 0
    
    p = win_rate
    q = 1 - win_rate
    b = avg_win / avg_loss
    
    kelly = (p * b - q) / b
    
    # Cap at 25% for safety (half-Kelly is common)
    return max(0, min(kelly, 0.25))

def calculate_position_size(account_value, risk_per_trade, entry_price, stop_loss_price):
    """
    Calculate position size based on risk
    
    Args:
        account_value: Total account value
        risk_per_trade: Risk percentage per trade (e.g., 0.02 for 2%)
        entry_price: Entry price
        stop_loss_price: Stop loss price
    
    Returns:
        Int: Number of shares to buy
    """
    risk_amount = account_value * risk_per_trade
    risk_per_share = abs(entry_price - stop_loss_price)
    
    if risk_per_share == 0:
        return 0
    
    shares = int(risk_amount / risk_per_share)
    return shares

# ═══════════════════════════════════════════════════════════════
# STATISTICAL FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def calculate_profit_factor(wins, losses):
    """
    Calculate Profit Factor = Total Wins / Total Losses
    
    Args:
        wins: List or Series of winning trades
        losses: List or Series of losing trades
    
    Returns:
        Float: Profit factor (>1.0 is good)
    """
    total_wins = sum(w for w in wins if w > 0)
    total_losses = abs(sum(l for l in losses if l < 0))
    
    if total_losses == 0:
        return float('inf') if total_wins > 0 else 0
    
    return total_wins / total_losses

def calculate_expectancy(win_rate, avg_win, avg_loss):
    """
    Calculate expectancy (average $ per trade)
    
    Args:
        win_rate: Win rate (0-1)
        avg_win: Average winning trade
        avg_loss: Average losing trade (positive number)
    
    Returns:
        Float: Expected value per trade
    """
    return (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

def bootstrap_sample(data, n_samples):
    """
    Bootstrap sampling for Monte Carlo simulation
    
    Args:
        data: Original data
        n_samples: Number of samples to draw
    
    Returns:
        Array: Bootstrapped samples
    """
    return np.random.choice(data, size=n_samples, replace=True)

def calculate_confidence_interval(data, confidence=0.95):
    """
    Calculate confidence interval
    
    Args:
        data: Data array
        confidence: Confidence level (default 0.95)
    
    Returns:
        Tuple: (lower_bound, upper_bound)
    """
    data = np.array(data)
    n = len(data)
    
    if n < 2:
        return (np.nan, np.nan)
    
    mean = np.mean(data)
    std_err = np.std(data, ddof=1) / np.sqrt(n)
    
    # Use t-distribution for small samples
    from scipy import stats
    t_value = stats.t.ppf((1 + confidence) / 2, n - 1)
    
    margin = t_value * std_err
    
    return (mean - margin, mean + margin)

# ═══════════════════════════════════════════════════════════════
# DATA VALIDATION
# ═══════════════════════════════════════════════════════════════

def validate_ohlc_data(df):
    """
    Validate OHLC data integrity
    
    Args:
        df: DataFrame with OHLC data
    
    Returns:
        Tuple: (is_valid, error_messages)
    """
    errors = []
    
    # Check required columns
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        errors.append(f"Missing columns: {missing_cols}")
    
    if len(errors) > 0:
        return (False, errors)
    
    # Check for NaN values
    nan_counts = df[required_cols].isna().sum()
    if nan_counts.any():
        errors.append(f"NaN values found: {nan_counts[nan_counts > 0].to_dict()}")
    
    # Check High >= Low
    invalid_hl = df[df['High'] < df['Low']]
    if len(invalid_hl) > 0:
        errors.append(f"High < Low in {len(invalid_hl)} rows")
    
    # Check High >= Open, Close
    invalid_high = df[(df['High'] < df['Open']) | (df['High'] < df['Close'])]
    if len(invalid_high) > 0:
        errors.append(f"High < Open/Close in {len(invalid_high)} rows")
    
    # Check Low <= Open, Close
    invalid_low = df[(df['Low'] > df['Open']) | (df['Low'] > df['Close'])]
    if len(invalid_low) > 0:
        errors.append(f"Low > Open/Close in {len(invalid_low)} rows")
    
    is_valid = len(errors) == 0
    return (is_valid, errors)

def clean_ohlc_data(df):
    """
    Clean and fix OHLC data issues
    
    Args:
        df: DataFrame with OHLC data
    
    Returns:
        DataFrame: Cleaned data
    """
    df = df.copy()
    
    # Forward fill NaN values
    df = df.fillna(method='ffill')
    
    # Remove rows where all OHLC are zero
    df = df[~((df['Open'] == 0) & (df['High'] == 0) & (df['Low'] == 0) & (df['Close'] == 0))]
    
    # Fix High/Low inconsistencies
    df['High'] = df[['High', 'Open', 'Close']].max(axis=1)
    df['Low'] = df[['Low', 'Open', 'Close']].min(axis=1)
    
    return df

# ═══════════════════════════════════════════════════════════════
# FORMATTING HELPERS
# ═══════════════════════════════════════════════════════════════

def format_currency(value):
    """Format value as currency"""
    if pd.isna(value):
        return "N/A"
    return f"${value:,.2f}"

def format_percentage(value):
    """Format value as percentage"""
    if pd.isna(value):
        return "N/A"
    return f"{value:.2f}%"

def format_number(value, decimals=2):
    """Format number with decimals"""
    if pd.isna(value):
        return "N/A"
    return f"{value:.{decimals}f}"

# ═══════════════════════════════════════════════════════════════
# DATE HELPERS
# ═══════════════════════════════════════════════════════════════

def get_trading_days_between(start_date, end_date):
    """
    Calculate number of trading days between two dates
    
    Args:
        start_date: Start date
        end_date: End date
    
    Returns:
        Int: Number of trading days
    """
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    # Generate business days
    trading_days = pd.bdate_range(start, end)
    
    return len(trading_days)

def is_market_open(date):
    """
    Check if market was open on date (simple check for weekends)
    
    Args:
        date: Date to check
    
    Returns:
        Bool: True if market open
    """
    date = pd.to_datetime(date)
    return date.weekday() < 5  # Monday=0, Friday=4

if __name__ == "__main__":
    print("✅ backtest_utils.py loaded successfully")
    print("📊 Available functions:")
    print("   - Technical Indicators: ATR, RSI, MACD, Bollinger Bands")
    print("   - Regime Detection: VIX-based classification")
    print("   - Position Sizing: Kelly Criterion, Risk-based")
    print("   - Statistics: Profit Factor, Expectancy, Bootstrap")
    print("   - Data Validation: OHLC integrity checks")

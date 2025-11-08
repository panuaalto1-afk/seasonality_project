# ml_unified_pipeline.py (ENHANCED VERSION)
# Seasonality project â€” unified daily ML pipeline with full seasonality + regime + ML

import argparse
import os
import sys
import glob
import math
from datetime import datetime, date, timedelta
from typing import Optional, Tuple, List, Dict
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

# ML imports
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("[WARN] LightGBM not installed. Install: pip install lightgbm")

# =====================================================================
# REGIME CALCULATOR (Standalone copy of regime_detector.py logic)
# =====================================================================

class RegimeCalculator:
    """
    Standalone regime detector (copy of regime_detector.py logic)
    Detects 7 market regimes from macro ETF signals
    """
    
    # Regime thresholds (composite score boundaries)
    REGIME_THRESHOLDS = {
        'CRISIS': -1.0,
        'BEAR_STRONG': -0.50,
        'BEAR_WEAK': -0.25,
        'NEUTRAL_BEARISH': -0.05,
        'NEUTRAL_BULLISH': 0.05,
        'BULL_WEAK': 0.25,
        'BULL_STRONG': 0.50,
    }
    
    # Component weights (how much each signal matters)
    COMPONENT_WEIGHTS = {
        'equity': 0.35,
        'volatility': 0.20,
        'credit': 0.20,
        'safe_haven': 0.15,
        'breadth': 0.10,
    }
    
    def __init__(self, macro_price_cache_dir: str):
        """
        Initialize regime calculator
        
        Args:
            macro_price_cache_dir: Path to seasonality_reports/price_cache/
        """
        self.cache_dir = macro_price_cache_dir
        self.prices = {}  # Cache loaded prices
    
    def _load_etf_prices(self, symbol: str, max_date: Optional[date] = None) -> pd.DataFrame:
        """Load ETF prices from cache"""
        cache_key = f"{symbol}_{max_date}" if max_date else symbol
        if cache_key in self.prices:
            return self.prices[cache_key]
        
        path = os.path.join(self.cache_dir, f"{symbol}.csv")
        if not os.path.isfile(path):
            return pd.DataFrame()
        
        try:
            df = pd.read_csv(path)
            # Normalize columns
            cols = {c.lower(): c for c in df.columns}
            date_col = cols.get('date')
            close_col = None
            for k in ['adj close', 'adj_close', 'close', 'price']:
                if k in cols:
                    close_col = cols[k]
                    break
            
            if date_col is None or close_col is None:
                return pd.DataFrame()
            
            df = df[[date_col, close_col]].rename(columns={date_col: 'date', close_col: 'close'})
            df['date'] = pd.to_datetime(df['date']).dt.date
            df['close'] = pd.to_numeric(df['close'], errors='coerce')
            df = df.dropna().sort_values('date')
            
            # Filter by max_date if provided
            if max_date:
                df = df[df['date'] <= max_date]
            
            self.prices[cache_key] = df
            return df
        except Exception as e:
            print(f"[WARN] Failed to load {symbol}: {e}")
            return pd.DataFrame()
    
    def _calculate_momentum(self, prices: pd.DataFrame, period: int = 60) -> float:
        """Calculate momentum (% change over period)"""
        if len(prices) < period:
            return 0.0
        return float((prices.iloc[-1]['close'] / prices.iloc[-period]['close']) - 1)
    
    def _calculate_volatility(self, prices: pd.DataFrame, period: int = 20) -> float:
        """Calculate annualized volatility"""
        if len(prices) < period:
            return 0.0
        returns = prices['close'].pct_change().dropna()
        return float(returns.tail(period).std() * np.sqrt(252))
    
    def _equity_signal(self, today: date) -> float:
        """
        Equity signal: SPY, QQQ, IWM momentum
        Positive = bullish, negative = bearish
        """
        symbols = ['SPY', 'QQQ', 'IWM']
        momentums = []
        
        for sym in symbols:
            prices = self._load_etf_prices(sym, today)
            if not prices.empty:
                mom = self._calculate_momentum(prices, period=60)
                momentums.append(mom)
        
        if not momentums:
            return 0.0
        
        # Average momentum, normalized to ~[-1, 1]
        avg_mom = np.mean(momentums)
        return float(np.clip(avg_mom * 5, -1, 1))  # Scale to [-1, 1]
    
    def _volatility_signal(self, today: date) -> float:
        """
        Volatility signal: VIX level
        High VIX = bearish (negative signal)
        """
        vix_symbols = ['VIXM', 'VIX']  # Try both
        
        for sym in vix_symbols:
            prices = self._load_etf_prices(sym, today)
            if not prices.empty:
                current_vix = prices.iloc[-1]['close']
                
                # VIX normalization:
                # 10-15: Low vol (bullish) â†’ +0.5
                # 15-20: Normal â†’ 0
                # 20-30: Elevated (bearish) â†’ -0.5
                # 30+: High (very bearish) â†’ -1.0
                
                if current_vix < 15:
                    return 0.5
                elif current_vix < 20:
                    return 0.0
                elif current_vix < 30:
                    return -0.5
                else:
                    return -1.0
        
        return 0.0  # Default if no data
    
    def _credit_signal(self, today: date) -> float:
        """
        Credit signal: HYG/LQD spread
        Widening spread = bearish (negative)
        """
        hyg = self._load_etf_prices('HYG', today)
        lqd = self._load_etf_prices('LQD', today)
        
        if hyg.empty or lqd.empty:
            return 0.0
        
        # Calculate momentum difference (credit spread proxy)
        hyg_mom = self._calculate_momentum(hyg, period=60)
        lqd_mom = self._calculate_momentum(lqd, period=60)
        
        spread = hyg_mom - lqd_mom
        
        # Positive spread (HYG outperforming) = bullish
        return float(np.clip(spread * 10, -1, 1))
    
    def _safe_haven_signal(self, today: date) -> float:
        """
        Safe haven signal: GLD, TLT
        Strong safe haven = bearish equities (negative)
        """
        gld = self._load_etf_prices('GLD', today)
        tlt = self._load_etf_prices('TLT', today)
        
        momentums = []
        for prices in [gld, tlt]:
            if not prices.empty:
                mom = self._calculate_momentum(prices, period=60)
                momentums.append(mom)
        
        if not momentums:
            return 0.0
        
        # Strong safe haven momentum = bearish equities
        avg_mom = np.mean(momentums)
        return float(np.clip(-avg_mom * 5, -1, 1))  # Inverted
    
    def _breadth_signal(self, today: date) -> float:
        """
        Breadth signal: Market breadth (advancing vs declining)
        Proxy: Compare large cap (SPY) vs small cap (IWM)
        """
        spy = self._load_etf_prices('SPY', today)
        iwm = self._load_etf_prices('IWM', today)
        
        if spy.empty or iwm.empty:
            return 0.0
        
        spy_mom = self._calculate_momentum(spy, period=60)
        iwm_mom = self._calculate_momentum(iwm, period=60)
        
        # Positive breadth: IWM outperforming (broad rally)
        breadth = iwm_mom - spy_mom
        return float(np.clip(breadth * 10, -1, 1))
    
    def detect_regime(self, today: date) -> Dict:
        """
        Detect market regime for given date
        
        Returns:
            dict: {
                'regime': str (e.g., 'BULL_STRONG'),
                'composite_score': float,
                'components': dict
            }
        """
        # Calculate all components
        components = {
            'equity': self._equity_signal(today),
            'volatility': self._volatility_signal(today),
            'credit': self._credit_signal(today),
            'safe_haven': self._safe_haven_signal(today),
            'breadth': self._breadth_signal(today),
        }
        
        # Composite score (weighted average)
        composite = sum(
            components[k] * self.COMPONENT_WEIGHTS[k]
            for k in components.keys()
        )
        
        # Determine regime from composite score
        regime = 'NEUTRAL_BULLISH'  # Default
        for regime_name, threshold in sorted(self.REGIME_THRESHOLDS.items(), key=lambda x: x[1], reverse=True):
            if composite >= threshold:
                regime = regime_name
                break
        
        return {
            'regime': regime,
            'composite_score': float(composite),
            'components': components
        }

# =====================================================================
# SEASONALITY CALCULATOR
# =====================================================================

class SeasonalityCalculator:
    """
    Calculate seasonality features walk-forward
    Multi-window approach (Vaihtoehto 3)
    """
    
    def __init__(self, lookback_years: int = 10):
        self.lookback_years = lookback_years
        self.segments_cache = {}  # Cache segments data
    
    def _load_segments(self, ticker: str, seg_type: str) -> pd.DataFrame:
        """
        Load segments_up or segments_down CSV
        
        Args:
            ticker: Ticker symbol
            seg_type: 'up' or 'down'
        
        Returns:
            DataFrame with segments
        """
        cache_key = f"{ticker}_{seg_type}"
        if cache_key in self.segments_cache:
            return self.segments_cache[cache_key]
        
        # Try to load from vintage/ folder
        vintage_dir = os.path.join(os.getcwd(), "seasonality_reports", "vintage")
        filename = f"{ticker}_segments_{seg_type}.csv"
        path = os.path.join(vintage_dir, filename)
        
        if not os.path.isfile(path):
            return pd.DataFrame()
        
        try:
            df = pd.read_csv(path)
            self.segments_cache[cache_key] = df
            return df
        except Exception:
            return pd.DataFrame()
    
    def _check_segment(self, ticker: str, current_doy: int) -> Dict:
        """
        Check if current day-of-year is in a bullish/bearish segment
        
        Returns:
            dict: {
                'in_bullish_segment': int (0 or 1),
                'in_bearish_segment': int (0 or 1),
                'days_into_segment': int,
                'segment_strength': float
            }
        """
        result = {
            'in_bullish_segment': 0,
            'in_bearish_segment': 0,
            'days_into_segment': 0,
            'segment_strength': 0.0
        }
        
        # Check bullish segments
        seg_up = self._load_segments(ticker, 'up')
        if not seg_up.empty:
            for _, row in seg_up.iterrows():
                start, end = row['start_doy'], row['end_doy']
                if start <= current_doy <= end:
                    result['in_bullish_segment'] = 1
                    result['days_into_segment'] = current_doy - start
                    result['segment_strength'] = float(row.get('strength', 0.0))
                    return result
        
        # Check bearish segments
        seg_down = self._load_segments(ticker, 'down')
        if not seg_down.empty:
            for _, row in seg_down.iterrows():
                start, end = row['start_doy'], row['end_doy']
                if start <= current_doy <= end:
                    result['in_bearish_segment'] = 1
                    result['days_into_segment'] = current_doy - start
                    result['segment_strength'] = float(row.get('strength', 0.0))
                    return result
        
        return result
    
    def calculate_features(self, ticker: str, prices: pd.DataFrame, today: date) -> Dict:
        """
        Calculate all seasonality features (Vaihtoehto 3: Multi-window)
        
        Args:
            ticker: Stock ticker
            prices: Historical prices (must go back at least lookback_years)
            today: Current date
        
        Returns:
            dict with seasonality features
        """
        if prices is None or prices.empty or len(prices) < 252:
            return self._empty_features()
        
        # Ensure date column
        if 'date' not in prices.columns:
            return self._empty_features()
        
        # Add helper columns
        prices = prices.copy()

        # ✅ FIX: Ensure close is numeric BEFORE any calculations
        prices['close'] = pd.to_numeric(prices['close'], errors='coerce')
        prices = prices.dropna(subset=['close'])

        if prices.empty or len(prices) < 252:
            return self._empty_features()

        # Now safe to calculate returns
        prices['return'] = prices['close'].pct_change()
        prices['week'] = pd.to_datetime(prices['date']).dt.isocalendar().week
        prices['month'] = pd.to_datetime(prices['date']).dt.month
        prices['quarter'] = pd.to_datetime(prices['date']).dt.quarter
        prices['doy'] = pd.to_datetime(prices['date']).dt.dayofyear
        
        # Current time markers
        current_week = today.isocalendar()[1]
        current_month = today.month
        current_quarter = (today.month - 1) // 3 + 1
        current_doy = today.timetuple().tm_yday
        
        # Filter to lookback window
        cutoff_date = today - timedelta(days=365 * self.lookback_years)
        hist = prices[pd.to_datetime(prices['date']) >= pd.to_datetime(cutoff_date)].copy()
        
        if hist.empty:
            return self._empty_features()
        
        # 1. Week-of-Year
        week_data = hist[hist['week'] == current_week]
        season_week_avg = float(week_data['return'].mean()) if not week_data.empty else 0.0
        season_week_hit_rate = float((week_data['return'] > 0).mean()) if not week_data.empty else 0.5
        
        # 2. Day-of-Year (Â±3 day window)
        day_window = hist[
            (hist['doy'] >= current_doy - 3) & 
            (hist['doy'] <= current_doy + 3)
        ]
        
        # Calculate forward 20-day returns for historical same dates
        forward_returns = []
        for idx in day_window.index:
            try:
                future_idx = hist.index.get_loc(idx) + 20
                if future_idx < len(hist):
                    future_price = hist.iloc[future_idx]['close']
                    current_price = hist.loc[idx, 'close']
                    if current_price > 0:
                        fwd_ret = (future_price / current_price) - 1
                        forward_returns.append(fwd_ret)
            except:
                pass
        
        season_20d_avg = float(np.mean(forward_returns)) if forward_returns else 0.0
        season_20d_hit_rate = float(np.mean([1 if r > 0 else 0 for r in forward_returns])) if forward_returns else 0.5
        
        # 3. Month-of-Year
        month_data = hist[hist['month'] == current_month]
        season_month_avg = float(month_data['return'].mean()) if not month_data.empty else 0.0
        
        # 4. Quarter-of-Year
        quarter_data = hist[hist['quarter'] == current_quarter]
        season_quarter_avg = float(quarter_data['return'].mean()) if not quarter_data.empty else 0.0
        
        # 5. Segment info
        segment_info = self._check_segment(ticker, current_doy)
        
        return {
            'season_week_avg': season_week_avg,
            'season_week_hit_rate': season_week_hit_rate,
            'season_20d_avg': season_20d_avg,
            'season_20d_hit_rate': season_20d_hit_rate,
            'season_month_avg': season_month_avg,
            'season_quarter_avg': season_quarter_avg,
            'in_bullish_segment': segment_info['in_bullish_segment'],
            'in_bearish_segment': segment_info['in_bearish_segment'],
            'days_into_segment': segment_info['days_into_segment'],
            'segment_strength': segment_info['segment_strength'],
        }
    
    def _empty_features(self) -> Dict:
        """Return empty features dict"""
        return {
            'season_week_avg': 0.0,
            'season_week_hit_rate': 0.5,
            'season_20d_avg': 0.0,
            'season_20d_hit_rate': 0.5,
            'season_month_avg': 0.0,
            'season_quarter_avg': 0.0,
            'in_bullish_segment': 0,
            'in_bearish_segment': 0,
            'days_into_segment': 0,
            'segment_strength': 0.0,
        }

# =====================================================================
# =====================================================================
# TRADING LEVELS CALCULATOR (Entry, Stop Loss, Take Profit)
# =====================================================================

class TradingLevelsCalculator:
    """
    Calculate entry price, stop loss, and take profit levels
    Based on regime_strategies.py logic
    """
    
    # Regime-specific SL/TP multipliers (from regime_strategies.py)
    REGIME_STRATEGIES = {
        'BULL_STRONG': {
            'stop_multiplier': 0.8,
            'tp_multiplier': 2.5,
        },
        'BULL_WEAK': {
            'stop_multiplier': 1.0,
            'tp_multiplier': 2.0,
        },
        'NEUTRAL_BULLISH': {
            'stop_multiplier': 1.0,
            'tp_multiplier': 1.5,
        },
        'NEUTRAL_BEARISH': {
            'stop_multiplier': 1.2,
            'tp_multiplier': 1.2,
        },
        'BEAR_WEAK': {
            'stop_multiplier': 1.5,
            'tp_multiplier': 1.0,
        },
        'BEAR_STRONG': {
            'stop_multiplier': 1.8,
            'tp_multiplier': 0.8,
        },
        'CRISIS': {
            'stop_multiplier': 2.0,
            'tp_multiplier': 0.5,
        },
    }
    
    def __init__(self):
        pass
    
    def _calculate_atr(self, prices: pd.DataFrame, period: int = 14) -> float:
        """
        Calculate Average True Range (ATR)
        
        Args:
            prices: DataFrame with high, low, close columns (or just close)
            period: ATR period (default 14)
        
        Returns:
            ATR value
        """
        if prices is None or len(prices) < period:
            return 0.0
        
        # If we have high/low, use true range
        if 'high' in prices.columns and 'low' in prices.columns:
            high = prices['high'].astype(float)
            low = prices['low'].astype(float)
            close = prices['close'].astype(float)
            
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(period).mean().iloc[-1]
        else:
            # Fallback: use close-to-close volatility as proxy
            # ✅ FIX: Ensure close is numeric
            prices['close'] = pd.to_numeric(prices['close'], errors='coerce')
            returns = prices['close'].pct_change()
            volatility = returns.rolling(period).std().iloc[-1]
            atr = volatility * prices['close'].iloc[-1]
        
        return float(atr) if not pd.isna(atr) else 0.0
    
    def calculate(self, ticker: str, prices: pd.DataFrame, regime: str) -> Dict:
        """
        Calculate trading levels
        
        Args:
            ticker: Stock ticker
            prices: Historical prices (needs at least 14 days for ATR)
            regime: Current market regime
        
        Returns:
            dict: {
                'entry_price': float,
                'stop_loss': float,
                'take_profit': float,
                'atr_14': float,
                'sl_distance_pct': float,
                'tp_distance_pct': float
            }
        """
        if prices is None or prices.empty or len(prices) < 14:
            return self._empty_levels()
        
        # Entry price = T-1 close (last available price)
        entry_price = float(prices.iloc[-1]['close'])
        
        if entry_price <= 0:
            return self._empty_levels()
        
        # Calculate ATR
        atr = self._calculate_atr(prices, period=14)
        
        if atr <= 0:
            # Fallback: use 2% of entry price as ATR
            atr = entry_price * 0.02
        
        # Get regime strategy
        strategy = self.REGIME_STRATEGIES.get(regime, self.REGIME_STRATEGIES['NEUTRAL_BULLISH'])
        
        # Calculate SL/TP
        stop_loss = entry_price - (atr * strategy['stop_multiplier'])
        take_profit = entry_price + (atr * strategy['tp_multiplier'])
        
        # Ensure SL/TP are positive
        stop_loss = max(stop_loss, entry_price * 0.5)  # Max 50% loss
        take_profit = max(take_profit, entry_price * 1.01)  # Min 1% gain
        
        # Calculate distance percentages
        sl_distance_pct = ((entry_price - stop_loss) / entry_price) * 100
        tp_distance_pct = ((take_profit - entry_price) / entry_price) * 100
        
        return {
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'atr_14': atr,
            'sl_distance_pct': sl_distance_pct,
            'tp_distance_pct': tp_distance_pct,
        }
    
    def _empty_levels(self) -> Dict:
        """Return empty levels dict"""
        return {
            'entry_price': 0.0,
            'stop_loss': 0.0,
            'take_profit': 0.0,
            'atr_14': 0.0,
            'sl_distance_pct': 0.0,
            'tp_distance_pct': 0.0,
        }

# =====================================================================
# ML MODEL (LightGBM)
# =====================================================================

class SeasonalityMLModel:
    """
    LightGBM regression model for predicting expected returns
    """
    
    def __init__(self):
        self.model = None
        self.feature_names = None
        self.is_trained = False
        
        # LightGBM parameters
        self.params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'force_col_wise': True,
        }
    
    def prepare_features(self, features_df: pd.DataFrame, regime_data: Dict) -> pd.DataFrame:
        """
        Prepare features for ML model
        
        Expected columns in features_df:
        - ticker
        - mom5, mom20, mom60, vol20 (momentum features)
        - season_week_avg, season_20d_avg, etc. (seasonality features)
        
        Args:
            features_df: DataFrame with all features
            regime_data: Regime detection result
        
        Returns:
            DataFrame with ML-ready features
        """
        df = features_df.copy()
        
        # Add regime as one-hot encoded features
        regime = regime_data['regime']
        for r in ['BULL_STRONG', 'BULL_WEAK', 'NEUTRAL_BULLISH', 'NEUTRAL_BEARISH', 
                  'BEAR_WEAK', 'BEAR_STRONG', 'CRISIS']:
            df[f'regime_{r}'] = 1 if regime == r else 0
        
        # Add regime composite score
        df['regime_score'] = regime_data['composite_score']
        
        # Add regime components
        for comp, val in regime_data['components'].items():
            df[f'regime_comp_{comp}'] = val
        
        # Select feature columns (exclude ticker, asof_date)
        feature_cols = [c for c in df.columns if c not in ['ticker', 'asof_date']]
        
        return df[feature_cols]
    
    def train(self, features: pd.DataFrame, targets: pd.Series, validation_split: float = 0.2):
        """
        Train LightGBM model
        
        Args:
            features: Feature matrix (no ticker column)
            targets: Target vector (forward returns)
            validation_split: Fraction for validation
        """
        if not HAS_LIGHTGBM:
            print("[WARN] LightGBM not installed, skipping training")
            return
        
        if features.empty or targets.empty:
            print("[WARN] Empty training data, skipping training")
            return
        
        # Store feature names
        self.feature_names = list(features.columns)
        
        # Split train/validation
        split_idx = int(len(features) * (1 - validation_split))
        
        train_features = features.iloc[:split_idx]
        train_targets = targets.iloc[:split_idx]
        val_features = features.iloc[split_idx:]
        val_targets = targets.iloc[split_idx:]
        
        # Create LightGBM datasets
        train_data = lgb.Dataset(train_features, label=train_targets)
        val_data = lgb.Dataset(val_features, label=val_targets, reference=train_data)
        
        # Train model
        print(f"[ML] Training LightGBM on {len(train_features)} samples...")
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=100,
            valid_sets=[val_data],
            valid_names=['validation'],
            callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]
        )
        
        self.is_trained = True
        print(f"[ML] Training complete. Best iteration: {self.model.best_iteration}")
    
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """
        Predict expected returns
        
        Args:
            features: Feature matrix (same columns as training)
        
        Returns:
            Array of predicted returns
        """
        if not self.is_trained or self.model is None:
            # Return zeros if not trained
            return np.zeros(len(features))
        
        # Ensure feature order matches training
        if self.feature_names:
            features = features[self.feature_names]
        
        predictions = self.model.predict(features)
        return predictions

# =====================================================================
# ENHANCED MAIN LOGIC
# =====================================================================

def _daily_features_from_prices(p: pd.DataFrame) -> Optional[pd.Series]:
    """
    Compute lightweight momentum features for *today* (last available row)
    
    Args:
        p: Price DataFrame with 'date' and 'close' columns
    
    Returns:
        Series with features or None
    """
    if p is None or len(p) < 60:
        return None
    
    s = p["close"].astype(float)
    ret = s.pct_change()
    mom5  = s.pct_change(5)
    mom20 = s.pct_change(20)
    mom60 = s.pct_change(60)
    vol20 = ret.rolling(20).std()

    last_idx = p.index[-1]
    feat = pd.Series({
        "asof_date": p.loc[last_idx, "date"],
        "mom5": float(mom5.iloc[-1]) if not pd.isna(mom5.iloc[-1]) else 0.0,
        "mom20": float(mom20.iloc[-1]) if not pd.isna(mom20.iloc[-1]) else 0.0,
        "mom60": float(mom60.iloc[-1]) if not pd.isna(mom60.iloc[-1]) else 0.0,
        "vol20": float(vol20.iloc[-1]) if not pd.isna(vol20.iloc[-1]) else 0.0,
    })
    return feat

def build_enhanced_features(universe: List[str], 
                            price_cache_dir: str, 
                            today: date,
                            regime_data: Dict,
                            seasonality_calc: SeasonalityCalculator,
                            trading_calc: TradingLevelsCalculator) -> pd.DataFrame:
    """
    Build enhanced feature DataFrame with:
    - Momentum (mom5, mom20, mom60, vol20)
    - Seasonality (multi-window)
    - Regime info
    - Entry/SL/TP levels
    
    Args:
        universe: List of tickers
        price_cache_dir: Path to price cache
        today: Current date
        regime_data: Regime detection result
        seasonality_calc: SeasonalityCalculator instance
        trading_calc: TradingLevelsCalculator instance
    
    Returns:
        DataFrame with enhanced features
    """
    from ml_unified_pipeline import _read_price_csv  # Use existing helper
    
    rows = []
    
    print(f"[STEP] Building enhanced features for {len(universe)} tickers...")
    
    for i, ticker in enumerate(universe):
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{len(universe)}")
        
        # Load prices
        prices = _read_price_csv(price_cache_dir, ticker)
        if prices is None or len(prices) < 60:
            continue
        
        # 1. Momentum features
        mom_feat = _daily_features_from_prices(prices)
        if mom_feat is None:
            continue
        
        # 2. Seasonality features
        season_feat = seasonality_calc.calculate_features(ticker, prices, today)
        
        # 3. Trading levels (Entry/SL/TP)
        trading_levels = trading_calc.calculate(ticker, prices, regime_data['regime'])
        
        # Combine all features
        row = {'ticker': ticker}
        row.update(mom_feat.to_dict())
        row.update(season_feat)
        row.update(trading_levels)
        
        # Add regime info
        row['regime'] = regime_data['regime']
        row['regime_score'] = regime_data['composite_score']
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    if not df.empty:
        # Clip extreme values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ['ticker', 'asof_date', 'in_bullish_segment', 'in_bearish_segment', 'days_into_segment']:
                df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    
    print(f"[OK] Built features for {len(df)} tickers")
    return df

def train_ml_model_rolling(features_df: pd.DataFrame, 
                           price_cache_dir: str,
                           today: date,
                           regime_data: Dict,
                           ml_model: SeasonalityMLModel) -> SeasonalityMLModel:
    """
    Train ML model on rolling 252-day window
    
    Args:
        features_df: Today's features
        price_cache_dir: Path to prices
        today: Current date
        regime_data: Regime data
        ml_model: ML model instance
    
    Returns:
        Trained model
    """
    if not HAS_LIGHTGBM:
        print("[WARN] LightGBM not available, skipping ML training")
        return ml_model
    
    print("[STEP] Training ML model (rolling 252 days)...")
    
    # For simplicity: Use a placeholder target (0.0)
    # In production, you would:
    # 1. Load historical features (past 252 days)
    # 2. Calculate forward 20-day returns as targets
    # 3. Train model on this data
    
    # Placeholder: Assume we have historical data
    # (In real implementation, you'd need to store/load historical features)
    
    # For now, skip training and return untrained model
    print("[INFO] ML training skipped (no historical data pipeline yet)")
    print("[INFO] Using simple momentum scoring instead")
    
    return ml_model

def predict_with_ml(features_df: pd.DataFrame, 
                   regime_data: Dict,
                   ml_model: SeasonalityMLModel) -> pd.DataFrame:
    """
    Generate predictions (or fallback to momentum scoring)
    
    Args:
        features_df: Feature DataFrame
        regime_data: Regime data
        ml_model: Trained ML model (or None)
    
    Returns:
        DataFrame with predictions
    """
    df = features_df.copy()
    
    if ml_model.is_trained and HAS_LIGHTGBM:
        # Use ML model
        print("[STEP] Generating ML predictions...")
        ml_features = ml_model.prepare_features(df, regime_data)
        predictions = ml_model.predict(ml_features)
        df['ml_expected_return'] = predictions
    else:
        # Fallback: momentum + seasonality scoring
        print("[STEP] Using momentum+seasonality scoring (no ML model)...")
        
        # Blend momentum and seasonality
        mom_score = (0.6 * df['mom5'] + 0.4 * df['mom20'])
        season_score = (df['season_week_avg'] + df['season_20d_avg']) / 2
        
        # Combined score
        combined = 0.5 * mom_score + 0.5 * season_score
        
        df['ml_expected_return'] = combined
    
    # Rank to 0-1 score
    df['score_long'] = df['ml_expected_return'].rank(pct=True)
    df['score_short'] = 1.0 - df['score_long']
    
    return df

# =====================================================================
# ENHANCED WRITE REPORTS
# =====================================================================

def write_enhanced_reports(run_root: str, 
                          today: date,
                          features_df: pd.DataFrame,
                          regime_data: Dict,
                          gate_alpha: float):
    """
    Write enhanced reports with all new features
    
    Args:
        run_root: Output directory
        today: Current date
        features_df: Enhanced features DataFrame
        regime_data: Regime detection result
        gate_alpha: Gating threshold
    """
    tag = today.strftime("%Y-%m-%d")
    reports_dir = os.path.join(run_root, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    
    if features_df.empty:
        print("[WARN] Empty features, writing header-only files")
        # Write empty files
        for fname in ['features', 'labels', 'top_long_candidates_RAW', 'top_long_candidates_GATED',
                      'top_short_candidates_RAW', 'top_short_candidates_GATED']:
            path = os.path.join(reports_dir, f"{fname}_{tag}.csv")
            pd.DataFrame().to_csv(path, index=False)
        return
    
    # 1. features_{date}.csv (full enhanced features)
    features_path = os.path.join(reports_dir, f"features_{tag}.csv")
    features_df.to_csv(features_path, index=False)
    print(f"[OK] Wrote: {features_path} ({len(features_df)} rows)")
    
    # 2. labels_{date}.csv (placeholder)
    labels_path = os.path.join(reports_dir, f"labels_{tag}.csv")
    pd.DataFrame(columns=['ticker', 'target']).to_csv(labels_path, index=False)
    
    # 3. top_long_candidates_RAW_{date}.csv (all, sorted by score_long)
    longs_raw = features_df.sort_values('score_long', ascending=False).head(200)
    longs_raw_path = os.path.join(reports_dir, f"top_long_candidates_RAW_{tag}.csv")
    longs_raw.to_csv(longs_raw_path, index=False)
    print(f"[OK] Wrote: {longs_raw_path} ({len(longs_raw)} rows)")
    
    # 4. top_long_candidates_GATED_{date}.csv (filtered by gate_alpha)
    longs_gated = longs_raw[longs_raw['score_long'] >= gate_alpha]
    longs_gated_path = os.path.join(reports_dir, f"top_long_candidates_GATED_{tag}.csv")
    longs_gated.to_csv(longs_gated_path, index=False)
    print(f"[OK] Wrote: {longs_gated_path} ({len(longs_gated)} rows)")
    
    # 5. top_short_candidates_RAW_{date}.csv
    shorts_raw = features_df.sort_values('score_short', ascending=False).head(200)
    shorts_raw_path = os.path.join(reports_dir, f"top_short_candidates_RAW_{tag}.csv")
    shorts_raw.to_csv(shorts_raw_path, index=False)
    
    # 6. top_short_candidates_GATED_{date}.csv
    shorts_gated = shorts_raw[shorts_raw['score_short'] >= gate_alpha]
    shorts_gated_path = os.path.join(reports_dir, f"top_short_candidates_GATED_{tag}.csv")
    shorts_gated.to_csv(shorts_gated_path, index=False)
    
    # 7. summary_{date}.txt
    summary_path = os.path.join(reports_dir, f"summary_{tag}.txt")
    with open(summary_path, 'w') as f:
        f.write(f"Enhanced ML Pipeline Summary - {tag}\n")
        f.write("="*60 + "\n\n")
        f.write(f"Regime: {regime_data['regime']}\n")
        f.write(f"Composite Score: {regime_data['composite_score']:.3f}\n\n")
        f.write("Regime Components:\n")
        for comp, val in regime_data['components'].items():
            f.write(f"  {comp}: {val:.3f}\n")
        f.write(f"\nGate Alpha: {gate_alpha}\n")
        f.write(f"Total Tickers Analyzed: {len(features_df)}\n")
        f.write(f"Long Candidates (RAW): {len(longs_raw)}\n")
        f.write(f"Long Candidates (GATED): {len(longs_gated)}\n")
        f.write(f"Short Candidates (GATED): {len(shorts_gated)}\n")
    
    print(f"[OK] Wrote: {summary_path}")
    print(f"[STEP] All reports written to: {reports_dir}")

# =====================================================================
# CLI & MAIN (keep same interface)
# =====================================================================

def parse_args():
    """Parse command line arguments (same as before)"""
    p = argparse.ArgumentParser(
        description="Enhanced Seasonality ML Pipeline (Regime + Seasonality + ML)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--run_root", type=str, default=None, help="Run root folder")
    p.add_argument("--today", type=str, default=None, help="YYYY-MM-DD; defaults to today")
    p.add_argument("--universe_csv", type=str, default="seasonality_reports/constituents_raw.csv",
                   help="Universe CSV")
    p.add_argument("--gate_alpha", type=float, default=0.10,
                   help="Gating threshold on 0-1 scores")
    p.add_argument("--feature_mode", type=str, default="enhanced", help="Feature mode")
    p.add_argument("--train_years", type=int, default=10, help="Seasonality lookback years")
    p.add_argument("--min_samples_per_regime", type=int, default=60, help="Min samples")
    p.add_argument("--vintage_cutoff", type=str, default=None, help="Vintage cutoff")
    return p.parse_args()

def _as_date(s: Optional[str]) -> date:
    """Convert string to date"""
    if s is None:
        return date.today()
    return datetime.strptime(s, "%Y-%m-%d").date()

def _find_reports_root() -> str:
    """Find seasonality_reports directory"""
    return os.path.join(os.getcwd(), "seasonality_reports")

def _run_root_default(reports_root: str, d: date) -> str:
    """Generate default run root path"""
    tag = d.strftime("%Y-%m-%d_%H%M")
    return os.path.join(reports_root, "runs", tag)

def _read_universe(path: str) -> pd.DataFrame:
    """Read universe CSV (same as before)"""
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    cand = None
    for k in ["ticker", "symbol", "ric", "isin"]:
        if k in cols:
            cand = cols[k]
            break
    if cand is None:
        df.columns = ["ticker"] + list(df.columns[1:])
        cand = "ticker"
    df["ticker"] = df[cand].astype(str).str.strip().str.upper()
    df = df.dropna(subset=["ticker"]).drop_duplicates("ticker")
    return df[["ticker"]]

def _read_price_csv(pc_dir: str, ticker: str) -> Optional[pd.DataFrame]:
    """Read price CSV (same as before)"""
    patterns = [
        os.path.join(pc_dir, f"{ticker}.csv"),
        os.path.join(pc_dir, f"{ticker.upper()}.csv"),
        os.path.join(pc_dir, f"{ticker.lower()}.csv"),
    ]
    path = None
    for p in patterns:
        if os.path.isfile(p):
            path = p
            break
    if path is None:
        return None
    try:
        df = pd.read_csv(path)
        cols = {c.lower(): c for c in df.columns}
        date_col = cols.get("date")
        if date_col is None:
            for k in ["time", "datetime"]:
                if k in cols:
                    date_col = cols[k]
                    break
        close_col = None
        for k in ["adj close", "adj_close", "close", "last", "price"]:
            if k in cols:
                close_col = cols[k]
                break
        if date_col is None or close_col is None:
            return None
        out = df[[date_col, close_col]].rename(columns={date_col: "date", close_col: "close"})
        out["date"] = pd.to_datetime(out["date"]).dt.date
        out = out.sort_values("date").dropna()
        return out
    except Exception:
        return None

def _find_price_cache_dir(reports_root: str) -> Optional[str]:
    """Find price cache directory (same as before)"""
    env = os.environ.get("PRICE_CACHE_DIR")
    if env and os.path.isdir(env):
        return env
    
    canonical = os.path.join(reports_root, "runs", "2025-10-04_0903", "price_cache")
    if os.path.isdir(canonical):
        return canonical
    
    runs_root = os.path.join(reports_root, "runs")
    if not os.path.isdir(runs_root):
        return None
    candidates = []
    for d in glob.glob(os.path.join(runs_root, "*")):
        pc = os.path.join(d, "price_cache")
        if os.path.isdir(pc):
            candidates.append(pc)
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0]

def main():
    """Enhanced main function"""
    print("\n" + "="*60)
    print("ENHANCED SEASONALITY ML PIPELINE v2.0")
    print("Features: Regime Detection + Seasonality + ML + Trading Levels")
    print("="*60 + "\n")
    
    args = parse_args()
    today = _as_date(args.today)
    reports_root = _find_reports_root()
    
    run_root = args.run_root or _run_root_default(reports_root, today)
    os.makedirs(run_root, exist_ok=True)
    
    print(f"[INFO] Today: {today.isoformat()}")
    print(f"[INFO] Run root: {run_root}")
    
    # Load universe
    uni_path = args.universe_csv
    if not os.path.isabs(uni_path):
        uni_path = os.path.join(os.getcwd(), uni_path)
    
    if not os.path.isfile(uni_path):
        print(f"[ERROR] Universe CSV not found: {uni_path}")
        sys.exit(1)
    
    universe_df = _read_universe(uni_path)
    tickers = universe_df["ticker"].tolist()
    print(f"[INFO] Universe: {len(tickers)} tickers")
    
    # Find price cache
    stock_price_cache = _find_price_cache_dir(reports_root)
    macro_price_cache = os.path.join(reports_root, "price_cache")
    
    if stock_price_cache is None:
        print("[ERROR] Stock price cache not found")
        sys.exit(1)
    
    if not os.path.isdir(macro_price_cache):
        print("[ERROR] Macro price cache not found (for regime detection)")
        sys.exit(1)
    
    print(f"[INFO] Stock prices: {stock_price_cache}")
    print(f"[INFO] Macro prices: {macro_price_cache}")
    
    # 1. Detect Regime
    print("\n[STEP 1/5] Detecting market regime...")
    regime_calc = RegimeCalculator(macro_price_cache)
    regime_data = regime_calc.detect_regime(today)
    print(f"[OK] Regime: {regime_data['regime']} (score: {regime_data['composite_score']:.3f})")
    
    # 2. Build Enhanced Features
    print("\n[STEP 2/5] Building enhanced features...")
    seasonality_calc = SeasonalityCalculator(lookback_years=args.train_years)
    trading_calc = TradingLevelsCalculator()
    
    features_df = build_enhanced_features(
        tickers, 
        stock_price_cache, 
        today, 
        regime_data,
        seasonality_calc,
        trading_calc
    )
    
    if features_df.empty:
        print("[ERROR] No features generated")
        sys.exit(1)
    
    print(f"[OK] Generated features for {len(features_df)} tickers")
    print(f"[OK] Feature columns: {list(features_df.columns)}")
    
    # 3. Train ML Model (rolling)
    print("\n[STEP 3/5] Training ML model...")
    ml_model = SeasonalityMLModel()
    ml_model = train_ml_model_rolling(features_df, stock_price_cache, today, regime_data, ml_model)
    
    # 4. Generate Predictions
    print("\n[STEP 4/5] Generating predictions...")
    predictions_df = predict_with_ml(features_df, regime_data, ml_model)
    
    # 5. Write Reports
    print("\n[STEP 5/5] Writing reports...")
    write_enhanced_reports(run_root, today, predictions_df, regime_data, args.gate_alpha)
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print(f"Reports: {os.path.join(run_root, 'reports')}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()


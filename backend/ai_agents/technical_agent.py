"""
Volatility-Based Technical Entry Point Optimizer

This module provides advanced technical analysis for identifying optimal entry points
in penny stocks during volatile market conditions. It calculates multiple technical
indicators (RSI, Bollinger Bands, MACD) and generates entry signals based on
quantitative criteria.

Designed specifically for BSE Bhavcopy data with handling for:
- Stagnant price movements (zero-division errors)
- Illiquid stocks (volume filtering)
- Penny stock volatility patterns

Dependencies:
    - pandas>=1.5.0
    - numpy>=1.23.0
    - scipy>=1.9.0
"""

import logging
import warnings
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import math

import pandas as pd
import numpy as np
from scipy.signal import argrelextrema


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class EntrySignal:
    """Data class to represent a technical entry signal."""
    signal_type: str  # 'BUY', 'SELL', 'HOLD', 'WEAK_BUY', 'STRONG_BUY'
    confidence: float  # 0-1
    entry_price: float
    stop_loss: float
    target_price: float
    rsi_value: float
    bb_lower: float
    bb_upper: float
    bb_position: float  # 0-1, price position within bands
    macd_signal: str  # 'BULLISH', 'BEARISH', 'NEUTRAL'
    volume_ma_ratio: float  # current_volume / volume_ma
    timestamp: datetime
    indicators_detail: Dict = None


class VolatilityEntryOptimizer:
    """
    Quantitative technical analysis engine for penny stock entry optimization.
    
    This class implements a comprehensive technical analysis framework specifically
    designed for Indian penny stocks. It handles:
    
    - RSI calculation with stagnant price movement handling
    - Bollinger Bands for volatility measurement
    - MACD for trend confirmation
    - Volume analysis for liquidity filtering
    - Multi-factor entry signal generation
    
    Attributes:
        rsi_period (int): Period for RSI calculation (default: 14)
        bb_period (int): Period for Bollinger Bands SMA (default: 20)
        bb_std_dev (float): Standard deviations for BB width (default: 2.0)
        macd_fast (int): Fast EMA period for MACD (default: 12)
        macd_slow (int): Slow EMA period for MACD (default: 26)
        macd_signal (int): Signal line period for MACD (default: 9)
        volume_ma_period (int): Period for volume moving average (default: 5)
        
    Example:
        >>> optimizer = VolatilityEntryOptimizer()
        >>> entry_signal = optimizer.get_entry_signal(bse_data)
        >>> print(f"Signal: {entry_signal.signal_type}")
        >>> print(f"Entry Price: {entry_signal.entry_price:.2f}")
    """
    
    # RSI Threshold levels
    RSI_OVERBOUGHT = 70
    RSI_OVERSOLD = 30
    RSI_STRONG_OVERSOLD = 20
    
    # Entry signal thresholds
    BB_LOWER_THRESHOLD = 0.10  # 10% above lower band triggers buy
    VOLUME_MA_MULTIPLIER = 1.0  # Current volume must be >= this * volume MA
    MIN_PRICE_CHANGE_THRESHOLD = 0.001  # Minimum price change to avoid division errors
    
    def __init__(
        self,
        rsi_period: int = 14,
        bb_period: int = 20,
        bb_std_dev: float = 2.0,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        volume_ma_period: int = 5
    ):
        """
        Initialize the Volatility Entry Optimizer.
        
        Args:
            rsi_period: Period for RSI calculation
            bb_period: Period for Bollinger Bands
            bb_std_dev: Standard deviations for Bollinger Bands
            macd_fast: Fast EMA period for MACD
            macd_slow: Slow EMA period for MACD
            macd_signal: Signal line period for MACD
            volume_ma_period: Period for volume moving average
        """
        self.rsi_period = rsi_period
        self.bb_period = bb_period
        self.bb_std_dev = bb_std_dev
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal_period = macd_signal
        self.volume_ma_period = volume_ma_period
        
        logger.info(
            f"Initialized VolatilityEntryOptimizer: RSI={rsi_period}, "
            f"BB={bb_period}, MACD=({macd_fast},{macd_slow},{macd_signal})"
        )
    
    def calculate_rsi(
        self,
        prices: pd.Series,
        period: int = None
    ) -> pd.Series:
        """
        Calculate the 14-day Relative Strength Index (RSI).
        
        Handles penny stocks with stagnant price movements by:
        - Detecting zero-division scenarios
        - Using smoothed gains/losses
        - Returning NaN for insufficiently volatile periods
        
        Args:
            prices: Series of price data (typically closing prices)
            period: RSI period (uses self.rsi_period if None)
        
        Returns:
            Series with RSI values (0-100)
        
        Raises:
            ValueError: If prices series is empty or too short
            
        Example:
            >>> prices = pd.Series([100, 101, 99, 102, 100, 103])
            >>> rsi = optimizer.calculate_rsi(prices)
        """
        if period is None:
            period = self.rsi_period
        
        if len(prices) < period + 1:
            raise ValueError(
                f"Prices series length ({len(prices)}) must be > period ({period})"
            )
        
        logger.debug(f"Calculating RSI with period={period}")
        
        # Calculate price changes
        delta = prices.diff()
        
        # Separate gains and losses
        gains = delta.where(delta > 0, 0.0)
        losses = -delta.where(delta < 0, 0.0)
        
        # Calculate average gains and losses using EMA
        avg_gains = gains.ewm(span=period, adjust=False).mean()
        avg_losses = losses.ewm(span=period, adjust=False).mean()
        
        # Handle zero-division: when avg_losses is 0 (stagnant market)
        rs = pd.Series(index=prices.index, dtype=float)
        
        # Where losses exist, calculate RS normally
        mask_with_losses = avg_losses != 0
        rs[mask_with_losses] = avg_gains[mask_with_losses] / avg_losses[mask_with_losses]
        
        # Where losses are 0 but gains exist, RS approaches infinity
        mask_no_losses = (avg_losses == 0) & (avg_gains > 0)
        rs[mask_no_losses] = 100.0  # Maximum RSI for uptrend with no losses
        
        # Where both are 0 (completely stagnant), RSI = 50 (neutral)
        mask_stagnant = (avg_losses == 0) & (avg_gains == 0)
        rs[mask_stagnant] = 50.0
        
        # Calculate RSI
        rsi = 100 - (100 / (1 + rs))
        
        # Ensure RSI stays within valid bounds
        rsi = rsi.clip(0, 100)
        
        logger.debug(f"RSI calculated. Min: {rsi.min():.2f}, Max: {rsi.max():.2f}, "
                    f"Current: {rsi.iloc[-1]:.2f}")
        
        return rsi
    
    def calculate_bollinger_bands(
        self,
        prices: pd.Series,
        period: int = None,
        num_std_dev: float = None
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands (20-day SMA with 2 std deviations).
        
        Returns:
            Tuple of (middle_band, upper_band, lower_band, bandwidth)
        
        Args:
            prices: Series of price data
            period: BB period (uses self.bb_period if None)
            num_std_dev: Standard deviations (uses self.bb_std_dev if None)
        
        Returns:
            Tuple containing:
                - middle_band: 20-day Simple Moving Average
                - upper_band: Middle + (2 * Std Dev)
                - lower_band: Middle - (2 * Std Dev)
                - bandwidth: (Upper - Lower) / Middle (volatility measure)
        
        Raises:
            ValueError: If prices series is too short
        """
        if period is None:
            period = self.bb_period
        if num_std_dev is None:
            num_std_dev = self.bb_std_dev
        
        if len(prices) < period:
            raise ValueError(
                f"Prices series length ({len(prices)}) must be >= period ({period})"
            )
        
        logger.debug(f"Calculating Bollinger Bands: period={period}, std_dev={num_std_dev}")
        
        # Calculate middle band (SMA)
        middle_band = prices.rolling(window=period).mean()
        
        # Calculate standard deviation
        std_dev = prices.rolling(window=period).std()
        
        # Calculate bands
        upper_band = middle_band + (std_dev * num_std_dev)
        lower_band = middle_band - (std_dev * num_std_dev)
        
        # Calculate bandwidth (volatility indicator)
        # Avoid division by zero
        bandwidth = pd.Series(index=prices.index, dtype=float)
        mask = middle_band != 0
        bandwidth[mask] = (upper_band[mask] - lower_band[mask]) / middle_band[mask]
        bandwidth[~mask] = 0.0
        
        logger.debug(f"Bollinger Bands calculated. Bandwidth range: "
                    f"{bandwidth.min():.4f} - {bandwidth.max():.4f}")
        
        return middle_band, upper_band, lower_band, bandwidth
    
    def calculate_macd(
        self,
        prices: pd.Series,
        fast: int = None,
        slow: int = None,
        signal_period: int = None
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        Args:
            prices: Series of price data
            fast: Fast EMA period (uses self.macd_fast if None)
            slow: Slow EMA period (uses self.macd_slow if None)
            signal_period: Signal line EMA period (uses self.macd_signal_period if None)
        
        Returns:
            Tuple containing:
                - macd_line: MACD line (fast EMA - slow EMA)
                - signal_line: 9-period EMA of MACD
                - histogram: MACD - Signal line
        
        Raises:
            ValueError: If prices series is too short
        """
        if fast is None:
            fast = self.macd_fast
        if slow is None:
            slow = self.macd_slow
        if signal_period is None:
            signal_period = self.macd_signal_period
        
        min_required = slow + signal_period
        if len(prices) < min_required:
            raise ValueError(
                f"Prices series length ({len(prices)}) must be >= {min_required}"
            )
        
        logger.debug(f"Calculating MACD: fast={fast}, slow={slow}, signal={signal_period}")
        
        # Calculate EMAs
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        
        # Calculate MACD line
        macd_line = ema_fast - ema_slow
        
        # Calculate signal line
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        
        # Calculate histogram
        histogram = macd_line - signal_line
        
        logger.debug(f"MACD calculated. Current MACD: {macd_line.iloc[-1]:.6f}, "
                    f"Signal: {signal_line.iloc[-1]:.6f}")
        
        return macd_line, signal_line, histogram
    
    def _calculate_volume_analysis(
        self,
        df: pd.DataFrame,
        volume_column: str = 'volume'
    ) -> pd.Series:
        """
        Calculate volume-based metrics.
        
        Args:
            df: DataFrame with volume data
            volume_column: Name of volume column
        
        Returns:
            Series with volume moving average
        """
        if volume_column not in df.columns:
            logger.warning(f"Volume column '{volume_column}' not found")
            return pd.Series(1.0, index=df.index)
        
        volume_ma = df[volume_column].rolling(window=self.volume_ma_period).mean()
        
        # Forward-fill first few values
        volume_ma = volume_ma.fillna(method='bfill')
        
        return volume_ma
    
    def _calculate_price_position_in_bands(
        self,
        close_price: float,
        lower_band: float,
        upper_band: float
    ) -> float:
        """
        Calculate price position within Bollinger Bands (0-1).
        
        Args:
            close_price: Current close price
            lower_band: Lower Bollinger Band
            upper_band: Upper Bollinger Band
        
        Returns:
            Position between 0 (at lower band) and 1 (at upper band)
        """
        band_width = upper_band - lower_band
        
        if band_width == 0:
            return 0.5  # Neutral position if bands are collapsed
        
        position = (close_price - lower_band) / band_width
        return np.clip(position, 0.0, 1.0)
    
    def _determine_macd_signal(
        self,
        macd_line: float,
        signal_line: float,
        histogram: float
    ) -> str:
        """
        Determine MACD signal type.
        
        Args:
            macd_line: Current MACD line value
            signal_line: Current signal line value
            histogram: Current histogram value
        
        Returns:
            Signal type: 'BULLISH', 'BEARISH', or 'NEUTRAL'
        """
        # MACD above signal line = bullish
        if macd_line > signal_line and histogram > 0:
            return 'BULLISH'
        # MACD below signal line = bearish
        elif macd_line < signal_line and histogram < 0:
            return 'BEARISH'
        else:
            return 'NEUTRAL'
    
    def get_entry_signal(
        self,
        df: pd.DataFrame,
        close_column: str = 'close_price',
        volume_column: str = 'volume'
    ) -> EntrySignal:
        """
        Identify optimal entry point during market volatility.
        
        Generates a 'BUY' signal when:
        1. Price drops near or below Lower Bollinger Band (within 10%)
        2. RSI indicates oversold condition (< 30)
        3. Current volume > 5-day volume moving average (filters illiquid stocks)
        
        Returns additional signals:
        - STRONG_BUY: All three conditions + RSI < 20 + MACD bullish
        - WEAK_BUY: Two conditions met
        - SELL: Overbought conditions (RSI > 70) + price near upper band
        - HOLD: No clear signal
        
        Args:
            df: BSE Bhavcopy DataFrame with OHLCV data
            close_column: Name of close price column
            volume_column: Name of volume column
        
        Returns:
            EntrySignal object with signal details
        
        Raises:
            ValueError: If required columns missing or insufficient data
            
        Example:
            >>> signal = optimizer.get_entry_signal(bse_data)
            >>> if signal.signal_type == 'BUY':
            ...     print(f"Enter at {signal.entry_price}")
            ...     print(f"Stop loss at {signal.stop_loss}")
        """
        # Validate required columns
        required_cols = ['open_price', 'high_price', 'low_price', close_column, volume_column]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Ensure sufficient data
        min_required = max(self.bb_period + 1, self.macd_slow + self.macd_signal_period)
        if len(df) < min_required:
            raise ValueError(
                f"DataFrame length ({len(df)}) must be >= {min_required}"
            )
        
        logger.info(f"Analyzing entry signal for {len(df)} rows")
        
        # Calculate technical indicators
        close_prices = df[close_column]
        
        # RSI
        try:
            rsi_series = self.calculate_rsi(close_prices)
            rsi_value = rsi_series.iloc[-1]
        except Exception as e:
            logger.error(f"RSI calculation failed: {e}")
            rsi_value = 50.0
        
        # Bollinger Bands
        try:
            middle_band, upper_band, lower_band, bandwidth = self.calculate_bollinger_bands(
                close_prices
            )
            bb_middle = middle_band.iloc[-1]
            bb_upper = upper_band.iloc[-1]
            bb_lower = lower_band.iloc[-1]
        except Exception as e:
            logger.error(f"Bollinger Bands calculation failed: {e}")
            bb_middle = close_prices.iloc[-1]
            bb_upper = close_prices.iloc[-1] * 1.05
            bb_lower = close_prices.iloc[-1] * 0.95
        
        # MACD
        try:
            macd_line, signal_line, histogram = self.calculate_macd(close_prices)
            macd_value = macd_line.iloc[-1]
            signal_value = signal_line.iloc[-1]
            histogram_value = histogram.iloc[-1]
            macd_signal = self._determine_macd_signal(macd_value, signal_value, histogram_value)
        except Exception as e:
            logger.error(f"MACD calculation failed: {e}")
            macd_signal = 'NEUTRAL'
        
        # Volume analysis
        try:
            volume_ma = self._calculate_volume_analysis(df, volume_column)
            current_volume = df[volume_column].iloc[-1]
            volume_ma_value = volume_ma.iloc[-1]
            volume_ma_ratio = current_volume / volume_ma_value if volume_ma_value > 0 else 1.0
        except Exception as e:
            logger.error(f"Volume analysis failed: {e}")
            current_volume = df[volume_column].iloc[-1]
            volume_ma_ratio = 1.0
        
        # Current prices
        current_close = close_prices.iloc[-1]
        current_low = df['low_price'].iloc[-1]
        
        # Calculate price position in bands
        bb_position = self._calculate_price_position_in_bands(
            current_close, bb_lower, bb_upper
        )
        
        logger.info(
            f"Entry Signal Analysis - RSI: {rsi_value:.2f}, "
            f"BB Position: {bb_position:.2f}, Volume Ratio: {volume_ma_ratio:.2f}, "
            f"MACD Signal: {macd_signal}"
        )
        
        # Determine entry signal
        signal_type = 'HOLD'
        confidence = 0.0
        
        # BUY Signal Criteria
        near_lower_band = current_close <= (bb_lower + (bb_upper - bb_lower) * self.BB_LOWER_THRESHOLD)
        oversold = rsi_value < self.RSI_OVERSOLD
        sufficient_volume = volume_ma_ratio >= self.VOLUME_MA_MULTIPLIER
        strong_oversold = rsi_value < self.RSI_STRONG_OVERSOLD
        
        buy_conditions = [near_lower_band, oversold, sufficient_volume]
        buy_count = sum(buy_conditions)
        
        if buy_count >= 3:  # All conditions met
            if strong_oversold and macd_signal == 'BULLISH':
                signal_type = 'STRONG_BUY'
                confidence = 0.9
            else:
                signal_type = 'BUY'
                confidence = 0.75
        elif buy_count == 2:
            signal_type = 'WEAK_BUY'
            confidence = 0.55
        
        # SELL Signal Criteria
        near_upper_band = current_close >= (bb_upper - (bb_upper - bb_lower) * self.BB_LOWER_THRESHOLD)
        overbought = rsi_value > self.RSI_OVERBOUGHT
        
        if near_upper_band and overbought and macd_signal == 'BEARISH':
            signal_type = 'SELL'
            confidence = 0.75
        
        # Calculate entry price and stops
        if signal_type in ['BUY', 'STRONG_BUY', 'WEAK_BUY']:
            entry_price = current_close
            # Stop loss at lower band or 2% below current, whichever is lower
            stop_loss = min(bb_lower, current_close * 0.98)
            # Target at upper band or 3% above current
            target_price = max(bb_upper, current_close * 1.03)
        elif signal_type == 'SELL':
            entry_price = current_close
            stop_loss = bb_upper
            target_price = bb_lower
        else:  # HOLD
            entry_price = current_close
            stop_loss = current_close * 0.95
            target_price = current_close * 1.05
        
        # Create signal object
        signal = EntrySignal(
            signal_type=signal_type,
            confidence=confidence,
            entry_price=entry_price,
            stop_loss=stop_loss,
            target_price=target_price,
            rsi_value=rsi_value,
            bb_lower=bb_lower,
            bb_upper=bb_upper,
            bb_position=bb_position,
            macd_signal=macd_signal,
            volume_ma_ratio=volume_ma_ratio,
            timestamp=datetime.now(),
            indicators_detail={
                'near_lower_band': bool(near_lower_band),
                'oversold': bool(oversold),
                'sufficient_volume': bool(sufficient_volume),
                'buy_conditions_met': int(buy_count),
                'bb_middle': bb_middle,
                'macd_line': macd_value,
                'signal_line': signal_value,
                'histogram': histogram_value
            }
        )
        
        logger.info(f"Entry Signal Generated: {signal_type} (Confidence: {confidence:.2f})")
        
        return signal
    
    def batch_analyze_multiple_symbols(
        self,
        data_dict: Dict[str, pd.DataFrame]
    ) -> Dict[str, EntrySignal]:
        """
        Analyze entry signals for multiple stock symbols.
        
        Args:
            data_dict: Dictionary mapping symbol -> DataFrame
        
        Returns:
            Dictionary mapping symbol -> EntrySignal
            
        Example:
            >>> signals = optimizer.batch_analyze_multiple_symbols({
            ...     'PENNY1': df1,
            ...     'PENNY2': df2,
            ...     'PENNY3': df3
            ... })
            >>> for symbol, signal in signals.items():
            ...     if signal.signal_type == 'STRONG_BUY':
            ...         print(f"{symbol}: {signal.signal_type}")
        """
        results = {}
        
        for symbol, df in data_dict.items():
            try:
                logger.info(f"Analyzing symbol: {symbol}")
                signal = self.get_entry_signal(df)
                results[symbol] = signal
            except Exception as e:
                logger.error(f"Failed to analyze {symbol}: {e}")
                results[symbol] = EntrySignal(
                    signal_type='HOLD',
                    confidence=0.0,
                    entry_price=0.0,
                    stop_loss=0.0,
                    target_price=0.0,
                    rsi_value=0.0,
                    bb_lower=0.0,
                    bb_upper=0.0,
                    bb_position=0.5,
                    macd_signal='NEUTRAL',
                    volume_ma_ratio=0.0,
                    timestamp=datetime.now()
                )
        
        return results
    
    def create_entry_report(
        self,
        signal: EntrySignal
    ) -> str:
        """
        Generate a formatted entry signal report.
        
        Args:
            signal: EntrySignal object
        
        Returns:
            Formatted string report
        """
        report = []
        report.append("=" * 80)
        report.append("TECHNICAL ENTRY SIGNAL REPORT")
        report.append("=" * 80)
        report.append(f"Timestamp: {signal.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Signal Summary
        report.append("SIGNAL SUMMARY")
        report.append("-" * 80)
        report.append(f"Signal Type: {signal.signal_type}")
        report.append(f"Confidence: {signal.confidence * 100:.1f}%")
        report.append("")
        
        # Price Levels
        report.append("PRICE LEVELS")
        report.append("-" * 80)
        report.append(f"Entry Price: ₹{signal.entry_price:.2f}")
        report.append(f"Stop Loss: ₹{signal.stop_loss:.2f}")
        report.append(f"Target Price: ₹{signal.target_price:.2f}")
        
        risk_reward = (signal.target_price - signal.entry_price) / (signal.entry_price - signal.stop_loss) if signal.entry_price != signal.stop_loss else 0
        report.append(f"Risk/Reward Ratio: {risk_reward:.2f}:1")
        report.append("")
        
        # Technical Indicators
        report.append("TECHNICAL INDICATORS")
        report.append("-" * 80)
        report.append(f"RSI (14): {signal.rsi_value:.2f}")
        if signal.rsi_value < 30:
            report.append("  └─ Status: OVERSOLD ⬇️")
        elif signal.rsi_value > 70:
            report.append("  └─ Status: OVERBOUGHT ⬆️")
        else:
            report.append("  └─ Status: NEUTRAL")
        
        report.append(f"Bollinger Bands Position: {signal.bb_position * 100:.1f}%")
        report.append(f"  Lower Band: ₹{signal.bb_lower:.2f}")
        report.append(f"  Upper Band: ₹{signal.bb_upper:.2f}")
        
        report.append(f"MACD Signal: {signal.macd_signal}")
        
        report.append(f"Volume (MA Ratio): {signal.volume_ma_ratio:.2f}x")
        if signal.volume_ma_ratio >= 1.0:
            report.append("  └─ Status: ABOVE AVERAGE ✓")
        else:
            report.append("  └─ Status: BELOW AVERAGE ⚠️")
        report.append("")
        
        # Conditions Met
        if signal.indicators_detail:
            report.append("CONDITIONS MET")
            report.append("-" * 80)
            detail = signal.indicators_detail
            report.append(f"Near Lower Band: {'✓' if detail.get('near_lower_band') else '✗'}")
            report.append(f"Oversold (RSI<30): {'✓' if detail.get('oversold') else '✗'}")
            report.append(f"Sufficient Volume: {'✓' if detail.get('sufficient_volume') else '✗'}")
            report.append(f"Total Conditions Met: {detail.get('buy_conditions_met', 0)}/3")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)


if __name__ == "__main__":
    # Example usage
    logger.info("Starting VolatilityEntryOptimizer demonstration")
    
    # Initialize optimizer
    optimizer = VolatilityEntryOptimizer()
    
    # Generate sample BSE Bhavcopy-like data
    print("\n" + "=" * 80)
    print("GENERATING SAMPLE BSE BHAVCOPY DATA")
    print("=" * 80)
    
    np.random.seed(42)
    dates = pd.date_range('2025-01-01', periods=100)
    
    # Generate realistic penny stock prices with volatility
    base_price = 25.0
    prices = [base_price]
    
    for i in range(1, 100):
        # Random walk with volatility
        change = np.random.randn() * 1.5  # Higher volatility for penny stocks
        prices.append(max(prices[-1] + change, 5.0))  # Ensure price doesn't go negative
    
    sample_data = pd.DataFrame({
        'date': dates,
        'open_price': [p + np.random.randn() * 0.5 for p in prices],
        'high_price': [p + abs(np.random.randn() * 1.0) for p in prices],
        'low_price': [max(p - abs(np.random.randn() * 1.0), 5.0) for p in prices],
        'close_price': prices,
        'volume': np.random.randint(100000, 1000000, 100)
    })
    
    print("Sample data shape:", sample_data.shape)
    print("\nFirst 5 rows:")
    print(sample_data.head())
    print("\nLast 5 rows:")
    print(sample_data.tail())
    
    # Analyze entry signal
    print("\n" + "=" * 80)
    print("ANALYZING ENTRY SIGNAL")
    print("=" * 80)
    
    try:
        signal = optimizer.get_entry_signal(sample_data)
        print(optimizer.create_entry_report(signal))
    except Exception as e:
        logger.error(f"Signal analysis failed: {e}")
    
    # Test with stagnant price data (zero-division handling)
    print("\n" + "=" * 80)
    print("TESTING WITH STAGNANT PRICE DATA (Zero-Division Handling)")
    print("=" * 80)
    
    stagnant_data = sample_data.copy()
    stagnant_data['close_price'] = 25.0  # Completely flat prices
    
    try:
        stagnant_signal = optimizer.get_entry_signal(stagnant_data)
        print(f"Signal Type: {stagnant_signal.signal_type}")
        print(f"RSI Value: {stagnant_signal.rsi_value:.2f}")
        print("✓ Successfully handled stagnant price movement")
    except Exception as e:
        logger.error(f"Stagnant price handling failed: {e}")

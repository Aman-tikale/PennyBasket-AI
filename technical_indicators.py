"""
Technical Analysis Indicators Calculator for BSE Bhavcopy Data

This module calculates common technical indicators:
- Relative Strength Index (RSI) - 14-day default
- Bollinger Bands (20-day SMA with 2 standard deviations)

Designed for use with BSE Bhavcopy and NSE market data.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, Union


def calculate_rsi(
    series: pd.Series,
    period: int = 14,
    column_name: str = 'rsi'
) -> pd.Series:
    """
    Calculate the Relative Strength Index (RSI).
    
    RSI measures the magnitude of recent price changes to evaluate overbought
    or oversold conditions. Values range from 0 to 100.
    - RSI > 70: Overbought (potential reversal/pullback)
    - RSI < 30: Oversold (potential bounce/rally)
    - RSI = 50: Neutral midpoint
    
    Args:
        series: pd.Series of price data (typically closing prices)
        period: Period for RSI calculation (default: 14 days)
        column_name: Name for the output column (default: 'rsi')
    
    Returns:
        pd.Series with RSI values
    
    Raises:
        ValueError: If period is less than 2 or greater than series length
        
    Example:
        >>> prices = pd.Series([100, 102, 101, 103, 105, 104, 106, 108, 107, 109])
        >>> rsi = calculate_rsi(prices, period=14)
    """
    if period < 2:
        raise ValueError(f"RSI period must be at least 2, got {period}")
    
    if len(series) < period + 1:
        raise ValueError(f"Series length ({len(series)}) must be greater than period ({period})")
    
    # Calculate price changes
    delta = series.diff()
    
    # Separate gains and losses
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)
    
    # Calculate average gains and losses using EMA
    avg_gains = gains.ewm(span=period, adjust=False).mean()
    avg_losses = losses.ewm(span=period, adjust=False).mean()
    
    # Avoid division by zero
    rs = np.where(avg_losses != 0, avg_gains / avg_losses, 0)
    
    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))
    
    return pd.Series(rsi, index=series.index, name=column_name)


def calculate_bollinger_bands(
    series: pd.Series,
    period: int = 20,
    num_std_dev: float = 2.0,
    prefix: str = 'bb'
) -> pd.DataFrame:
    """
    Calculate Bollinger Bands.
    
    Bollinger Bands consist of:
    - Middle Band: SMA of the price over the period
    - Upper Band: Middle Band + (Std Dev * num_std_dev)
    - Lower Band: Middle Band - (Std Dev * num_std_dev)
    - Bandwidth: (Upper Band - Lower Band) / Middle Band (volatility measure)
    
    Args:
        series: pd.Series of price data (typically closing prices)
        period: Period for SMA and standard deviation (default: 20 days)
        num_std_dev: Number of standard deviations for band width (default: 2.0)
        prefix: Prefix for output column names (default: 'bb')
    
    Returns:
        pd.DataFrame with columns:
            - '{prefix}_middle': Simple Moving Average (Middle Band)
            - '{prefix}_upper': Upper Band
            - '{prefix}_lower': Lower Band
            - '{prefix}_bandwidth': Band width (volatility indicator)
            - '{prefix}_position': Position of price within bands (0-1)
    
    Raises:
        ValueError: If period is less than 2 or num_std_dev is negative
        
    Example:
        >>> prices = pd.Series([100, 102, 101, 103, 105, 104, 106, 108, 107, 109])
        >>> bb = calculate_bollinger_bands(prices, period=20, num_std_dev=2.0)
    """
    if period < 2:
        raise ValueError(f"Bollinger Bands period must be at least 2, got {period}")
    
    if num_std_dev < 0:
        raise ValueError(f"Number of standard deviations must be non-negative, got {num_std_dev}")
    
    if len(series) < period:
        raise ValueError(f"Series length ({len(series)}) must be at least period ({period})")
    
    # Calculate middle band (SMA)
    middle_band = series.rolling(window=period).mean()
    
    # Calculate standard deviation
    std_dev = series.rolling(window=period).std()
    
    # Calculate upper and lower bands
    upper_band = middle_band + (std_dev * num_std_dev)
    lower_band = middle_band - (std_dev * num_std_dev)
    
    # Calculate bandwidth (volatility measure)
    bandwidth = (upper_band - lower_band) / middle_band
    
    # Calculate position of price within bands (Bollinger Band %B)
    # 0 = at lower band, 1 = at upper band, 0.5 = at middle band
    position = np.where(
        (upper_band - lower_band) != 0,
        (series - lower_band) / (upper_band - lower_band),
        np.nan
    )
    
    # Clip position to 0-1 range for interpretation
    position = np.clip(position, 0, 1)
    
    result_df = pd.DataFrame({
        f'{prefix}_middle': middle_band,
        f'{prefix}_upper': upper_band,
        f'{prefix}_lower': lower_band,
        f'{prefix}_bandwidth': bandwidth,
        f'{prefix}_position': position
    }, index=series.index)
    
    return result_df


def add_technical_indicators(
    df: pd.DataFrame,
    close_column: str = 'CLOSE',
    rsi_period: int = 14,
    bb_period: int = 20,
    bb_std_dev: float = 2.0
) -> pd.DataFrame:
    """
    Add RSI and Bollinger Bands to a BSE Bhavcopy DataFrame.
    
    BSE Bhavcopy typically has columns like: SYMBOL, SERIES, OPEN, HIGH, LOW, CLOSE, etc.
    
    Args:
        df: DataFrame with stock price data
        close_column: Name of the close price column (default: 'CLOSE')
        rsi_period: Period for RSI calculation (default: 14)
        bb_period: Period for Bollinger Bands SMA (default: 20)
        bb_std_dev: Standard deviations for Bollinger Bands (default: 2.0)
    
    Returns:
        DataFrame with added RSI and Bollinger Bands columns
    
    Raises:
        ValueError: If close_column not found in DataFrame
        
    Example:
        >>> bse_data = pd.read_csv('bse_bhavcopy.csv')
        >>> technical_df = add_technical_indicators(bse_data, close_column='CLOSE')
    """
    if close_column not in df.columns:
        raise ValueError(f"Column '{close_column}' not found in DataFrame. Available columns: {df.columns.tolist()}")
    
    result_df = df.copy()
    
    # Calculate RSI
    result_df['rsi_14'] = calculate_rsi(df[close_column], period=rsi_period)
    
    # Calculate Bollinger Bands
    bb_df = calculate_bollinger_bands(df[close_column], period=bb_period, num_std_dev=bb_std_dev)
    result_df = pd.concat([result_df, bb_df], axis=1)
    
    return result_df


def add_rsi_by_symbol(
    df: pd.DataFrame,
    symbol_column: str = 'SYMBOL',
    close_column: str = 'CLOSE',
    rsi_period: int = 14
) -> pd.DataFrame:
    """
    Calculate RSI grouped by symbol for multi-symbol datasets.
    
    Useful for BSE Bhavcopy files containing multiple securities.
    
    Args:
        df: DataFrame with stock price data (must have symbol column)
        symbol_column: Name of the symbol/ticker column (default: 'SYMBOL')
        close_column: Name of the close price column (default: 'CLOSE')
        rsi_period: Period for RSI calculation (default: 14)
    
    Returns:
        DataFrame with RSI calculated per symbol
        
    Example:
        >>> multi_symbol_data = pd.read_csv('multi_stock_bhavcopy.csv')
        >>> data_with_rsi = add_rsi_by_symbol(multi_symbol_data)
    """
    if symbol_column not in df.columns or close_column not in df.columns:
        raise ValueError(f"Required columns not found. Looking for: {symbol_column}, {close_column}")
    
    result_df = df.copy()
    
    # Calculate RSI for each symbol separately
    result_df['rsi_14'] = result_df.groupby(symbol_column)[close_column].transform(
        lambda x: calculate_rsi(x, period=rsi_period)
    )
    
    return result_df


def add_bollinger_bands_by_symbol(
    df: pd.DataFrame,
    symbol_column: str = 'SYMBOL',
    close_column: str = 'CLOSE',
    bb_period: int = 20,
    bb_std_dev: float = 2.0
) -> pd.DataFrame:
    """
    Calculate Bollinger Bands grouped by symbol for multi-symbol datasets.
    
    Useful for BSE Bhavcopy files containing multiple securities.
    
    Args:
        df: DataFrame with stock price data (must have symbol column)
        symbol_column: Name of the symbol/ticker column (default: 'SYMBOL')
        close_column: Name of the close price column (default: 'CLOSE')
        bb_period: Period for Bollinger Bands SMA (default: 20)
        bb_std_dev: Standard deviations for Bollinger Bands (default: 2.0)
    
    Returns:
        DataFrame with Bollinger Bands calculated per symbol
    """
    if symbol_column not in df.columns or close_column not in df.columns:
        raise ValueError(f"Required columns not found. Looking for: {symbol_column}, {close_column}")
    
    result_df = df.copy()
    
    # Calculate Bollinger Bands for each symbol separately
    bb_data = result_df.groupby(symbol_column)[close_column].transform(
        lambda x: calculate_bollinger_bands(x, period=bb_period, num_std_dev=bb_std_dev).set_index(x.index)
    )
    
    # Handle the nested DataFrame from transform
    if isinstance(bb_data, pd.DataFrame):
        for col in bb_data.columns:
            result_df[col] = bb_data[col]
    
    return result_df


def identify_trading_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate trading signals based on RSI and Bollinger Bands.
    
    Signal Rules:
    - RSI Oversold: RSI < 30 (Buy signal)
    - RSI Overbought: RSI > 70 (Sell signal)
    - BB Lower Touch: Price touches or crosses lower band (Buy signal)
    - BB Upper Touch: Price touches or crosses upper band (Sell signal)
    - BB Squeeze: Bandwidth < 20th percentile (Low volatility, breakout expected)
    
    Args:
        df: DataFrame with RSI and Bollinger Bands already calculated
    
    Returns:
        DataFrame with added signal columns:
            - 'signal_rsi': 'BUY', 'SELL', or 'NEUTRAL'
            - 'signal_bb': 'BUY', 'SELL', or 'NEUTRAL'
            - 'signal_combined': Combined signal strength (0-2)
            
    Example:
        >>> data = add_technical_indicators(bse_data)
        >>> signals = identify_trading_signals(data)
    """
    result_df = df.copy()
    
    # RSI Signals
    result_df['signal_rsi'] = 'NEUTRAL'
    result_df.loc[result_df['rsi_14'] < 30, 'signal_rsi'] = 'BUY'
    result_df.loc[result_df['rsi_14'] > 70, 'signal_rsi'] = 'SELL'
    
    # Bollinger Bands Signals
    result_df['signal_bb'] = 'NEUTRAL'
    
    if 'bb_position' in result_df.columns:
        result_df.loc[result_df['bb_position'] < 0.2, 'signal_bb'] = 'BUY'
        result_df.loc[result_df['bb_position'] > 0.8, 'signal_bb'] = 'SELL'
    
    # Combined Signal Strength
    result_df['signal_combined'] = 0
    result_df.loc[result_df['signal_rsi'] == 'BUY', 'signal_combined'] += 1
    result_df.loc[result_df['signal_bb'] == 'BUY', 'signal_combined'] += 1
    
    # Negative for sell signals
    result_df.loc[result_df['signal_rsi'] == 'SELL', 'signal_combined'] -= 1
    result_df.loc[result_df['signal_bb'] == 'SELL', 'signal_combined'] -= 1
    
    # Calculate Bollinger Band squeeze (low volatility)
    if 'bb_bandwidth' in result_df.columns:
        bandwidth_percentile = result_df['bb_bandwidth'].quantile(0.2)
        result_df['bb_squeeze'] = result_df['bb_bandwidth'] < bandwidth_percentile
    
    return result_df


def generate_technical_report(df: pd.DataFrame, symbol: str = None) -> str:
    """
    Generate a formatted report of technical indicators for a stock or symbol.
    
    Args:
        df: DataFrame with technical indicators already calculated
        symbol: Specific symbol to report on (if multi-symbol data)
    
    Returns:
        Formatted string report
    """
    if symbol and 'SYMBOL' in df.columns:
        df = df[df['SYMBOL'] == symbol].copy()
    
    if df.empty:
        return f"No data found for symbol: {symbol}"
    
    # Get latest values
    latest = df.iloc[-1]
    
    report = []
    report.append("=" * 80)
    report.append("TECHNICAL ANALYSIS REPORT")
    report.append("=" * 80)
    report.append("")
    
    # Header info
    if 'SYMBOL' in df.columns:
        report.append(f"Symbol: {latest['SYMBOL']}")
    
    report.append(f"Latest Close: {latest.get('CLOSE', 'N/A')}")
    report.append(f"Date: {df.index[-1] if hasattr(df.index[-1], 'strftime') else 'N/A'}")
    report.append("")
    
    # RSI Analysis
    report.append("RELATIVE STRENGTH INDEX (RSI-14)")
    report.append("-" * 80)
    rsi_value = latest.get('rsi_14', np.nan)
    if not np.isnan(rsi_value):
        report.append(f"RSI Value: {rsi_value:.2f}")
        if rsi_value > 70:
            report.append("Status: OVERBOUGHT (Potential sell signal)")
        elif rsi_value < 30:
            report.append("Status: OVERSOLD (Potential buy signal)")
        else:
            report.append("Status: NEUTRAL")
    else:
        report.append("RSI: Insufficient data")
    report.append("")
    
    # Bollinger Bands Analysis
    report.append("BOLLINGER BANDS (20-day, 2 Std Dev)")
    report.append("-" * 80)
    if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
        close = latest.get('CLOSE', np.nan)
        upper = latest.get('bb_upper', np.nan)
        middle = latest.get('bb_middle', np.nan)
        lower = latest.get('bb_lower', np.nan)
        position = latest.get('bb_position', np.nan)
        bandwidth = latest.get('bb_bandwidth', np.nan)
        
        report.append(f"Upper Band: {upper:.2f}")
        report.append(f"Middle Band (SMA): {middle:.2f}")
        report.append(f"Lower Band: {lower:.2f}")
        report.append(f"Current Close: {close:.2f}")
        report.append(f"Band Position: {position*100:.1f}% (0=Lower, 100=Upper)")
        report.append(f"Bandwidth: {bandwidth*100:.2f}% (Volatility Measure)")
        
        if position < 0.2:
            report.append("Status: NEAR LOWER BAND (Potential buy)")
        elif position > 0.8:
            report.append("Status: NEAR UPPER BAND (Potential sell)")
        else:
            report.append("Status: WITHIN BANDS")
    else:
        report.append("Bollinger Bands: Not calculated")
    report.append("")
    
    # Trading Signals
    if 'signal_rsi' in df.columns:
        report.append("TRADING SIGNALS")
        report.append("-" * 80)
        report.append(f"RSI Signal: {latest.get('signal_rsi', 'N/A')}")
        report.append(f"BB Signal: {latest.get('signal_bb', 'N/A')}")
        if 'signal_combined' in df.columns:
            combined = latest.get('signal_combined', 0)
            if combined > 1:
                report.append(f"Combined Signal: STRONG BUY")
            elif combined == 1:
                report.append(f"Combined Signal: BUY")
            elif combined < -1:
                report.append(f"Combined Signal: STRONG SELL")
            elif combined == -1:
                report.append(f"Combined Signal: SELL")
            else:
                report.append(f"Combined Signal: NEUTRAL")
    report.append("")
    
    report.append("=" * 80)
    
    return "\n".join(report)


if __name__ == "__main__":
    # Example usage with sample BSE Bhavcopy-like data
    np.random.seed(42)
    dates = pd.date_range('2025-01-01', periods=100)
    
    # Generate sample price data for multiple symbols
    symbols = ['INFY', 'TCS', 'RELIANCE']
    data_list = []
    
    for symbol in symbols:
        # Generate realistic stock prices
        prices = np.cumsum(np.random.randn(100) * 0.5) + 100
        
        data_list.append(pd.DataFrame({
            'SYMBOL': symbol,
            'DATE': dates,
            'OPEN': prices - np.abs(np.random.randn(100) * 0.3),
            'HIGH': prices + np.abs(np.random.randn(100) * 0.3),
            'LOW': prices - np.abs(np.random.randn(100) * 0.3),
            'CLOSE': prices,
            'VOLUME': np.random.randint(1000000, 10000000, 100)
        }))
    
    bse_data = pd.concat(data_list, ignore_index=True)
    
    print("Original Data:")
    print(bse_data.head(10))
    print("\n")
    
    # Add RSI for each symbol
    data_with_rsi = add_rsi_by_symbol(bse_data, close_column='CLOSE', rsi_period=14)
    print("Data with RSI:")
    print(data_with_rsi[['SYMBOL', 'DATE', 'CLOSE', 'rsi_14']].head(20))
    print("\n")
    
    # Add Bollinger Bands
    data_with_bb = add_bollinger_bands_by_symbol(data_with_rsi, close_column='CLOSE')
    print("Data with Bollinger Bands:")
    print(data_with_bb[['SYMBOL', 'CLOSE', 'bb_lower', 'bb_middle', 'bb_upper', 'bb_position']].tail(10))
    print("\n")
    
    # Generate trading signals
    signals = identify_trading_signals(data_with_bb)
    print("Trading Signals:")
    print(signals[['SYMBOL', 'CLOSE', 'rsi_14', 'signal_rsi', 'signal_bb', 'signal_combined']].tail(10))
    print("\n")
    
    # Generate report for specific symbol
    print(generate_technical_report(signals[signals['SYMBOL'] == 'INFY']))

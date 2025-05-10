import pandas as pd
import numpy as np


def atr(df: pd.DataFrame, window: int = 14, high_col: str = 'High', low_col: str = 'Low', close_col: str = 'Close') -> pd.Series:
    """
    Calculates the Average True Range (ATR), a volatility indicator that measures market volatility
    by decomposing the entire range of an asset price for a given period.

    Args:
        df (pd.DataFrame): The input DataFrame.
        close (pd.Series): The closing prices of the period.
        window (int): The lookback period for the calculation. Default is 14.
        high_col (str): The column name for high prices. Default is 'High'.
        low_col (str): The column name for low prices. Default is 'Low'.

    Returns:
        pd.Series: ATR values for the given input series.

    The ATR is calculated in three steps:

    1. Calculate the True Range (TR) for each period:
       TR = max(high - low, abs(high - prev_close), abs(low - prev_close))
       Where:
       - high - low = current period range
       - abs(high - prev_close) = current high minus previous close
       - abs(low - prev_close) = current low minus previous close

    2. For the first ATR value, take the simple average of TR values over the specified window.
    
    3. For subsequent ATR values, use a smoothing technique:
       ATR = ((Prior ATR * (window-1)) + Current TR) / window

    The ATR is primarily used to measure volatility, not to indicate trend direction.

    Use Cases:

    - Volatility measurement: Higher ATR values indicate higher volatility, while lower ATR values
      indicate lower volatility.
    - Position sizing: Traders often use ATR to determine position size and set stop-loss orders
      that adjust for volatility.
    - Trend confirmation: ATR can be used to confirm trend strength; increasing ATR during price
      moves may indicate stronger trends.
    - Breakout identification: Significant increases in ATR may precede or confirm breakouts.
    - Entry/exit signals: Some trading systems use ATR-based indicators for trade signals.
    """
    high = df[high_col]
    low = df[low_col]
    close = df[close_col]
    
    # Calculate True Range
    prev_close = close.shift(1)
    tr1 = high - low  # Current high - current low
    tr2 = (high - prev_close).abs()  # Current high - previous close
    tr3 = (low - prev_close).abs()  # Current low - previous close
    
    # True Range is the maximum of the three calculations
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calculate ATR using Wilder's smoothing method
    # First ATR value is the simple average of TR over the window
    atr_values = pd.Series(index=close.index)
    
    # First ATR value is simple average of first 'window' TRs
    first_atr = tr.iloc[:window].mean()
    
    # Use the first value to start the smoothing process
    atr_values.iloc[window-1] = first_atr
    
    # Apply Wilder's smoothing method for the rest of the values
    for i in range(window, len(close)):
        atr_values.iloc[i] = ((atr_values.iloc[i-1] * (window-1)) + tr.iloc[i]) / window

    atr_values.name = f'ATR_{window}'
    
    return atr_values

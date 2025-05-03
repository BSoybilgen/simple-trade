import pandas as pd
import numpy as np
from ..trend.ema import ema


def chaikin_volatility(high: pd.Series, low: pd.Series, ema_window: int = 10, 
                       roc_window: int = 10) -> pd.Series:
    """
    Calculates the Chaikin Volatility (CV) indicator, which measures volatility by 
    calculating the rate of change of the high-low price range.
    
    Args:
        high (pd.Series): The high prices of the period.
        low (pd.Series): The low prices of the period.
        ema_window (int): The window for calculating the EMA of the high-low range. Default is 10.
        roc_window (int): The window for calculating the rate of change. Default is 10.
    
    Returns:
        pd.Series: A Series containing the Chaikin Volatility values.
    
    The Chaikin Volatility is calculated in three steps:
    
    1. Calculate the daily high-low range: high - low
    2. Calculate an exponential moving average (EMA) of the high-low range
    3. Calculate the rate of change of this EMA over the specified period
    
    A higher CV value indicates higher volatility, while a lower value indicates lower volatility.
    
    Use Cases:
    
    - Volatility measurement: Identifies periods of increasing or decreasing volatility.
    - Market turning points: Rising volatility may precede market tops, while falling volatility
      may precede market bottoms.
    - Range expansion/contraction: Helps identify when markets are transitioning from quiet to 
      active periods.
    - Breakout confirmation: Sharp increases in volatility can confirm breakout movements.
    - Risk management: Adjust position sizing based on current volatility conditions.
    """
    # Make sure inputs have the same index
    high = high.copy()
    low = low.copy()
    
    # Calculate the daily high-low range
    hl_range = high - low
    
    # Calculate the EMA of the high-low range
    range_ema = ema(hl_range, window=ema_window)
    
    # Calculate the percentage rate of change over roc_window days
    # (Current EMA - EMA roc_window days ago) / (EMA roc_window days ago) * 100
    roc = ((range_ema - range_ema.shift(roc_window)) / range_ema.shift(roc_window)) * 100
    
    return roc

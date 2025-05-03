import pandas as pd
import numpy as np


def vma(close: pd.Series, volume: pd.Series, window: int = 14) -> pd.Series:
    """
    Calculates the Volume Moving Average (VMA), which is a weighted moving average
    that uses volume as the weighting factor.
    
    Args:
        close (pd.Series): The closing prices of the period.
        volume (pd.Series): The volume traded of the period.
        window (int): The lookback period for calculation. Default is 14.
    
    Returns:
        pd.Series: The Volume Moving Average values.
    
    VMA gives more weight to price moves accompanied by higher volume, making it
    more responsive to significant market movements than a simple moving average.
    
    The formula is:
    VMA = Σ(Price * Volume) / Σ(Volume), calculated over the specified window.
    
    Use Cases:
    
    - Trend identification: VMA can be used similarly to other moving averages to
      identify trends, but with more emphasis on volume-supported price movements.
    - Dynamic support/resistance: VMA can act as support in uptrends and resistance
      in downtrends.
    - Entry/exit signals: Crossovers between price and VMA or between multiple VMAs
      with different periods can generate trading signals.
    - Volume-validated price movement: VMA filters out price movements that occur on
      low volume, focusing on more significant market activity.
    - Divergence analysis: Comparing VMA to other moving averages can highlight periods
      where price moves are or aren't supported by volume.
    """
    # Ensure both series have the same index
    close = close.copy()
    volume = volume.copy()
    
    # Calculate the volume-weighted price
    weighted_price = close * volume
    
    # Calculate the VMA using rolling windows
    # For each window, sum(price * volume) / sum(volume)
    vma_values = weighted_price.rolling(window=window).sum() / volume.rolling(window=window).sum()
    
    return vma_values

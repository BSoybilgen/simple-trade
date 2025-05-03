import pandas as pd
import numpy as np


def vpt(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Calculates the Volume Price Trend (VPT), a volume-based indicator that relates
    volume to price change percentage to create a cumulative indicator of buying/selling
    pressure.
    
    Args:
        close (pd.Series): The closing prices of the period.
        volume (pd.Series): The volume traded of the period.
    
    Returns:
        pd.Series: The Volume Price Trend values.
    
    VPT is similar to OBV but instead of just using the direction of price change,
    it uses the percentage change in price to give more weight to more significant
    price movements.
    
    Calculation:
    1. Calculate percentage price change for each period:
       Price Change % = (Today's Close - Yesterday's Close) / Yesterday's Close
    
    2. For each period, multiply the percentage price change by volume:
       VPT = Previous VPT + (Price Change % * Volume)
    
    Interpretation:
    - Rising VPT: Indicates buying pressure (accumulation)
    - Falling VPT: Indicates selling pressure (distribution)
    - The steepness of the VPT line indicates the strength of the buying/selling pressure
    
    Use Cases:
    
    - Trend confirmation: VPT should move in the same direction as price in a valid trend.
    - Divergence analysis: If price makes new highs/lows but VPT doesn't, it suggests
      the trend may be weakening.
    - Volume analysis: VPT gives a cumulative view of volume weighted by price change %,
      providing insight into the conviction behind price movements.
    - Breakout validation: Significant volume should accompany breakouts, visible as
      a steep change in the VPT.
    - Accumulation/distribution identification: VPT can help identify periods of
      accumulation or distribution before major price moves.
    """
    # Ensure both series have the same index
    close = close.copy()
    volume = volume.copy()
    
    # Calculate the percentage price change
    price_change_pct = close.pct_change()
    
    # Calculate VPT as a running sum of (volume * percentage price change)
    # First item is just the first volume value (there's no price change for first item)
    vpt_change = price_change_pct * volume
    vpt_values = pd.Series(index=close.index, dtype=float)
    vpt_values.iloc[0] = 0  # Start with 0
    
    # Calculate the cumulative sum
    vpt_values = vpt_change.cumsum()
    
    return vpt_values

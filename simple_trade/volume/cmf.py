import pandas as pd
import numpy as np


def cmf(df: pd.DataFrame, period: int = 20, high_col: str = 'High', low_col: str = 'Low', close_col: str = 'Close', volume_col: str = 'Volume') -> pd.Series:
    """
    Calculates the Chaikin Money Flow (CMF), a volume-based indicator that measures
    the amount of Money Flow Volume over a specific period.
    
    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        period (int): The lookback period for calculation. Default is 20.
        high_col (str): The column name for high prices. Default is 'High'.
        low_col (str): The column name for low prices. Default is 'Low'.
        close_col (str): The column name for closing prices. Default is 'Close'.
        volume_col (str): The column name for volume. Default is 'Volume'.
    
    Returns:
        pd.Series: The Chaikin Money Flow values.
    
    CMF is derived from the Accumulation/Distribution Line (A/D Line) but instead of
    being cumulative, it sums up the Money Flow Volume over a specific period and
    divides it by the total volume over that same period.
    
    Calculation steps:
    1. Calculate Money Flow Multiplier (MFM) for each period:
       MFM = ((Close - Low) - (High - Close)) / (High - Low)
       
    2. Calculate Money Flow Volume (MFV) for each period:
       MFV = MFM * Volume
       
    3. Calculate CMF by summing MFV over the period and dividing by the sum of Volume:
       CMF = Sum(MFV, n) / Sum(Volume, n)
    
    Interpretation:
    - CMF > 0: Accumulation (buying pressure)
    - CMF < 0: Distribution (selling pressure)
    - The further from zero, the stronger the pressure
    
    Use Cases:
    
    - Market strength assessment: CMF helps identify if buying or selling pressure 
      is driving price movements.
    - Trend confirmation: CMF should be positive in uptrends and negative in downtrends.
    - Divergence analysis: If price makes new highs but CMF fails to do so, it suggests
      the trend may be weakening.
    - Support/resistance validation: Strong volume should accompany breakouts, visible
      as a stronger CMF reading.
    - Overbought/oversold identification: Extreme CMF values may indicate potential
      reversal points.
    """
    high = df[high_col]
    low = df[low_col]
    close = df[close_col]
    volume = df[volume_col]
    
    # Handle division by zero - if high and low are the same,
    # money flow multiplier is zero (neutral)
    price_range = high - low
    price_range_nonzero = price_range.replace(0, np.nan)
    
    # Calculate Money Flow Multiplier (MFM)
    # MFM = ((Close - Low) - (High - Close)) / (High - Low)
    # This simplifies to MFM = (2*Close - High - Low) / (High - Low)
    mfm = ((2 * close - high - low) / price_range_nonzero).fillna(0)
    
    # Calculate Money Flow Volume (MFV)
    mfv = mfm * volume
    
    # Calculate CMF as sum(MFV)/sum(Volume) over the period
    cmf_values = mfv.rolling(window=period).sum() / volume.rolling(window=period).sum()
    cmf_values.name = f'CMF_{period}'

    return cmf_values

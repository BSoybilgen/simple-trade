import pandas as pd
import numpy as np


def adline(df, high_col: str = 'High', low_col: str = 'Low', close_col: str = 'Close', volume_col: str = 'Volume') -> pd.Series:
    """
    Calculates the Accumulation/Distribution Line (A/D Line), a volume-based indicator
    that measures the cumulative flow of money into and out of a security.
    
    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        high_col (str): The column name for high prices. Default is 'High'.
        low_col (str): The column name for low prices. Default is 'Low'.
        close_col (str): The column name for closing prices. Default is 'Close'.
        volume_col (str): The column name for volume. Default is 'Volume'.
    
    Returns:
        pd.Series: The Accumulation/Distribution Line values.
    
    Unlike OBV which only considers price direction (up or down), the A/D Line
    considers the position of the close relative to the trading range (high-low)
    to determine the amount of buying or selling pressure.
    
    Calculation steps:
    1. Calculate Money Flow Multiplier (MFM) for each period:
       MFM = ((Close - Low) - (High - Close)) / (High - Low)
       
    2. Calculate Money Flow Volume (MFV) for each period:
       MFV = MFM * Volume
       
    3. Sum the Money Flow Volume cumulatively to create the A/D Line:
       A/D Line = Previous A/D Line + Current MFV
    
    Interpretation:
    - Rising A/D Line: Accumulation (buying pressure exceeds selling pressure)
    - Falling A/D Line: Distribution (selling pressure exceeds buying pressure)
    
    Use Cases:
    
    - Trend confirmation: A/D Line should move in the same direction as the price trend.
    - Divergence detection: If the A/D Line fails to confirm new price highs or lows,
      it may signal a potential reversal.
    - Volume analysis: Helps to understand if price moves are supported by volume.
    - Market sentiment: Indicates whether money is flowing into or out of a security.
    - Early warning: A/D Line often leads price movements, making it useful for
      identifying potential trend changes before they occur in price.
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
    
    # Calculate A/D Line as cumulative sum of Money Flow Volume
    ad_line = mfv.cumsum()
    ad_line.name = 'ADLINE'
    
    return ad_line

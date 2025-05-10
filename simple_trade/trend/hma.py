import pandas as pd
import numpy as np
from .wma import wma

def hma(df: pd.DataFrame, window: int = 14, close_col: str = 'Close') -> pd.Series:
    """
    Calculates the Hull Moving Average (HMA) of a series.

    The HMA is a moving average that reduces lag and improves smoothing.
    It is calculated using weighted moving averages (WMAs) with specific
    window lengths to achieve this effect.

    Args:
        df (pd.DataFrame): The dataframe containing price data. Must have close column.
        window (int): The window size for the HMA.
        close_col (str): The name of the close price column (default: 'Close').

    Returns:
        pd.Series: The HMA of the series.

    The Hull Moving Average (HMA) is a type of moving average that is designed
    to reduce lag and improve smoothing compared to traditional moving averages.
    It achieves this by using a combination of weighted moving averages (WMAs)
    with different window lengths.

    The formula for calculating the HMA is as follows:

    1. Calculate a WMA of the input series with a window length of half the
       specified window size (half_length).
    2. Calculate a WMA of the input series with the full specified window size.
    3. Calculate the difference between 2 times the first WMA and the second WMA.
    4. Calculate a WMA of the result from step 3 with a window length equal to
       the square root of the specified window size.

    Use Cases:

    - Identifying trends: The HMA can be used to identify the direction of a
      price trend.
    - Smoothing price data: The HMA can smooth out short-term price fluctuations
      to provide a clearer view of the underlying trend.
    - Generating buy and sell signals: The HMA can be used in crossover systems
      to generate buy and sell signals.
    """
    series = df[close_col]
    half_length = int(window / 2)
    sqrt_length = int(np.sqrt(window))
    wma_half = wma(df, half_length, close_col=close_col)
    wma_full = wma(df, window, close_col=close_col)
    df = pd.DataFrame(2 * wma_half - wma_full, columns=[close_col])
    hma_ = wma(df, sqrt_length, close_col=close_col)
    hma_.name = f'HMA_{window}'
    return hma_
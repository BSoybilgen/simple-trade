import pandas as pd
import numpy as np
from .wma import wma

def hma(series: pd.Series, window: int = 14) -> pd.Series:
    """
    Calculates the Hull Moving Average (HMA) of a series.

    The HMA is a moving average that reduces lag and improves smoothing.
    It is calculated using weighted moving averages (WMAs) with specific
    window lengths to achieve this effect.

    Args:
        series (pd.Series): The input series.
        window (int): The window size for the HMA.

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
    half_length = int(window / 2)
    sqrt_length = int(np.sqrt(window))
    wma_half = wma(series, half_length)
    wma_full = wma(series, window)
    hma_ = wma(2 * wma_half - wma_full, sqrt_length)
    return hma_
import pandas as pd
import numpy as np


def wma(df: pd.DataFrame, window: int = 14, close_col: str = 'Close') -> pd.Series:
    """
    Calculates the Weighted Moving Average (WMA) of a series.

    Args:
        df (pd.DataFrame): The dataframe containing price data. Must have close column.
        window (int): The window size for the WMA.
        close_col (str): The name of the close price column (default: 'Close').

    Returns:
        pd.Series: The WMA of the series.

    The (WMA) is a type of moving average that assigns different 
    weights to the data points in the window, with more recent 
    data points receiving higher weights. This makes the WMA more 
    responsive to recent price changes compared to a Simple Moving 
    Average (SMA), which gives equal weight to all data points.

    The formula for calculating the WMA is as follows:

    1. weights = np.arange(1, window + 1): This line creates an array of 
    weights, where the weight for each data point increases linearly 
    from 1 to window. So, the most recent data point has a weight of 
    window, the second most recent has a weight of window - 1, and so 
    on.
    2. series.rolling(window).apply(lambda prices: np.dot(prices, weights)
    / weights.sum(), raw=True): This line calculates the WMA. It first 
    creates a rolling window of size window over the input series. 
    Then, for each window, it calculates the weighted average by 
    taking the dot product of the prices in the window and the weights,
    and dividing by the sum of the weights.

    Use Cases:

    - Price Trend Identification: WMAs are commonly used to identify 
    the direction of a price trend. Because they give more weight 
    to recent prices, they react more quickly to changes in trend 
    than SMAs. Traders might use multiple WMAs with different window 
    sizes to identify potential buy or sell signals.
    - Smoothing Price Data: WMAs can smooth out short-term price 
    fluctuations to provide a clearer view of the underlying trend.
    - Crossover Systems: Traders may use WMA crossovers (e.g., a 
    short-term WMA crossing above a long-term WMA) as buy signals, 
    and vice versa for sell signals.
    """
    series = df[close_col]
    weights = np.arange(1, window + 1)
    series = series.rolling(window).apply(lambda prices: np.dot(prices, weights) / weights.sum(), raw=True)
    series.name = f'WMA_{window}'
    return series
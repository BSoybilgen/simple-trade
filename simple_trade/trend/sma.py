import pandas as pd

def sma(series: pd.Series, window: int = 14) -> pd.Series:
    """
    Calculates the Simple Moving Average (SMA) of a series.

    The SMA is a moving average that is calculated by taking the arithmetic
    mean of a given set of values over a specified period.

    Args:
        series (pd.Series): The input series.
        window (int): The window size for the SMA.

    Returns:
        pd.Series: The SMA of the series.

    The Simple Moving Average (SMA) is a type of moving average that is calculated
    by taking the arithmetic mean of a given set of values over a specified
    period. It is the simplest form of moving average and is often used to
    smooth out price data or other time series data.

    The formula for calculating the SMA is as follows:

    SMA = (Sum of values in the period) / (Number of values in the period)

    Use Cases:

    - Identifying trends: The SMA can be used to identify the direction of a
      price trend.
    - Smoothing price data: The SMA can smooth out short-term price fluctuations
      to provide a clearer view of the underlying trend.
    - Generating buy and sell signals: The SMA can be used in crossover systems
      to generate buy and sell signals.
    """
    # Return the raw Series
    return series.rolling(window=window).mean()
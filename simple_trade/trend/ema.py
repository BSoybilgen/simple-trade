import pandas as pd

def ema(series: pd.Series, window: int = 14) -> pd.Series:
    """
    Calculates the Exponential Moving Average (EMA) of a series.

    The EMA is a type of moving average that gives more weight to recent
    prices, making it more responsive to new information than the SMA.

    Args:
        series (pd.Series): The input series.
        window (int): The window size for the EMA.

    Returns:
        pd.Series: The EMA of the series.

    The Exponential Moving Average (EMA) is a type of moving average that
    gives more weight to recent prices, making it more responsive to new
    information than the Simple Moving Average (SMA). The weighting applied
    to the most recent price depends on the specified period, with a shorter
    period giving more weight to recent prices.

    The formula for calculating the EMA is as follows:

    EMA = (Price(today) * k) + (EMA(yesterday) * (1 - k))
    where:
    k = 2 / (window + 1)

    Use Cases:

    - Identifying trends: The EMA can be used to identify the direction of a
      price trend.
    - Smoothing price data: The EMA can smooth out short-term price fluctuations
      to provide a clearer view of the underlying trend.
    - Generating buy and sell signals: The EMA can be used in crossover systems
      to generate buy and sell signals.
    - Reacting quickly to price changes: The EMA's responsiveness makes it
      suitable for identifying entry and exit points in fast-moving markets.
    """
    # Return the raw Series
    return series.ewm(span=window, adjust=False).mean()
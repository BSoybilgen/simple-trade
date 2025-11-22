import numpy as np
import pandas as pd


def ama(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Adaptive Moving Average (AMA), also known as Kaufman's Adaptive Moving Average (KAMA).
    AMA adjusts its smoothing factor based on market noise using an Efficiency Ratio (ER).

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - window (int): The lookback period for Efficiency Ratio. Default is 10.
            - fast_period (int): The fast EMA period limit. Default is 2.
            - slow_period (int): The slow EMA period limit. Default is 30.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the AMA series and a list of column names.

    The Adaptive Moving Average is calculated as follows:

    1. Calculate Efficiency Ratio (ER):
       Change = Abs(Price - Price(n periods ago))
       Volatility = Sum(Abs(Price - Prev Price), n)
       ER = Change / Volatility

    2. Calculate Smoothing Constant (SC):
       Fast SC = 2 / (fast_period + 1)
       Slow SC = 2 / (slow_period + 1)
       Scaled SC = (ER * (Fast SC - Slow SC) + Slow SC)^2

    3. Calculate AMA:
       AMA = Previous AMA + Scaled SC * (Price - Previous AMA)

    Interpretation:
    - When market moves directionally (high ER), AMA adapts quickly.
    - When market is choppy (low ER), AMA flattens out to avoid false signals.

    Use Cases:
    - Trend Following: Identifying the trend with reduced noise in sideways markets.
    - Stop Loss: The flat nature of AMA in ranges makes it a good trailing stop level.
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}

    close_col = columns.get('close_col', 'Close')
    er_window = int(parameters.get('window', 10))
    fast_period = int(parameters.get('fast_period', 2))
    slow_period = int(parameters.get('slow_period', 30))

    close = df[close_col]

    # Efficiency Ratio
    direction = close.diff(er_window).abs()
    volatility = close.diff().abs().rolling(er_window).sum()
    er = direction / volatility
    er = er.fillna(0)

    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    smoothing_constant = (er * (fast_sc - slow_sc) + slow_sc) ** 2

    values = close.to_numpy(dtype=float)
    sc_values = smoothing_constant.to_numpy(dtype=float)
    ama_values = np.full_like(values, np.nan)

    valid_idx = np.where(~np.isnan(values))[0]
    if valid_idx.size:
        start = valid_idx[0]
        ama_values[start] = values[start]
        for i in range(start + 1, len(values)):
            previous = ama_values[i - 1]
            price = values[i]
            sc = sc_values[i]
            if np.isnan(price):
                ama_values[i] = previous
                continue
            if np.isnan(sc):
                sc = 0.0
            ama_values[i] = previous + sc * (price - previous)

    ama_series = pd.Series(ama_values, index=close.index,
                           name=f'AMA_{er_window}_{fast_period}_{slow_period}')

    return ama_series, [ama_series.name]

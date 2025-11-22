import numpy as np
import pandas as pd


def kma(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Kaufman Adaptive Moving Average (KAMA).
    KAMA dynamically adjusts its smoothing factor based on the Efficiency Ratio (ER).
    When price action is smooth, KAMA reacts faster; when price action is noisy, 
    it smooths more aggressively to filter out whipsaws.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - window (int): The lookback period for Efficiency Ratio. Default is 10.
            - fast_period (int): The fast EMA period limit. Default is 2.
            - slow_period (int): The slow EMA period limit. Default is 30.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the KAMA series and a list of column names.

    The Kaufman Adaptive Moving Average is calculated as follows:

    1. Calculate Efficiency Ratio (ER):
       Change = Abs(Price - Price(n periods ago))
       Volatility = Sum(Abs(Price - Prev Price), n)
       ER = Change / Volatility

    2. Calculate Smoothing Constant (SC):
       Fast SC = 2 / (fast_period + 1)
       Slow SC = 2 / (slow_period + 1)
       Scaled SC = (ER * (Fast SC - Slow SC) + Slow SC)^2

    3. Calculate KAMA:
       KAMA = Previous KAMA + Scaled SC * (Price - Previous KAMA)

    Interpretation:
    - ER near 1: Market is trending efficiently (Directional). KAMA tracks price closely.
    - ER near 0: Market is sideways or noisy (Inefficient). KAMA remains flat.

    Use Cases:
    - Trend Filtering: Helps stay in trends while ignoring sideways noise.
    - Trailing Stop: The flatness of KAMA in congestion makes it a good stop level.
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

    direction = close.diff(er_window).abs()
    volatility = close.diff().abs().rolling(er_window).sum()
    er = direction / volatility
    er = er.replace([np.inf, -np.inf], 0).fillna(0)

    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    smoothing_constant = (er * (fast_sc - slow_sc) + slow_sc) ** 2

    values = close.to_numpy(dtype=float)
    sc_values = smoothing_constant.to_numpy(dtype=float)
    kama_values = np.full_like(values, np.nan)

    valid_idx = np.where(~np.isnan(values))[0]
    if valid_idx.size:
        start = valid_idx[0]
        kama_values[start] = values[start]
        for i in range(start + 1, len(values)):
            price = values[i]
            prev = kama_values[i - 1]
            sc = sc_values[i]
            if np.isnan(price):
                kama_values[i] = prev
                continue
            if np.isnan(sc):
                sc = 0.0
            kama_values[i] = prev + sc * (price - prev)

    kama_series = pd.Series(
        kama_values,
        index=close.index,
        name=f'KMA_{er_window}_{fast_period}_{slow_period}'
    )

    return kama_series, [kama_series.name]

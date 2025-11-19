import numpy as np
import pandas as pd


def ama(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Adaptive Moving Average (AMA, also known as Kaufman AMA).

    AMA adjusts its smoothing factor based on market noise using an Efficiency
    Ratio (ER). When price movement is directional (high ER), AMA reacts faster;
    when movement is choppy (low ER), AMA smooths more aggressively.
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

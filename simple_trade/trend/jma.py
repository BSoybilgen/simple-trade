import numpy as np
import pandas as pd


def jma(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Jurík Moving Average (JMA).

    JMA is designed to remain smooth while maintaining low lag. This
    implementation approximates the JMA by dynamically adjusting a smoothing
    constant based on user-specified length and phase parameters.
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}

    close_col = columns.get('close_col', 'Close')
    length = max(1, int(parameters.get('length', 21)))
    phase = float(parameters.get('phase', 0.0))
    power = float(parameters.get('power', 2.0))

    phase = max(-100.0, min(100.0, phase))
    phase_ratio = phase / 100.0

    close = df[close_col].astype(float)
    values = close.to_numpy(dtype=float)
    jma_values = np.full_like(values, np.nan)

    smoothing_constant = (2.0 / (length + 1.0)) ** power
    smoothing_constant = np.clip(smoothing_constant, 0.0, 1.0)

    non_nan_idx = np.where(~np.isnan(values))[0]
    if non_nan_idx.size == 0:
        series = pd.Series(jma_values, index=close.index, name=f'JMA_{length}')
        return series, [series.name]

    start = non_nan_idx[0]
    jma_values[start] = values[start]

    for i in range(start + 1, len(values)):
        price = values[i]
        prev = jma_values[i - 1]
        if np.isnan(price):
            jma_values[i] = prev
            continue
        if np.isnan(prev):
            prev = price

        base = prev + smoothing_constant * (price - prev)
        jma_values[i] = base + phase_ratio * (price - base)

    jma_series = pd.Series(jma_values, index=close.index, name=f'JMA_{length}')
    return jma_series, [jma_series.name]

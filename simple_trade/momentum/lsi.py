import numpy as np
import pandas as pd


def lsi(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """Calculate the Laguerre RSI (LRSI) indicator."""
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}

    gamma = float(parameters.get('gamma', 0.5))
    close_col = columns.get('close_col', 'Close')

    series = df[close_col]
    prices = series.to_numpy(dtype=float)
    length = len(series)

    l0 = np.full(length, np.nan)
    l1 = np.full(length, np.nan)
    l2 = np.full(length, np.nan)
    l3 = np.full(length, np.nan)
    lrsi = np.full(length, np.nan)

    prev_l0 = prev_l1 = prev_l2 = prev_l3 = None

    for idx, price in enumerate(prices):
        if np.isnan(price):
            continue

        if prev_l0 is None:
            prev_l0 = prev_l1 = prev_l2 = prev_l3 = price
            l0[idx] = prev_l0
            l1[idx] = prev_l1
            l2[idx] = prev_l2
            l3[idx] = prev_l3
            continue

        new_l0 = (1 - gamma) * price + gamma * prev_l0
        new_l1 = -gamma * new_l0 + prev_l0 + gamma * prev_l1
        new_l2 = -gamma * new_l1 + prev_l1 + gamma * prev_l2
        new_l3 = -gamma * new_l2 + prev_l2 + gamma * prev_l3

        l0[idx] = new_l0
        l1[idx] = new_l1
        l2[idx] = new_l2
        l3[idx] = new_l3

        diff0 = new_l0 - new_l1
        diff1 = new_l1 - new_l2
        diff2 = new_l2 - new_l3

        cu = max(diff0, 0.0) + max(diff1, 0.0) + max(diff2, 0.0)
        cd = max(-diff0, 0.0) + max(-diff1, 0.0) + max(-diff2, 0.0)

        denom = cu + cd
        lrsi[idx] = np.nan if denom == 0 else 100 * cu / denom

        prev_l0, prev_l1, prev_l2, prev_l3 = new_l0, new_l1, new_l2, new_l3

    gamma_str = f"{gamma:g}"
    lrsi_series = pd.Series(lrsi, index=series.index, name=f'LRSI_{gamma_str}')

    columns_list = [lrsi_series.name]
    return lrsi_series, columns_list

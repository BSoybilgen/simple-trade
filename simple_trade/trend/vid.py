import numpy as np
import pandas as pd


def vid(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """Calculate the Variable Index Dynamic Average (VIDYA) indicator."""
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}

    window = int(parameters.get('window', 21))
    cmo_window = int(parameters.get('cmo_window', 9))
    close_col = columns.get('close_col', 'Close')

    series = df[close_col]

    delta = series.diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)

    sum_up = up.rolling(window=cmo_window).sum()
    sum_down = down.rolling(window=cmo_window).sum()
    denominator = sum_up + sum_down
    denominator = denominator.replace(0, np.nan)

    cmo = ((sum_up - sum_down) / denominator) * 100
    cmo = cmo.fillna(0.0)

    base_alpha = 2 / (window + 1)

    vidya = pd.Series(np.nan, index=series.index)
    prev_vidya = None

    for idx, price in enumerate(series):
        cmo_value = cmo.iat[idx]
        if np.isnan(price) or np.isnan(cmo_value):
            continue

        alpha = base_alpha * abs(cmo_value) / 100
        if prev_vidya is None:
            prev_vidya = price

        vid_value = alpha * price + (1 - alpha) * prev_vidya
        vidya.iat[idx] = vid_value
        prev_vidya = vid_value

    name = f'VID_{window}_{cmo_window}'
    vidya.name = name

    columns_list = [name]
    return vidya, columns_list

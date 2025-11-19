import pandas as pd
from typing import Iterable, List


def gma(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Guppy Multiple Moving Average (GMMA).

    GMMA applies two sets of exponential moving averages (EMAs):
    short-term averages that react quickly to price changes and long-term
    averages that capture the broader trend. The relationship between the two
    groups helps identify trend strength and potential transitions.
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}

    close_col = columns.get('close_col', 'Close')
    short_windows = _normalize_windows(
        parameters.get('short_windows', (3, 5, 8, 10, 12, 15))
    )
    long_windows = _normalize_windows(
        parameters.get('long_windows', (30, 35, 40, 45, 50, 60))
    )

    close = df[close_col]

    ema_data = {}
    for window in short_windows:
        ema_series = close.ewm(span=window, adjust=False).mean()
        ema_series.name = f'GMA_short_{window}'
        ema_data[ema_series.name] = ema_series

    for window in long_windows:
        ema_series = close.ewm(span=window, adjust=False).mean()
        ema_series.name = f'GMA_long_{window}'
        ema_data[ema_series.name] = ema_series

    result_df = pd.DataFrame(ema_data, index=close.index)
    columns_list = list(result_df.columns)

    return result_df, columns_list


def _normalize_windows(value: Iterable) -> List[int]:
    try:
        windows = list(value)
    except TypeError:
        windows = [value]

    cleaned: List[int] = []
    for window in windows:
        try:
            window_int = int(window)
        except (TypeError, ValueError):
            continue
        if window_int > 0:
            cleaned.append(window_int)

    if not cleaned:
        cleaned = [1]

    return cleaned

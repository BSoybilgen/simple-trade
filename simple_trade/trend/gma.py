import pandas as pd
from typing import Iterable, List


def gma(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Guppy Multiple Moving Average (GMMA).
    GMMA applies two sets of exponential moving averages (EMAs):
    short-term averages that react quickly to price changes and long-term
    averages that capture the broader trend. The relationship between the two
    groups helps identify trend strength and potential transitions.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - short_windows (iterable): List of periods for short-term EMAs. Default is (3, 5, 8, 10, 12, 15).
            - long_windows (iterable): List of periods for long-term EMAs. Default is (30, 35, 40, 45, 50, 60).
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the GMMA DataFrame (multiple EMA columns) and a list of column names.

    The GMMA is calculated as follows:

    1. Calculate Short-Term EMAs:
       Calculate EMA for each period in short_windows.

    2. Calculate Long-Term EMAs:
       Calculate EMA for each period in long_windows.

    Interpretation:
    - Compression: When EMAs bunch together, it indicates agreement on price and potential for a breakout.
    - Expansion: When EMAs spread out, it indicates a strong trend.
    - Crossovers: Short-term group crossing above long-term group indicates a bullish trend reversal.
    - Separation: The space between the short-term and long-term groups indicates the strength of the trend.

    Use Cases:
    - Trend Identification: Determining the long-term trend direction.
    - Entry Signals: Short-term pullbacks into the long-term group during a strong trend.
    - Reversal Signals: Crossover of the two groups.
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

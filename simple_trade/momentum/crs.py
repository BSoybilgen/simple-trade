import numpy as np
import pandas as pd


def crs(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """Calculate the Connors RSI (CRSI).

    CRSI combines three components:
        1. RSI of closing prices (short period)
        2. RSI of streak length (consecutive up/down closes)
        3. Percent rank of one-day price change

    The final value is the average of the three sub-indicators, yielding a value
    between 0 and 100 useful for short-term overbought/oversold analysis.

    Args:
        df (pd.DataFrame): Input price data.
        parameters (dict, optional): Calculation parameters.
            - rsi_window (int): Window for close-price RSI (default 3).
            - streak_window (int): Window for streak RSI (default 2).
            - rank_window (int): Lookback for percent rank of price change (default 100).
        columns (dict, optional): Column overrides.
            - close_col (str): Column with closing prices (default 'Close').

    Returns:
        tuple: (crsi_series, [column_name])
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}

    rsi_window = int(parameters.get('rsi_window', 3))
    streak_window = int(parameters.get('streak_window', 2))
    rank_window = int(parameters.get('rank_window', 100))
    close_col = columns.get('close_col', 'Close')

    close = df[close_col]

    def _rsi(series: pd.Series, window: int) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=window, min_periods=window).mean()
        avg_loss = loss.rolling(window=window, min_periods=window).mean()
        rs = avg_gain / avg_loss.replace({0: np.nan})
        rsi_values = 100 - (100 / (1 + rs))
        return rsi_values.astype(float)

    # Component 1: RSI of closing prices
    price_rsi = _rsi(close, rsi_window)

    # Component 2: RSI of streak length
    delta = close.diff()
    streak = pd.Series(0.0, index=close.index)
    for i in range(1, len(close)):
        if delta.iloc[i] > 0:
            streak.iloc[i] = streak.iloc[i - 1] + 1 if streak.iloc[i - 1] > 0 else 1
        elif delta.iloc[i] < 0:
            streak.iloc[i] = streak.iloc[i - 1] - 1 if streak.iloc[i - 1] < 0 else -1
        else:
            streak.iloc[i] = 0
    streak_rsi = _rsi(streak, streak_window)

    # Component 3: Percent rank of price change
    price_change = close.diff()

    def _percent_rank(window_values):
        valid = window_values[~pd.isna(window_values)]
        if len(valid) < len(window_values) or len(valid) <= 1:
            return np.nan
        last = valid.iloc[-1]
        less_count = (valid.iloc[:-1] < last).sum()
        equal_count = (valid.iloc[:-1] == last).sum()
        denom = len(valid) - 1
        if denom <= 0:
            return np.nan
        rank = (less_count + 0.5 * equal_count) / denom
        return rank * 100

    percent_rank = price_change.rolling(window=rank_window, min_periods=rank_window).apply(
        lambda x: _percent_rank(pd.Series(x)), raw=False
    )

    percent_rank = percent_rank.astype(float)

    crsi_values = (price_rsi + streak_rsi + percent_rank) / 3
    crsi_values.name = f'CRSI_{rsi_window}_{streak_window}_{rank_window}'

    columns_list = [crsi_values.name]
    return crsi_values, columns_list

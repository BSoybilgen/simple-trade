import numpy as np
import pandas as pd


def crs(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Connors RSI (CRSI), a composite indicator designed by Larry Connors 
    to better identify short-term overbought and oversold conditions.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - rsi_window (int): Window for the close-price RSI component. Default is 3.
            - streak_window (int): Window for the streak RSI component. Default is 2.
            - rank_window (int): Lookback period for the percent rank component. Default is 100.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the Connors RSI series and a list of column names.

    The Connors RSI is calculated as the average of three components:

    1. Calculate the RSI of Closing Prices:
       RSI_Price = RSI(Close, rsi_window)

    2. Calculate the RSI of the Streak:
       - Determine the Streak (consecutive days up or down).
       - RSI_Streak = RSI(Streak, streak_window)

    3. Calculate the Percent Rank of Price Change:
       - Calculate one-day price change (Percent Change).
       - Rank = Percent Rank of today's change over rank_window.

    4. Calculate CRSI:
       CRSI = (RSI_Price + RSI_Streak + Rank) / 3

    Interpretation:
    - Range: 0 to 100.
    - Overbought: Values above 90 (or 95) indicate potential short-term exhaustion/overbought.
    - Oversold: Values below 10 (or 5) indicate potential short-term oversold conditions.

    Use Cases:
    - Mean Reversion: Identifying short-term pullback opportunities in a broader trend.
    - Exit Signals: Exiting positions when CRSI reaches extreme levels.
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

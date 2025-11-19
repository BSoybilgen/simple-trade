import pandas as pd


def ult(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """Calculate the Ultimate Oscillator (ULTOSC).

    The Ultimate Oscillator combines short, medium, and long-term buying pressure
    averages to reduce volatility and false signals compared to using a single-period
    oscillator. Values above 70 indicate potential overbought conditions, while
    values below 30 indicate potential oversold conditions.

    Args:
        df (pd.DataFrame): Input OHLC data.
        parameters (dict, optional): Calculation parameters.
            - short_window (int): Short-term period length. Default is 7.
            - medium_window (int): Medium-term period length. Default is 14.
            - long_window (int): Long-term period length. Default is 28.
        columns (dict, optional): Column overrides.
            - close_col (str): Close price column (default 'Close').
            - high_col (str): High price column (default 'High').
            - low_col (str): Low price column (default 'Low').

    Returns:
        tuple: (ultimate_oscillator_series, [column_name])
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}

    short_window = int(parameters.get('short_window', 7))
    medium_window = int(parameters.get('medium_window', 14))
    long_window = int(parameters.get('long_window', 28))

    close_col = columns.get('close_col', 'Close')
    high_col = columns.get('high_col', 'High')
    low_col = columns.get('low_col', 'Low')

    close = df[close_col]
    high = df[high_col]
    low = df[low_col]
    prev_close = close.shift(1)

    min_low_close = pd.concat([low, prev_close], axis=1).min(axis=1)
    max_high_close = pd.concat([high, prev_close], axis=1).max(axis=1)

    buying_pressure = close - min_low_close
    true_range = max_high_close - min_low_close

    def _avg_bp_tr(window: int):
        bp_sum = buying_pressure.rolling(window=window, min_periods=window).sum()
        tr_sum = true_range.rolling(window=window, min_periods=window).sum()
        return bp_sum / tr_sum.where(tr_sum != 0)

    avg_short = _avg_bp_tr(short_window)
    avg_medium = _avg_bp_tr(medium_window)
    avg_long = _avg_bp_tr(long_window)

    ultimate = 100 * ((4 * avg_short) + (2 * avg_medium) + avg_long) / 7
    ultimate.name = f'ULTOSC_{short_window}_{medium_window}_{long_window}'

    columns_list = [ultimate.name]
    return ultimate, columns_list

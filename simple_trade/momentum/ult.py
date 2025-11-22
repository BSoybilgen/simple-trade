import pandas as pd


def ult(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Ultimate Oscillator (ULTOSC), developed by Larry Williams.
    It combines short, medium, and long-term buying pressure to reduce volatility 
    and false signals associated with single-timeframe oscillators.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - short_window (int): Short-term period length. Default is 7.
            - medium_window (int): Medium-term period length. Default is 14.
            - long_window (int): Long-term period length. Default is 28.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.
            - high_col (str): The column name for high prices. Default is 'High'.
            - low_col (str): The column name for low prices. Default is 'Low'.

    Returns:
        tuple: A tuple containing the Ultimate Oscillator series and a list of column names.

    The Ultimate Oscillator is calculated as follows:

    1. Calculate Buying Pressure (BP):
       BP = Close - Min(Low, Prev Close)

    2. Calculate True Range (TR):
       TR = Max(High, Prev Close) - Min(Low, Prev Close)

    3. Calculate Averages for each window:
       Avg7 = Sum(BP, 7) / Sum(TR, 7)
       Avg14 = Sum(BP, 14) / Sum(TR, 14)
       Avg28 = Sum(BP, 28) / Sum(TR, 28)

    4. Calculate Ultimate Oscillator:
       UltOsc = 100 * ((4 * Avg7) + (2 * Avg14) + Avg28) / 7

    Interpretation:
    - Range: 0 to 100.
    - Overbought: Values above 70 indicate potential overbought conditions.
    - Oversold: Values below 30 indicate potential oversold conditions.

    Use Cases:
    - Divergence: Bullish divergence (Price lower low, Oscillator higher low) in oversold territory is a strong buy signal.
    - Breakouts: Can be used to confirm trend reversals.
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

import pandas as pd


def stc(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """Calculate the Schaff Trend Cycle (STC) indicator.

    Args:
        df (pd.DataFrame): Input price data containing close prices.
        parameters (dict, optional): Calculation parameters.
            - window_fast (int): Fast EMA length used in MACD. Default is 23.
            - window_slow (int): Slow EMA length used in MACD. Default is 50.
            - cycle (int): Look-back window for stochastic calculations. Default is 10.
            - smooth (int): EMA smoothing factor for the cycle. Default is 3.
        columns (dict, optional): Column overrides.
            - close_col (str): Column name for closing prices. Default is 'Close'.

    Returns:
        tuple: (stc_series, [column_name])
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}

    window_fast = int(parameters.get('window_fast', 23))
    window_slow = int(parameters.get('window_slow', 50))
    cycle = int(parameters.get('cycle', 10))
    smooth = int(parameters.get('smooth', 3))
    close_col = columns.get('close_col', 'Close')

    series = pd.to_numeric(df[close_col], errors='coerce')
    ema_fast = series.ewm(span=window_fast, adjust=False).mean()
    ema_slow = series.ewm(span=window_slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow

    macd_min = macd_line.rolling(window=cycle, min_periods=1).min()
    macd_max = macd_line.rolling(window=cycle, min_periods=1).max()
    macd_range = macd_max - macd_min
    macd_range = macd_range.replace(0, float('nan'))

    percent_k = ((macd_line - macd_min) / macd_range) * 100
    percent_k = percent_k.fillna(0).clip(lower=0, upper=100).astype(float)

    smoothed_k = percent_k.ewm(span=smooth, adjust=False).mean()

    smoothed_min = smoothed_k.rolling(window=cycle, min_periods=1).min()
    smoothed_max = smoothed_k.rolling(window=cycle, min_periods=1).max()
    smoothed_range = smoothed_max - smoothed_min
    smoothed_range = smoothed_range.replace(0, float('nan'))

    stc_values = ((smoothed_k - smoothed_min) / smoothed_range) * 100
    stc_values = stc_values.fillna(0).clip(lower=0, upper=100).astype(float)
    stc_values.name = f'STC_{window_fast}_{window_slow}_{cycle}'

    columns_list = [stc_values.name]
    return stc_values, columns_list

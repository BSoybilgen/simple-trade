import pandas as pd


def dpo(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """Calculate the Detrended Price Oscillator (DPO).

    The DPO removes the longer-term trend from price to highlight shorter-term cycles.
    It subtracts a displaced moving average from the closing price, oscillating around zero.

    Args:
        df (pd.DataFrame): Input price DataFrame.
        parameters (dict, optional): Calculation parameters.
            - window (int): Lookback period for the SMA. Default is 20.
        columns (dict, optional): Column overrides.
            - close_col (str): Column name for closing prices. Default is 'Close'.

    Returns:
        tuple: (dpo_series, [column_name])
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}

    window = int(parameters.get('window', 20))
    close_col = columns.get('close_col', 'Close')

    series = df[close_col]
    sma = series.rolling(window=window, min_periods=window).mean()

    displacement = (window // 2) + 1
    displaced_sma = sma.shift(displacement)

    dpo_values = series - displaced_sma
    dpo_values.name = f'DPO_{window}'

    columns_list = [dpo_values.name]
    return dpo_values, columns_list

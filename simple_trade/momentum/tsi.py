import pandas as pd


def tsi(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """Calculate the True Strength Index (TSI).

    TSI double smooths price momentum and its absolute value to create a bounded
    oscillator between -100 and +100 that filters out short-term noise while
    highlighting trend strength.

    Args:
        df (pd.DataFrame): Input DataFrame containing price data.
        parameters (dict, optional): Calculation parameters.
            - slow (int): Span for the first (slow) EMA smoothing of momentum. Default 25.
            - fast (int): Span for the second (fast) EMA smoothing. Default 13.
        columns (dict, optional): Column overrides.
            - close_col (str): Column name for closing prices. Default 'Close'.

    Returns:
        tuple: (tsi_series, [column_name])
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}

    slow = int(parameters.get('slow', 25))
    fast = int(parameters.get('fast', 13))
    close_col = columns.get('close_col', 'Close')

    series = df[close_col]
    momentum = series.diff()

    def _double_ema(values: pd.Series) -> pd.Series:
        first = values.ewm(span=slow, adjust=False, min_periods=slow).mean()
        return first.ewm(span=fast, adjust=False, min_periods=slow + fast - 1).mean()

    smoothed_momentum = _double_ema(momentum)
    smoothed_abs_momentum = _double_ema(momentum.abs())

    tsi_series = 100 * smoothed_momentum / smoothed_abs_momentum.replace({0: pd.NA})
    tsi_series.name = f'TSI_{slow}_{fast}'

    columns_list = [tsi_series.name]
    return tsi_series, columns_list

import numpy as np
import pandas as pd


def cog(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """Calculate the Center of Gravity (COG) momentum indicator.

    Args:
        df (pd.DataFrame): Input price data containing close prices.
        parameters (dict, optional): Calculation parameters.
            - window (int): Lookback length for the COG calculation. Default is 10.
        columns (dict, optional): Column overrides.
            - close_col (str): Column name for closing prices. Default is 'Close'.

    Returns:
        tuple: (cog_series, [column_name])
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}

    window = int(parameters.get('window', 10))
    close_col = columns.get('close_col', 'Close')

    series = df[close_col]

    def _cog(values: np.ndarray) -> float:
        if len(values) < window:
            return np.nan
        denom = values.sum()
        if denom == 0:
            return np.nan
        weights = np.arange(1, len(values) + 1, dtype=float)
        weighted_sum = (values[::-1] * weights).sum()
        return -weighted_sum / denom + (len(values) + 1) / 2

    cog_series = series.rolling(window=window, min_periods=window).apply(_cog, raw=True)
    cog_series.name = f'COG_{window}'

    columns_list = [cog_series.name]
    return cog_series, columns_list

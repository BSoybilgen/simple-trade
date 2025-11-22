import numpy as np
import pandas as pd


def cog(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Center of Gravity (COG), a momentum indicator developed by John Ehlers.
    It is designed to spot turning points in prices with zero lag by analogy to the physical center of gravity.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - window (int): The lookback period for the calculation. Default is 10.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the COG series and a list of column names.

    The Center of Gravity is calculated using a weighted sum of prices:

    1. Assign weights to prices in the window:
       Weights range from 1 to window (most recent).

    2. Calculate Weighted Sums:
       Numerator = -Sum(Price[i] * (i + 1)) over the window (conceptually adjusted for lag)
       Denominator = Sum(Price[i]) over the window

    3. Calculate COG:
       COG = -Numerator / Denominator + (window + 1) / 2
       
       (Note: The specific implementation details may vary to align with Ehler's filter formula).

    Interpretation:
    - The COG acts as a leading indicator, helping to identify market turning points.
    - It helps in visualizing the "center of mass" of the price action over the window.

    Use Cases:
    - Turning Points: Identifying potential peaks and valleys in price action.
    - Cycle Analysis: Used in conjunction with other cycle-based indicators.
    - Trend Reversals: Sharp changes in COG can signal immediate trend reversals.
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

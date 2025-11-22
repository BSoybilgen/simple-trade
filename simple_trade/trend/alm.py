import math
import numpy as np
import pandas as pd


def alm(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Arnaud Legoux Moving Average (ALMA).
    ALMA uses a Gaussian distribution to apply weights to the moving window,
    allowing for smoothness and responsiveness with minimal lag.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - window (int): The lookback period. Default is 9.
            - sigma (float): The standard deviation of the Gaussian distribution. Default is 6.
            - offset (float): The offset of the Gaussian distribution (0 to 1). Default is 0.85.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the ALMA series and a list of column names.

    The ALMA is calculated as follows:

    1. Determine Weights using Gaussian Distribution:
       Weight(i) = exp( - (i - m)^2 / (2 * s^2) )
       Where:
       m = offset * (window - 1)
       s = window / sigma

    2. Normalize Weights:
       Normalized Weight(i) = Weight(i) / Sum(Weights)

    3. Calculate ALMA:
       ALMA = Sum(Price(i) * Normalized Weight(i)) over the window

    Interpretation:
    - ALMA is designed to be smoother than SMA/EMA but with less lag.
    - Offset > 0.5 makes it more responsive to recent prices (less lag, more overshoot).
    - Sigma controls the width of the filter (smoothness).

    Use Cases:
    - Trend Identification: Similar to other moving averages but with better fidelity.
    - Support/Resistance: Acts as dynamic support and resistance.
    - Crossovers: Fast ALMA crossing slow ALMA.
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}

    close_col = columns.get('close_col', 'Close')
    window = int(parameters.get('window', 9))
    sigma = float(parameters.get('sigma', 6))
    offset = float(parameters.get('offset', 0.85))

    series = df[close_col]

    m = offset * (window - 1)
    s = window / sigma

    w = []
    for i in range(window):
        weight = math.exp(-((i - m) ** 2) / (2 * (s ** 2)))
        w.append(weight)

    weights = np.array(w)
    weights = weights / weights.sum()  # Normalize

    def weighted_avg(x):
        return np.dot(x, weights)

    alma_series = series.rolling(window=window).apply(weighted_avg, raw=True)
    alma_series.name = f'ALMA_{window}'

    columns_list = [alma_series.name]
    return alma_series, columns_list

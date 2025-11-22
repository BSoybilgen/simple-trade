import math
import numpy as np
import pandas as pd


def swm(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Sine Weighted Moving Average (SWMA).
    SWMA uses sine wave weighting to emphasize the middle portion of the
    moving window, providing smooth transitions with reduced lag compared
    to simple moving averages.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - window (int): The lookback period. Default is 20.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the SWMA series and a list of column names.

    The SWMA is calculated as follows:

    1. Generate Sine Weights:
       Weight(i) = sin((i + 1) * PI / (window + 1))
       (Weights follow a sine curve, peaking in the middle of the window)

    2. Normalize Weights:
       Normalized Weight(i) = Weight(i) / Sum(Weights)

    3. Calculate SWMA:
       SWMA = Sum(Price(i) * Normalized Weight(i)) over the window

    Interpretation:
    - SWMA gives less weight to recent data compared to WMA or EMA, but more than the beginning of the window.
    - It aims to extract the cyclical component of the price action.

    Use Cases:
    - Cycle Analysis: Better suited for identifying cyclical turning points than trend following.
    - Smoothing: Provides a very smooth average.
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}

    close_col = columns.get('close_col', 'Close')
    window = int(parameters.get('window', 20))

    series = df[close_col]

    # Generate sine weights
    weights = []
    for i in range(window):
        # Sine wave from 0 to π
        weight = math.sin((i + 1) * math.pi / (window + 1))
        weights.append(weight)
    
    weights = np.array(weights)
    weights = weights / weights.sum()  # Normalize

    def weighted_avg(x):
        return np.dot(x, weights)

    swma_series = series.rolling(window=window).apply(weighted_avg, raw=True)
    swma_series.name = f'SWMA_{window}'

    columns_list = [swma_series.name]
    return swma_series, columns_list

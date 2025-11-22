import pandas as pd
import numpy as np


def wma(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Weighted Moving Average (WMA) of a series.
    The WMA is a moving average that assigns different weights to data points,
    typically giving more weight to recent data.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - window (int): The lookback period for the WMA. Default is 20.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the WMA series and a list of column names.

    Calculation Steps:
    1. Generate Weights:
       Create an array of weights from 1 to window.
       Weights = [1, 2, ..., window]
    2. Calculate Weighted Average:
       WMA = Sum(Price * Weight) / Sum(Weights)

    Interpretation:
    - More responsive to recent price changes than SMA due to linear weighting.
    - Less lag than SMA but more than EMA (typically).

    Use Cases:
    - Trend Identification: Identifying direction with less lag than SMA.
    - Crossovers: Using WMA in crossover strategies for faster signals.
    - Smoothing: General price smoothing.
    """
    # Set default values
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    # Extract parameters with defaults
    close_col = columns.get('close_col', 'Close')
    window = int(parameters.get('window', 20))
    
    series = df[close_col]
    weights = np.arange(1, window + 1)
    series = series.rolling(window).apply(lambda prices: np.dot(prices, weights) / weights.sum(), raw=True)
    series.name = f'WMA_{window}'
    columns = [series.name]
    return series, columns
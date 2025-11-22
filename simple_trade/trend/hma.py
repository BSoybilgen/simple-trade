import pandas as pd
import numpy as np
from .wma import wma

def hma(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Hull Moving Average (HMA) of a series.
    The HMA is a moving average that reduces lag and improves smoothing.
    It is calculated using weighted moving averages (WMAs) with specific
    window lengths to achieve this effect.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - window (int): The lookback period for the HMA. Default is 20.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the HMA series and a list of column names.

    Calculation Steps:
    1. Calculate WMA of Half Length:
       WMA1 = WMA(Close, window / 2)

    2. Calculate WMA of Full Length:
       WMA2 = WMA(Close, window)

    3. Calculate Raw HMA:
       Raw = 2 * WMA1 - WMA2

    4. Calculate Final HMA:
       HMA = WMA(Raw, sqrt(window))

    Interpretation:
    - HMA hugs the price action much closer than SMA or EMA.
    - The turning points in HMA are often sharper and more timely.

    Use Cases:
    - Identifying trends: The HMA can be used to identify the direction of a
      price trend.
    - Smoothing price data: The HMA can smooth out short-term price fluctuations
      to provide a clearer view of the underlying trend.
    - Generating buy and sell signals: The HMA can be used in crossover systems
      to generate buy and sell signals.
    """
    # Set default values
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    # Extract parameters with defaults
    close_col = columns.get('close_col', 'Close')
    window = int(parameters.get('window', 20))

    half_length = int(window / 2)
    sqrt_length = int(np.sqrt(window))
    # Create parameter dicts for wma function
    wma_params_half = {'window': half_length}
    wma_params_full = {'window': window}
    wma_cols = {'close_col': close_col}
    
    # wma now returns a tuple (series, columns)
    wma_half_series = wma(df, parameters=wma_params_half, columns=wma_cols)[0]
    wma_full_series = wma(df, parameters=wma_params_full, columns=wma_cols)[0]

    df_mid = pd.DataFrame(2 * wma_half_series - wma_full_series, columns=[close_col])
    
    wma_params_sqrt = {'window': sqrt_length}
    hma_series = wma(df_mid, parameters=wma_params_sqrt, columns=wma_cols)[0]
    hma_series.name = f'HMA_{window}'

    columns_list = [hma_series.name]
    return hma_series, columns_list
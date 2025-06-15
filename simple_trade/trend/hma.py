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
        df (pd.DataFrame): The dataframe containing price data. Must have close column.
        parameter (dict): The parameter dictionary that includes the window size for the HMA.
        columns (dict): The column dictionary that includes close column name.

    Returns:
        tuple: A tuple containing the HMA series and a list of column names.

    The Hull Moving Average (HMA) is a type of moving average that is designed
    to reduce lag and improve smoothing compared to traditional moving averages.
    It achieves this by using a combination of weighted moving averages (WMAs)
    with different window lengths.

    The formula for calculating the HMA is as follows:

    1. Calculate a WMA of the input series with a window length of half the
       specified window size (half_length).
    2. Calculate a WMA of the input series with the full specified window size.
    3. Calculate the difference between 2 times the first WMA and the second WMA.
    4. Calculate a WMA of the result from step 3 with a window length equal to
       the square root of the specified window size.

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
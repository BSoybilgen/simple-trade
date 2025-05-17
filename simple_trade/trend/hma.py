import pandas as pd
import numpy as np
from .wma import wma

def hma(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> pd.Series:
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
        pd.Series: The HMA of the series.

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
    window = parameters.get('window', 20)

    half_length = int(window / 2)
    sqrt_length = int(np.sqrt(window))
    # Create parameter dicts for wma function
    wma_params_half = {'window': half_length}
    wma_params_full = {'window': window}
    wma_cols = {'close_col': close_col}
    
    wma_half = wma(df, parameters=wma_params_half, columns=wma_cols)
    wma_full = wma(df, parameters=wma_params_full, columns=wma_cols)
    df = pd.DataFrame(2 * wma_half - wma_full, columns=[close_col])
    
    wma_params_sqrt = {'window': sqrt_length}
    hma_ = wma(df, parameters=wma_params_sqrt, columns=wma_cols)
    hma_.name = f'HMA_{window}'
    return hma_
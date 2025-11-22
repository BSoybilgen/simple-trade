import pandas as pd
import numpy as np


def wad(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates Williams Accumulation/Distribution (WAD), an indicator that uses
    price changes (True Range) to determine accumulation or distribution.
    It measures market pressure by comparing the close to the true range.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters.
            No parameters are used.
        columns (dict, optional): Dictionary containing column name mappings:
            - high_col (str): The column name for high prices. Default is 'High'.
            - low_col (str): The column name for low prices. Default is 'Low'.
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the WAD series and a list of column names.

    The Williams Accumulation/Distribution is calculated as follows:

    1. Calculate True Range High (TRH) and True Range Low (TRL):
       TRH = Max(Current High, Previous Close)
       TRL = Min(Current Low, Previous Close)

    2. Calculate Price Move (PM):
       - If Close > Previous Close: PM = Close - TRL
       - If Close < Previous Close: PM = Close - TRH
       - If Close = Previous Close: PM = 0

    3. Calculate WAD (Cumulative):
       WAD = Previous WAD + PM

    Interpretation:
    - Rising WAD: Accumulation (Buying pressure).
    - Falling WAD: Distribution (Selling pressure).
    - Divergence: Price vs WAD divergence signals potential reversals.

    Use Cases:
    - Trend Confirmation: WAD should align with price direction.
    - Divergence Analysis: Powerful tool for spotting tops and bottoms.
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    high_col = columns.get('high_col', 'High')
    low_col = columns.get('low_col', 'Low')
    close_col = columns.get('close_col', 'Close')
    
    high = df[high_col]
    low = df[low_col]
    close = df[close_col]
    prev_close = close.shift(1)
    
    # 1. True Range High and Low
    # We use numpy fmax/fmin for element-wise comparison, filling NaN prev_close first
    # For the first element, prev_close is NaN. We can fill it with current close or high/low
    # Standard practice is usually to start accumulation from 0 or match price
    
    trh = np.maximum(high, prev_close.fillna(high))
    trl = np.minimum(low, prev_close.fillna(low))
    
    # 2. Calculate Price Move
    ad = pd.Series(0.0, index=df.index)
    
    # Close > Prev Close
    mask_up = close > prev_close
    ad[mask_up] = close[mask_up] - trl[mask_up]
    
    # Close < Prev Close
    mask_down = close < prev_close
    ad[mask_down] = close[mask_down] - trh[mask_down]
    
    # Close == Prev Close is already 0
    
    # 3. Cumulative Sum
    wad_values = ad.cumsum()
    
    wad_values.name = 'WAD'
    columns_list = [wad_values.name]
    return wad_values, columns_list

import pandas as pd
import numpy as np


def cho(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Choppiness Index (CHOP), a volatility indicator designed to determine
    whether the market is trending or trading sideways (choppy). It measures the market's
    trendiness on a scale from 0 to 100.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - period (int): The lookback period for calculations. Default is 14.
        columns (dict, optional): Dictionary containing column name mappings:
            - high_col (str): The column name for high prices. Default is 'High'.
            - low_col (str): The column name for low prices. Default is 'Low'.
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the Choppiness Index series and a list of column names.

    Calculation Steps:
    1. Calculate True Range (TR) for each period.
    2. Sum True Range over the period: SumTR = Sum(TR, period)
    3. Calculate High-Low Range over the period: Range = Highest High - Lowest Low
    4. Calculate CHOP: CHOP = 100 * log10(SumTR / Range) / log10(period)

    Interpretation:
    - High CHOP (>61.8): Market is consolidating (choppy), avoid trend-following strategies.
    - Low CHOP (<38.2): Market is trending, favorable for trend-following strategies.
    - Rising CHOP: Trend is weakening, market entering consolidation.
    - Falling CHOP: Consolidation is ending, potential breakout approaching.

    Use Cases:
    - Trend vs. Range identification: Determine market regime.
    - Trade filtering: Filter out trend trades during high choppiness.
    - Breakout anticipation: Low volatility/high chop often precedes breakouts.
    """
    # Set default values
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    # Extract parameters with defaults
    period = int(parameters.get('period', 14))
    high_col = columns.get('high_col', 'High')
    low_col = columns.get('low_col', 'Low')
    close_col = columns.get('close_col', 'Close')
    
    high = df[high_col]
    low = df[low_col]
    close = df[close_col]
    
    # Calculate True Range
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Sum of True Range over period
    sum_tr = tr.rolling(window=period).sum()
    
    # Highest high and lowest low over period
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()
    
    # Calculate high-low range
    hl_range = highest_high - lowest_low
    
    # Calculate Choppiness Index
    # CHOP = 100 * log10(sum_tr / hl_range) / log10(period)
    # Handle division by zero
    chop_values = pd.Series(index=close.index, dtype=float)
    
    # Only calculate where hl_range > 0 to avoid division by zero
    valid_mask = hl_range > 0
    
    chop_values[valid_mask] = (
        100 * np.log10(sum_tr[valid_mask] / hl_range[valid_mask]) / np.log10(period)
    )
    
    chop_values.name = f'CHOP_{period}'
    columns_list = [chop_values.name]
    return chop_values, columns_list

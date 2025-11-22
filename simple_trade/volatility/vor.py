import pandas as pd


def vor(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Volatility Ratio (VR), which compares short-term volatility to
    long-term volatility to identify changes in volatility regimes. A ratio above 1
    indicates increasing volatility, while below 1 indicates decreasing volatility.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - short_period (int): Short-term volatility period. Default is 5.
            - long_period (int): Long-term volatility period. Default is 20.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the Volatility Ratio series and a list of column names.

    Calculation Steps:
    1. Calculate Short-Term Volatility:
       Short Vol = Standard Deviation(Close, short_period)
    2. Calculate Long-Term Volatility:
       Long Vol = Standard Deviation(Close, long_period)
    3. Calculate Ratio:
       VR = Short Vol / Long Vol

    Interpretation:
    - VR > 1: Short-term volatility is expanding relative to long-term.
    - VR < 1: Short-term volatility is contracting (consolidation).
    - Breakout Signal: A crossing above 1 often signals a price breakout from range.

    Use Cases:
    - Volatility regime detection.
    - Strategy switching (Breakout vs. Mean Reversion).
    - Breakout confirmation.

    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    short_period = int(parameters.get('short_period', 5))
    long_period = int(parameters.get('long_period', 20))
    close_col = columns.get('close_col', 'Close')
    
    close = df[close_col]
    
    # Calculate short-term and long-term standard deviation
    short_std = close.rolling(window=short_period).std()
    long_std = close.rolling(window=long_period).std()
    
    # Calculate volatility ratio
    vr_values = short_std / long_std
    vr_values = vr_values.fillna(1.0)
    
    vr_values.name = f'VR_{short_period}_{long_period}'
    columns_list = [vr_values.name]
    return vr_values, columns_list

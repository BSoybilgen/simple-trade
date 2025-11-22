import pandas as pd


def nat(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Normalized Average True Range (NATR), which expresses ATR as a
    percentage of the closing price, similar to ATRP but with additional normalization
    commonly used in technical analysis platforms.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - window (int): The lookback period for ATR calculation. Default is 14.
        columns (dict, optional): Dictionary containing column name mappings:
            - high_col (str): The column name for high prices. Default is 'High'.
            - low_col (str): The column name for low prices. Default is 'Low'.
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the NATR series and a list of column names.

    Calculation Steps:
    1. Calculate ATR:
       Using Wilder's smoothing method over the specified window.
    2. Normalize:
       NATR = (ATR / Close) * 100

    Interpretation:
    - Low NATR (<2%): Low relative volatility.
    - Medium NATR (2-5%): Normal volatility.
    - High NATR (>5%): High relative volatility.

    Use Cases:
    - Cross-asset volatility comparison.
    - Normalized position sizing.
    - Historical volatility analysis.
    - Percentage-based stop-loss placement.
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    window = int(parameters.get('window', 14))
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
    
    # Calculate ATR using Wilder's smoothing
    atr_values = pd.Series(index=close.index, dtype=float)
    first_atr = tr.iloc[:window].mean()
    atr_values.iloc[window-1] = first_atr
    
    for i in range(window, len(close)):
        atr_values.iloc[i] = ((atr_values.iloc[i-1] * (window-1)) + tr.iloc[i]) / window
    
    # Normalize to percentage
    natr_values = (atr_values / close) * 100
    
    natr_values.name = f'NATR_{window}'
    columns_list = [natr_values.name]
    return natr_values, columns_list

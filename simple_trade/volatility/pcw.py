import pandas as pd


def pcw(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates Price Channel Width (PCW), which measures the width of a price channel
    (Donchian-style) as a percentage of the closing price. It provides a simple measure
    of volatility based on the high-low range over a period.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - period (int): The lookback period for channel calculation. Default is 20.
        columns (dict, optional): Dictionary containing column name mappings:
            - high_col (str): The column name for high prices. Default is 'High'.
            - low_col (str): The column name for low prices. Default is 'Low'.
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the PCW series and a list of column names.

    Calculation Steps:
    1. Find Highest High and Lowest Low over the period.
    2. Calculate Channel Width as Percentage:
       PCW = ((Highest High - Lowest Low) / Close) * 100

    Interpretation:
    - Low PCW: Narrow price channel, low volatility, consolidation.
    - High PCW: Wide price channel, high volatility, trending.
    - Increasing PCW: Expanding volatility.
    - Decreasing PCW: Contracting volatility.

    Use Cases:
    - Simple volatility measurement.
    - Breakout identification (low PCW precedes breakouts).
    - Channel-based trading strategies.
    - Volatility comparison across assets.
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    period = int(parameters.get('period', 20))
    high_col = columns.get('high_col', 'High')
    low_col = columns.get('low_col', 'Low')
    close_col = columns.get('close_col', 'Close')
    
    high = df[high_col]
    low = df[low_col]
    close = df[close_col]
    
    # Calculate highest high and lowest low over period
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()
    
    # Calculate channel width as percentage of close
    pcw_values = ((highest_high - lowest_low) / close) * 100
    
    pcw_values.name = f'PCW_{period}'
    columns_list = [pcw_values.name]
    return pcw_values, columns_list

import pandas as pd

def wil(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Williams %R (Williams Percent Range), a momentum oscillator that measures
    overbought and oversold levels based on recent closing prices relative to the high-low range.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - window (int): The lookback period used for the highest high and lowest low. Default is 14.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.
            - high_col (str): The column name for high prices. Default is 'High'.
            - low_col (str): The column name for low prices. Default is 'Low'.

    Returns:
        tuple: A tuple containing the Williams %R series and a list of column names.

    The Williams %R is calculated as follows:

    1. Identify Highest High and Lowest Low over the window.

    2. Calculate %R:
       %R = ((Highest High - Close) / (Highest High - Lowest Low)) * -100

    Interpretation:
    - Range: 0 to -100.
    - Overbought: Values above -20 (closer to 0) indicate overbought conditions.
    - Oversold: Values below -80 (closer to -100) indicate oversold conditions.
    - Momentum Failure: If price makes a new high but %R fails to move above -20, it signals weak momentum.

    Use Cases:
    - Overbought/Oversold: Identifying potential reversal points.
    - Momentum Confirmation: Strong trends tend to keep %R near the extremes (-20 in uptrends, -80 in downtrends).
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}

    window = int(parameters.get('window', 14))
    close_col = columns.get('close_col', 'Close')
    high_col = columns.get('high_col', 'High')
    low_col = columns.get('low_col', 'Low')

    close = df[close_col]
    high = df[high_col]
    low = df[low_col]

    highest_high = high.rolling(window=window, min_periods=window).max()
    lowest_low = low.rolling(window=window, min_periods=window).min()
    range_values = (highest_high - lowest_low).where(lambda x: x != 0)

    williams_r = ((highest_high - close) / range_values) * -100
    williams_r.name = f'WILLR_{window}'

    columns_list = [williams_r.name]
    return williams_r, columns_list

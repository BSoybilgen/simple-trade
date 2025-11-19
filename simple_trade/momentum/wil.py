import pandas as pd

def wil(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Williams %R (Williams Percent Range), a momentum oscillator that measures
    overbought and oversold levels based on recent closing prices relative to the high-low range.

    Args:
        df (pd.DataFrame): The input DataFrame containing price data.
        parameters (dict, optional): Calculation parameters.
            - window (int): Lookback period used for the highest high and lowest low. Default is 14.
        columns (dict, optional): Column mapping for price data.
            - close_col (str): Column name for closing prices. Default is 'Close'.
            - high_col (str): Column name for high prices. Default is 'High'.
            - low_col (str): Column name for low prices. Default is 'Low'.

    Returns:
        tuple: A tuple containing the Williams %R series and a list with the resulting column name.

    Williams %R is calculated using the formula:

        %R = ((Highest High - Close) / (Highest High - Lowest Low)) * -100

    The oscillator ranges from 0 to -100, where values above -20 typically indicate overbought
    conditions and values below -80 indicate oversold conditions.
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

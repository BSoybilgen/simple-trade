import pandas as pd


def eri(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """Calculate the Elder-Ray Index (ERI).

    ERI combines a trend component (EMA) with buying and selling pressure estimates.
    It outputs Bull Power (High - EMA) and Bear Power (Low - EMA) to quantify the
    strength of bulls and bears relative to the prevailing trend.

    Args:
        df (pd.DataFrame): Input OHLC data.
        parameters (dict, optional): Calculation parameters.
            - window (int): Period for the EMA baseline. Default is 13.
        columns (dict, optional): Column overrides.
            - close_col (str): Close price column. Default is 'Close'.
            - high_col (str): High price column. Default is 'High'.
            - low_col (str): Low price column. Default is 'Low'.

    Returns:
        tuple: (eri_dataframe, [bull_col, bear_col])
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}

    window = int(parameters.get('window', 13))
    close_col = columns.get('close_col', 'Close')
    high_col = columns.get('high_col', 'High')
    low_col = columns.get('low_col', 'Low')

    close = df[close_col]
    high = df[high_col]
    low = df[low_col]

    ema = close.ewm(span=window, adjust=False, min_periods=window).mean()

    bull_col = f'ERI_BULL_{window}'
    bear_col = f'ERI_BEAR_{window}'
    result = pd.DataFrame({
        bull_col: high - ema,
        bear_col: low - ema
    })

    columns_list = [bull_col, bear_col]
    return result, columns_list

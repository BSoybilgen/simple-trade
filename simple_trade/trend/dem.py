import pandas as pd


def dem(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Double Exponential Moving Average (DEMA).
    DEMA reduces the lag of traditional EMAs by subtracting the EMA of the EMA from the doubled EMA.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - window (int): The lookback period for the calculation. Default is 20.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the DEMA series and a list of column names.

    The DEMA is calculated as follows:

    1. Calculate EMA1:
       EMA1 = EMA(Close, window)

    2. Calculate EMA2:
       EMA2 = EMA(EMA1, window)

    3. Calculate DEMA:
       DEMA = 2 * EMA1 - EMA2

    Interpretation:
    - DEMA is faster and more responsive than a standard EMA.
    - It aims to reduce the inherent lag of moving averages.

    Use Cases:
    - Trend Following: Identifying trends earlier than SMA/EMA.
    - Crossovers: DEMA crossings can signal entries/exits faster.
    - Support/Resistance: Can act as dynamic support/resistance levels.
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}

    close_col = columns.get('close_col', 'Close')
    window = int(parameters.get('window', 20))

    series = df[close_col]

    ema1 = series.ewm(span=window, adjust=False).mean()
    ema2 = ema1.ewm(span=window, adjust=False).mean()

    dema_series = 2 * ema1 - ema2
    dema_series.name = f'DEMA_{window}'

    columns_list = [dema_series.name]
    return dema_series, columns_list

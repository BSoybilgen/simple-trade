import pandas as pd


def soa(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Smoothed Moving Average (SmMA) of a series.
    The Smoothed Moving Average is similar to Wilder's moving average and applies
    heavier smoothing than a simple moving average. Each new value is a blend of
    the previous SmMA and the current price.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - window (int): The lookback period for the SmMA calculation. Default is 20.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the SmMA series and a list of column names.

    The Smoothed Moving Average is calculated as follows:

    1. Calculate the first SmMA (usually SMA of the first window).

    2. Calculate subsequent SmMAs:
       SmMA(i) = (SmMA(i-1) * (n - 1) + Price(i)) / n
       where n is the window size.

    Interpretation:
    - The SmMA gives recent prices an equal weight to historic prices.
    - It does not respond to price changes as quickly as EMA.
    - Useful for identifying long-term trends.

    Use Cases:
    - Trend Identification: Filtering out short-term noise to see the long-term trend.
    - Support/Resistance: Often acts as a significant dynamic support or resistance level.
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}

    close_col = columns.get('close_col', 'Close')
    window = int(parameters.get('window', 20))

    series = df[close_col]
    smoothed = series.ewm(alpha=1 / window, adjust=False).mean()
    smoothed.name = f'SOA_{window}'

    columns_list = [smoothed.name]

    return smoothed, columns_list

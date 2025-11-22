import pandas as pd

def sma(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Simple Moving Average (SMA) of a series.
    The SMA is a moving average that is calculated by taking the arithmetic
    mean of a given set of values over a specified period.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - window (int): The window size for the SMA calculation. Default is 20.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the SMA series and a list of column names.

    The Simple Moving Average is calculated as follows:

    1. Sum the values over the specified window.
    2. Divide the sum by the window size.
       SMA = Sum(Price, window) / window

    Interpretation:
    - Rising SMA: Uptrend.
    - Falling SMA: Downtrend.
    - Price > SMA: Bullish.
    - Price < SMA: Bearish.

    Use Cases:
    - Identifying trends: The SMA can be used to identify the direction of a
      price trend.
    - Smoothing price data: The SMA can smooth out short-term price fluctuations
      to provide a clearer view of the underlying trend.
    - Generating buy and sell signals: The SMA can be used in crossover systems
      to generate buy and sell signals (e.g. Price crosses SMA).
    - Support/Resistance: Often acts as dynamic support or resistance.
    """
    # Set default values
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    # Extract parameters with defaults
    close_col = columns.get('close_col', 'Close')
    window = int(parameters.get('window', 20))

    series = df[close_col]
    series =series.rolling(window=window).mean()
    series.name = f'SMA_{window}'

    columns_list = [series.name]

    return series, columns_list
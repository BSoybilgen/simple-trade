import pandas as pd


def eri(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Elder-Ray Index (ERI), a technical indicator developed by Dr. Alexander Elder.
    It measures the amount of buying and selling pressure in the market.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - window (int): The period for the EMA baseline. Default is 13.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.
            - high_col (str): The column name for high prices. Default is 'High'.
            - low_col (str): The column name for low prices. Default is 'Low'.

    Returns:
        tuple: A tuple containing the ERI DataFrame (with Bull and Bear power) and a list of column names.

    The Elder-Ray Index consists of three components (calculated as follows):

    1. Calculate the Exponential Moving Average (EMA):
       EMA = EMA(Close, window) (Often a 13-period EMA)

    2. Calculate Bull Power:
       Bull Power = High - EMA

    3. Calculate Bear Power:
       Bear Power = Low - EMA

    Interpretation:
    - Bull Power: Measures the ability of buyers to push prices above the average consensus of value (EMA).
      Positive values indicate strength.
    - Bear Power: Measures the ability of sellers to push prices below the average consensus of value (EMA).
      Negative values indicate weakness.
    - EMA Slope: Indicates the direction of the main trend.

    Use Cases:
    - Trend Following: Buy when the trend is up (EMA rising) and Bear Power is negative but rising.
      Sell when the trend is down (EMA falling) and Bull Power is positive but falling.
    - Divergence: Divergence between prices and Bull/Bear Power can signal reversals.
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

import pandas as pd


def tem(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Triple Exponential Moving Average (TEMA).
    TEMA uses triple smoothing to further reduce lag compared to DEMA and EMA.
    It was developed by Patrick Mulloy.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - window (int): The lookback period for the calculation. Default is 20.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the TEMA series and a list of column names.

    The TEMA is calculated as follows:

    1. Calculate EMA1:
       EMA1 = EMA(Close, window)

    2. Calculate EMA2:
       EMA2 = EMA(EMA1, window)

    3. Calculate EMA3:
       EMA3 = EMA(EMA2, window)

    4. Calculate TEMA:
       TEMA = 3 * EMA1 - 3 * EMA2 + EMA3

    Interpretation:
    - TEMA reacts extremely quickly to price changes.
    - It eliminates the lag associated with single and double EMAs.

    Use Cases:
    - Scalping/Day Trading: Highly responsive indicator for short-term trades.
    - Trend Confirmation: Fast confirmation of new trends.
    - Crossovers: Using TEMA in place of EMA in MACD or other crossover strategies.
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
    ema3 = ema2.ewm(span=window, adjust=False).mean()

    tema_series = 3 * ema1 - 3 * ema2 + ema3
    tema_series.name = f'TEMA_{window}'

    columns_list = [tema_series.name]
    return tema_series, columns_list

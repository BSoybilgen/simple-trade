import pandas as pd


def qst(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Qstick Indicator, a technical indicator developed by Tushar Chande.
    It quantifies the buying and selling pressure by averaging the difference between closing and opening prices.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - window (int): The rolling window length for averaging. Default is 10.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.
            - open_col (str): The column name for opening prices. Default is 'Open'.

    Returns:
        tuple: A tuple containing the Qstick series and a list of column names.

    The Qstick Indicator is calculated as follows:

    1. Calculate Candle Body:
       Body = Close - Open

    2. Calculate Moving Average of Body:
       Qstick = SMA(Body, window)

    Interpretation:
    - Positive Qstick: Buying pressure is dominant (Closes > Opens on average).
    - Negative Qstick: Selling pressure is dominant (Opens > Closes on average).
    - Zero Crossing: Crossing the zero line acts as a signal for trend change.

    Use Cases:
    - Trend Confirmation: Confirming the validity of a trend (e.g., price rising but Qstick falling is a divergence).
    - Signal Generation: Crossovers of the zero line.
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}

    window = int(parameters.get('window', 10))
    close_col = columns.get('close_col', 'Close')
    open_col = columns.get('open_col', 'Open')

    close = df[close_col]
    open_price = df[open_col]
    body = close - open_price

    qstick_values = body.rolling(window=window, min_periods=window).mean()
    qstick_values.name = f'QSTICK_{window}'

    columns_list = [qstick_values.name]
    return qstick_values, columns_list

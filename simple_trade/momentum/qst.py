import pandas as pd


def qst(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """Calculate the Qstick Indicator.

    Qstick averages the difference between closing and opening prices over a
    specified window to highlight bullish or bearish pressure. Positive values
    indicate that closes are generally above opens over the period, while
    negative values indicate the opposite.

    Args:
        df (pd.DataFrame): Input DataFrame containing price data.
        parameters (dict, optional): Calculation parameters.
            - window (int): Rolling window length for averaging. Default is 10.
        columns (dict, optional): Column overrides.
            - close_col (str): Column for closing prices. Default 'Close'.
            - open_col (str): Column for opening prices. Default 'Open'.

    Returns:
        tuple: (qstick_series, [column_name])
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

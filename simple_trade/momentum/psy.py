import pandas as pd


def psy(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Psychological Line (PSY), a ratio of the number of rising periods 
    over the last N periods to the total number of periods.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - window (int): The lookback period N. Default is 12.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the PSY series and a list of column names.

    The Psychological Line is calculated as follows:

    1. Identify Rising Periods:
       Rising = 1 if Close > Prev Close else 0

    2. Sum Rising Periods over Window:
       Sum Rising = Sum(Rising, window)

    3. Calculate PSY:
       PSY = (Sum Rising / window) * 100

    Interpretation:
    - Range: 0 to 100.
    - Equilibrium: 50 indicates a balance between rising and falling periods.
    - Overbought: Values above 75 indicate potential overbought conditions.
    - Oversold: Values below 25 indicate potential oversold conditions.

    Use Cases:
    - Sentiment Analysis: Gauging market sentiment (bullish vs bearish days).
    - Reversal Signals: Extreme values suggest that the current trend may be overextended.
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}

    window = int(parameters.get('window', 12))
    close_col = columns.get('close_col', 'Close')

    close = df[close_col]

    # Rising periods: Close > Prev Close
    is_rising = (close > close.shift(1)).astype(float)

    psy_val = is_rising.rolling(window=window).sum() / window * 100
    psy_val.name = f'PSY_{window}'

    return psy_val, [psy_val.name]

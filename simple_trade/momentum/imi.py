import pandas as pd


def imi(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Intraday Momentum Index (IMI), a technical indicator developed by Tushar Chande 
    that combines candlestick analysis with the Relative Strength Index (RSI).

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - window (int): The lookback period for calculation. Default is 14.
        columns (dict, optional): Dictionary containing column name mappings:
            - open_col (str): The column name for open prices. Default is 'Open'.
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the IMI series and a list of column names.

    The Intraday Momentum Index is calculated as follows:

    1. Calculate Intraday Gains and Losses:
       If Close > Open: Gain = Close - Open, Loss = 0
       If Close < Open: Loss = Open - Close, Gain = 0

    2. Sum Gains and Losses over the Window:
       Sum Gains = Sum(Gain, window)
       Sum Losses = Sum(Loss, window)

    3. Calculate IMI:
       IMI = (Sum Gains / (Sum Gains + Sum Losses)) * 100

    Interpretation:
    - Range: 0 to 100.
    - Overbought: Values above 70 indicate potential overbought conditions.
    - Oversold: Values below 30 indicate potential oversold conditions.
    - Momentum: High IMI suggests strong buying pressure within the day (white/green candles dominate).
      Low IMI suggests strong selling pressure (black/red candles dominate).

    Use Cases:
    - Reversal Signals: Identifying overbought/oversold levels for potential reversals.
    - Confirmation: Using IMI to confirm support/resistance levels.
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}

    window = int(parameters.get('window', 14))
    open_col = columns.get('open_col', 'Open')
    close_col = columns.get('close_col', 'Close')

    open_vals = df[open_col]
    close_vals = df[close_col]

    # Calculate gains (up candles) and losses (down candles)
    # Gain = Close - Open (if Close > Open)
    # Loss = Open - Close (if Close < Open)

    diff = close_vals - open_vals
    gains = diff.where(diff > 0, 0)
    losses = -diff.where(diff < 0, 0)

    sum_gains = gains.rolling(window=window).sum()
    sum_losses = losses.rolling(window=window).sum()

    imi_val = (sum_gains / (sum_gains + sum_losses)) * 100
    imi_val.name = f'IMI_{window}'

    return imi_val, [imi_val.name]

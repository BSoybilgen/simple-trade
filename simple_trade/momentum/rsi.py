import pandas as pd

def rsi(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Relative Strength Index (RSI), a momentum indicator used in technical analysis.
    It measures the magnitude of recent price changes to evaluate overbought or oversold conditions.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - window (int): The window size for the RSI calculation. Default is 14.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the RSI series and a list of column names.

    Calculation Steps:
    1. Calculate the difference between consecutive values in the series (Diff).
    2. Separate gains (Diff > 0) and losses (Diff < 0, as positive).
    3. Calculate the Average Gain and Average Loss over the specified window (Smoothed).
    4. Calculate the Relative Strength (RS):
       RS = Average Gain / Average Loss
    5. Calculate the RSI:
       RSI = 100 - (100 / (1 + RS))

    Interpretation:
    - Range: 0 to 100.
    - Overbought: Values above 70 are often interpreted as overbought.
    - Oversold: Values below 30 are often interpreted as oversold.
    - Centerline: 50 acts as a neutral level.

    Use Cases:
    - Identifying overbought and oversold conditions: Potential reversal zones.
    - Identifying trend direction: RSI > 50 generally indicates uptrend, < 50 downtrend.
    - Generating buy and sell signals: Divergences between the RSI and price (e.g. Price higher high, RSI lower high).
    - Failure Swings: Specific patterns in RSI that signal reversals.
    """
    # Set default values
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    # Extract parameters with defaults
    window = int(parameters.get('window', 14))
    close_col = columns.get('close_col', 'Close')
    
    series = df[close_col]
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    rsi.name = f'RSI_{window}'
    columns_list = [rsi.name]
    return rsi, columns_list
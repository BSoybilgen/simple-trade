import pandas as pd

def rsi(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Relative Strength Index (RSI) of a series.

    The RSI is a momentum indicator used in technical analysis that measures the magnitude of recent price changes to evaluate overbought or oversold conditions in the price of a stock or other asset.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - window (int): The window size for the RSI calculation. Default is 14.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the RSI series and a list of column names.

    The RSI is calculated as follows:

    1. Calculate the difference between consecutive values in the series.
    2. Calculate the average gain and average loss over the specified window.
    3. Calculate the relative strength (RS) by dividing the average gain by the average loss.
    4. Calculate the RSI using the formula: RSI = 100 - (100 / (1 + RS)).

    Use Cases:

    - Identifying overbought and oversold conditions: RSI values above 70 are often interpreted as overbought, while values below 30 are often interpreted as oversold.
    - Identifying trend direction: The RSI can be used to confirm the direction of a price trend.
    - Generating buy and sell signals: Divergences between the RSI and price can be used to generate buy and sell signals.
    """
    # Set default values
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    # Extract parameters with defaults
    window = parameters.get('window', 14)
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
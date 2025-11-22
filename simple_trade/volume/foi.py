import pandas as pd


def foi(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Force Index (FI), an indicator that uses price and volume
    to assess the power behind a price move and identify potential turning points.
    It combines price change, extent of price change, and trading volume.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - period (int): The EMA smoothing period. Default is 13.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.
            - volume_col (str): The column name for volume. Default is 'Volume'.

    Returns:
        tuple: A tuple containing the Force Index series and a list of column names.

    The Force Index is calculated as follows:

    1. Calculate Raw Force Index:
       Raw FI = (Current Close - Previous Close) * Current Volume

    2. Calculate Smoothed Force Index:
       FI = EMA(Raw FI, period)

    Interpretation:
    - Positive FI: Buying pressure dominates (Bulls are in control).
    - Negative FI: Selling pressure dominates (Bears are in control).
    - Zero Line Crossovers: Signal trend changes.

    Use Cases:
    - Trend Confirmation: Positive FI confirms uptrend; negative FI confirms downtrend.
    - Divergence: Price making new highs/lows while FI fails to do so signals weakness.
    - Entry Signals: Buying on negative spikes in uptrends (pullbacks).
    - Breakout Validation: High volume moves produce large FI spikes.
    """
    # Set default values
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    # Extract parameters with defaults
    period = parameters.get('period', 13)
    close_col = columns.get('close_col', 'Close')
    volume_col = columns.get('volume_col', 'Volume')
    
    close = df[close_col]
    volume = df[volume_col]
    
    # Calculate price change
    price_change = close.diff()
    
    # Calculate Raw Force Index
    raw_force_index = price_change * volume
    
    # Apply EMA smoothing
    force_index = raw_force_index.ewm(span=period, adjust=False).mean()
    
    force_index.name = f'FI_{period}'
    columns_list = [force_index.name]
    return force_index, columns_list

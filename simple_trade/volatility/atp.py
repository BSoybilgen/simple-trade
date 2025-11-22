import pandas as pd


def atp(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Average True Range Percent (ATRP), which expresses the Average True Range
    as a percentage of the closing price. This normalization allows for comparison of volatility
    across different assets and price levels.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - window (int): The lookback period for ATR calculation. Default is 14.
        columns (dict, optional): Dictionary containing column name mappings:
            - high_col (str): The column name for high prices. Default is 'High'.
            - low_col (str): The column name for low prices. Default is 'Low'.
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the ATRP series and a list of column names.

    Calculation Steps:
    1. Calculate the True Range (TR) for each period:
       TR = max(high - low, abs(high - prev_close), abs(low - prev_close))
    2. Calculate the Average True Range (ATR):
       Smoothed average of TR over the specified window (typically using Wilder's smoothing).
    3. Convert ATR to percentage of closing price:
       ATRP = (ATR / Close) * 100

    Interpretation:
    - Low ATRP (<2%): Low volatility, stable price movements.
    - Medium ATRP (2-5%): Normal volatility, typical market conditions.
    - High ATRP (5-10%): Elevated volatility, increased price swings.
    - Very High ATRP (>10%): Extreme volatility, highly unstable market.

    Use Cases:
    - Cross-asset comparison: Compare volatility across assets with different prices.
    - Position sizing: Normalize position sizes across different assets based on relative volatility.
    - Stop-loss placement: Set percentage-based stops using ATRP multiples.
    - Volatility screening: Screen for low or high volatility assets.
    """
    # Set default values
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    # Extract parameters with defaults
    window = int(parameters.get('window', 14))
    high_col = columns.get('high_col', 'High')
    low_col = columns.get('low_col', 'Low')
    close_col = columns.get('close_col', 'Close')
    
    high = df[high_col]
    low = df[low_col]
    close = df[close_col]
    
    # Calculate True Range
    prev_close = close.shift(1)
    tr1 = high - low  # Current high - current low
    tr2 = (high - prev_close).abs()  # Current high - previous close
    tr3 = (low - prev_close).abs()  # Current low - previous close
    
    # True Range is the maximum of the three calculations
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calculate ATR using Wilder's smoothing method
    atr_values = pd.Series(index=close.index, dtype=float)
    
    # First ATR value is simple average of first 'window' TRs
    first_atr = tr.iloc[:window].mean()
    
    # Use the first value to start the smoothing process
    atr_values.iloc[window-1] = first_atr
    
    # Apply Wilder's smoothing method for the rest of the values
    for i in range(window, len(close)):
        atr_values.iloc[i] = ((atr_values.iloc[i-1] * (window-1)) + tr.iloc[i]) / window
    
    # Convert ATR to percentage of closing price
    atrp_values = (atr_values / close) * 100
    
    atrp_values.name = f'ATRP_{window}'
    columns_list = [atrp_values.name]
    return atrp_values, columns_list

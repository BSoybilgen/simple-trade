import pandas as pd


def rvi(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Relative Volatility Index (RVI), a volatility-based momentum indicator that
    measures the direction of volatility. It applies the RSI formula to standard deviation
    instead of price changes.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - window (int): The lookback period for standard deviation. Default is 10.
            - rvi_period (int): The period for RVI smoothing (similar to RSI period). Default is 14.
            - ddof (int): Delta Degrees of Freedom for std calculation. Default is 0.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the RVI series and a list of column names.

    Calculation Steps:
    1. Calculate Rolling Standard Deviation (over window).
    2. Separate Standard Deviations:
       - Upward Std: Std Dev when Price > Previous Price.
       - Downward Std: Std Dev when Price <= Previous Price.
    3. Smooth Standard Deviations (EMA over rvi_period).
    4. Calculate RVI:
       RVI = 100 * (Avg Upward Std) / (Avg Upward Std + Avg Downward Std)

    Interpretation:
    - RVI > 50: Volatility is associated with rising prices (Bullish).
    - RVI < 50: Volatility is associated with falling prices (Bearish).
    - Overbought: > 70 (or 80).
    - Oversold: < 30 (or 20).

    Use Cases:
    - Trend confirmation: RVI direction confirms price trend.
    - Divergence detection.
    - Entry/exit signals (crossovers).
    - Volatility direction analysis.
    """
    # Set default values
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    # Extract parameters with defaults
    window = int(parameters.get('window', 10))
    rvi_period = int(parameters.get('rvi_period', 14))
    ddof = int(parameters.get('ddof', 0))
    close_col = columns.get('close_col', 'Close')
    
    close = df[close_col]
    
    # Calculate rolling standard deviation
    std_values = close.rolling(window=window).std(ddof=ddof)
    
    # Determine price direction (up or down)
    price_change = close.diff()
    
    # Separate std into upward and downward components
    upward_std = pd.Series(index=close.index, dtype=float)
    downward_std = pd.Series(index=close.index, dtype=float)
    
    upward_std[price_change > 0] = std_values[price_change > 0]
    downward_std[price_change <= 0] = std_values[price_change <= 0]
    
    # Fill remaining NaN values with 0 for EMA calculation
    upward_std = upward_std.fillna(0.0)
    downward_std = downward_std.fillna(0.0)
    
    # Calculate EMA of upward and downward std
    # Using span parameter: span = rvi_period corresponds to N-period EMA
    # Set min_periods to ensure we have enough data before starting EMA
    avg_upward_std = upward_std.ewm(span=rvi_period, adjust=False, min_periods=rvi_period).mean()
    avg_downward_std = downward_std.ewm(span=rvi_period, adjust=False, min_periods=rvi_period).mean()
    
    # Calculate RVI using RSI formula
    # RVI = 100 * (avg_upward_std / (avg_upward_std + avg_downward_std))
    rvi_values = 100 * avg_upward_std / (avg_upward_std + avg_downward_std)
    
    rvi_values.name = f'RVI_{window}_{rvi_period}'
    columns_list = [rvi_values.name]
    return rvi_values, columns_list

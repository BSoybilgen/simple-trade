import pandas as pd


def dvi(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Dynamic Volatility Indicator (DVI), a composite indicator that combines
    multiple volatility measures and price momentum to create a normalized oscillator that
    identifies overbought/oversold conditions based on volatility-adjusted price movements.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - magnitude_period (int): Period for magnitude calculation (price position). Default is 5.
            - stretch_period (int): Period for stretch calculation (consecutive moves). Default is 100.
            - smooth_period (int): Period for final smoothing. Default is 3.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the DVI series and a list of column names.

    Calculation Steps:
    1. Calculate Magnitude Component:
       - Ratio = Close / SMA(Close, magnitude_period)
       - Normalize to 0-1 rank over stretch_period.

    2. Calculate Stretch Component:
       - Net consecutive up/down days count.
       - Normalize to 0-1 rank over stretch_period.

    3. Combine Components:
       - DVI = 0.5 * Magnitude + 0.5 * Stretch

    4. Apply Smoothing:
       - Final DVI = SMA(DVI, smooth_period) * 100

    Interpretation:
    - DVI < 30: Oversold (potential buy).
    - DVI > 70: Overbought (potential sell).
    - DVI near 50: Neutral.

    Use Cases:
    - Mean reversion trading: Counter-trend entries at extremes.
    - Divergence detection: Reversals.
    - Trend filtering: Confirming entries in direction of larger trend.
    """
    # Set default values
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    # Extract parameters with defaults
    magnitude_period = int(parameters.get('magnitude_period', 5))
    stretch_period = int(parameters.get('stretch_period', 100))
    smooth_period = int(parameters.get('smooth_period', 3))
    close_col = columns.get('close_col', 'Close')
    
    close = df[close_col]
    
    # Calculate Magnitude component
    # Ratio of current price to its moving average
    sma = close.rolling(window=magnitude_period).mean()
    magnitude_ratio = close / sma
    
    # Normalize magnitude using percentile rank over stretch_period
    def percentile_rank(series, period):
        """Calculate percentile rank of current value within rolling window"""
        result = pd.Series(index=series.index, dtype=float)
        for i in range(period - 1, len(series)):
            window = series.iloc[max(0, i - period + 1):i + 1]
            current_value = series.iloc[i]
            if len(window) > 0:
                rank = (window < current_value).sum() / len(window)
                result.iloc[i] = rank
        return result
    
    magnitude_normalized = percentile_rank(magnitude_ratio, stretch_period)
    
    # Calculate Stretch component
    # Count consecutive up/down moves
    price_change = close.diff()
    
    # Calculate consecutive up days
    consecutive_up = pd.Series(0, index=close.index, dtype=int)
    consecutive_down = pd.Series(0, index=close.index, dtype=int)
    
    up_count = 0
    down_count = 0
    
    for i in range(1, len(close)):
        if price_change.iloc[i] > 0:
            up_count += 1
            down_count = 0
        elif price_change.iloc[i] < 0:
            down_count += 1
            up_count = 0
        else:
            # No change, maintain previous counts
            pass
        
        consecutive_up.iloc[i] = up_count
        consecutive_down.iloc[i] = down_count
    
    # Calculate stretch score (negative for down moves)
    stretch_score = consecutive_up - consecutive_down
    
    # Normalize stretch using percentile rank over stretch_period
    stretch_normalized = percentile_rank(stretch_score, stretch_period)
    
    # Combine Magnitude and Stretch (equal weighting)
    dvi_raw = 0.5 * magnitude_normalized + 0.5 * stretch_normalized
    
    # Apply smoothing
    dvi_smoothed = dvi_raw.rolling(window=smooth_period).mean()
    
    # Scale to 0-100 range
    dvi_values = dvi_smoothed * 100
    
    dvi_values.name = f'DVI_{magnitude_period}_{stretch_period}_{smooth_period}'
    columns_list = [dvi_values.name]
    return dvi_values, columns_list

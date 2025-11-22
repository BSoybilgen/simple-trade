import pandas as pd


def ads(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Adaptive Deviation-Scaled Moving Average (ADSMA).
    ADSMA adjusts its smoothing factor based on price changes relative to the moving average.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - window (int): Base period for EMA calculation. Default is 20.
            - sensitivity (float): Multiplier for price change impact. Default is 2.0.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the ADSMA series and a list of column names.

    The ADSMA is calculated as follows:

    1. Calculate Price Change Ratio:
       Ratio = Abs(Price - Previous ADSMA) / Previous ADSMA

    2. Scale the Ratio:
       Scaled Change = Ratio * sensitivity

    3. Calculate Adaptive Alpha:
       Base Alpha = 2 / (window + 1)
       Adaptive Alpha = Base Alpha * (1 + Scaled Change)
       (Capped at 1.0)

    4. Calculate ADSMA:
       ADSMA = (Adaptive Alpha * Price) + ((1 - Adaptive Alpha) * Previous ADSMA)

    Interpretation:
    - When price changes are large, ADSMA becomes more responsive (higher alpha) to capture the move.
    - When price changes are small (consolidation), ADSMA becomes smoother (lower alpha) to avoid noise.

    Use Cases:
    - Trend Following: Filters out noise during consolidation while reacting quickly to breakouts.
    - Dynamic Support/Resistance: Acts as a support line in uptrends and resistance in downtrends.
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}

    close_col = columns.get('close_col', 'Close')
    window = int(parameters.get('window', 20))
    sensitivity = float(parameters.get('sensitivity', 2.0))

    series = df[close_col].copy()
    
    # Initialize ADSMA with first value
    adsma = pd.Series(index=series.index, dtype=float)
    adsma.iloc[0] = series.iloc[0]
    
    # Calculate adaptive EMA with dynamic alpha based on price changes
    for i in range(1, len(series)):
        if pd.notna(series.iloc[i]) and pd.notna(adsma.iloc[i-1]):
            # Calculate price change ratio
            price_change = abs(series.iloc[i] - adsma.iloc[i-1]) / (adsma.iloc[i-1] + 1e-10)
            
            # Scale by sensitivity
            scaled_change = price_change * sensitivity
            
            # Convert to alpha: larger changes = higher alpha (more responsive)
            # Smaller changes = lower alpha (more smoothing)
            alpha = 2.0 / (window + 1)  # Base EMA alpha
            adaptive_alpha = alpha * (1.0 + scaled_change)
            adaptive_alpha = min(adaptive_alpha, 1.0)  # Cap at 1.0
            
            # Apply adaptive EMA formula
            adsma.iloc[i] = adaptive_alpha * series.iloc[i] + (1 - adaptive_alpha) * adsma.iloc[i-1]
        else:
            adsma.iloc[i] = series.iloc[i]
    
    adsma.name = f'ADSMA_{window}'
    
    columns_list = [adsma.name]
    return adsma, columns_list

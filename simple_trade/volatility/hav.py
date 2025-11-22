import pandas as pd


def hav(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Heikin-Ashi Volatility (HAV), a volatility indicator that applies
    the Heikin-Ashi smoothing technique to price data and then measures the volatility
    of the smoothed candles to filter out market noise.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - period (int): The lookback period for volatility calculation. Default is 14.
            - method (str): Volatility calculation method - 'atr' or 'std'. Default is 'atr'.
        columns (dict, optional): Dictionary containing column name mappings:
            - open_col (str): The column name for open prices. Default is 'Open'.
            - high_col (str): The column name for high prices. Default is 'High'.
            - low_col (str): The column name for low prices. Default is 'Low'.
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the HAV series and a list of column names.

    Calculation Steps:
    1. Calculate Heikin-Ashi candles:
       HA_Close = (Open + High + Low + Close) / 4
       HA_Open = (Previous HA_Open + Previous HA_Close) / 2
       HA_High = max(High, HA_Open, HA_Close)
       HA_Low = min(Low, HA_Open, HA_Close)

    2. Calculate Volatility on Heikin-Ashi candles:
       
       If method = 'atr':
       - Calculate True Range on HA candles:
         TR = max(HA_High - HA_Low, abs(HA_High - prev_HA_Close), abs(HA_Low - prev_HA_Close))
       - Apply Wilder's smoothing:
         HAV = smoothed average of TR over period
       
       If method = 'std':
       - Calculate standard deviation of HA_Close over period:
         HAV = std(HA_Close, period)

    Interpretation:
    - Lower HAV: Low volatility, potential consolidation.
    - Higher HAV: High volatility, potential trending conditions.
    - Rising HAV: Increasing volatility, potential breakout.
    - Falling HAV: Decreasing volatility, potential consolidation.

    Use Cases:
    - Trend identification: High HAV indicates active trends.
    - Breakout detection: Sharp increases in HAV.
    - Noise reduction: Smoother than standard ATR.
    - Volatility compression: Very low HAV precedes explosive moves.
    - Position sizing: Use HAV to adjust position sizes based on current market volatility levels.
    - Market regime filtering: Filter trades based on HAV levels - avoid trading during extremely high or low volatility periods.
    - Smoother signals: HAV provides cleaner volatility signals compared to regular ATR.
    """
    # Set default values
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    # Extract parameters with defaults
    period = int(parameters.get('period', 14))
    method = parameters.get('method', 'atr').lower()
    open_col = columns.get('open_col', 'Open')
    high_col = columns.get('high_col', 'High')
    low_col = columns.get('low_col', 'Low')
    close_col = columns.get('close_col', 'Close')
    
    open_price = df[open_col]
    high = df[high_col]
    low = df[low_col]
    close = df[close_col]
    
    # Calculate Heikin-Ashi candles
    ha_close = (open_price + high + low + close) / 4
    
    # Initialize HA_Open
    ha_open = pd.Series(index=close.index, dtype=float)
    ha_open.iloc[0] = (open_price.iloc[0] + close.iloc[0]) / 2
    
    # Calculate HA_Open iteratively
    for i in range(1, len(close)):
        ha_open.iloc[i] = (ha_open.iloc[i-1] + ha_close.iloc[i-1]) / 2
    
    # Calculate HA_High and HA_Low
    ha_high = pd.concat([high, ha_open, ha_close], axis=1).max(axis=1)
    ha_low = pd.concat([low, ha_open, ha_close], axis=1).min(axis=1)
    
    # Calculate volatility based on method
    if method == 'atr':
        # Calculate True Range on Heikin-Ashi candles
        prev_ha_close = ha_close.shift(1)
        tr1 = ha_high - ha_low
        tr2 = (ha_high - prev_ha_close).abs()
        tr3 = (ha_low - prev_ha_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Apply Wilder's smoothing method
        hav_values = pd.Series(index=close.index, dtype=float)
        first_hav = tr.iloc[:period].mean()
        hav_values.iloc[period-1] = first_hav
        
        for i in range(period, len(close)):
            hav_values.iloc[i] = ((hav_values.iloc[i-1] * (period-1)) + tr.iloc[i]) / period
    
    elif method == 'std':
        # Calculate standard deviation of HA_Close
        hav_values = ha_close.rolling(window=period).std()
    
    else:
        raise ValueError(f"Invalid method '{method}'. Must be 'atr' or 'std'.")
    
    hav_values.name = f'HAV_{period}_{method.upper()}'
    columns_list = [hav_values.name]
    return hav_values, columns_list
